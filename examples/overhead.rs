//! What is the budget actually costing? cargo run --release --example overhead
//!
//! Same workload three ways, so "is rambudget fast?" becomes a ratio:
//!   1. raw DashMap insert (floor — no accounting, no gate)
//!   2. rambudget put (accounting + CAS admission gate)
//!   3. Mutex<HashMap> (the "obvious" safe alternative)
use dashmap::DashMap;
use rambudget::{BudgetStore, Cost};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

struct FixedCost;
impl Cost<u64, [u8; 32]> for FixedCost {
    #[inline]
    fn entry_cost(_k: &u64, _v: &[u8; 32]) -> usize {
        40
    }
}

const N: u64 = 2_000_000;

fn main() {
    let v = [0u8; 32];

    // single-thread: pure per-op cost
    let raw = DashMap::<u64, [u8; 32]>::new();
    let t = Instant::now();
    for i in 0..N {
        raw.insert(i, v);
    }
    let d_raw = t.elapsed();

    let s: BudgetStore<u64, [u8; 32], FixedCost> = BudgetStore::new(usize::MAX);
    let t = Instant::now();
    for i in 0..N {
        let _ = s.put(i, v);
    }
    let d_rb = t.elapsed();

    let m = Mutex::new(HashMap::<u64, [u8; 32]>::new());
    let t = Instant::now();
    for i in 0..N {
        m.lock().unwrap().insert(i, v);
    }
    let d_mu = t.elapsed();

    let rate = |d: std::time::Duration| N as f64 / d.as_secs_f64() / 1e6;
    println!(
        "1T   raw DashMap    : {:>6.2} Mops/s  ({:>5.1} ns/op)  <- floor",
        rate(d_raw),
        d_raw.as_nanos() as f64 / N as f64
    );
    println!(
        "1T   rambudget put   : {:>6.2} Mops/s  ({:>5.1} ns/op)  {:.1}% of floor",
        rate(d_rb),
        d_rb.as_nanos() as f64 / N as f64,
        rate(d_rb) / rate(d_raw) * 100.0
    );
    println!(
        "1T   Mutex<HashMap>  : {:>6.2} Mops/s  ({:>5.1} ns/op)",
        rate(d_mu),
        d_mu.as_nanos() as f64 / N as f64
    );
    println!(
        "     budget overhead : {:.1} ns/op",
        (d_rb.as_nanos() - d_raw.as_nanos()) as f64 / N as f64
    );

    // 8-thread: where sharding vs global-mutex designs diverge
    let n_threads = 8;
    let per = N / n_threads as u64;
    let t = Instant::now();
    let raw = std::sync::Arc::new(DashMap::<u64, [u8; 32]>::new());
    let hs: Vec<_> = (0..n_threads)
        .map(|th| {
            let raw = std::sync::Arc::clone(&raw);
            std::thread::spawn(move || {
                let base = th as u64 * per;
                for i in 0..per {
                    raw.insert(base + i, v);
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let d_raw8 = t.elapsed();

    let t = Instant::now();
    let s = std::sync::Arc::new(BudgetStore::<u64, [u8; 32], FixedCost>::new(usize::MAX));
    let hs: Vec<_> = (0..n_threads)
        .map(|th| {
            let s = std::sync::Arc::clone(&s);
            std::thread::spawn(move || {
                let base = th as u64 * per;
                for i in 0..per {
                    let _ = s.put(base + i, v);
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let d_rb8 = t.elapsed();

    let t = Instant::now();
    let m = std::sync::Arc::new(Mutex::new(HashMap::<u64, [u8; 32]>::new()));
    let hs: Vec<_> = (0..n_threads)
        .map(|th| {
            let m = std::sync::Arc::clone(&m);
            std::thread::spawn(move || {
                let base = th as u64 * per;
                for i in 0..per {
                    m.lock().unwrap().insert(base + i, v);
                }
            })
        })
        .collect();
    for h in hs {
        h.join().unwrap();
    }
    let d_mu8 = t.elapsed();

    let total = N as f64 / n_threads as f64 * n_threads as f64;
    let rate8 = |d: std::time::Duration| total / d.as_secs_f64() / 1e6;
    println!();
    println!("8T   raw DashMap    : {:>6.2} Mops/s", rate8(d_raw8));
    println!(
        "8T   rambudget put   : {:>6.2} Mops/s  ({:.1}% of floor)",
        rate8(d_rb8),
        rate8(d_rb8) / rate8(d_raw8) * 100.0
    );
    println!(
        "8T   Mutex<HashMap>  : {:>6.2} Mops/s  (the 'obvious safe' design — one global lock)",
        rate8(d_mu8)
    );
}
