//! throughput sanity: cargo run --release --example bench
//! (std-only; criterion-class stats not needed to compare designs)
use rambudget::{BudgetStore, Cost};
use std::time::Instant;

struct KeyCost;
impl Cost<u64, [u8; 64]> for KeyCost {
    #[inline]
    fn entry_cost(_key: &u64, _value: &[u8; 64]) -> usize {
        8 + 64
    }
}

fn main() {
    let n_ops = 2_000_000u64;
    let cap = 1_000_000usize; // ~15k entries resident, steady evict churn
    let s: BudgetStore<u64, [u8; 64], KeyCost> = BudgetStore::new(cap);
    let v = [0u8; 64];

    // 1) admission churn (put + evict pressure path)
    let t = Instant::now();
    let mut stored = 0u64;
    for i in 0..n_ops {
        if s.put(i, v).is_ok() {
            stored += 1;
        } else {
            s.evict(64);
        }
        if i % 4096 == 0 {
            s.evict(16);
        }
    }
    let d = t.elapsed();
    println!(
        "put churn : {:.2}M ops/s  stored={} resident={} reserved<=cap: {}",
        n_ops as f64 / d.as_secs_f64() / 1e6,
        stored,
        s.len(),
        s.reserved() <= s.capacity()
    );

    // 2) read hot path
    let t = Instant::now();
    let mut hits = 0u64;
    for i in 0..n_ops {
        hits += s.get(&(i % n_ops)).is_some() as u64;
    }
    let d = t.elapsed();
    println!(
        "get       : {:.2}M ops/s  hits={}",
        n_ops as f64 / d.as_secs_f64() / 1e6,
        hits
    );

    // 3) single-thread batched replace (no contention, CAS-free shrink path)
    let s2: BudgetStore<u64, [u8; 64], KeyCost> = BudgetStore::new(usize::MAX);
    for i in 0..100_000u64 {
        s2.put(i, v).unwrap();
    }
    let t = Instant::now();
    for _ in 0..500_000u64 {
        for i in 0..100 {
            s2.put(i, v).unwrap(); // same cost → need>=old delta-0 path
        }
    }
    let d = t.elapsed();
    println!(
        "replace   : {:.2}M ops/s",
        500_000.0 * 100.0 / d.as_secs_f64() / 1e6
    );
}
