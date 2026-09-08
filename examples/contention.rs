//! Contention ceiling measurement: cargo run --release --example contention
//!
//! Sweeps thread count on the two global serialization points
//! (reserved CAS + queue mutex) so scaling numbers are measured, not claimed.
use rambudget::{BudgetStore, Cost};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

struct FixedCost;
impl Cost<u64, u64> for FixedCost {
    #[inline]
    fn entry_cost(_k: &u64, _v: &u64) -> usize {
        16
    }
}

fn main() {
    let ops_per_thread = 250_000u64;
    let cap = ops_per_thread as usize * 16; // sized so churn can't starve on budget
    println!(
        "{:<6} {:>14} {:>14}",
        "threads", "put Mops/s", "per-core Mops/s"
    );
    let mut prev = 0f64;
    for threads in [1usize, 2, 4, 8, 16] {
        let s = Arc::new(BudgetStore::<u64, u64, FixedCost>::new(cap));
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::new();
        let t0 = Instant::now();
        for _ in 0..threads {
            let s = Arc::clone(&s);
            let c = Arc::clone(&counter);
            handles.push(std::thread::spawn(move || {
                let base = c.fetch_add(ops_per_thread, AtomicOrdering::Relaxed);
                for i in 0..ops_per_thread {
                    let _ = s.put(base + i, i); // unique keys → queue + CAS path
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let secs = t0.elapsed().as_secs_f64();
        let total = ops_per_thread as f64 * threads as f64;
        let mops = total / secs / 1e6;
        let scaling = if prev > 0.0 {
            format!(" (x{:.1} vs 1T)", mops / prev)
        } else {
            String::new()
        };
        prev = mops;
        println!(
            "{:<6} {:>10.2}    {:>10.2}{}",
            threads,
            mops,
            mops / threads as f64,
            scaling
        );
    }
    // sanity invariants after all that racing
    let s: BudgetStore<u64, u64, FixedCost> = BudgetStore::new(0);
    assert_eq!(s.denials(), 0);
    let _ = s;
}
