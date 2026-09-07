//! Is per-read `Instant::now()` free? cargo run --release --example clock_tax
//!
//! Measures vDSO clock_gettime alone, then the store read path with and
//! without a TTL-bearing entry, to see what the expiry check actually costs.
use rambudget::{BudgetStore, Clock, Cost, SystemClock};
use std::time::{Duration, Instant};

struct FixedCost;
impl Cost<u64, u64> for FixedCost {
    #[inline]
    fn entry_cost(_k: &u64, _v: &u64) -> usize {
        16
    }
}

/// Same store, but `now()` is a constant — isolates the clock syscall cost.
#[derive(Clone, Copy)]
struct FrozenClock(Instant);
impl Clock for FrozenClock {
    #[inline]
    fn now(&self) -> Instant {
        self.0
    }
}

fn main() {
    let n = 5_000_000u64;

    // 1) raw clock
    let t = Instant::now();
    let mut black = Instant::now();
    for _ in 0..n {
        black = Instant::max(black, Instant::now());
    }
    let d = t.elapsed();
    println!(
        "Instant::now()      : {:.1} ns/call  ({:.2}M/s) {:?}",
        d.as_nanos() as f64 / n as f64,
        n as f64 / d.as_secs_f64() / 1e6,
        black
    );

    // 2) store read, SystemClock (real clock per read), entry has TTL
    let s: BudgetStore<u64, u64, FixedCost, SystemClock> = BudgetStore::new(1 << 30);
    s.put(1, 42).unwrap();
    let t = Instant::now();
    let mut hits = 0u64;
    for _ in 0..n {
        hits += s.get(&1).is_some() as u64;
    }
    let d = t.elapsed();
    println!(
        "get, SystemClock    : {:.1} ns/call  hits={} ({:.2}M/s)",
        d.as_nanos() as f64 / n as f64,
        hits,
        n as f64 / d.as_secs_f64() / 1e6
    );

    // 3) same, frozen clock (no syscall) — the difference IS the clock tax
    let f: BudgetStore<u64, u64, FixedCost, FrozenClock> =
        BudgetStore::with_clock(1 << 30, FrozenClock(Instant::now()));
    f.put(1, 42).unwrap();
    let t = Instant::now();
    let mut hits2 = 0u64;
    for _ in 0..n {
        hits2 += f.get(&1).is_some() as u64;
    }
    let d = t.elapsed();
    println!(
        "get, FrozenClock    : {:.1} ns/call  hits={} ({:.2}M/s)",
        d.as_nanos() as f64 / n as f64,
        hits2,
        n as f64 / d.as_secs_f64() / 1e6
    );

    // 4) the no-TTL fast path: expiry check should be a branch, not a syscall
    //    if we short-circuit on ttl_count == 0
    let t = Instant::now();
    let mut hits3 = 0u64;
    for _ in 0..n {
        hits3 += f.get(&1).is_some() as u64;
    }
    let d = t.elapsed();
    println!(
        "get, no-ttl-store   : {:.1} ns/call  hits={}",
        d.as_nanos() as f64 / n as f64,
        hits3
    );
    let _ = Duration::from_secs(1);
}
