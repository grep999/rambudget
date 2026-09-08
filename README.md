# rambudget

> **A RAM store that degrades by policy, not by kernel.**

Every in-memory cache eventually re-implements memory safety badly — a size
counter, a global mutex, or nothing at all — and ends in an OOMKill.
`rambudget` moves admission control *into* the data structure: each insert
reserves its measured bytes under a hard cap through a CAS gate, budget
exhaustion **denies** the insert — your pair comes back untouched with the
reason, the process stays up — expired-but-unreclaimed entries stay charged
(time cannot forge a refund), and the degradation order is your policy behind
a seam.

This is the semantics of `OOM_SCORE_ADJ`, cgroups, and Redis `maxmemory` at
the only layer that sees real per-entry costs: yours. The rules are extracted
from [RamShield](https://github.com/grep999/ramshield)'s `BoundedEventStore`,
a store whose semantics were hardened where memory exhaustion is an attack
surface, not an accident. The category is **RAM governance at the library
layer**; the unique part is shipping it as a two-dependency,
`forbid(unsafe_code)`, ~117 ns-tax library.

---

## The idea in 30 seconds

> OOM is not an accident. It is a missing policy — and the store is where
> that policy belongs.

```
   your cache today              with rambudget
   ────────────────              ─────────────────────────
   grow, grow, grow              reserve → charge → admit
   kernel SIGKILL                or DENY (pair returned, alive)
   crash, or silent loss         pressure() + denials() observable
```

## Quick start — six lines in

```rust
use rambudget::{BudgetStore, Cost, InsertOutcome};

struct ByteCost;
impl Cost<String, Vec<u8>> for ByteCost {
    fn entry_cost(k: &String, v: &Vec<u8>) -> usize { k.len() + v.capacity() }
}

let payload = vec![0u8; 128];
let store: BudgetStore<String, Vec<u8>, ByteCost> = BudgetStore::new(1 << 30); // 1 GiB cap

match store.put("session:42".into(), payload) {
    Ok(InsertOutcome::Stored { charged }) => {
        // admitted: `charged` bytes now reserved
    }
    Ok(InsertOutcome::Replaced { freed, charged }) => {
        // old budget refunded, new one gated
    }
    Err(denied) => {
        // the budget said no — you decide what that means
        let (key, value) = denied.into_pair();
    }
}
```

No allocations you didn't ask for. No lock you can see. `Result` is the only
error path.

---

## Guarantees

| # | Rule |
|---|------|
| 1 | `reserved()` never exceeds `capacity()` — unless you lower the cap below usage, which is a visible state (`pressure() > 1.0`), never a surprise. |
| 2 | `reserved() == Σ cost(live entries)` at every quiescent point. Refunds are exact: remove, evict, and shrink-refund can never underflow. |
| 3 | A denied admission changes **nothing** and returns the moved pair. |
| 4 | Expired entries are invisible to readers but still charged until reclaimed. Time can never forge a budget refund. |

---

## Semantics that took production scars to learn

Ported from [RamShield](https://github.com/grep999/ramshield)'s
`BoundedEventStore` — a DDoS shield where memory exhaustion is attack surface.

| Property | What it means |
|----------|---------------|
| **Delta-only admission** | Growing a value gates just the delta; replacing at equal cost touches no atomic; shrinking refunds immediately. Full stores still accept shrink-replaces. |
| **Lazy TTL, batch reclaim** | Expiry is a *read predicate* (`get` sees nothing), reclamation is explicit (`evict_expired`, O(1) skip when no TTLs exist). A `ttl_count` sidecar keeps the population exact through replaces, removes, pins, and evicts. |
| **`pin`** | Cancel an entry's clock without touching its bytes — the block-until-manually-unblocked pattern. |
| **FIFO reclaim with a real queue** | `evict(n)` is O(1) amortized (stale keys skipped lazily); `evict_older_than(window)` stops at the first fresh key. |
| **One shard lock per insert** | The DashMap `entry` handle spans read-old-cost → CAS-gate → mutate. No `get`+`insert` double-lock, no reserve race, no same-shard self-deadlock — those are measured bugs this shape prevents, not folklore. |
| **The clock short-circuit** | TTL checks sample the clock only for entries that have one. Unconditionally calling `Instant::now()` per read cost +90% read latency on this machine; the guard is documented so nobody "simplifies" it back. |

---

## Measured — the honest numbers, and what they mean

The question is never "is it fast?" — it's **"what does the guarantee cost?"**
Same workload, three designs, laptop-class hardware
(`cargo run --release --example overhead`):

```
                  1 thread        8 threads
─────────────────────────────────────────────────────────────
raw DashMap       3.16 Mops/s     3.33 Mops/s   ← no accounting (the floor)
rambudget put     2.31 Mops/s     2.29 Mops/s   ← 434 ns/op: ~117 ns guarantee
Mutex<HashMap>    2.54 Mops/s     1.36 Mops/s   ← the "obvious safe" design
```

- **The guarantee costs ~117 ns/op** — one CAS + one queue push. That is the
  entire tax over an unaccounted map (62% faster than the v0.1 200 ns: ahash
  replaced SipHash, denial paths recycle their internal key clone). Read
  path pays nothing: zero-copy `get_with` is 29 ns and TTL-less stores never
  touch the clock.
- **At 8 threads rambudget beats Mutex\<HashMap\> 1.7×** — same sharded-locks
  family as raw DashMap, plus the gate. The point: the design everyone
  actually hand-rolls for safety (a global mutex around a size counter) is
  the slow one. That is why budgets don't exist in most caches: the
  safe-by-mutex price looks unacceptable, so teams ship "unsafe + hope" and
  get OOMKilled. rambudget takes DashMap's concurrency and spends ~117 ns
  of it on the guarantee.
- Absolute rates here are bandwidth-bound (~150 MB working set on a laptop,
  raw DashMap itself caps at 3 Mops/s); a server-class box shifts the
  numbers, not the ~70%-of-floor ratio.
- The unit that actually matters: a denied admission decides in 434 ns; an
  OOMKill costs a restart, a cold cache, and millions of lost ops after.
  Degrading by policy is the performance feature.

**Scaling ceiling** (`example contention`):
2.3 Mops/s 1T → 9.9 Mops/s 8T → 11.5 16T
Two global serialization points (budget CAS + queue mutex); documented
upgrade path if you ever live there.

---

## Why not just…

| Alternative | Gap |
|-------------|-----|
| Redis `maxmemory` | Eviction happens inside someone else's process; costs are Redis's model, not yours; no denial-with-pair; a sidecar for a library problem. |
| `cgroups` / `memory.max` | The kernel's answer to "what do I do when RAM is gone" is SIGKILL — the store still dies, just later. |
| `OOM_SCORE_ADJ` | Chooses *who* dies. Never prevents it. |
| `HashMap` + a size counter | The counter is the bug: accounting, races, refunds, TTL and eviction all become your production incident. |

`rambudget` is the layer between "the OS kills you" and "you over-provision
in fear."

---

## Facts

| | |
|---|---|
| **Dependencies** | `dashmap` + `ahash` — everything else is std |
| **Safety** | `#![forbid(unsafe_code)]` — this safety argument does not need an audit |
| **Edition** | Rust 2024 (stable ≥ 1.85), `DashMap` v6 |
| **Tests** | 18 unit tests, `clippy -D warnings` clean |
| **License** | MIT — not affiliated with the RamShield project's runtime; semantics are shared lineage |

---

## Install

```
cargo add rambudget --git https://github.com/grep999/rambudget   # until crates.io
```

---

## Status

**v0.1.0** — core semantics final. Priority/LRU eviction policies and
pressure-reactive admission are the next rings: added when a consumer needs
deterministic drop order, not before. `Denied` is `#[non_exhaustive]`
(opaque, accessor-based); `InsertOutcome` is exhaustively matchable by design.
