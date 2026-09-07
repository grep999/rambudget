//! rambudget — byte-accounted, budget-enforced in-memory store.
//!
//! The store owns its RAM budget: every entry's cost is reserved at insert,
//! refunded at removal, and admission is gated on the reservation. On budget
//! exhaustion the insert is *denied* — the pair comes back, no state changed —
//! instead of the process being OOMKilled. That is the whole thesis: a store
//! that degrades by policy, not by kernel.
//!
//! Semantic core — admission control, byte accounting, lazy TTL expiry,
//! TTL-population counter, pin-to-clear — extracted from RamShield's
//! `BoundedEventStore` (reserve → charge → rollback-on-denied; expiry checked
//! at read, reclaimed in batch, never swept eagerly).
//!
//! # Invariants (what the type guarantees)
//!
//! 1. `reserved()` never exceeds `capacity()` unless the cap was lowered
//!    below usage via [`set_capacity`](BudgetStore::set_capacity).
//! 2. Every byte counted in `reserved()` is owned by exactly one live entry;
//!    `reserved() == Σ cost(entry)` at every quiescent point.
//! 3. A denied insert changes no store state and returns the moved pair.
//! 4. Refunds are exact: remove/evict/shrink-refund can never underflow the
//!    counter (stale handles refund nothing — the entry is simply absent).
//! 5. Expired entries are invisible to readers but still *charged* until
//!    reclaimed (RamShield semantics: time must never be able to forge a
//!    budget refund).
//!
//! # Design notes (why this shape)
//!
//! - One shard write-lock per `put` via a single DashMap `entry` handle: the
//!   old slot is read from the same handle. No `get`+`insert` double-lock, no
//!   reserve/insert race (the handle holds the shard across both), and no
//!   `get`-ref-across-`insert` self-deadlock.
//! - Growing replace gates only the delta; same-cost replace touches no
//!   atomic at all; shrinking replace is a single `fetch_sub`.
//! - `evict` is FIFO O(1) amortized via an insertion queue — no collect-all-
//!   keys scan. Stale queue entries are skipped lazily.
//! - Reads default to zero-copy ([`get_with`](BudgetStore::get_with));
//!   [`get`](BudgetStore::get) exists for `V: Clone` ergonomics.
//! - TTL is lazy: expiry is a *read predicate*, reclamation is batched
//!   ([`evict_expired`](BudgetStore::evict_expired), O(1) skipped when the
//!   TTL population counter is zero). `pin` cancels expiry without touching
//!   bytes (blocked records outlive their window; only the clock does).
//!
//! # Lock discipline (keep it this way)
//!
//! `evict*` nest map-under-queue (queue → map). `put` must therefore NEVER
//! acquire the queue lock while a map handle is alive: `Entry::insert`
//! consumes the handle (releasing the shard) before the queue push.
//! Reversing that order is a deadlock.
//!
//! # Memory overhead (unbudgeted)
//!
//! The FIFO queue holds one `(K clone, Instant)` per net-new insert
//! (≤ 40–72 B for typical keys). For million-key stores that is low-MB;
//! account it in your own capacity math. `ponytail:` upgrade path =
//! per-shard queues or intrusive list, add only when the queue mutex shows
//! in profiles.
//!
//! # Contention ceiling (measured, see examples/contention.rs)
//!
//! Two global serialization points: the `reserved` CAS and the queue mutex
//! (net-new inserts only). Both are uncontended-fast; the number, not a
//! guess, decides when to shard.
//!
//! # Example
//! ```
//! use rambudget::{BudgetStore, Cost};
//!
//! struct ByteCost;
//! impl Cost<String, Vec<u8>> for ByteCost {
//!     fn entry_cost(key: &String, value: &Vec<u8>) -> usize {
//!         key.len() + value.capacity()
//!     }
//! }
//!
//! let store: BudgetStore<String, Vec<u8>, ByteCost> = BudgetStore::new(64);
//! store.put("k".into(), vec![0u8; 32]).unwrap();
//! assert_eq!(store.reserved(), 33);
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use dashmap::mapref::entry::Entry;
use std::collections::VecDeque;
use std::fmt;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Entry byte-cost calculator. Implement per key/value pair; include heap
/// the allocator actually holds (e.g. `Vec::capacity`, `String::len`), not
/// `size_of_val` alone.
pub trait Cost<K, V> {
    /// Bytes this entry occupies in the store. Must be stable for a given
    /// (key, value) pair across calls.
    fn entry_cost(key: &K, value: &V) -> usize;
}

/// `size_of`-only cost: fine for `Copy` payloads, wrong for heap types.
#[derive(Debug, Clone, Copy)]
pub struct InlineCost;

impl<K, V> Cost<K, V> for InlineCost {
    #[inline]
    fn entry_cost(_key: &K, _value: &V) -> usize {
        std::mem::size_of::<(K, V)>()
    }
}

/// Clock abstraction so TTL behaviour is testable without sleeping.
pub trait Clock: Clone + Send + Sync + 'static {
    /// The current instant.
    fn now(&self) -> Instant;
}

/// Wall clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    #[inline]
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// Outcome of a successful insert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// Fresh entry, `charged` bytes reserved.
    Stored {
        /// Bytes newly reserved for this entry.
        charged: usize,
    },
    /// Existing key replaced: `freed` old bytes refunded, `charged` new bytes reserved.
    Replaced {
        /// Bytes the previous entry had reserved.
        freed: usize,
        /// Bytes the new entry reserves.
        charged: usize,
    },
}

/// Admission denied: the entry would exceed the byte budget.
///
/// Carries the moved pair back untouched. Constructed only by this crate.
#[derive(PartialEq, Eq)]
#[non_exhaustive]
pub struct Denied<K, V> {
    key: K,
    value: V,
    required: usize,
    available: usize,
}

impl<K, V> Denied<K, V> {
    /// The rejected key.
    #[inline]
    pub fn key(&self) -> &K {
        &self.key
    }
    /// The rejected value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }
    /// Bytes the entry needed.
    #[inline]
    pub fn required(&self) -> usize {
        self.required
    }
    /// Bytes left under the cap at denial time (may be stale under
    /// concurrency — a diagnostic, not a gate).
    #[inline]
    pub fn available(&self) -> usize {
        self.available
    }
    /// Reclaim the moved pair.
    #[inline]
    pub fn into_pair(self) -> (K, V) {
        (self.key, self.value)
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for Denied<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Denied")
            .field("key", &self.key)
            .field("required", &self.required)
            .field("available", &self.available)
            .finish_non_exhaustive()
    }
}

impl<K: fmt::Debug, V> fmt::Display for Denied<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "budget denied insert for key {:?}: required {} bytes, available {}",
            self.key, self.required, self.available
        )
    }
}

impl<K: fmt::Debug, V: fmt::Debug> std::error::Error for Denied<K, V> {}

/// Live slot: value, its byte cost, optional expiry (RamShield `Entry`).
struct Slot<V> {
    value: V,
    cost: usize,
    expires_at: Option<Instant>,
}

impl<V> Slot<V> {
    /// Lazy expiry: samples the clock ONLY if a TTL exists. Keep the
    /// `is_some_and` shape — an eager `expired(now)` argument pays a
    /// ~33ns clock call on every read even for TTL-less stores.
    #[inline]
    fn expired_by<F: FnOnce() -> Instant>(&self, now: F) -> bool {
        self.expires_at.is_some_and(|e| now() > e)
    }
}

/// Cheap consistent-enough snapshot (metrics/logging surface).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Byte cap.
    pub capacity: usize,
    /// Bytes reserved by live entries (expired-but-unreclaimed included).
    pub reserved: usize,
    /// Live entry count (includes expired-but-unreclaimed).
    pub len: usize,
    /// Entries carrying a TTL. `evict_expired` early-exits when this is 0.
    pub ttl_count: usize,
    /// Lifetime denied admissions.
    pub denials: usize,
}

/// Byte-accounted concurrent map with hard capacity and lazy TTL.
///
/// `K: Hash + Eq + Clone` for sharding; `V: Clone` is only needed for
/// [`get`](BudgetStore::get). `C` computes per-entry cost, `W` is the clock
/// (defaults to [`SystemClock`]).
pub struct BudgetStore<K, V, C = InlineCost, W = SystemClock>
where
    K: Hash + Eq + Clone,
    C: Cost<K, V>,
    W: Clock,
{
    capacity: AtomicUsize,
    reserved: AtomicUsize,
    denials: AtomicUsize,
    /// Count of live slots with `expires_at: Some(_)` (RamShield `ttl_entries`).
    ttl_count: AtomicUsize,
    total_inserts: AtomicU64,
    map: dashmap::DashMap<K, Slot<V>>,
    /// FIFO of `(key, insert_time)` for net-new inserts. May contain stale
    /// keys (removed or evicted since); skipped lazily on pop.
    queue: Mutex<VecDeque<(K, Instant)>>,
    clock: W,
    _cost: std::marker::PhantomData<C>,
}

impl<K, V, C> BudgetStore<K, V, C, SystemClock>
where
    K: Hash + Eq + Clone,
    C: Cost<K, V>,
{
    /// Empty store with a hard byte `capacity` on the wall clock.
    /// `capacity == 0` admits nothing.
    pub fn new(capacity: usize) -> Self {
        Self::with_clock(capacity, SystemClock)
    }
}

impl<K, V, C, W> BudgetStore<K, V, C, W>
where
    K: Hash + Eq + Clone,
    C: Cost<K, V>,
    W: Clock,
{
    /// Empty store with a hard byte `capacity` and an injected clock.
    pub fn with_clock(capacity: usize, clock: W) -> Self {
        Self {
            capacity: AtomicUsize::new(capacity),
            reserved: AtomicUsize::new(0),
            denials: AtomicUsize::new(0),
            ttl_count: AtomicUsize::new(0),
            total_inserts: AtomicU64::new(0),
            map: dashmap::DashMap::new(),
            queue: Mutex::new(VecDeque::new()),
            clock,
            _cost: std::marker::PhantomData,
        }
    }

    #[inline]
    fn now(&self) -> Instant {
        self.clock.now()
    }

    #[inline]
    fn queue_lock(&self) -> MutexGuard<'_, VecDeque<(K, Instant)>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// CAS-gate: move reserved `need` bytes higher under `capacity`.
    /// Nothing changes on failure. `checked_add` so absurd needs can't wrap.
    fn reserve(&self, need: usize) -> Result<(), ()> {
        let cap = self.capacity.load(Ordering::Relaxed);
        let mut cur = self.reserved.load(Ordering::Relaxed);
        loop {
            let Some(next) = cur.checked_add(need) else {
                return Err(());
            };
            if next > cap {
                return Err(());
            }
            match self.reserved.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => cur = actual,
            }
        }
    }

    #[cold]
    fn denial<K2, V2>(&self, key: K2, value: V2, required: usize) -> Denied<K2, V2> {
        self.denials.fetch_add(1, Ordering::Relaxed);
        Denied {
            key,
            value,
            required,
            available: self.available(),
        }
    }

    /// TTL-population delta bookkeeping (RamShield: applied only after the
    /// insert is known to stick; denials never touch it).
    #[inline]
    fn ttl_delta(&self, old_had_ttl: bool, new_has_ttl: bool) {
        match (old_had_ttl, new_has_ttl) {
            (false, true) => {
                self.ttl_count.fetch_add(1, Ordering::Relaxed);
            }
            (true, false) => {
                self.ttl_count.fetch_sub(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    /// Insert or replace, no expiry. Takes ownership; on budget denial the
    /// entry comes back as [`Denied`] and no store state changed.
    ///
    /// The `entry` handle holds the key's shard write-lock across the CAS
    /// reserve and the map mutation, so two threads racing on the same new
    /// key serialize: the loser sees `Occupied` and takes the replace path.
    #[inline]
    pub fn put(&self, key: K, value: V) -> Result<InsertOutcome, Denied<K, V>> {
        self.put_with_ttl(key, value, None)
    }

    /// Insert or replace with a TTL. Expired entries are invisible to reads
    /// but hold their budget until reclaimed ([`evict_expired`] or removal) —
    /// time must never be able to forge a refund.
    ///
    /// [`evict_expired`]: BudgetStore::evict_expired
    pub fn put_with_ttl(
        &self,
        key: K,
        value: V,
        ttl: Option<Duration>,
    ) -> Result<InsertOutcome, Denied<K, V>> {
        let need = C::entry_cost(&key, &value);
        let expires_at = ttl.map(|d| self.now() + d);
        let new_has_ttl = expires_at.is_some();
        match self.map.entry(key.clone()) {
            Entry::Occupied(mut e) => {
                let old = e.get().cost;
                let old_had_ttl = e.get().expires_at.is_some();
                // three cases, least atomic work first
                if need > old {
                    // grow: gate only the delta
                    if self.reserve(need - old).is_err() {
                        return Err(self.denial(key, value, need));
                    }
                } else if need < old {
                    // shrink: always allowed, refund the difference
                    self.reserved.fetch_sub(old - need, Ordering::Relaxed);
                }
                // equal cost: touch no budget atomic at all
                e.insert(Slot {
                    value,
                    cost: need,
                    expires_at,
                }); // handle consumed → shard lock released
                self.ttl_delta(old_had_ttl, new_has_ttl);
                self.total_inserts.fetch_add(1, Ordering::Relaxed);
                Ok(InsertOutcome::Replaced {
                    freed: old,
                    charged: need,
                })
            }
            Entry::Vacant(e) => {
                if self.reserve(need).is_err() {
                    return Err(self.denial(key, value, need));
                }
                let stamp = self.now();
                e.insert(Slot {
                    value,
                    cost: need,
                    expires_at,
                }); // handle consumed → shard lock released
                self.ttl_delta(false, new_has_ttl);
                self.total_inserts.fetch_add(1, Ordering::Relaxed);
                // queue lock AFTER the map handle is gone (see lock discipline)
                self.queue_lock().push_back((key, stamp));
                Ok(InsertOutcome::Stored { charged: need })
            }
        }
    }

    /// Read without cloning: run `f` over the *unexpired* value while the
    /// shard read-lock is held. Expired entries read as absent (lazy expiry,
    /// RamShield `get`). The default zero-copy read.
    ///
    /// Do NOT remove the slot from inside `f` — the handle is held; use
    /// [`remove`](BudgetStore::remove) after.
    ///
    /// Clock discipline: the expiry check samples the clock only for slots
    /// that actually carry a TTL (`is_some_and` short-circuit, RamShield's
    /// `is_expired` shape). Measured: an unconditional `now()` costs ~42ns on
    /// a 47ns DashMap read (examples/clock_tax.rs) — a 90% tax on stores with
    /// no TTLs at all. Keep the short-circuit.
    #[inline]
    pub fn get_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        self.map.get(key).and_then(|e| {
            if e.expired_by(|| self.now()) {
                None
            } else {
                Some(f(&e.value))
            }
        })
    }

    /// Clone out the value for `key`, if present and unexpired.
    #[inline]
    pub fn get(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_with(key, V::clone)
    }

    /// Present AND unexpired probe.
    #[inline]
    pub fn contains_key(&self, key: &K) -> bool {
        self.map
            .get(key)
            .is_some_and(|e| !e.expired_by(|| self.now()))
    }

    /// Remove `key` (expired or not); refunds its cost. Returns the value.
    /// The FIFO queue entry goes stale and is skipped lazily on `evict`.
    #[inline]
    pub fn remove(&self, key: &K) -> Option<V> {
        let (_, slot) = self.map.remove(key)?;
        self.reserved.fetch_sub(slot.cost, Ordering::Relaxed);
        if slot.expires_at.is_some() {
            self.ttl_count.fetch_sub(1, Ordering::Relaxed);
        }
        Some(slot.value)
    }

    /// Cancel expiry on a live entry without touching bytes — the
    /// block-until-manual-unblock pattern: the record's *time* is cleared,
    /// its budget charge stays. Returns true if a TTL was cleared.
    pub fn pin(&self, key: &K) -> bool {
        if let Some(mut e) = self.map.get_mut(key)
            && e.expires_at.take().is_some()
        {
            drop(e); // release before counter touch (ordering doesn't
                     // matter, but keep handles short)
            self.ttl_count.fetch_sub(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    /// Drop every entry and zero the counters.
    pub fn clear(&self) {
        self.map.clear();
        self.queue_lock().clear();
        self.reserved.store(0, Ordering::Relaxed);
        self.ttl_count.store(0, Ordering::Relaxed);
    }

    /// Reclaim ALL expired entries (read-predicate expiry, batch removal).
    /// Returns `(count, bytes)`. O(1) early-exit when no entry carries a TTL
    /// (RamShield `evict_expired`). Queue entries go stale; `evict` skips
    /// them lazily.
    pub fn evict_expired(&self) -> (usize, usize) {
        if self.ttl_count.load(Ordering::Relaxed) == 0 {
            return (0, 0);
        }
        let now = self.now();
        let stale: Vec<K> = self
            .map
            .iter()
            .filter(|e| e.expired_by(|| now))
            .map(|e| e.key().clone())
            .collect(); // refs released before any removal
        let mut n = 0;
        let mut bytes = 0;
        for key in &stale {
            // remove refunds budget + TTL counter exactly once; a racing
            // remover already freed the slot gets None and skips.
            if let Some((_, slot)) = self.map.remove(key) {
                self.reserved.fetch_sub(slot.cost, Ordering::Relaxed);
                if slot.expires_at.is_some() {
                    self.ttl_count.fetch_sub(1, Ordering::Relaxed);
                }
                n += 1;
                bytes += slot.cost;
            }
        }
        (n, bytes)
    }

    /// Evict up to `n` *live* entries in FIFO order (oldest net-new insert
    /// first; replacements keep the original position), refunding each cost.
    /// Returns bytes reclaimed. Stale queue keys are skipped without
    /// counting toward `n`.
    pub fn evict(&self, n: usize) -> usize {
        let mut reclaimed = 0usize;
        let mut live = 0usize;
        let mut q = self.queue_lock(); // queue held, then map: the ONLY nesting allowed
        while live < n {
            let Some((k, _)) = q.pop_front() else { break };
            // entry() may block on the shard write-lock — that's why put
            // releases its handle BEFORE touching this queue (no cycle).
            if let Entry::Occupied(e) = self.map.entry(k) {
                let slot = e.remove_entry().1;
                self.reserved.fetch_sub(slot.cost, Ordering::Relaxed);
                if slot.expires_at.is_some() {
                    self.ttl_count.fetch_sub(1, Ordering::Relaxed);
                }
                reclaimed += slot.cost;
                live += 1;
            }
        }
        reclaimed
    }

    /// Reclaim everything older than `window` (FIFO-front scan, RamShield
    /// `evict_batch` windowed gate). Stops at the first in-window key — the
    /// queue is ordered, so nothing later can be older. Returns
    /// `(count, bytes)`.
    pub fn evict_older_than(&self, window: Duration) -> (usize, usize) {
        // checked_sub: window may exceed time-since-boot on a fresh process —
        // nothing is older than the clock's own origin.
        let Some(cutoff) = self.now().checked_sub(window) else {
            return (0, 0);
        };
        let mut n = 0;
        let mut bytes = 0;
        let mut q = self.queue_lock();
        while let Some((_, t)) = q.front() {
            if *t > cutoff {
                break; // queue ordered: rest is newer
            }
            let (k, _) = q.pop_front().expect("front checked");
            if let Entry::Occupied(e) = self.map.entry(k) {
                let slot = e.remove_entry().1;
                self.reserved.fetch_sub(slot.cost, Ordering::Relaxed);
                if slot.expires_at.is_some() {
                    self.ttl_count.fetch_sub(1, Ordering::Relaxed);
                }
                n += 1;
                bytes += slot.cost;
            }
        }
        (n, bytes)
    }

    /// Hot-reconfigure the cap. Shrinking below `reserved` does NOT evict —
    /// the store simply refuses new admissions until usage drops (visible as
    /// `pressure() > 1.0`).
    #[inline]
    pub fn set_capacity(&self, capacity: usize) {
        self.capacity.store(capacity, Ordering::Relaxed);
    }

    /// Current byte cap.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Bytes currently reserved by live entries (expired-unreclaimed included).
    #[inline]
    pub fn reserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// Bytes left under the cap (saturating: never negative after a shrink).
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity().saturating_sub(self.reserved())
    }

    /// Lifetime count of denied admissions — the policy-fire signal, distinct
    /// from `pressure()` (how full). Nonzero means the cap is real policy,
    /// not decoration.
    #[inline]
    pub fn denials(&self) -> usize {
        self.denials.load(Ordering::Relaxed)
    }

    /// Lifetime successful inserts (fresh + replacements).
    #[inline]
    pub fn total_inserts(&self) -> u64 {
        self.total_inserts.load(Ordering::Relaxed)
    }

    /// Entries currently carrying a TTL.
    #[inline]
    pub fn ttl_count(&self) -> usize {
        self.ttl_count.load(Ordering::Relaxed)
    }

    /// Reserved / capacity. `1.0` at full; `> 1.0` after a shrink below usage;
    /// `0.0` when capacity is 0.
    #[inline]
    pub fn pressure(&self) -> f64 {
        let cap = self.capacity();
        if cap == 0 {
            return 0.0;
        }
        self.reserved() as f64 / cap as f64
    }

    /// Live entry count (includes expired-but-unreclaimed).
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True when no live entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Snapshot for metrics/logging.
    pub fn stats(&self) -> Stats {
        Stats {
            capacity: self.capacity(),
            reserved: self.reserved(),
            len: self.len(),
            ttl_count: self.ttl_count(),
            denials: self.denials(),
        }
    }
}

impl<K, V, C, W> Default for BudgetStore<K, V, C, W>
where
    K: Hash + Eq + Clone,
    C: Cost<K, V>,
    W: Clock + Default,
{
    fn default() -> Self {
        Self::with_clock(0, W::default())
    }
}

impl<K, V, C> std::fmt::Debug for BudgetStore<K, V, C, SystemClock>
where
    K: Hash + Eq + Clone,
    C: Cost<K, V>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetStore")
            .field("capacity", &self.capacity())
            .field("reserved", &self.reserved())
            .field("denials", &self.denials())
            .field("ttl_count", &self.ttl_count())
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 byte per key char + exact Vec length (tests use len==capacity).
    struct ByteCost;
    impl Cost<String, Vec<u8>> for ByteCost {
        #[inline]
        fn entry_cost(key: &String, value: &Vec<u8>) -> usize {
            key.len() + value.len()
        }
    }

    fn store(cap: usize) -> BudgetStore<String, Vec<u8>, ByteCost> {
        BudgetStore::new(cap)
    }

    fn entry(k: &str, n: usize) -> (String, Vec<u8>) {
        (k.to_string(), vec![7u8; n])
    }

    #[test]
    fn admits_until_full_then_denies_without_state_change() {
        let s = store(10);
        let (k, v) = entry("a", 4); // cost 5
        assert_eq!(s.put(k, v), Ok(InsertOutcome::Stored { charged: 5 }));
        let (k, v) = entry("bb", 8); // cost 10 → 15 > 10 → denied, pair returned
        let d = s.put(k.clone(), v.clone()).unwrap_err();
        assert_eq!(d.key(), &k);
        assert_eq!(d.required(), 10);
        assert_eq!(d.into_pair(), (k, v));
        assert_eq!(s.reserved(), 5);
        assert_eq!(s.len(), 1);
        assert_eq!(s.denials(), 1);
    }

    #[test]
    fn zero_capacity_admits_nothing() {
        let s = store(0);
        let (k, v) = entry("a", 1);
        assert!(s.put(k.clone(), v.clone()).is_err());
        assert_eq!(s.reserved(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn replace_grow_charges_delta_only() {
        let s = store(10);
        let (k, v) = entry("k", 2); // cost 3
        s.put(k.clone(), v).unwrap();
        let (k, v) = entry("k", 7); // cost 8, delta 5, total 8 ≤ 10
        assert_eq!(
            s.put(k, v),
            Ok(InsertOutcome::Replaced {
                freed: 3,
                charged: 8
            })
        );
        assert_eq!(s.reserved(), 8);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn replace_denied_keeps_old_entry_intact() {
        let s = store(10);
        let (k, v) = entry("k", 2); // cost 3
        s.put(k.clone(), v).unwrap();
        let (k2, v2) = entry("k", 20); // cost 21, delta 18 → denied
        let d = s.put(k2.clone(), v2.clone()).unwrap_err();
        assert_eq!(d.into_pair(), (k2, v2));
        assert_eq!(s.reserved(), 3);
        assert_eq!(s.get(&k).unwrap().len(), 2);
    }

    #[test]
    fn replace_shrink_refunds() {
        let s = store(10);
        let (k, v) = entry("k", 8); // cost 9
        s.put(k.clone(), v).unwrap();
        let (k, v) = entry("k", 1); // cost 2
        assert_eq!(
            s.put(k, v),
            Ok(InsertOutcome::Replaced {
                freed: 9,
                charged: 2
            })
        );
        assert_eq!(s.reserved(), 2);
    }

    #[test]
    fn replace_same_cost_touches_no_gate() {
        // Full store: delta-0 replace must still succeed (no reserve needed).
        let s = store(5);
        let (k, v) = entry("a", 4); // cost 5, full
        s.put(k.clone(), v).unwrap();
        let (k, v) = entry("a", 4); // same cost
        assert_eq!(
            s.put(k, v),
            Ok(InsertOutcome::Replaced {
                freed: 5,
                charged: 5
            })
        );
        assert_eq!(s.reserved(), 5);
        assert_eq!(s.denials(), 0);
    }

    #[test]
    fn get_with_reads_without_clone() {
        let s = store(100);
        s.put(entry("k", 3).0, entry("k", 3).1).unwrap();
        let sum = s.get_with(&"k".to_string(), |v| v.iter().sum::<u8>());
        assert_eq!(sum, Some(21)); // 3 * 7u8
        assert!(s.get_with(&"missing".to_string(), |_| 0).is_none());
    }

    #[test]
    fn remove_refunds_exactly() {
        let s = store(10);
        let (k, v) = entry("x", 5); // cost 6
        s.put(k.clone(), v).unwrap();
        assert_eq!(s.remove(&k).unwrap().len(), 5);
        assert_eq!(s.reserved(), 0);
        assert!(s.remove(&k).is_none()); // double-remove: no negative refund
    }

    #[test]
    fn shrink_capacity_refuses_admission_not_evicts() {
        let s = store(100);
        let (k, v) = entry("a", 50); // cost 51
        s.put(k, v).unwrap();
        s.set_capacity(30);
        assert_eq!(s.len(), 1); // nothing evicted
        let (k, v) = entry("b", 1);
        assert!(s.put(k, v).is_err()); // no headroom
        assert!(s.pressure() > 1.0); // over capacity is observable
    }

    #[test]
    fn evict_is_fifo_and_refunds() {
        let s = store(12);
        s.put(entry("a", 3).0, entry("a", 3).1).unwrap(); // cost 4
        s.put(entry("b", 3).0, entry("b", 3).1).unwrap(); // cost 4
        s.put(entry("c", 3).0, entry("c", 3).1).unwrap(); // cost 4, full
        let (k, v) = entry("d", 3);
        assert!(s.put(k.clone(), v.clone()).is_err());
        let reclaimed = s.evict(1);
        assert_eq!(reclaimed, 4);
        assert_eq!(s.reserved(), 8);
        assert!(s.contains_key(&"b".to_string())); // middle untouched
        assert!(!s.contains_key(&"a".to_string())); // oldest gone
        s.put(k, v).unwrap(); // room now
    }

    #[test]
    fn evict_skips_stale_queue_entries() {
        let s = store(100);
        for i in 0..10 {
            let (k, v) = entry(&format!("k{i}"), 3); // cost 5 each
            s.put(k, v).unwrap();
        }
        s.remove(&"k2".to_string());
        s.remove(&"k7".to_string());
        assert_eq!(s.reserved(), 40);
        let reclaimed = s.evict(3); // pops k0,k1,(k2 stale),k3
        assert_eq!(reclaimed, 15);
        assert_eq!(s.reserved(), 25);
        assert_eq!(s.len(), 5);
        assert!(!s.contains_key(&"k3".to_string()));
        assert!(s.contains_key(&"k4".to_string()));
    }

    #[test]
    fn replace_keeps_original_fifo_position() {
        let s = store(100);
        s.put(entry("a", 3).0, entry("a", 3).1).unwrap();
        s.put(entry("b", 3).0, entry("b", 3).1).unwrap();
        let (k, v) = entry("a", 9);
        s.put(k, v).unwrap(); // replace a — no second queue entry
        assert_eq!(s.queue_lock().len(), 2);
        assert_eq!(s.evict(1), 10); // a still goes first, at its NEW cost
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn denied_implements_error_trait() {
        let s = store(1);
        let d = s.put(entry("a", 9).0, entry("a", 9).1).unwrap_err();
        let boxed: Box<dyn std::error::Error> = Box::new(d);
        assert!(boxed.to_string().contains("denied"));
    }

    // --- TTL semantics (fake clock, no sleeping) ---

    #[derive(Clone)]
    struct FakeClock(std::sync::Arc<Mutex<Instant>>);
    impl FakeClock {
        fn start() -> (Self, std::sync::Arc<Mutex<Instant>>) {
            let t = std::sync::Arc::new(Mutex::new(Instant::now()));
            (Self(t.clone()), t)
        }
    }
    impl Clock for FakeClock {
        fn now(&self) -> Instant {
            *self.0.lock().unwrap()
        }
    }

    type FakeTime = std::sync::Arc<Mutex<Instant>>;

    fn fake_store(
        cap: usize,
    ) -> (
        BudgetStore<String, Vec<u8>, ByteCost, FakeClock>,
        FakeTime,
    ) {
        let (clk, t) = FakeClock::start();
        (BudgetStore::with_clock(cap, clk), t)
    }

    fn advance(t: &std::sync::Arc<Mutex<Instant>>, d: Duration) {
        *t.lock().unwrap() += d;
    }

    #[test]
    fn expired_reads_as_absent_but_holds_budget() {
        let (s, t) = fake_store(100);
        let k = "a".to_string();
        s.put_with_ttl(k.clone(), vec![0u8; 9], Some(Duration::from_secs(10)))
            .unwrap(); // cost 10
        assert_eq!(s.ttl_count(), 1);
        advance(&t, Duration::from_secs(11));
        assert!(s.get(&k).is_none()); // invisible…
        assert!(!s.contains_key(&k));
        assert_eq!(s.reserved(), 10); // …but still charged (invariant 5)
        let (n, bytes) = s.evict_expired(); // batch reclaim
        assert_eq!((n, bytes), (1, 10));
        assert_eq!(s.reserved(), 0);
        assert_eq!(s.ttl_count(), 0);
    }

    #[test]
    fn evict_expired_early_exits_without_ttl() {
        let (s, _t) = fake_store(100);
        s.put(entry("a", 3).0, entry("a", 3).1).unwrap();
        assert_eq!(s.evict_expired(), (0, 0)); // O(1) via ttl_count
    }

    #[test]
    fn ttl_delta_bookkeeping_across_replace_and_remove() {
        let (s, _t) = fake_store(100);
        let k = "a".to_string();
        s.put(k.clone(), vec![0u8; 4]).unwrap(); // no ttl
        assert_eq!(s.ttl_count(), 0);
        s.put_with_ttl(k.clone(), vec![0u8; 4], Some(Duration::from_secs(5)))
            .unwrap(); // replace: none → ttl
        assert_eq!(s.ttl_count(), 1);
        s.put(k.clone(), vec![0u8; 4]).unwrap(); // replace: ttl → none
        assert_eq!(s.ttl_count(), 0);
        s.put_with_ttl(k.clone(), vec![0u8; 4], Some(Duration::from_secs(5)))
            .unwrap();
        assert_eq!(s.ttl_count(), 1);
        s.remove(&k).unwrap(); // removal decrements
        assert_eq!(s.ttl_count(), 0);
        assert_eq!(s.reserved(), 0);
    }

    #[test]
    fn denied_insert_never_touches_ttl_count() {
        let (s, _t) = fake_store(5);
        let d = s
            .put_with_ttl("a".to_string(), vec![0u8; 9], Some(Duration::from_secs(5)))
            .unwrap_err(); // cost 10 > 5
        drop(d);
        assert_eq!(s.ttl_count(), 0);
    }

    #[test]
    fn pin_cancels_expiry_keeps_bytes() {
        let (s, t) = fake_store(100);
        let k = "a".to_string();
        s.put_with_ttl(k.clone(), vec![0u8; 9], Some(Duration::from_secs(10)))
            .unwrap();
        assert!(s.pin(&k));
        assert!(!s.pin(&k)); // second pin: nothing to clear
        advance(&t, Duration::from_secs(99));
        assert!(s.get(&k).is_some()); // still visible
        assert_eq!(s.reserved(), 10); // bytes never moved
        assert_eq!(s.ttl_count(), 0);
        assert_eq!(s.evict_expired(), (0, 0)); // nothing expired
    }

    #[test]
    fn evict_older_than_stops_at_first_fresh() {
        let (s, t) = fake_store(1000);
        for i in 0..5 {
            let (k, v) = entry(&format!("k{i}"), 3); // cost 5
            s.put(k, v).unwrap();
            advance(&t, Duration::from_secs(10));
        }
        // now: k0=50s … k4=10s old; cutoff = 50-25 = 25 → k0(t0),k1(t10),k2(t20) stale
        let (n, bytes) = s.evict_older_than(Duration::from_secs(25));
        assert_eq!((n, bytes), (3, 15));
        assert_eq!(s.len(), 2);
        assert!(!s.contains_key(&"k2".to_string()));
        assert!(s.contains_key(&"k3".to_string()));
    }

    #[test]
    fn ttl_entries_reclaimed_by_fifo_evict_too() {
        let (s, t) = fake_store(100);
        s.put_with_ttl("a".to_string(), vec![0u8; 9], Some(Duration::from_secs(10)))
            .unwrap();
        advance(&t, Duration::from_secs(11));
        assert_eq!(s.evict(1), 10); // plain evict reclaims an expired one too
        assert_eq!(s.ttl_count(), 0); // …and fixes the counter
        assert_eq!(s.reserved(), 0);
    }

    #[test]
    fn stats_snapshot() {
        let s = store(100);
        s.put(entry("a", 3).0, entry("a", 3).1).unwrap();
        let d = s.put(entry("b", 99).0, entry("b", 99).1).unwrap_err();
        drop(d);
        let st = s.stats();
        assert_eq!(
            st,
            Stats {
                capacity: 100,
                reserved: 4,
                len: 1,
                ttl_count: 0,
                denials: 1
            }
        );
        assert_eq!(s.total_inserts(), 1);
    }

    #[test]
    fn concurrent_admission_never_breaks_the_cap() {
        use std::sync::Arc;
        let s = Arc::new(store(100));
        let mut ts = Vec::new();
        for t in 0..8 {
            let s = Arc::clone(&s);
            ts.push(std::thread::spawn(move || {
                for i in 0..50 {
                    let (k, v) = (format!("k{t}_{i}"), vec![0u8; 3]);
                    let _ = s.put(k, v);
                }
            }));
        }
        for t in ts {
            t.join().unwrap();
        }
        assert!(s.reserved() <= 100, "cap breached: {}", s.reserved());
        assert_eq!(s.reserved(), s.map.iter().map(|e| e.cost).sum::<usize>());
    }

    #[test]
    fn concurrent_same_key_put_serializes_to_one_charge() {
        use std::sync::Arc;
        let s = Arc::new(store(100));
        let mut ts = Vec::new();
        for _ in 0..8 {
            let s = Arc::clone(&s);
            ts.push(std::thread::spawn(move || {
                let (k, v) = ("hot".to_string(), vec![0u8; 4]);
                let _ = s.put(k, v); // 8 racers, cost 7 each
            }));
        }
        for t in ts {
            t.join().unwrap();
        }
        assert_eq!(s.len(), 1);
        assert_eq!(s.reserved(), 7);
    }

    #[test]
    fn concurrent_evict_and_put_conservation() {
        use std::sync::Arc;
        let s = Arc::new(store(200));
        let mut ts = Vec::new();
        for t in 0..4 {
            let s = Arc::clone(&s);
            ts.push(std::thread::spawn(move || {
                for i in 0..100 {
                    let (k, v) = (format!("p{t}_{i}"), vec![0u8; 4]);
                    let _ = s.put(k, v);
                }
            }));
        }
        let s2 = Arc::clone(&s);
        ts.push(std::thread::spawn(move || {
            for _ in 0..100 {
                s2.evict(3);
            }
        }));
        for t in ts {
            t.join().unwrap();
        }
        assert!(s.reserved() <= 200);
        assert_eq!(s.reserved(), s.map.iter().map(|e| e.cost).sum::<usize>());
    }
}
