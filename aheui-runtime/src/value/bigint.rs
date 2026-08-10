//! Big-integer backend. Parity with `rpaheui/aheui/int/bigint.py`.
//!
//! `bigint.py` sets `Int = rbigint` and wraps RPython's rbigint. This
//! is the path RPython translates. Our Rust port uses a tagged-pointer
//! layout so the hot small-int path stays allocation-free:
//!
//!   bit 0 = 1  →  small integer, value = word >> 1  (arithmetic shift)
//!   bit 0 = 0  →  pointer to a heap `BigInt` (always aligned)
//!
//! Small-integer range: −2^62 .. 2^62 − 1 (±4.6 × 10^18).

#[cfg(feature = "malachite-bigint")]
use malachite_bigint::BigInt;
#[cfg(all(feature = "num-bigint", not(feature = "malachite-bigint")))]
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const SMALL_MIN: i64 = -(1i64 << 62);
const SMALL_MAX: i64 = (1i64 << 62) - 1;

pub type AheuiBigInt = BigInt;
pub type BigIntAllocHook = fn(BigInt) -> *mut BigInt;
pub type BigIntMaybeCollectHook = fn();

static BIGINT_ALLOC_HOOK: AtomicUsize = AtomicUsize::new(0);
static BIGINT_MAYBE_COLLECT_HOOK: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static BIGINT_TRANSIENT_ROOTS: RefCell<Vec<*mut Val>> = const { RefCell::new(Vec::new()) };
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Val(i64);

/// Raw tagged-word AND for JIT fast paths.
///
/// This compares/ANDs the packed `i64` representation directly. It is correct
/// only when operands share the small tag, and is not general-purpose `Val`
/// semantics.
impl std::ops::BitAnd for Val {
    type Output = i64;

    #[inline(always)]
    fn bitand(self, r: Val) -> i64 {
        self.0 & r.0
    }
}

/// Raw tagged-word arithmetic shift for JIT fast paths.
///
/// This shifts the packed `i64` representation directly. For small integers,
/// shifting right by one untags the value. This is not general-purpose `Val`
/// semantics.
impl std::ops::Shr<i64> for Val {
    type Output = i64;

    #[inline(always)]
    fn shr(self, s: i64) -> i64 {
        self.0 >> s
    }
}

/// Raw tagged-word ordering for JIT fast paths.
///
/// This compares the packed `i64` representation directly. It is correct only
/// when operands share the small tag, and is not general-purpose `Val`
/// semantics. `PartialEq` is derived over the same representation.
impl PartialOrd for Val {
    #[inline(always)]
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&o.0)
    }
}

/// Mode-0 machine-word arithmetic, `None` when the true result does not fit
/// the word.
///
/// These read the packed representation as the plain `i64` it is in mode 0, so
/// they are meaningless in mode 1. They exist in this shape because the JIT
/// recognises `match a.checked_add(b) { Some(_) => _, None => _ }` and fuses it
/// into one overflow-branching add — which is what mode 0 is for. The `None`
/// arm must be the only route out of mode 0 in JIT-visible code: leaving it
/// anywhere else inside compiled code would convert a storage whose values the
/// trace is still holding raw.
impl Val {
    #[inline(always)]
    pub fn checked_add(self, o: Val) -> Option<Val> {
        self.0.checked_add(o.0).map(Val)
    }

    #[inline(always)]
    pub fn checked_sub(self, o: Val) -> Option<Val> {
        self.0.checked_sub(o.0).map(Val)
    }

    #[inline(always)]
    pub fn checked_mul(self, o: Val) -> Option<Val> {
        self.0.checked_mul(o.0).map(Val)
    }
}

impl Val {
    #[inline(always)]
    fn from_small(v: i64) -> Self {
        debug_assert!(v >= SMALL_MIN && v <= SMALL_MAX);
        Val((v << 1) | 1)
    }

    #[inline(always)]
    fn from_big(b: BigInt) -> Self {
        let hook_addr = BIGINT_ALLOC_HOOK.load(Ordering::Relaxed);
        let ptr = if hook_addr != 0 {
            let hook: BigIntAllocHook = unsafe { std::mem::transmute(hook_addr) };
            hook(b)
        } else {
            Box::into_raw(Box::new(b))
        };
        debug_assert!(!ptr.is_null());
        debug_assert_eq!(ptr as usize & 1, 0);
        Val(ptr as i64) // aligned pointer, bit 0 = 0
    }

    #[inline(always)]
    fn from_i64_promoting(v: i64) -> Self {
        if v >= SMALL_MIN && v <= SMALL_MAX {
            Self::from_small(v)
        } else {
            Self::from_big(BigInt::from(v))
        }
    }

    #[inline(always)]
    pub fn is_small(self) -> bool {
        self.0 & 1 != 0
    }

    #[inline(always)]
    fn as_i64_unchecked(self) -> i64 {
        self.0 >> 1
    }

    pub fn to_i64(self) -> i64 {
        if self.is_small() {
            self.as_i64_unchecked()
        } else {
            self.as_big().to_i64().expect("BigInt too large for i64")
        }
    }

    /// Try to convert to i64, returning None for BigInt values that overflow.
    #[inline]
    pub fn try_to_i64(self) -> Option<i64> {
        if !bigint_mode() {
            return Some(self.0);
        }
        if self.is_small() {
            Some(self.as_i64_unchecked())
        } else {
            self.as_big().to_i64()
        }
    }

    #[inline(always)]
    fn as_big(&self) -> &BigInt {
        debug_assert!(!self.is_small());
        unsafe { &*(self.0 as *const BigInt) }
    }

    fn to_bigint(self) -> BigInt {
        if self.is_small() {
            BigInt::from(self.as_i64_unchecked())
        } else {
            self.as_big().clone()
        }
    }

    fn normalize_big(b: BigInt) -> Val {
        match b.to_i64() {
            Some(v) if v >= SMALL_MIN && v <= SMALL_MAX => Val::from_small(v),
            _ => Val::from_big(b),
        }
    }
}

impl From<i64> for Val {
    #[inline(always)]
    fn from(v: i64) -> Self {
        if bigint_mode() {
            Val::from_i64_promoting(v)
        } else {
            Val(v)
        }
    }
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !bigint_mode() {
            return write!(f, "{}", self.0);
        }
        if self.is_small() {
            write!(f, "{}", self.as_i64_unchecked())
        } else {
            write!(f, "{}", self.as_big())
        }
    }
}

impl std::fmt::Debug for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !bigint_mode() {
            return write!(f, "Val::Raw({})", self.0);
        }
        if self.is_small() {
            write!(f, "Val::Small({})", self.as_i64_unchecked())
        } else {
            write!(f, "Val::Big({})", self.as_big())
        }
    }
}

pub fn register_bigint_alloc_hook(hook: BigIntAllocHook) {
    BIGINT_ALLOC_HOOK.store(hook as usize, Ordering::Relaxed);
}

pub fn register_bigint_maybe_collect_hook(hook: BigIntMaybeCollectHook) {
    BIGINT_MAYBE_COLLECT_HOOK.store(hook as usize, Ordering::Relaxed);
}

#[inline(always)]
pub fn val_bigint_addr(v: &Val) -> Option<usize> {
    // In mode 0 the word is a raw integer, and an even one has bit 0 clear
    // exactly like a pointer does. No bigint exists before the flip, so the
    // collector must be told there is nothing here rather than left to read
    // the tag.
    if !bigint_mode() {
        return None;
    }
    (!v.is_small()).then_some(v.0 as usize)
}

#[inline]
pub fn with_bigint_transient_root<R>(value: &mut Val, f: impl FnOnce() -> R) -> R {
    struct RootGuard;

    impl Drop for RootGuard {
        fn drop(&mut self) {
            BIGINT_TRANSIENT_ROOTS.with(|roots| {
                roots.borrow_mut().pop();
            });
        }
    }

    BIGINT_TRANSIENT_ROOTS.with(|roots| {
        roots.borrow_mut().push(value as *mut Val);
    });
    let _guard = RootGuard;
    f()
}

pub fn walk_bigint_transient_roots(visit: &mut dyn FnMut(&mut Val)) {
    let roots = BIGINT_TRANSIENT_ROOTS.with(|roots| roots.borrow().clone());
    for root in roots {
        if !root.is_null() {
            visit(unsafe { &mut *root });
        }
    }
}

/// False while values are raw machine words (mode 0), true once they are
/// tagged (mode 1). One-way: set by [`leave_raw_mode`] and never cleared.
static BIGINT_MODE: AtomicBool = AtomicBool::new(true);

/// True when values carry the tag bit.
#[inline(always)]
pub fn bigint_mode() -> bool {
    BIGINT_MODE.load(Ordering::Relaxed)
}

/// Number of completed flips out of mode 0. At most one per process; a second
/// would mean a value was read in a mode it was not written in.
static RAW_MODE_EXITS: AtomicUsize = AtomicUsize::new(0);

/// How many times this process left mode 0.
///
/// Which mode a run used is invisible in its output — both modes compute the
/// same integers — so a corpus that passes proves nothing about which one
/// produced the answers. This is what a caller reads to tell them apart.
pub fn raw_mode_exit_count() -> usize {
    RAW_MODE_EXITS.load(Ordering::Relaxed)
}

/// Report each dual-mode transition on stderr under `AHEUI_DUAL_LOG`, the way
/// `AHEUI_GC_LOG` reports collections.
fn dual_log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AHEUI_DUAL_LOG").is_some())
}

/// `AHEUI_DUAL=0` makes [`start_in_raw_mode`] a no-op, so the run stays tagged
/// from the start.
///
/// This is the control arm. Both modes compute the same integers, so a run's
/// output cannot tell you which encoding produced it, and one binary that can
/// be asked for either is what makes the comparison a comparison rather than a
/// diff between two builds.
fn dual_mode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("AHEUI_DUAL").map(|v| v != "0").unwrap_or(true))
}

/// Start the process in mode 0, before any value exists.
///
/// ⚠ Only valid before the first value is created: it reinterprets nothing, it
/// only changes how subsequent words are read. Flipping back is not offered —
/// the reverse direction has no meaning once a bigint has been allocated.
///
/// A no-op once the process has left mode 0. The mode is process-global while
/// each mainloop owns its own storage, so a second mainloop in a process that
/// already flipped must not re-enter mode 0 — the first one's boxed values
/// would be reread as raw words. (One mainloop per process is the standing
/// assumption anyway: `set_gc_roots` is a single global slot.)
pub fn start_in_raw_mode() {
    if !dual_mode_enabled() {
        if dual_log_enabled() {
            eprintln!("@@@DUAL start refused: AHEUI_DUAL=0");
        }
        return;
    }
    if RAW_MODE_EXITS.load(Ordering::Relaxed) != 0 {
        if dual_log_enabled() {
            eprintln!("@@@DUAL start refused: this process already left mode 0");
        }
        return;
    }
    if dual_log_enabled() {
        eprintln!("@@@DUAL start mode=0");
    }
    BIGINT_MODE.store(false, Ordering::Relaxed);
}

/// Leave mode 0: convert every value the storage can reach, then switch.
///
/// The caller is responsible for two things this cannot see — any operand it
/// has already popped, and staying inside a [`with_no_collect`] region until
/// its own in-flight values are tagged and reachable again.
fn leave_raw_mode() {
    debug_assert!(no_collect_active(), "the flip must not be collectable");
    // Not a `debug_assert`: an unregistered storage converts nothing while
    // reporting nothing, and the mode switch below would then reread every live
    // raw word as a tagged one. The check costs one branch on a path that runs
    // at most once per process, and the alternative to failing here is silent
    // memory corruption.
    assert!(
        crate::storage::promote_all_root_values(),
        "mode 0 was entered without a registered storage, so the flip cannot \
         reach the live values — call `storage::set_gc_roots` before \
         `start_in_raw_mode`"
    );
    BIGINT_MODE.store(true, Ordering::Relaxed);
    RAW_MODE_EXITS.fetch_add(1, Ordering::Relaxed);
    if dual_log_enabled() {
        eprintln!("@@@DUAL flip mode=1");
    }
}

/// Leave mode 0 on behalf of a binary op that has already popped `a` and `b`,
/// then apply `f` to the promoted operands.
///
/// `a` and `b` are not reachable from any root, so the whole sequence — the
/// walk, their own promotion, and `f`, which allocates again — runs inside one
/// suppressed region rather than leaving a freshly boxed operand unrooted.
#[cold]
#[inline(never)]
fn promote_and_apply(a: Val, b: Val, f: impl FnOnce(Val, Val) -> Val) -> Val {
    with_no_collect(|| {
        leave_raw_mode();
        let mut a = a;
        let mut b = b;
        promote_raw_value(&mut a);
        promote_raw_value(&mut b);
        debug_assert!(bigint_mode());
        f(a, b)
    })
}

/// Nesting depth of [`with_no_collect`]. Non-zero suppresses bignum
/// collection.
static NO_COLLECT_DEPTH: AtomicUsize = AtomicUsize::new(0);

/// True while a [`with_no_collect`] region is open.
#[inline(always)]
pub fn no_collect_active() -> bool {
    NO_COLLECT_DEPTH.load(Ordering::Relaxed) != 0
}

/// Run `f` with bignum collection suppressed.
///
/// `rgc.no_collect` states the same property statically, as
/// `func._gc_no_collect_`; a Rust build has no translation step to enforce it
/// against, so the region is marked dynamically instead.
///
/// The region this exists for is the dual-mode flip. Mid-walk the storage
/// holds visited values in tagged form and unvisited ones as raw words, and a
/// raw even number has bit 0 clear — indistinguishable from a bigint pointer.
/// A collection there would follow it. The flip allocates (every value past
/// ±2^62 boxes), and `alloc_bigint_oldgen` collects *before* each allocation,
/// so the two meet unless the region is closed.
pub fn with_no_collect<R>(f: impl FnOnce() -> R) -> R {
    struct DepthGuard;

    impl Drop for DepthGuard {
        fn drop(&mut self) {
            NO_COLLECT_DEPTH.fetch_sub(1, Ordering::Relaxed);
        }
    }

    NO_COLLECT_DEPTH.fetch_add(1, Ordering::Relaxed);
    let _guard = DepthGuard;
    f()
}

#[inline]
pub fn maybe_collect_bigints() {
    if no_collect_active() {
        return;
    }
    let hook_addr = BIGINT_MAYBE_COLLECT_HOOK.load(Ordering::Relaxed);
    if hook_addr != 0 {
        let hook: BigIntMaybeCollectHook = unsafe { std::mem::transmute(hook_addr) };
        hook();
    }
}

/// Drop both hooks. Tests register process-global hooks and must not leak them
/// into the next test.
#[cfg(test)]
pub(crate) fn clear_bigint_hooks_for_test() {
    BIGINT_ALLOC_HOOK.store(0, Ordering::Relaxed);
    BIGINT_MAYBE_COLLECT_HOOK.store(0, Ordering::Relaxed);
}

// ── Public API ──────────────────────────────────────────────────────

#[inline(always)]
pub fn val_from_i32(v: i32) -> Val {
    if bigint_mode() {
        Val::from_small(v as i64)
    } else {
        Val(v as i64)
    }
}

/// Raw tagged-word retag for JIT fast paths.
///
/// The caller guarantees `untagged` is in the small-integer range.
#[inline(always)]
pub fn val_retag_small(untagged: i64) -> Val {
    Val((untagged << 1) | 1)
}

// ── Dual mode ───────────────────────────────────────────────────────
//
// Dual mode runs untagged until the first arithmetic overflow. While it is in
// mode 0 the same word carries a plain `i64`, so nothing tags, untags or
// allocates; the flip converts every reachable value once and the tagged
// encoding above takes over for the rest of the run.

/// Mode-0 constructor: the word holds a plain untagged `i64`.
///
/// The full `i64` range is representable, unlike [`Val::from_small`] — mode 0
/// is left by an overflow of the machine word, not by leaving ±2^62.
#[inline(always)]
pub fn val_from_raw_i64(v: i64) -> Val {
    Val(v)
}

/// Mode-0 `>=` as a value, 1 or 0.
///
/// Raw word ordering, which is what mode 0 wants and what mode 1's tagged
/// encoding would get wrong for a heap value. Cannot leave mode 0.
#[inline(always)]
pub fn val_ge_raw(a: Val, b: Val) -> Val {
    Val((a.0 >= b.0) as i64)
}

/// Mode-0 division that cannot leave mode 0.
///
/// A zero divisor yields zero, as it does everywhere. The caller must have
/// ruled out `i64::MIN / -1`, the one quotient that does not fit a machine
/// word — [`val_div`] handles that by leaving mode 0, which is not available
/// from inside compiled code.
#[inline(always)]
pub fn val_div_raw(a: Val, b: Val) -> Val {
    if b.0 == 0 {
        return Val(0);
    }
    Val(a.0.wrapping_div(b.0))
}

/// Mode-0 remainder. A zero divisor yields zero; nothing else can fail.
#[inline(always)]
pub fn val_mod_raw(a: Val, b: Val) -> Val {
    if b.0 == 0 {
        return Val(0);
    }
    Val(a.0.wrapping_rem(b.0))
}

/// Convert one mode-0 raw word into its mode-1 tagged form.
///
/// ⚠ **Not idempotent.** A second application reads an already-tagged word as
/// a raw integer and doubles it. Every reachable value must be visited exactly
/// once; `storage::walk_bigint_root_values` is what guarantees that, by
/// walking the stack chains with the queue and port slots excluded and then
/// those two chains once each — `pools[VAL_QUEUE]`/`pools[VAL_PORT]` alias
/// them, so a walk over all 28 slots would visit those values twice.
#[inline]
pub fn promote_raw_value(v: &mut Val) {
    *v = Val::from_i64_promoting(v.0);
}

#[inline(always)]
pub fn val_is_zero(v: &Val) -> bool {
    if !bigint_mode() {
        return v.0 == 0;
    }
    // Small 0 is encoded as (0 << 1) | 1 = 1.
    if v.is_small() {
        v.0 == 1
    } else {
        v.as_big().is_zero()
    }
}

#[inline(always)]
pub fn val_to_i64(v: &Val) -> i64 {
    if !bigint_mode() {
        return v.0;
    }
    v.to_i64()
}

pub fn val_to_i32_saturating(v: &Val) -> i32 {
    if v.is_small() || !bigint_mode() {
        let raw = if bigint_mode() {
            v.as_i64_unchecked()
        } else {
            v.0
        };
        if raw > i32::MAX as i64 {
            i32::MAX
        } else if raw < i32::MIN as i64 {
            i32::MIN
        } else {
            raw as i32
        }
    } else if v.as_big().is_positive() {
        i32::MAX
    } else {
        i32::MIN
    }
}

// ── Arithmetic ──────────────────────────────────────────────────────

#[inline(always)]
fn binop_fast(
    a: Val,
    b: Val,
    f_small: impl FnOnce(i64, i64) -> Option<i64>,
    f_big: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Val {
    if a.is_small() & b.is_small() {
        let av = a.as_i64_unchecked();
        let bv = b.as_i64_unchecked();
        if let Some(r) = f_small(av, bv) {
            return Val::from_i64_promoting(r);
        }
        return Val::normalize_big(f_big(BigInt::from(av), BigInt::from(bv)));
    }
    Val::normalize_big(f_big(a.to_bigint(), b.to_bigint()))
}

// Mode 0 is a plain `checked_*` on the machine word. The overflow arm is the
// only exit from it, and it is cold: it fires at most once per process.

#[inline(always)]
pub fn val_add(a: Val, b: Val) -> Val {
    if !bigint_mode() {
        return match a.0.checked_add(b.0) {
            Some(r) => Val(r),
            None => promote_and_apply(a, b, val_add),
        };
    }
    binop_fast(a, b, |a, b| a.checked_add(b), |a, b| a + b)
}

#[inline(always)]
pub fn val_sub(a: Val, b: Val) -> Val {
    if !bigint_mode() {
        return match a.0.checked_sub(b.0) {
            Some(r) => Val(r),
            None => promote_and_apply(a, b, val_sub),
        };
    }
    binop_fast(a, b, |a, b| a.checked_sub(b), |a, b| a - b)
}

#[inline(always)]
pub fn val_mul(a: Val, b: Val) -> Val {
    if !bigint_mode() {
        return match a.0.checked_mul(b.0) {
            Some(r) => Val(r),
            None => promote_and_apply(a, b, val_mul),
        };
    }
    binop_fast(a, b, |a, b| a.checked_mul(b), |a, b| a * b)
}

#[inline(always)]
pub fn val_div(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return val_from_i32(0);
    }
    if !bigint_mode() {
        // `checked_div` also rejects `i64::MIN / -1`, whose true quotient does
        // not fit the word — that one is a promotion, not a zero.
        return match a.0.checked_div(b.0) {
            Some(r) => Val(r),
            None => promote_and_apply(a, b, val_div),
        };
    }
    binop_fast(a, b, |a, b| Some(a.wrapping_div(b)), |a, b| a / b)
}

#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return val_from_i32(0);
    }
    if !bigint_mode() {
        // `wrapping_rem`, not `checked_rem`: the one case the checked form
        // rejects is `i64::MIN % -1`, whose remainder is 0 and fits the word
        // perfectly — it is rejected only because the division instruction
        // traps on it. No remainder fails to fit, so mode 0 is never left here.
        return Val(a.0.wrapping_rem(b.0));
    }
    binop_fast(a, b, |a, b| Some(a.wrapping_rem(b)), |a, b| a % b)
}

#[inline(always)]
pub fn val_ge(a: &Val, b: &Val) -> bool {
    if !bigint_mode() {
        return a.0 >= b.0;
    }
    if a.is_small() & b.is_small() {
        return a.as_i64_unchecked() >= b.as_i64_unchecked();
    }
    // Debug: detect Val(0) which is an invalid tagged pointer
    assert!(
        a.0 != 0,
        "val_ge: a is Val(0) — invalid tagged pointer! b.0={}",
        b.0
    );
    assert!(
        b.0 != 0,
        "val_ge: b is Val(0) — invalid tagged pointer! a.0={}",
        a.0
    );
    val_ge_slow(a, b)
}

#[cold]
fn val_ge_slow(a: &Val, b: &Val) -> bool {
    let a = if a.is_small() {
        BigInt::from(a.as_i64_unchecked())
    } else {
        a.as_big().clone()
    };
    let b = if b.is_small() {
        BigInt::from(b.as_i64_unchecked())
    } else {
        b.as_big().clone()
    };
    a >= b
}

pub fn val_from_str(s: &str) -> Option<Val> {
    if !bigint_mode() {
        if let Ok(v) = s.parse::<i64>() {
            return Some(Val(v));
        }
        // Input wider than a machine word leaves mode 0 on its own, with no
        // arithmetic involved. There is no popped operand to fix up here.
        let big = s.parse::<BigInt>().ok()?;
        return Some(with_no_collect(|| {
            leave_raw_mode();
            Val::normalize_big(big)
        }));
    }
    if let Ok(v) = s.parse::<i64>() {
        return Some(Val::from_i64_promoting(v));
    }
    s.parse::<BigInt>().ok().map(Val::normalize_big)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every raw word that stays inside the small range becomes a tagged
    /// small; the ones that do not become heap bigints, and both readings
    /// agree with the raw word they came from.
    #[test]
    fn test_promote_raw_value_covers_both_encodings() {
        for raw in [
            0i64,
            1,
            -1,
            42,
            -42,
            SMALL_MAX,
            SMALL_MIN,
            SMALL_MAX + 1,
            SMALL_MIN - 1,
            i64::MAX,
            i64::MIN,
        ] {
            let mut v = val_from_raw_i64(raw);
            promote_raw_value(&mut v);
            assert_eq!(
                v.is_small(),
                (SMALL_MIN..=SMALL_MAX).contains(&raw),
                "wrong encoding chosen for {raw}"
            );
            assert_eq!(v.try_to_i64(), Some(raw), "value changed promoting {raw}");
        }
    }

    /// The zero a mode-0 storage holds is the raw word `0`, which as a tagged
    /// word would be a null bigint pointer — so promotion, not reinterpretation,
    /// is what makes an untouched storage readable after the flip.
    #[test]
    fn test_promote_raw_zero_is_not_a_null_pointer() {
        let mut v = val_from_raw_i64(0);
        assert!(!v.is_small(), "raw 0 has no tag bit set");
        promote_raw_value(&mut v);
        assert!(v.is_small());
        assert!(val_is_zero(&v));
    }

    /// Pinned because the walk's correctness rests on it: promoting twice is
    /// wrong, so a value reachable from two roots must not be walked twice.
    #[test]
    fn test_promote_raw_value_is_not_idempotent() {
        let mut v = val_from_raw_i64(21);
        promote_raw_value(&mut v);
        assert_eq!(v.try_to_i64(), Some(21));
        promote_raw_value(&mut v);
        assert_eq!(v.try_to_i64(), Some(43), "(21 << 1) | 1 read as a raw word");
    }
}
