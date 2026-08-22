//! Arithmetic over operand words held outside a node chain.
//!
//! `linkedlist_jit.rs`'s helpers each do two things: reach the top two nodes of
//! a chain, and combine their values. When the operands already sit in a
//! virtualizable band there is no chain to reach through, so only the second
//! half is wanted. These are that half — the same fast paths, over the packed
//! `Val` word rather than over `Node.value`.
//!
//! Every entry takes and returns the packed word, never a `Val`. A `Val` is
//! `#[repr(transparent)]` over that word, so the two are the same bits; taking
//! the word is what keeps the fast paths free of a conversion the lowerer would
//! have to spell as a call. The slow paths, reached only when an operand is a
//! heap value or the result leaves the fast range, convert and call the shared
//! `val_*` implementation.
//!
//! Each operation comes in a mode-1 form, which works on the tagged encoding,
//! and a mode-0 `_raw` twin, which works on the plain machine word. The caller
//! picks on the same `bigint_mode` green the chain helpers are picked on.
//!
//! These carry no `#[jit_inline]` attribute. A helper that returns a value
//! reaches the trace through the graph pipeline instead, the way
//! `linkedlist::pop_base_known_nonempty` does; the caller names the policy.

use crate::value::*;
use crate::value::{floor_div_i64, floor_mod_i64};

/// Tag a value known to fit the small range.
///
/// `Val::from_small`'s encoding, written out: the low bit marks a small
/// integer, so the value shifts up one and the marker goes in underneath.
/// Spelling it here rather than calling the constructor keeps the fast path
/// free of a call the lowerer would otherwise have to emit or fold.
#[cfg(feature = "bigint-backend")]
macro_rules! tag_small {
    ($v:expr) => {
        ($v << 1) | 1
    };
}

/// Both operands are tagged small integers.
///
/// The tag bit is the low bit of each word, so one AND tests both at once.
#[cfg(feature = "bigint-backend")]
macro_rules! both_small {
    ($a:expr, $b:expr) => {
        (($a & $b) & 1) != 0
    };
}

/// The value survives a round trip through the tag, i.e. it fits 63 bits.
#[cfg(feature = "bigint-backend")]
macro_rules! fits_small {
    ($v:expr) => {
        ((($v << 1) >> 1) == $v)
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_add(r2: i64, r1: i64) -> i64 {
    let sum = (r2 >> 1) + (r1 >> 1);
    if both_small!(r1, r2) & fits_small!(sum) {
        tag_small!(sum)
    } else {
        promote_add(r2, r1)
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_sub(r2: i64, r1: i64) -> i64 {
    let diff = (r2 >> 1) - (r1 >> 1);
    if both_small!(r1, r2) & fits_small!(diff) {
        tag_small!(diff)
    } else {
        promote_sub(r2, r1)
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_mul(r2: i64, r1: i64) -> i64 {
    // Sign-extending each operand to 32 bits bounds the product to a word, so
    // the fast path needs no overflow check beyond the range tests themselves.
    let av = r2 >> 1;
    let bv = r1 >> 1;
    let av32 = (av << 32) >> 32;
    let bv32 = (bv << 32) >> 32;
    let prod = av32 * bv32;
    if both_small!(r1, r2) & (av32 == av) & (bv32 == bv) & fits_small!(prod) {
        tag_small!(prod)
    } else {
        promote_mul(r2, r1)
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_div(r2: i64, r1: i64) -> i64 {
    promote_div(r2, r1)
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_mod(r2: i64, r1: i64) -> i64 {
    promote_mod(r2, r1)
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_cmp(r2: i64, r1: i64) -> i64 {
    if both_small!(r1, r2) {
        // Two tagged small integers order the same way their words do, so the
        // comparison needs no untag.
        tag_small!((r2 >= r1) as i64)
    } else {
        compare_ge(r2, r1)
    }
}

/// `@jit.dont_look_inside` for [`compare_ge`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_compare_ge: bool = false;

/// Heap-value escape for [`band_cmp`].
///
/// Out of line for the same reason as the `promote_*` escapes, and because a
/// second branch inside the slow arm would leave the jitcode ending in a join
/// rather than in the typed return `inline_pipeline_int` reads.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn compare_ge(r2: i64, r1: i64) -> i64 {
    // `val_ge` answers a bool; `cmp` pushes it as a value word.
    let ge = val_ge(&val_from_raw_i64(r2), &val_from_raw_i64(r1));
    val_as_raw_i64(val_from_i32(ge as i32))
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_add_raw(r2: i64, r1: i64) -> i64 {
    // A mode-0 word is the value, so an overflow is the only thing that can
    // leave the mode, and `wrapping_` plus the check the compiler already
    // emits is what `checked_` is.
    match r2.checked_add(r1) {
        Some(sum) => sum,
        None => promote_add(r2, r1),
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_sub_raw(r2: i64, r1: i64) -> i64 {
    match r2.checked_sub(r1) {
        Some(diff) => diff,
        None => promote_sub(r2, r1),
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_mul_raw(r2: i64, r1: i64) -> i64 {
    match r2.checked_mul(r1) {
        Some(prod) => prod,
        None => promote_mul(r2, r1),
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_div_raw(r2: i64, r1: i64) -> i64 {
    // Divisor −1 is guarded out wholesale: it is the only divisor whose
    // quotient can fail to fit the word (`i64::MIN / -1`), and handling that
    // means leaving mode 0, which compiled code cannot do.
    if r1 == -1 {
        promote_div(r2, r1)
    } else if r1 == 0 {
        0
    } else {
        floor_div_i64(r2, r1)
    }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_mod_raw(r2: i64, r1: i64) -> i64 {
    // Every remainder fits the word, `i64::MIN % -1` included.
    if r1 == 0 { 0 } else { floor_mod_i64(r2, r1) }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
pub fn band_cmp_raw(r2: i64, r1: i64) -> i64 {
    // A mode-0 word orders as itself, and 1/0 is already its own word.
    (r2 >= r1) as i64
}

// The escapes every band operation leaves through when its fast path does not
// apply.
//
// Each is `#[inline(never)]` so the graph pipeline emits a call to it by path
// rather than lowering `val_*` — which reaches closures and generic helpers the
// pipeline has no address for. A host that names a band helper is therefore
// binding exactly these six paths, and nothing below them.
//
// Each carries a `_jit_look_inside_*` marker const: what `@jit.dont_look_inside`
// would emit, spelled out so it survives without the `jit` feature's proc macros
// in scope. It is what tells the pipeline to call the function rather than lower
// it.

/// `@jit.dont_look_inside` for [`promote_add`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_promote_add: bool = false;

/// Overflow escape for `add`, in both modes.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn promote_add(r2: i64, r1: i64) -> i64 {
    val_as_raw_i64(val_add(val_from_raw_i64(r2), val_from_raw_i64(r1)))
}

/// `@jit.dont_look_inside` for [`promote_sub`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_promote_sub: bool = false;

/// Overflow escape for `sub`.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn promote_sub(r2: i64, r1: i64) -> i64 {
    val_as_raw_i64(val_sub(val_from_raw_i64(r2), val_from_raw_i64(r1)))
}

/// `@jit.dont_look_inside` for [`promote_mul`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_promote_mul: bool = false;

/// Overflow escape for `mul`.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn promote_mul(r2: i64, r1: i64) -> i64 {
    val_as_raw_i64(val_mul(val_from_raw_i64(r2), val_from_raw_i64(r1)))
}

/// `@jit.dont_look_inside` for [`promote_div`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_promote_div: bool = false;

/// The whole of `div` in mode 1, and in mode 0 the one quotient that does not
/// fit a word.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn promote_div(r2: i64, r1: i64) -> i64 {
    val_as_raw_i64(val_div(val_from_raw_i64(r2), val_from_raw_i64(r1)))
}

/// `@jit.dont_look_inside` for [`promote_mod`].
#[doc(hidden)]
#[allow(non_upper_case_globals)]
#[cfg(feature = "bigint-backend")]
const _jit_look_inside_promote_mod: bool = false;

/// The whole of `mod` in mode 1.
#[cfg(feature = "bigint-backend")]
#[inline(never)]
fn promote_mod(r2: i64, r1: i64) -> i64 {
    val_as_raw_i64(val_mod(val_from_raw_i64(r2), val_from_raw_i64(r1)))
}

/// The mode-appropriate form of each operation, over `Val`.
///
/// `LinkedList`'s arithmetic is defined through these so the fast paths have
/// one implementation rather than two: the chain reaches them here, and a
/// caller holding its operands outside a chain reaches the word forms above.
#[cfg(feature = "bigint-backend")]
macro_rules! band_val_op {
    ($name:ident, $tagged:ident, $raw:ident) => {
        #[inline(always)]
        pub fn $name(a: Val, b: Val) -> Val {
            let (aw, bw) = (val_as_raw_i64(a), val_as_raw_i64(b));
            val_from_raw_i64(if bigint_mode() {
                $tagged(aw, bw)
            } else {
                $raw(aw, bw)
            })
        }
    };
}

#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_add, band_add, band_add_raw);
#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_sub, band_sub, band_sub_raw);
#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_mul, band_mul, band_mul_raw);
#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_div, band_div, band_div_raw);
#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_mod, band_mod, band_mod_raw);
#[cfg(feature = "bigint-backend")]
band_val_op!(band_val_cmp, band_cmp, band_cmp_raw);
