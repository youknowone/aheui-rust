//! Value abstraction layer for Aheui.
//!
//! Directory layout mirrors `rpaheui/aheui/int/`:
//!   * [`smallint`] — `smallint.py`. Active when no bigint feature is set.
//!   * [`bigint`] — `bigint.py`. Active with `num-bigint`,
//!     `malachite-bigint`, or `runtime-rbigint`.
//!
//! rpaheui picks the backend at import time (`from aheui.int import bigint`
//! when targeting RPython). We pick the backend at compile time via Cargo
//! features — the active module is re-exported below so downstream code
//! keeps referring to `crate::value::{Val, val_*}` regardless of which
//! backend is live.

#[cfg(all(feature = "num-bigint", feature = "malachite-bigint"))]
compile_error!("features `num-bigint` and `malachite-bigint` are mutually exclusive");

#[cfg(not(any(
    feature = "num-bigint",
    feature = "malachite-bigint",
    feature = "runtime-rbigint"
)))]
pub mod smallint;
#[cfg(any(
    feature = "num-bigint",
    feature = "malachite-bigint",
    feature = "runtime-rbigint"
))]
pub mod bigint;
#[cfg(any(
    feature = "num-bigint",
    feature = "malachite-bigint",
    feature = "runtime-rbigint"
))]
mod bigint_backend;

#[cfg(not(any(
    feature = "num-bigint",
    feature = "malachite-bigint",
    feature = "runtime-rbigint"
)))]
pub use smallint::*;
#[cfg(any(
    feature = "num-bigint",
    feature = "malachite-bigint",
    feature = "runtime-rbigint"
))]
pub use bigint::*;

// ── Floored division ────────────────────────────────────────────────
//
// Both backends divide the same way, because both upstream files do:
// `smallint.py` spells division `r1 // r2` and `bigint.py` calls
// `rbigint.div`, and those are the same convention — the quotient rounds
// toward negative infinity and the remainder carries the divisor's sign
// (`rbigint.divmod`: "a mod b has the value a - b*floor(a/b)").
//
// Rust's `/`, `wrapping_div` and `wrapping_rem` truncate toward zero
// instead, so a pair whose signs differ needs the correction below. The two
// conventions agree whenever the signs match or the division is exact, which
// is why they are told apart only by a mixed-sign inexact pair.
//
// `div_euclid`/`rem_euclid` are a *third* convention, not this one: they
// force a non-negative remainder, so they answer `7 / -2` with `-3` where
// flooring answers `-4`.

/// Whether a truncating division of `a` by `b` needs the floor correction.
///
/// `r` is the truncated remainder, which carries `a`'s sign.
#[inline(always)]
pub(crate) fn floor_correction_needed(r: i64, b: i64) -> bool {
    r != 0 && (r < 0) != (b < 0)
}

/// `a // b` for a non-zero `b`, floored.
#[inline(always)]
pub(crate) fn floor_div_i64(a: i64, b: i64) -> i64 {
    let q = a.wrapping_div(b);
    if floor_correction_needed(a.wrapping_rem(b), b) {
        q.wrapping_sub(1)
    } else {
        q
    }
}

/// `a % b` for a non-zero `b`, floored.
///
/// The corrected remainder cannot overflow: it is only computed when `r` and
/// `b` have opposite signs, so `|r + b| < |b|`.
#[inline(always)]
pub(crate) fn floor_mod_i64(a: i64, b: i64) -> i64 {
    let r = a.wrapping_rem(b);
    if floor_correction_needed(r, b) {
        r.wrapping_add(b)
    } else {
        r
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_val_from_i32() {
        let v = val_from_i32(42);
        assert_eq!(val_to_i64(&v), 42);
    }

    #[test]
    fn test_val_is_zero() {
        assert!(val_is_zero(&val_from_i32(0)));
        assert!(!val_is_zero(&val_from_i32(1)));
    }

    #[test]
    fn test_val_add() {
        let a = val_from_i32(10);
        let b = val_from_i32(20);
        let r = val_add(a, b);
        assert_eq!(val_to_i64(&r), 30);
    }

    #[test]
    fn test_val_sub() {
        let a = val_from_i32(30);
        let b = val_from_i32(10);
        let r = val_sub(a, b);
        assert_eq!(val_to_i64(&r), 20);
    }

    #[test]
    fn test_val_div_by_zero() {
        let a = val_from_i32(10);
        let b = val_from_i32(0);
        let r = val_div(a, b);
        assert_eq!(val_to_i64(&r), 0);
    }

    #[test]
    fn test_val_ge() {
        assert!(val_ge(&val_from_i32(5), &val_from_i32(3)));
        assert!(val_ge(&val_from_i32(5), &val_from_i32(5)));
        assert!(!val_ge(&val_from_i32(3), &val_from_i32(5)));
    }

    #[test]
    fn test_val_from_str() {
        assert_eq!(val_to_i64(&val_from_str("42").unwrap()), 42);
        assert_eq!(val_to_i64(&val_from_str("-7").unwrap()), -7);
        assert!(val_from_str("abc").is_none());
    }

    #[test]
    fn test_val_negative() {
        let v = val_from_i32(-100);
        assert_eq!(val_to_i64(&v), -100);
    }

    #[test]
    fn test_val_mul_no_overflow() {
        let a = val_from_i32(1000);
        let b = val_from_i32(1000);
        let r = val_mul(a, b);
        assert_eq!(val_to_i64(&r), 1_000_000);
    }
}
