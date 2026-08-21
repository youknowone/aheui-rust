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

#[cfg(feature = "bigint-backend")]
pub mod bigint;
#[cfg(feature = "bigint-backend")]
mod bigint_backend;
#[cfg(not(feature = "bigint-backend"))]
pub mod smallint;

#[cfg(feature = "bigint-backend")]
pub use bigint::*;
#[cfg(not(feature = "bigint-backend"))]
pub use smallint::*;

// Floored division shared by both value backends.
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

/// All ones when a truncating division of `a` by `b` needs the floor
/// correction, zero otherwise. `r` is the truncated remainder, which carries
/// `a`'s sign.
///
/// A mask rather than a `bool` because the caller adds it: spelled as a branch,
/// the correction is a run-time test on the remainder, and a JIT compiling the
/// division records one arm of it and guards the other. logo's banded DIV made
/// that guard fail 62115 times. Bit 63 of `r | -r` is set exactly when `r` is
/// non-zero, and bit 63 of `r ^ b` exactly when the two signs differ, so the
/// arithmetic shift broadcasts their conjunction with nothing to guard.
///
/// The negation is spelled as a subtraction because the graph pipeline lowers
/// binary integer arithmetic to IR ops but leaves `wrapping_neg` a call to a
/// path no host binds, which reaches compiled code as an unresolved target.
#[inline(always)]
pub(crate) fn floor_correction_mask(r: i64, b: i64) -> i64 {
    ((r | 0i64.wrapping_sub(r)) & (r ^ b)) >> 63
}

/// `a // b` for a non-zero `b`, floored.
#[inline(always)]
pub(crate) fn floor_div_i64(a: i64, b: i64) -> i64 {
    let q = a.wrapping_div(b);
    let r = a.wrapping_rem(b);
    q.wrapping_add(floor_correction_mask(r, b))
}

/// `a % b` for a non-zero `b`, floored.
///
/// The corrected remainder cannot overflow: it is only computed when `r` and
/// `b` have opposite signs, so `|r + b| < |b|`.
#[inline(always)]
pub(crate) fn floor_mod_i64(a: i64, b: i64) -> i64 {
    let r = a.wrapping_rem(b);
    r.wrapping_add(b & floor_correction_mask(r, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mask has to agree with the branching form it replaced everywhere,
    /// including where the remainder is zero and where either operand is
    /// negative — the cases the two spellings can disagree on.
    #[test]
    fn floor_correction_mask_matches_the_branching_form() {
        fn branching(a: i64, b: i64) -> (i64, i64) {
            let q = a.wrapping_div(b);
            let r = a.wrapping_rem(b);
            let corrected = r != 0 && (r < 0) != (b < 0);
            if corrected {
                (q.wrapping_sub(1), r.wrapping_add(b))
            } else {
                (q, r)
            }
        }
        let mut operands: Vec<i64> = (-40..=40).collect();
        operands.extend([
            i64::MIN,
            i64::MIN + 1,
            i64::MAX,
            i64::MAX - 1,
            1 << 40,
            -1 << 40,
        ]);
        for &a in &operands {
            for &b in &operands {
                if b == 0 {
                    continue;
                }
                assert_eq!(
                    (floor_div_i64(a, b), floor_mod_i64(a, b)),
                    branching(a, b),
                    "a={a} b={b}"
                );
            }
        }
    }

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
