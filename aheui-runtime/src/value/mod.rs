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

pub use ahsembler::consts::{floor_correction_mask, floor_div_i64, floor_mod_i64};

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
