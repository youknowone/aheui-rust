//! Value abstraction layer for Aheui.
//!
//! Directory layout mirrors `rpaheui/aheui/int/`:
//!   * [`smallint`] — `smallint.py`. Active when no bigint feature is set.
//!   * [`bigint`] — `bigint.py`. Active with `num-bigint` or
//!     `malachite-bigint`.
//!
//! rpaheui picks the backend at import time (`from aheui.int import bigint`
//! when targeting RPython). We pick the backend at compile time via Cargo
//! features — the active module is re-exported below so downstream code
//! keeps referring to `crate::value::{Val, val_*}` regardless of which
//! backend is live.

#[cfg(all(feature = "num-bigint", feature = "malachite-bigint"))]
compile_error!("features `num-bigint` and `malachite-bigint` are mutually exclusive");

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub mod bigint;
#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
pub mod smallint;

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub use bigint::*;
#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
pub use smallint::*;

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
