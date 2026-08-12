/// Value abstraction layer for Aheui.
///
/// Without a bigint backend feature: `Val = i64` (zero overhead).
/// With a bigint backend feature: `Val` is a tagged 64-bit word —
/// small integers are stored inline (shifted), big integers are
/// heap-allocated behind a leaked `Box<BigInt>` pointer.

#[cfg(all(feature = "num-bigint", feature = "malachite-bigint"))]
compile_error!("features `num-bigint` and `malachite-bigint` are mutually exclusive");

// Plain i64 mode.

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
pub type Val = i64;

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_from_i32(v: i32) -> Val {
    v as i64
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_is_zero(v: &Val) -> bool {
    *v == 0
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_to_i64(v: &Val) -> i64 {
    *v
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_to_i32_saturating(v: &Val) -> i32 {
    if *v > i32::MAX as i64 {
        i32::MAX
    } else if *v < i32::MIN as i64 {
        i32::MIN
    } else {
        *v as i32
    }
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_add(a: Val, b: Val) -> Val {
    a.wrapping_add(b)
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_sub(a: Val, b: Val) -> Val {
    a.wrapping_sub(b)
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_mul(a: Val, b: Val) -> Val {
    a.wrapping_mul(b)
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_div(a: Val, b: Val) -> Val {
    if b == 0 {
        0
    } else {
        ahsembler::consts::floor_div_i64(a, b)
    }
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if b == 0 {
        0
    } else {
        ahsembler::consts::floor_mod_i64(a, b)
    }
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
#[inline(always)]
pub fn val_ge(a: &Val, b: &Val) -> bool {
    *a >= *b
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
pub fn val_from_str(s: &str) -> Option<Val> {
    s.parse::<i64>().ok()
}

// Tagged-pointer bigint mode.
//
// Layout of the 64-bit word:
//   bit 0 = 1  →  small integer, value = word >> 1  (arithmetic shift)
//   bit 0 = 0  →  pointer to a leaked Box<BigInt>   (always aligned)
//
// Small integer range: −2^62 .. 2^62 − 1  (±4.6 × 10^18).

#[cfg(feature = "malachite-bigint")]
use malachite_bigint::BigInt;
#[cfg(all(feature = "num-bigint", not(feature = "malachite-bigint")))]
use num_bigint::BigInt;
#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
use num_traits::{Signed, ToPrimitive, Zero};

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
const SMALL_MIN: i64 = -(1i64 << 62);
#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
const SMALL_MAX: i64 = (1i64 << 62) - 1;

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Val(i64);

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
impl Val {
    #[inline(always)]
    fn from_small(v: i64) -> Self {
        debug_assert!(v >= SMALL_MIN && v <= SMALL_MAX);
        Val((v << 1) | 1)
    }

    #[inline(always)]
    fn from_big(b: BigInt) -> Self {
        let ptr = Box::into_raw(Box::new(b));
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
impl From<i64> for Val {
    #[inline(always)]
    fn from(v: i64) -> Self {
        Val::from_i64_promoting(v)
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_small() {
            write!(f, "{}", self.as_i64_unchecked())
        } else {
            write!(f, "{}", self.as_big())
        }
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
impl std::fmt::Debug for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_small() {
            write!(f, "Val::Small({})", self.as_i64_unchecked())
        } else {
            write!(f, "Val::Big({})", self.as_big())
        }
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_from_i32(v: i32) -> Val {
    Val::from_small(v as i64)
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_is_zero(v: &Val) -> bool {
    // Small 0 is encoded as (0 << 1) | 1 = 1.
    if v.is_small() {
        v.0 == 1
    } else {
        v.as_big().is_zero()
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_to_i64(v: &Val) -> i64 {
    v.to_i64()
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub fn val_to_i32_saturating(v: &Val) -> i32 {
    if v.is_small() {
        let raw = v.as_i64_unchecked();
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline]
fn floor_correction_needed_big(r: &BigInt, b: &BigInt) -> bool {
    !r.is_zero() && (r.is_negative() != b.is_negative())
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
fn floor_div_big(a: BigInt, b: BigInt) -> BigInt {
    let q = &a / &b;
    if floor_correction_needed_big(&(a % &b), &b) {
        q - BigInt::from(1)
    } else {
        q
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
fn floor_mod_big(a: BigInt, b: BigInt) -> BigInt {
    let r = a % &b;
    if floor_correction_needed_big(&r, &b) {
        r + b
    } else {
        r
    }
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_add(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_add(b), |a, b| a + b)
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_sub(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_sub(b), |a, b| a - b)
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_mul(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_mul(b), |a, b| a * b)
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_div(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return Val::from_small(0);
    }
    binop_fast(
        a,
        b,
        |a, b| {
            a.checked_div(b)
                .map(|_| ahsembler::consts::floor_div_i64(a, b))
        },
        floor_div_big,
    )
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return Val::from_small(0);
    }
    binop_fast(
        a,
        b,
        |a, b| Some(ahsembler::consts::floor_mod_i64(a, b)),
        floor_mod_big,
    )
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[inline(always)]
pub fn val_ge(a: &Val, b: &Val) -> bool {
    if a.is_small() & b.is_small() {
        return a.as_i64_unchecked() >= b.as_i64_unchecked();
    }
    // Zero is not a valid tagged representation.
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub fn val_from_str(s: &str) -> Option<Val> {
    if let Ok(v) = s.parse::<i64>() {
        return Some(Val::from_i64_promoting(v));
    }
    s.parse::<BigInt>().ok().map(Val::normalize_big)
}

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
    fn division_and_remainder_are_floored() {
        let cases = [
            (-7, 2, -4, 1),
            (7, -2, -4, -1),
            (-13, 10, -2, 7),
            (13, -10, -2, -7),
            (12, -3, -4, 0),
        ];

        for (a, b, quotient, remainder) in cases {
            let a = val_from_str(&a.to_string()).unwrap();
            let b = val_from_str(&b.to_string()).unwrap();
            assert_eq!(val_to_i64(&val_div(a, b)), quotient);
            assert_eq!(val_to_i64(&val_mod(a, b)), remainder);
        }
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
