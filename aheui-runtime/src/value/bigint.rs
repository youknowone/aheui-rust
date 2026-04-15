//! Big-integer backend. Parity with `rpaheui/aheui/int/bigint.py`.
//!
//! `bigint.py` sets `Int = rbigint` and wraps RPython's rbigint. This
//! is the path RPython translates. Our Rust port uses a tagged-pointer
//! layout so the hot small-int path stays allocation-free:
//!
//!   bit 0 = 1  →  small integer, value = word >> 1  (arithmetic shift)
//!   bit 0 = 0  →  pointer to a leaked `Box<BigInt>` (always aligned)
//!
//! Small-integer range: −2^62 .. 2^62 − 1 (±4.6 × 10^18).

#[cfg(feature = "malachite-bigint")]
use malachite_bigint::BigInt;
#[cfg(all(feature = "num-bigint", not(feature = "malachite-bigint")))]
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};

const SMALL_MIN: i64 = -(1i64 << 62);
const SMALL_MAX: i64 = (1i64 << 62) - 1;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Val(i64);

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

impl From<i64> for Val {
    #[inline(always)]
    fn from(v: i64) -> Self {
        Val::from_i64_promoting(v)
    }
}

impl std::fmt::Display for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_small() {
            write!(f, "{}", self.as_i64_unchecked())
        } else {
            write!(f, "{}", self.as_big())
        }
    }
}

impl std::fmt::Debug for Val {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_small() {
            write!(f, "Val::Small({})", self.as_i64_unchecked())
        } else {
            write!(f, "Val::Big({})", self.as_big())
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────

#[inline(always)]
pub fn val_from_i32(v: i32) -> Val {
    Val::from_small(v as i64)
}

#[inline(always)]
pub fn val_is_zero(v: &Val) -> bool {
    // Small 0 is encoded as (0 << 1) | 1 = 1.
    if v.is_small() {
        v.0 == 1
    } else {
        v.as_big().is_zero()
    }
}

#[inline(always)]
pub fn val_to_i64(v: &Val) -> i64 {
    v.to_i64()
}

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

#[inline(always)]
pub fn val_add(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_add(b), |a, b| a + b)
}

#[inline(always)]
pub fn val_sub(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_sub(b), |a, b| a - b)
}

#[inline(always)]
pub fn val_mul(a: Val, b: Val) -> Val {
    binop_fast(a, b, |a, b| a.checked_mul(b), |a, b| a * b)
}

#[inline(always)]
pub fn val_div(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return Val::from_small(0);
    }
    binop_fast(a, b, |a, b| Some(a.wrapping_div(b)), |a, b| a / b)
}

#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if val_is_zero(&b) {
        return Val::from_small(0);
    }
    binop_fast(a, b, |a, b| Some(a.wrapping_rem(b)), |a, b| a % b)
}

#[inline(always)]
pub fn val_ge(a: &Val, b: &Val) -> bool {
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
    if let Ok(v) = s.parse::<i64>() {
        return Some(Val::from_i64_promoting(v));
    }
    s.parse::<BigInt>().ok().map(Val::normalize_big)
}
