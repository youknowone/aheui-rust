//! Small-integer backend. Parity with `rpaheui/aheui/int/smallint.py`.
//!
//! `smallint.py` sets `Int = int` and wraps the plain Python `int`
//! operations; RPython never picks this path (it uses `bigint.py`).
//! Here we mirror the same role: when no bigint feature is enabled,
//! `Val = i64` and arithmetic wraps on overflow (matching CPython 2's
//! `int` range as rpaheui's smallint backend did).

pub type Val = i64;

#[inline(always)]
pub fn val_from_i32(v: i32) -> Val {
    v as i64
}

#[inline(always)]
pub fn val_is_zero(v: &Val) -> bool {
    *v == 0
}

#[inline(always)]
pub fn val_to_i64(v: &Val) -> i64 {
    *v
}

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

#[inline(always)]
pub fn val_add(a: Val, b: Val) -> Val {
    a.wrapping_add(b)
}

#[inline(always)]
pub fn val_sub(a: Val, b: Val) -> Val {
    a.wrapping_sub(b)
}

#[inline(always)]
pub fn val_mul(a: Val, b: Val) -> Val {
    a.wrapping_mul(b)
}

#[inline(always)]
pub fn val_div(a: Val, b: Val) -> Val {
    if b == 0 { 0 } else { a.wrapping_div(b) }
}

#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if b == 0 { 0 } else { a.wrapping_rem(b) }
}

#[inline(always)]
pub fn val_ge(a: &Val, b: &Val) -> bool {
    *a >= *b
}

pub fn val_from_str(s: &str) -> Option<Val> {
    s.parse::<i64>().ok()
}

#[inline(always)]
pub fn with_bigint_transient_root<R>(_value: &mut Val, f: impl FnOnce() -> R) -> R {
    f()
}

#[inline(always)]
pub fn maybe_collect_bigints() {}
