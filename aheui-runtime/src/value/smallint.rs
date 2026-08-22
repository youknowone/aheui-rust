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

/// Mode-0 constructor: `Val` is the word itself here, so this is identity.
#[inline(always)]
pub fn val_from_raw_i64(v: i64) -> Val {
    v
}

/// The packed word behind a `Val`. Identity in this backend.
#[inline(always)]
pub fn val_as_raw_i64(v: Val) -> i64 {
    v
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
    if b == 0 {
        0
    } else {
        super::floor_div_i64(a, b)
    }
}

#[inline(always)]
pub fn val_mod(a: Val, b: Val) -> Val {
    if b == 0 {
        0
    } else {
        super::floor_mod_i64(a, b)
    }
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

#[inline(always)]
pub fn no_collect_active() -> bool {
    false
}

/// This backend is permanently untagged, so mode 0 is the only mode it has.
#[inline(always)]
pub fn start_in_raw_mode() {}

/// Mode 0 is never left here: there is no tagged mode to leave it for.
#[inline(always)]
pub fn raw_mode_exit_count() -> usize {
    0
}

/// Always false: there is no tagged mode to reach.
#[inline(always)]
pub fn bigint_mode() -> bool {
    false
}

/// No bignum exists in this backend, so there is nothing to suppress.
#[inline(always)]
pub fn with_no_collect<R>(f: impl FnOnce() -> R) -> R {
    f()
}
