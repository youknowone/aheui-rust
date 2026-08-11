use num_traits::ToPrimitive;

const SMALL_MIN: i64 = -(1i64 << 62);
const SMALL_MAX: i64 = (1i64 << 62) - 1;

fn to_big(value: i64) -> BigInt {
    if value & 1 != 0 {
        BigInt::from(value >> 1)
    } else {
        unsafe { &*(value as *const BigInt) }.clone()
    }
}

fn normalize(value: BigInt) -> i64 {
    match value.to_i64() {
        Some(small) if (SMALL_MIN..=SMALL_MAX).contains(&small) => (small << 1) | 1,
        _ => Box::into_raw(Box::new(value)) as i64,
    }
}

fn divmod_floor(a: BigInt, b: BigInt) -> (BigInt, BigInt) {
    let mut q = a.clone() / b.clone();
    let mut r = a % b.clone();
    if r != BigInt::from(0) && (r < BigInt::from(0)) != (b < BigInt::from(0)) {
        q -= BigInt::from(1);
        r += b;
    }
    (q, r)
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_from_i64(value: i64) -> i64 {
    normalize(BigInt::from(value))
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_add(a: i64, b: i64) -> i64 {
    normalize(to_big(a) + to_big(b))
}
#[unsafe(no_mangle)]
pub extern "C" fn cbig_sub(a: i64, b: i64) -> i64 {
    normalize(to_big(a) - to_big(b))
}
#[unsafe(no_mangle)]
pub extern "C" fn cbig_mul(a: i64, b: i64) -> i64 {
    normalize(to_big(a) * to_big(b))
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_div(a: i64, b: i64) -> i64 {
    let (a, b) = (to_big(a), to_big(b));
    if b == BigInt::from(0) {
        1
    } else {
        normalize(divmod_floor(a, b).0)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_rem(a: i64, b: i64) -> i64 {
    let (a, b) = (to_big(a), to_big(b));
    if b == BigInt::from(0) {
        1
    } else {
        normalize(divmod_floor(a, b).1)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_ge(a: i64, b: i64) -> i32 {
    (to_big(a) >= to_big(b)) as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_to_i64(value: i64) -> i64 {
    unsafe { &*(value as *const BigInt) }.to_i64().unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_write_num(value: i64, emit: extern "C" fn(u8)) {
    for byte in unsafe { &*(value as *const BigInt) }.to_string().bytes() {
        emit(byte);
    }
}

unsafe extern "C" {
    fn compaheuiler_c_entry() -> i64;
}

fn main() {
    let result = unsafe { compaheuiler_c_entry() };
    std::process::exit(result as i32);
}
