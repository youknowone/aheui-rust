use num_traits::ToPrimitive;
use std::collections::VecDeque;

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
        _ => alloc_bigint(value),
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
    if value & 1 != 0 {
        value >> 1
    } else {
        unsafe { &*(value as *const BigInt) }.to_i64().unwrap_or(0)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn cbig_write_num(value: i64, emit: extern "C" fn(u8)) {
    let text = if value & 1 != 0 {
        (value >> 1).to_string()
    } else {
        unsafe { &*(value as *const BigInt) }.to_string()
    };
    for byte in text.bytes() {
        emit(byte);
    }
}

const QUEUE: usize = 21;
const PORT: usize = 27;

/// The queue (selector 21) and the port (selector 27).
///
/// Both back ends reach this one implementation: the generated Rust uses the
/// same `VecDeque`/`Vec` shape directly, and the generated C carries only an
/// opaque handle and calls the `csp_*` entry points below. Growing on demand
/// is why the C side has no capacity left to overflow.
struct SpecialStorage {
    queue: VecDeque<i64>,
    port: Vec<i64>,
    port_last: i64,
    big_mode: bool,
}

impl SpecialStorage {
    /// What a pop off an empty storage yields: the zero of the current
    /// representation, which in tagged mode is the small immediate 0.
    fn empty(&self) -> i64 {
        if self.big_mode { 1 } else { 0 }
    }
}

fn promote_val(value: i64) -> i64 {
    if (SMALL_MIN..=SMALL_MAX).contains(&value) {
        (value << 1) | 1
    } else {
        cbig_from_i64(value)
    }
}

/// The C side spells the storage as `void *`, so the type stays private here.
type Handle = *mut core::ffi::c_void;

fn storage<'a>(handle: Handle) -> &'a mut SpecialStorage {
    unsafe { &mut *(handle as *mut SpecialStorage) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn cbig_collect(bases: *const [*mut i64; 28], tops: *const [*mut i64; 28], handle: Handle) {
    let mut roots = Vec::new();
    unsafe { bigint_stack_roots(&mut roots, &*bases, &*tops); }
    let s = storage(handle);
    for &v in s.queue.iter().chain(s.port.iter()).chain(std::iter::once(&s.port_last)) {
        bigint_root(&mut roots, v);
    }
    collect_bigint_roots(roots);
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_new() -> Handle {
    let storage = Box::new(SpecialStorage {
        queue: VecDeque::new(),
        port: Vec::new(),
        port_last: 0,
        big_mode: false,
    });
    Box::into_raw(storage) as Handle
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_push(handle: Handle, sel: usize, value: i64) {
    let s = storage(handle);
    if sel == QUEUE {
        s.queue.push_back(value);
    } else {
        s.port_last = value;
        s.port.push(value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_pop(handle: Handle, sel: usize) -> i64 {
    let s = storage(handle);
    let empty = s.empty();
    if sel == QUEUE {
        s.queue.pop_front().unwrap_or(empty)
    } else {
        s.port.pop().unwrap_or(empty)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_depth(handle: Handle, sel: usize) -> usize {
    let s = storage(handle);
    if sel == QUEUE {
        s.queue.len()
    } else {
        s.port.len()
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_dup(handle: Handle, sel: usize) {
    let s = storage(handle);
    if sel == QUEUE {
        if let Some(&value) = s.queue.front() {
            s.queue.push_front(value);
        }
    } else {
        s.port.push(s.port_last);
    }
}

/// Duplicate the queue front onto the back: one rotation step, which is what a
/// duplicate immediately followed by a move to the same queue amounts to.
#[unsafe(no_mangle)]
pub extern "C" fn csp_dup_back(handle: Handle) {
    let s = storage(handle);
    if let Some(&value) = s.queue.front() {
        s.queue.push_back(value);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_swap(handle: Handle, sel: usize) {
    let s = storage(handle);
    if sel == QUEUE && s.queue.len() >= 2 {
        s.queue.swap(0, 1);
    } else if sel == PORT && s.port.len() >= 2 {
        let n = s.port.len();
        s.port.swap(n - 1, n - 2);
    }
}

/// Move every element up to and including the first zero to the back. A queue
/// holding no zero is left as it was.
#[unsafe(no_mangle)]
pub extern "C" fn csp_scan_to_zero(handle: Handle) {
    let s = storage(handle);
    let zero = s.empty();
    if let Some(pos) = s.queue.iter().position(|value| *value == zero) {
        s.queue.rotate_left(pos + 1);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn csp_promote(handle: Handle) {
    let s = storage(handle);
    for value in &mut s.queue {
        *value = promote_val(*value);
    }
    for value in &mut s.port {
        *value = promote_val(*value);
    }
    s.port_last = promote_val(s.port_last);
    s.big_mode = true;
}

unsafe extern "C" {
    fn compaheuiler_c_entry() -> i64;
}

fn main() {
    let result = unsafe { compaheuiler_c_entry() };
    clear_bigints();
    std::process::exit(result as i32);
}
