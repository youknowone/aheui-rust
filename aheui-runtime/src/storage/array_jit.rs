//! Monomorphic JIT helpers for array-backed storage ops.
//!
//! The linked-list counterpart is `linkedlist_jit`; this module is the same
//! surface over a contiguous buffer. Every helper here reads and writes
//! `data[size]` instead of walking a node chain, so a push is one element
//! store and a size bump rather than an allocation plus four stores, and a
//! two-operand op is two element loads and one element store rather than three
//! dependent pointer loads, three stores and a free.
//!
//! Growth is not expressible here. `data` moves when a push outgrows `cap`,
//! and a base pointer read before that move must not be reused after it. The
//! caller tests `size < cap` and leaves the reallocating push to the
//! interpreter, so the compiled path never reallocates and the trace exits
//! through an ordinary guard on the O(log n) pushes that do.
//!
//! Only the stack family is inlined. The queue is a ring: its element index is
//! `(front + i) % cap`, so every access carries a division the stack's does
//! not, and its pop moves `front` rather than `size` alone. Those helpers stay
//! residual, as the port already is.
//!
//! The arg type is `usize` (a raw pointer reinterpreted as an integer), not
//! `*mut Stack`: the JIT macro carries storage handles in the ref register
//! bank and takes the pointee shape from the `ref(...)` metadata, without
//! changing the concrete Rust ABI.

use super::array::{ArrayBase, ArrayStorage, Port, Queue};
#[cfg(not(feature = "bigint-backend"))]
use super::array::Stack;
use crate::value::*;

// The element descr is declared as `i64` rather than `Val`: the descr needs
// only the element width and signedness, `Val` is a `#[repr(transparent)]`
// wrapper around `i64` in the bigint backend and an alias for it in the
// smallint one, and the declaration site needs a type with `MIN` on it.

// Stack helpers.

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_push(stack: usize, value: Val) {
    // The caller has established `size < cap`; see the module header.
    stack.data[stack.size] = value;
    stack.size = stack.size + 1u32;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_pop_known_nonempty(stack: usize) -> Val {
    let top = stack.size - 1u32;
    stack.size = top;
    stack.data[top]
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_swap_known_two(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.data[top] = r2;
    stack.data[below] = r1;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_dup(stack: usize) {
    // The caller has established `size < cap`; see the module header.
    let top = stack.size - 1u32;
    let top_val = stack.data[top];
    stack.data[stack.size] = top_val;
    stack.size = stack.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_add => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_add(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    let sum = (r2 >> 1) + (r1 >> 1);
    stack.data[below] = if ((r1 & r2) & 1 != 0) & (((sum << 1) >> 1) == sum) {
        val_retag_small(sum)
    } else {
        val_add(r2, r1)
    };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_add => elidable_int,
    },
    native_int_binops = { val_add => IntAdd },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_add(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_add(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_sub => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_sub(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    let diff = (r2 >> 1) - (r1 >> 1);
    stack.data[below] = if ((r1 & r2) & 1 != 0) & (((diff << 1) >> 1) == diff) {
        val_retag_small(diff)
    } else {
        val_sub(r2, r1)
    };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_sub => elidable_int,
    },
    native_int_binops = { val_sub => IntSub },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_sub(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_sub(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_mul => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_mul(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    // Native smallint mul fast path: untag, sign-extend each operand to 32
    // bits so the i64 product can never overflow, multiply, and re-tag —
    // falling back to val_mul when an operand is a bigint or exceeds the
    // ±2^31 fast range.
    let av = r2 >> 1;
    let bv = r1 >> 1;
    let av32 = (av << 32) >> 32;
    let bv32 = (bv << 32) >> 32;
    let prod = av32 * bv32;
    stack.data[below] =
        if ((r1 & r2) & 1 != 0) & (av32 == av) & (bv32 == bv) & (((prod << 1) >> 1) == prod) {
            val_retag_small(prod)
        } else {
            val_mul(r2, r1)
        };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_mul => elidable_int,
    },
    native_int_binops = { val_mul => IntMul },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_mul(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_mul(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_div => elidable_int,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_div(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_div(r2, r1);
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
pub fn stack_div(stack: usize) {
    unsafe { (*(stack as *mut Stack)).div() }
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_mod => elidable_int,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_mod(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_mod(r2, r1);
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
pub fn stack_mod(stack: usize) {
    unsafe { (*(stack as *mut Stack)).modulo() }
}


#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_ge_jit => elidable_int_cannot_raise,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_cmp(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_ge_jit(r2, r1);
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_cmp(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = (r2 >= r1) as i64;
}

/// `val_ge` wrapper returning Val (0 or 1) instead of bool.
/// JIT type: `(Val, Val) -> Val`.
#[inline(always)]
pub fn val_ge_jit(a: Val, b: Val) -> Val {
    val_from_i32(val_ge(&a, &b) as i32)
}

// Mode-0 twins. The word is the value, so the arithmetic is native and the
// conversion appears only in the `None` arm of a `checked_*` — which the tracer
// records as a `guard_no_overflow`, running the conversion after the deopt, in
// the interpreter, where the storage is materialised.
//
// `>> 0` reads the mode-0 word unchanged, the way `>> 1` untags in the tagged
// twins. It is written this way — rather than through an accessor — because an
// unregistered call is silently skipped by the lowerer rather than rejected.

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_add => elidable_int,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_add_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = match r2.checked_add(r1) {
        Some(sum) => sum,
        None => val_add(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_sub => elidable_int,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_sub_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = match r2.checked_sub(r1) {
        Some(diff) => diff,
        None => val_sub(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_mul => elidable_int,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_mul_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = match r2.checked_mul(r1) {
        Some(prod) => prod,
        None => val_mul(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_ge_raw => elidable_int_cannot_raise,
    },
    native_int_binops = { val_ge_raw => IntGe },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_cmp_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    stack.data[below] = val_ge_raw(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_div => elidable_int,
        val_div_raw => elidable_int_cannot_raise,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_div_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    // Divisor −1 is guarded out wholesale: it is the only divisor for which a
    // quotient can fail to fit the word (`i64::MIN / -1`), and `val_div` leaves
    // mode 0 for that one. Testing the divisor alone keeps the test to a single
    // compare, and the arm the tracer records for it puts the conversion in the
    // interpreter.
    stack.data[below] = if (r1 >> 0) == -1 {
        val_div(r2, r1)
    } else {
        val_div_raw(r2, r1)
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::array::Stack),
    },
    int_fields = {
        super::array::ArrayBase::size => u32,
        super::array::ArrayBase::cap => u32,
    },
    array_fields = {
        super::array::ArrayBase::data => i64,
    },
    calls = {
        val_mod_raw => elidable_int_cannot_raise,
    },
    inlined_prefix = {
        super::array::Stack::base => super::array::ArrayBase,
    },
)]
pub fn stack_mod_raw(stack: usize) {
    let top = stack.size - 1u32;
    let below = stack.size - 2u32;
    let r1 = stack.data[top];
    let r2 = stack.data[below];
    stack.size = top;
    // No guard: every remainder fits the word, `i64::MIN % -1` included.
    stack.data[below] = val_mod_raw(r2, r1);
}

// Queue and port helpers.
//
// Concrete ABI shims over the trait implementations: the queue's ring index
// carries a division the stack's does not (module header), and the port is the
// I/O storage and off the inner hot path. Both reach the JIT as residual calls.

macro_rules! residual_storage_op {
    ($name:ident, $ty:ty, $method:ident) => {
        #[inline(always)]
        pub fn $name(storage: usize) {
            unsafe { (*(storage as *mut $ty)).$method() }
        }
    };
}

residual_storage_op!(queue_add, Queue, add);
residual_storage_op!(queue_sub, Queue, sub);
residual_storage_op!(queue_mul, Queue, mul);
residual_storage_op!(queue_div, Queue, div);
residual_storage_op!(queue_mod, Queue, modulo);
residual_storage_op!(queue_cmp, Queue, cmp);
residual_storage_op!(queue_dup, Queue, dup);
residual_storage_op!(queue_swap, Queue, swap);

#[inline(always)]
pub fn queue_push(queue: usize, value: Val) {
    unsafe { (*(queue as *mut Queue)).push(value) }
}

#[inline(always)]
pub fn queue_pop(queue: usize) -> Val {
    unsafe { (*(queue as *mut Queue)).pop() }
}

#[cfg(feature = "bigint-backend")]
mod raw_aliases {
    // Mode-0 twins of the residual queue ops. The residual path runs the same
    // trait method in either mode — the method itself branches on the mode —
    // so these exist to keep one name per (op, mode) pair at the call sites.
    pub use super::{
        queue_add as queue_add_raw, queue_cmp as queue_cmp_raw, queue_div as queue_div_raw,
        queue_mod as queue_mod_raw, queue_mul as queue_mul_raw, queue_sub as queue_sub_raw,
    };
}

#[cfg(feature = "bigint-backend")]
pub use raw_aliases::*;

#[inline(always)]
pub fn port_push(port: usize, value: Val) {
    unsafe { (*(port as *mut Port)).push(value) }
}

#[inline(always)]
pub fn port_pop(port: usize) -> Val {
    unsafe { (*(port as *mut Port)).pop() }
}

/// The element count of any storage, read off the base the three embed.
#[inline(always)]
pub fn base_size(base: usize) -> u32 {
    unsafe { (*(base as *const ArrayBase)).size }
}
