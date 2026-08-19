//! Monomorphic JIT helpers for array-backed storage ops.
//!
//! The linked-list counterpart is `linkedlist_jit`; this module is the same
//! surface over a contiguous buffer. Every helper here reads and writes
//! `data[size]` instead of walking a node chain, so a push is one element
//! store and a size bump rather than an allocation plus four stores, and a pop
//! is one element load rather than two dependent pointer loads.
//!
//! Growth is not expressible here. `data` may move when a push outgrows `cap`,
//! and a base pointer read before that move must not be reused after it. The
//! caller guards `size < cap` and leaves the reallocating push to the
//! interpreter, so the compiled path never reallocates and the trace exits
//! through an ordinary guard on the O(log n) pushes that do.
//!
//! The arg type is `usize` (a raw pointer reinterpreted as an integer), not
//! `*mut Stack`: the JIT macro carries storage handles in the ref register
//! bank and takes the pointee shape from the `ref(...)` metadata, without
//! changing the concrete Rust ABI.

use crate::value::*;

// The element descr is declared as `i64` rather than `Val`: the descr needs
// only the element width and signedness, `Val` is a `#[repr(transparent)]`
// wrapper around `i64` in the bigint backend and an alias for it in the
// smallint one, and the declaration site needs a type with `MIN` on it.

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
pub fn stack_push_known_room(stack: usize, value: Val) {
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
