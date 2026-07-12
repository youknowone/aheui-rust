//! Monomorphic JIT helpers for LinkedList storage ops.
//!
//! Phase D-1 (2026-04-28, design
//! `~/.claude/plans/2026-04-28-phase-d1-monomorphic-dispatch-design.md`).
//! aheui-jit's mainloop branches on the `is_queue` / `is_port` JIT
//! greens at each storage-op site to dispatch monomorphically through
//! these helpers; the trace IR then sees concrete reads / writes on
//! `*mut Stack` / `*mut Queue` instead of the polymorphic
//! `&dyn LinkedList` call which `#[jit_interp]` silent-skips.
//!
//! Port slot (`VAL_PORT`) keeps the trait-object path: it is the I/O
//! storage and not on the inner hot path.
//!
//! The arg type is `usize` (raw pointer reinterpreted as integer), not
//! `*mut Stack` / `*mut Queue`. The `#[jit_interp]` lowerer accepts
//! integer arguments fed from `int(usize)` state-fields directly, but
//! it rejects `as *mut Stack` casts at the call site (it only knows
//! `is_supported_int_cast`), so a typed-pointer signature would cause
//! every call to silent-skip during trace recording — producing an
//! empty trace that compiles to a `Label → Jump` infinite loop.

use super::linkedlist::{LinkedList, Queue, Stack};
use crate::value::*;

// ── Stack helpers ────────────────────────────────────────────────────

#[inline(always)]
pub fn stack_push(stack: usize, value: Val) {
    unsafe { (*(stack as *mut Stack)).push(value) }
}

#[inline(always)]
pub fn stack_pop(stack: usize) -> Val {
    unsafe { (*(stack as *mut Stack)).pop() }
}

#[inline(always)]
pub fn stack_add(stack: usize) {
    unsafe { (*(stack as *mut Stack)).add() }
}

#[inline(always)]
pub fn stack_sub(stack: usize) {
    unsafe { (*(stack as *mut Stack)).sub() }
}

#[inline(always)]
pub fn stack_mul(stack: usize) {
    unsafe { (*(stack as *mut Stack)).mul() }
}

#[inline(always)]
pub fn stack_div(stack: usize) {
    unsafe { (*(stack as *mut Stack)).div() }
}

#[inline(always)]
pub fn stack_mod(stack: usize) {
    unsafe { (*(stack as *mut Stack)).modulo() }
}

#[inline(always)]
pub fn stack_dup(stack: usize) {
    unsafe { (*(stack as *mut Stack)).dup() }
}

#[inline(always)]
pub fn stack_swap(stack: usize) {
    unsafe { (*(stack as *mut Stack)).swap() }
}

#[inline(always)]
pub fn stack_cmp(stack: usize) {
    unsafe { (*(stack as *mut Stack)).cmp() }
}

// ── Node alloc/free + val arithmetic — JIT-callable wrappers ────────
//
// These expose the storage nursery and value arithmetic to the JIT's
// field-level IR path.  `alloc_node_jit` / `free_node_jit` use `usize`
// (Ref-bank) for node pointers; `val_*_jit` use `i64` (Int-bank) for
// Val, matching the JIT's register banks.

/// Allocate a Node from the nursery free list.
/// JIT type: `(Val value, usize next) -> usize new_node`.
#[inline(always)]
pub fn alloc_node_jit(value: Val, next: usize) -> usize {
    super::alloc_node(value, next as *mut super::linkedlist::Node) as usize
}

/// Return a Node to the nursery free list.
/// JIT type: `(usize node) -> void`.
#[inline(always)]
pub fn free_node_jit(node: usize) {
    super::free_node(node as *mut super::linkedlist::Node)
}

/// `val_ge` wrapper returning Val (0 or 1) instead of bool.
/// JIT type: `(Val, Val) -> Val`.
#[inline(always)]
pub fn val_ge_jit(a: Val, b: Val) -> Val {
    if val_ge(&a, &b) {
        val_from_i32(1)
    } else {
        val_from_i32(0)
    }
}

// ── Queue helpers ────────────────────────────────────────────────────

#[inline(always)]
pub fn queue_push(queue: usize, value: Val) {
    unsafe { (*(queue as *mut Queue)).push(value) }
}

#[inline(always)]
pub fn queue_pop(queue: usize) -> Val {
    unsafe { (*(queue as *mut Queue)).pop() }
}

#[inline(always)]
pub fn queue_add(queue: usize) {
    unsafe { (*(queue as *mut Queue)).add() }
}

#[inline(always)]
pub fn queue_sub(queue: usize) {
    unsafe { (*(queue as *mut Queue)).sub() }
}

#[inline(always)]
pub fn queue_mul(queue: usize) {
    unsafe { (*(queue as *mut Queue)).mul() }
}

#[inline(always)]
pub fn queue_div(queue: usize) {
    unsafe { (*(queue as *mut Queue)).div() }
}

#[inline(always)]
pub fn queue_mod(queue: usize) {
    unsafe { (*(queue as *mut Queue)).modulo() }
}

#[inline(always)]
pub fn queue_dup(queue: usize) {
    unsafe { (*(queue as *mut Queue)).dup() }
}

#[inline(always)]
pub fn queue_swap(queue: usize) {
    unsafe { (*(queue as *mut Queue)).swap() }
}

#[inline(always)]
pub fn queue_cmp(queue: usize) {
    unsafe { (*(queue as *mut Queue)).cmp() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_push_pop_roundtrip() {
        let mut s = Stack::new();
        let p = &mut s as *mut Stack as usize;
        unsafe {
            stack_push(p, val_from_i32(1));
            stack_push(p, val_from_i32(2));
            stack_push(p, val_from_i32(3));
            assert_eq!(val_to_i64(&stack_pop(p)), 3);
            assert_eq!(val_to_i64(&stack_pop(p)), 2);
            assert_eq!(val_to_i64(&stack_pop(p)), 1);
        }
    }

    #[test]
    fn stack_arith_dispatches_through_pointer() {
        let mut s = Stack::new();
        let p = &mut s as *mut Stack as usize;
        unsafe {
            stack_push(p, val_from_i32(7));
            stack_push(p, val_from_i32(3));
            // add: r1=3, r2=7, push r2+r1=10 (replaces top)
            stack_add(p);
            assert_eq!(val_to_i64(&stack_pop(p)), 10);
        }
    }

    #[test]
    fn queue_push_pop_fifo_order() {
        let mut q = Queue::new();
        let p = &mut q as *mut Queue as usize;
        unsafe {
            queue_push(p, val_from_i32(1));
            queue_push(p, val_from_i32(2));
            queue_push(p, val_from_i32(3));
            assert_eq!(val_to_i64(&queue_pop(p)), 1);
            assert_eq!(val_to_i64(&queue_pop(p)), 2);
            assert_eq!(val_to_i64(&queue_pop(p)), 3);
        }
    }

    #[test]
    fn queue_add_pushes_sum_to_back() {
        let mut q = Queue::new();
        let p = &mut q as *mut Queue as usize;
        unsafe {
            queue_push(p, val_from_i32(7));
            queue_push(p, val_from_i32(3));
            // Queue::add pops front twice (r1=7, r2=3), pushes r2+r1=10 to back.
            queue_add(p);
            assert_eq!(val_to_i64(&queue_pop(p)), 10);
        }
    }
}
