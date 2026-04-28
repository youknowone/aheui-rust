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
//! Each helper is a thin `#[inline(always)]` wrapper around the
//! underlying `LinkedList` method — same logic, monomorphic call site.
//! When registered with `#[jit_interp(calls = { stack_push =>
//! residual_call_void, ... })]`, `aheui-jit` emits a residual call to
//! the concrete function pointer in the trace; a future slice can flip
//! individual helpers to `#[jit_inline]` once their bodies are
//! validated against the lowerer.

use super::linkedlist::{LinkedList, Queue, Stack};
use crate::value::*;

// ── Stack helpers ────────────────────────────────────────────────────

#[inline(always)]
pub unsafe fn stack_push(stack: *mut Stack, value: Val) {
    unsafe { (*stack).push(value) }
}

#[inline(always)]
pub unsafe fn stack_pop(stack: *mut Stack) -> Val {
    unsafe { (*stack).pop() }
}

#[inline(always)]
pub unsafe fn stack_add(stack: *mut Stack) {
    unsafe { (*stack).add() }
}

#[inline(always)]
pub unsafe fn stack_sub(stack: *mut Stack) {
    unsafe { (*stack).sub() }
}

#[inline(always)]
pub unsafe fn stack_mul(stack: *mut Stack) {
    unsafe { (*stack).mul() }
}

#[inline(always)]
pub unsafe fn stack_div(stack: *mut Stack) {
    unsafe { (*stack).div() }
}

#[inline(always)]
pub unsafe fn stack_mod(stack: *mut Stack) {
    unsafe { (*stack).modulo() }
}

#[inline(always)]
pub unsafe fn stack_dup(stack: *mut Stack) {
    unsafe { (*stack).dup() }
}

#[inline(always)]
pub unsafe fn stack_swap(stack: *mut Stack) {
    unsafe { (*stack).swap() }
}

#[inline(always)]
pub unsafe fn stack_cmp(stack: *mut Stack) {
    unsafe { (*stack).cmp() }
}

// ── Queue helpers ────────────────────────────────────────────────────

#[inline(always)]
pub unsafe fn queue_push(queue: *mut Queue, value: Val) {
    unsafe { (*queue).push(value) }
}

#[inline(always)]
pub unsafe fn queue_pop(queue: *mut Queue) -> Val {
    unsafe { (*queue).pop() }
}

#[inline(always)]
pub unsafe fn queue_add(queue: *mut Queue) {
    unsafe { (*queue).add() }
}

#[inline(always)]
pub unsafe fn queue_sub(queue: *mut Queue) {
    unsafe { (*queue).sub() }
}

#[inline(always)]
pub unsafe fn queue_mul(queue: *mut Queue) {
    unsafe { (*queue).mul() }
}

#[inline(always)]
pub unsafe fn queue_div(queue: *mut Queue) {
    unsafe { (*queue).div() }
}

#[inline(always)]
pub unsafe fn queue_mod(queue: *mut Queue) {
    unsafe { (*queue).modulo() }
}

#[inline(always)]
pub unsafe fn queue_dup(queue: *mut Queue) {
    unsafe { (*queue).dup() }
}

#[inline(always)]
pub unsafe fn queue_swap(queue: *mut Queue) {
    unsafe { (*queue).swap() }
}

#[inline(always)]
pub unsafe fn queue_cmp(queue: *mut Queue) {
    unsafe { (*queue).cmp() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_push_pop_roundtrip() {
        let mut s = Stack::new();
        let p = &mut s as *mut Stack;
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
        let p = &mut s as *mut Stack;
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
        let p = &mut q as *mut Queue;
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
        let p = &mut q as *mut Queue;
        unsafe {
            queue_push(p, val_from_i32(7));
            queue_push(p, val_from_i32(3));
            // Queue::add pops front twice (r1=7, r2=3), pushes r2+r1=10 to back.
            queue_add(p);
            assert_eq!(val_to_i64(&queue_pop(p)), 10);
        }
    }
}
