//! Monomorphic JIT helpers for LinkedList storage ops.
//!
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
//! `*mut Stack` / `*mut Queue`. The JIT macro carries stack handles in the
//! ref register bank and uses explicit `ref(Stack)` metadata on inline helpers
//! to lower field access without changing the concrete Rust ABI.

use super::linkedlist::{LinkedList, Queue, Stack};
use crate::value::*;

// Stack helpers.

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
)]
pub fn stack_push(stack: usize, value: Val) {
    let old_head = stack.head;
    let new_node = super::linkedlist::Node {
        value,
        next: old_head,
    };
    stack.head = new_node;
    stack.size = stack.size + 1u32;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
    },
)]
pub fn stack_pop(stack: usize) -> Val {
    let top_node = stack.head;
    let value = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    value
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
)]
pub fn stack_add(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    let sum = (r2 >> 1) + (r1 >> 1);
    next.value = if ((r1 & r2) & 1 != 0) & (((sum << 1) >> 1) == sum) {
        val_retag_small(sum)
    } else {
        val_add(r2, r1)
    };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
    },
    native_int_binops = { val_add => IntAdd },
)]
pub fn stack_add(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = val_add(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
)]
pub fn stack_sub(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    let diff = (r2 >> 1) - (r1 >> 1);
    next.value = if ((r1 & r2) & 1 != 0) & (((diff << 1) >> 1) == diff) {
        val_retag_small(diff)
    } else {
        val_sub(r2, r1)
    };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
    },
    native_int_binops = { val_sub => IntSub },
)]
pub fn stack_sub(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = val_sub(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
)]
pub fn stack_mul(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    // Native smallint mul fast path: untag, sign-extend each operand to 32
    // bits so the i64 product can never overflow, multiply, and re-tag —
    // falling back to val_mul when an operand is a bigint or exceeds the
    // ±2^31 fast range.
    let av = r2 >> 1;
    let bv = r1 >> 1;
    let av32 = (av << 32) >> 32;
    let bv32 = (bv << 32) >> 32;
    let prod = av32 * bv32;
    next.value =
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
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
    },
    native_int_binops = { val_mul => IntMul },
)]
pub fn stack_mul(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = val_mul(r2, r1);
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
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
)]
pub fn stack_dup(stack: usize) {
    let head = stack.head;
    let top_val = head.value;
    let new_node = super::linkedlist::Node {
        value: top_val,
        next: head,
    };
    stack.head = new_node;
    stack.size = stack.size + 1u32;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
)]
pub fn stack_swap(stack: usize) {
    let node1 = stack.head;
    let node2 = node1.next;
    let v1 = node1.value;
    let v2 = node2.value;
    node1.value = v2;
    node2.value = v1;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_ge_jit => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
    },
    native_tag_small = { val_retag_small },
)]
pub fn stack_cmp(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = if (r1 & r2) & 1 != 0 {
        val_retag_small((r2 >= r1) as i64)
    } else {
        val_ge_jit(r2, r1)
    };
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
    },
)]
pub fn stack_cmp(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = (r2 >= r1) as i64;
}

// JIT-callable wrappers for node allocation and value arithmetic.
//
// These expose the storage nursery and value arithmetic to the JIT's
// field-level IR path.  `alloc_node_jit` / `free_node_jit` use `usize`
// (Ref-bank) for node pointers; `val_*_jit` use `i64` (Int-bank) for
// Val, matching the JIT's register banks.

/// Allocate a Node from the shared node nursery.
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

// Queue helpers.

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        alloc_node_jit => nursery_alloc_ref,
        val_from_i32 => elidable_int_cannot_raise,
    },
)]
pub fn queue_push(queue: usize, value: Val) {
    // linkedlist.py:103-110 Queue.push — write the value into the current tail
    // (dummy) node and append a fresh null-next sentinel as the new tail.
    // Allocate the sentinel FIRST and read `tail` only AFTER, so no Node ref is
    // held across the collecting alloc: `alloc_node_jit` can collect on the
    // concrete path, and `collect` forwards the in-memory `queue.tail` slot, so
    // a post-alloc read yields the forwarded pointer. Passing `next = 0` (null)
    // lets the inline nursery bump init `next@8` from the arg, avoiding a
    // null-ref store the IR cannot lower — do not restore an
    // `alloc_node_jit(_, tail)` next-arg (that reintroduces the keep-root dance
    // this ordering replaces).
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = value;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
    },
)]
pub fn queue_pop(queue: usize) -> Val {
    let top_node = queue.head;
    let value = top_node.value;
    let next = top_node.next;
    queue.head = next;
    queue.size = queue.size - 1u32;
    free_node_jit(top_node);
    value
}

// Queue arithmetic (linkedlist.py:99-127): _get_2_values pops two front values,
// _put_value pushes the result at the tail. Inlined as queue_pop x2 + the native
// smallint fast path (mirroring stack_add/sub/mul) + the queue_push tail-sentinel
// append. The result Val is not a Node, so holding it across the sentinel alloc's
// collect is safe (node-collect never moves values); the popped nodes are freed
// before the alloc. `maybe_collect_bigints` is dropped, matching the inlined
// stack arithmetic.
#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_tag_small = { val_retag_small },
)]
pub fn queue_add(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let sum = (r2 >> 1) + (r1 >> 1);
    let result = if ((r1 & r2) & 1 != 0) & (((sum << 1) >> 1) == sum) {
        val_retag_small(sum)
    } else {
        val_add(r2, r1)
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_int_binops = { val_add => IntAdd },
)]
pub fn queue_add(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = val_add(r2, r1);
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_tag_small = { val_retag_small },
)]
pub fn queue_sub(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let diff = (r2 >> 1) - (r1 >> 1);
    let result = if ((r1 & r2) & 1 != 0) & (((diff << 1) >> 1) == diff) {
        val_retag_small(diff)
    } else {
        val_sub(r2, r1)
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_int_binops = { val_sub => IntSub },
)]
pub fn queue_sub(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = val_sub(r2, r1);
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_tag_small = { val_retag_small },
)]
pub fn queue_mul(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let av = r2 >> 1;
    let bv = r1 >> 1;
    let av32 = (av << 32) >> 32;
    let bv32 = (bv << 32) >> 32;
    let prod = av32 * bv32;
    let result =
        if ((r1 & r2) & 1 != 0) & (av32 == av) & (bv32 == bv) & (((prod << 1) >> 1) == prod) {
            val_retag_small(prod)
        } else {
            val_mul(r2, r1)
        };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_int_binops = { val_mul => IntMul },
)]
pub fn queue_mul(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = val_mul(r2, r1);
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
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
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
)]
pub fn queue_dup(queue: usize) {
    // linkedlist.py:112-116 Queue.dup — duplicate the head node (front). Same
    // shape as `stack_dup`; the queue's `tail` is untouched. Inlined (not a
    // residual call) so the head/size mutation is tracked by the optimizer and
    // a following pop reads the post-dup values instead of a stale cache.
    let head = queue.head;
    let top_val = head.value;
    let new_node = super::linkedlist::Node {
        value: top_val,
        next: head,
    };
    queue.head = new_node;
    queue.size = queue.size + 1u32;
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
)]
pub fn queue_swap(queue: usize) {
    let node1 = queue.head;
    let node2 = node1.next;
    let v1 = node1.value;
    let v2 = node2.value;
    node1.value = v2;
    node2.value = v1;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_ge_jit => elidable_int,
        val_retag_small => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_tag_small = { val_retag_small },
)]
pub fn queue_cmp(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = if (r1 & r2) & 1 != 0 {
        val_retag_small((r2 >= r1) as i64)
    } else {
        val_ge_jit(r2, r1)
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(not(feature = "bigint-backend"))]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_cmp(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = (r2 >= r1) as i64;
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

// Raw-word mode equivalents of the tagged-value helpers.
//
// Same storage work as the helpers above, raw-word arithmetic instead of
// tagged. The mainloop picks between a twin and its sibling on the `bm` green,
// so the choice is made once at record time and compiled code never tests the
// mode.
//
// Every route out of mode 0 must sit on the failure side of a guard. This is a
// correctness invariant rather than an optimization: the
// conversion walks the storage and retags every value it can reach, and from
// inside compiled code it can reach neither a value the trace has already
// loaded into a register nor a node the trace has not materialised yet. Both
// would keep their raw words while everything around them turned tagged.
//
// So `val_add`/`val_sub`/`val_mul` appear only in the `None` arm of a
// `checked_*`, which the tracer records as a `guard_no_overflow`, and `val_div`
// only behind an explicit divisor test. The conversion then runs after the
// deopt, in the interpreter, where the storage is materialised and no register
// holds a value.
//
// `>> 0` reads the mode-0 word unchanged, the way `>> 1` untags in the tagged
// twins. It is written this way — rather than through an accessor — because an
// unregistered call is silently skipped by the lowerer rather than rejected.

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
    },
)]
pub fn stack_add_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = match r2.checked_add(r1) {
        Some(sum) => sum,
        None => val_add(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
    },
)]
pub fn stack_sub_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = match r2.checked_sub(r1) {
        Some(diff) => diff,
        None => val_sub(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
    },
)]
pub fn stack_mul_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = match r2.checked_mul(r1) {
        Some(prod) => prod,
        None => val_mul(r2, r1),
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_ge_raw => elidable_int_cannot_raise,
    },
    native_int_binops = { val_ge_raw => IntGe },
)]
pub fn stack_cmp_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    next.value = val_ge_raw(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_div => elidable_int,
        val_div_raw => elidable_int_cannot_raise,
    },
)]
pub fn stack_div_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    // Divisor −1 is guarded out wholesale: it is the only divisor for which a
    // quotient can fail to fit the word (`i64::MIN / -1`), and `val_div` leaves
    // mode 0 for that one. Testing the divisor alone keeps the test to a single
    // compare, and the arm the tracer records for it puts the conversion in the
    // interpreter.
    next.value = if (r1 >> 0) == -1 {
        val_div(r2, r1)
    } else {
        val_div_raw(r2, r1)
    };
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::Stack::size => u32,
    },
    ref_fields = {
        super::linkedlist::Stack::head => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mod_raw => elidable_int_cannot_raise,
    },
)]
pub fn stack_mod_raw(stack: usize) {
    let top_node = stack.head;
    let r1 = top_node.value;
    let next = top_node.next;
    stack.head = next;
    stack.size = stack.size - 1u32;
    free_node_jit(top_node);
    let r2 = next.value;
    // No guard: every remainder fits the word, `i64::MIN % -1` included.
    next.value = val_mod_raw(r2, r1);
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_add => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_add_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = match r2.checked_add(r1) {
        Some(sum) => sum,
        None => val_add(r2, r1),
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_sub => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_sub_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = match r2.checked_sub(r1) {
        Some(diff) => diff,
        None => val_sub(r2, r1),
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mul => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_mul_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = match r2.checked_mul(r1) {
        Some(prod) => prod,
        None => val_mul(r2, r1),
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_ge_raw => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
    native_int_binops = { val_ge_raw => IntGe },
)]
pub fn queue_cmp_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = val_ge_raw(r2, r1);
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_div => elidable_int,
        val_div_raw => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_div_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    // Divisor −1 guarded out, as in `stack_div_raw`.
    let result = if (r1 >> 0) == -1 {
        val_div(r2, r1)
    } else {
        val_div_raw(r2, r1)
    };
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(feature = "bigint-backend")]
#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::Queue::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::head => super::linkedlist::Node,
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        free_node_jit => concrete_only_void,
        val_mod_raw => elidable_int_cannot_raise,
        val_from_i32 => elidable_int_cannot_raise,
        alloc_node_jit => nursery_alloc_ref,
    },
)]
pub fn queue_mod_raw(queue: usize) {
    let n1 = queue.head;
    let r1 = n1.value;
    let n2 = n1.next;
    queue.head = n2;
    queue.size = queue.size - 1u32;
    free_node_jit(n1);
    let r2 = n2.value;
    let n3 = n2.next;
    queue.head = n3;
    queue.size = queue.size - 1u32;
    free_node_jit(n2);
    let result = val_mod_raw(r2, r1);
    let sentinel_val = val_from_i32(0);
    let new_sentinel = alloc_node_jit(sentinel_val, 0);
    let tail = queue.tail;
    tail.value = result;
    tail.next = new_sentinel;
    queue.tail = new_sentinel;
    queue.size = queue.size + 1u32;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_push_pop_roundtrip() {
        let _lock = crate::storage::nursery_test_lock();
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
        let _lock = crate::storage::nursery_test_lock();
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
        let _lock = crate::storage::nursery_test_lock();
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
        let _lock = crate::storage::nursery_test_lock();
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
