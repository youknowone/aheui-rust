//! Concrete storage access for the generated Aheui dispatch loop.
//! Arithmetic is shared with the interpreter through `crate::band`.

use super::linkedlist::ListBase;
use crate::value::*;

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::ListBase::size => u32,
    },
    ref_fields = {
        super::linkedlist::ListBase::head => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
    inlined_prefix = {
        super::linkedlist::Stack::base => super::linkedlist::ListBase,
    },
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
pub fn pop_base_known_nonempty(list: usize) -> Val {
    unsafe { super::linkedlist::pop_base_known_nonempty(&mut *(list as *mut ListBase)) }
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        stack: ref(super::linkedlist::Stack),
    },
    int_fields = {
        super::linkedlist::ListBase::size => u32,
    },
    ref_fields = {
        super::linkedlist::ListBase::head => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
    inlined_prefix = {
        super::linkedlist::Stack::base => super::linkedlist::ListBase,
    },
)]
pub fn stack_dup(stack: usize) {
    // linkedlist.py Stack.dup -- `self.push(self.head.value)`, flattened
    // here into the three statements that push expands to.  The queue's `dup`
    // is a separate definition (:112-116) that happens to expand the same way;
    // they are two implementations of a method every storage overrides, not one
    // implementation reached twice, so they stay two helpers.
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
pub fn swap_base_known_two(list: usize) {
    unsafe { super::linkedlist::swap_base_known_two(&mut *(list as *mut ListBase)) }
}

#[inline(always)]
pub fn alloc_node_jit(value: Val, next: usize) -> usize {
    super::alloc_node(value, next as *mut super::linkedlist::Node) as usize
}

#[inline(always)]
#[majit_macros::jit_inline(
    ref_params = {
        queue: ref(super::linkedlist::Queue),
    },
    int_fields = {
        super::linkedlist::ListBase::size => u32,
    },
    ref_fields = {
        super::linkedlist::Queue::tail => super::linkedlist::Node,
        super::linkedlist::Node::next => super::linkedlist::Node,
    },
    calls = {
        alloc_node_jit => nursery_alloc_ref,
        val_from_i32 => elidable_int_cannot_raise,
    },
    headerless_structs = { super::linkedlist::Node, },
    inlined_prefix = {
        super::linkedlist::Queue::base => super::linkedlist::ListBase,
    },
)]
pub fn queue_push(queue: usize, value: Val) {
    // linkedlist.py Queue.push — write the value into the current tail
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
        super::linkedlist::ListBase::size => u32,
    },
    ref_fields = {
        super::linkedlist::ListBase::head => super::linkedlist::Node,
    },
    struct_allocs = { super::linkedlist::Node => alloc_node_jit, },
    headerless_structs = { super::linkedlist::Node, },
    inlined_prefix = {
        super::linkedlist::Queue::base => super::linkedlist::ListBase,
    },
)]
pub fn queue_dup(queue: usize) {
    // linkedlist.py Queue.dup — duplicate the head node (front). Same
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::linkedlist::{LinkedList, Queue, Stack};

    #[test]
    fn stack_push_pop_roundtrip() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        let p = &mut s as *mut Stack as usize;
        stack_push(p, val_from_i32(1));
        stack_push(p, val_from_i32(2));
        stack_push(p, val_from_i32(3));
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 3);
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 2);
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 1);
    }

    #[test]
    fn stack_arith_dispatches_through_pointer() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        let p = &mut s as *mut Stack as usize;
        stack_push(p, val_from_i32(7));
        stack_push(p, val_from_i32(3));
        // add: r1=3, r2=7, push r2+r1=10 (replaces top)
        s.add();
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 10);
    }

    #[test]
    fn queue_push_pop_fifo_order() {
        let _lock = crate::storage::nursery_test_lock();
        let mut q = Queue::new();
        let p = &mut q as *mut Queue as usize;
        queue_push(p, val_from_i32(1));
        queue_push(p, val_from_i32(2));
        queue_push(p, val_from_i32(3));
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 1);
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 2);
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 3);
    }

    #[test]
    fn queue_add_pushes_sum_to_back() {
        let _lock = crate::storage::nursery_test_lock();
        let mut q = Queue::new();
        let p = &mut q as *mut Queue as usize;
        queue_push(p, val_from_i32(7));
        queue_push(p, val_from_i32(3));
        // Queue::add pops front twice (r1=7, r2=3), pushes r2+r1=10 to back.
        q.add();
        assert_eq!(val_to_i64(&pop_base_known_nonempty(p)), 10);
    }
}
