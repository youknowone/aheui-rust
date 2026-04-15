//! Linked-list storage: Node / Stack / Queue / Port.
//!
//! Ported from `rpaheui/aheui/storage/linkedlist.py`. In RPython this is the
//! only storage backend (CPython-only `array.py` is excluded by the
//! `aheui._compat.PYR` guard). All structures share the same linked-list
//! node type.
use super::{StorageOps, alloc_node, free_node};
use crate::value::*;

/// Linked-list node. `rpaheui/aheui/storage/linkedlist.py::Node`.
///
/// JIT virtualizes these allocations via OptVirtualize:
///   New(NODE_SIZE_DESCR) + SetfieldGc(node, value) + SetfieldGc(node, next)
#[repr(C)]
pub struct StackNode {
    pub value: Val,
    pub next: *mut StackNode,
}

/// Layout constants for JIT access to `StackNode` fields.
pub const STACKNODE_SIZE: usize = std::mem::size_of::<StackNode>();
pub const STACKNODE_VALUE_OFFSET: usize = 0; // offset_of!(StackNode, value)
pub const STACKNODE_NEXT_OFFSET: usize = 8; // offset_of!(StackNode, next)

/// Stack (LIFO). `rpaheui/aheui/storage/linkedlist.py::Stack(LinkedList)`.
///
/// JIT accesses `head`/`size` via GetfieldGc/SetfieldGc on the
/// StorageKind ptr.
#[repr(C)]
pub struct AheuiStack {
    pub head: *mut StackNode,
    pub size: usize,
}

impl AheuiStack {
    pub fn new() -> Self {
        AheuiStack {
            head: std::ptr::null_mut(),
            size: 0,
        }
    }

    /// linkedlist.py: push(value) → Node(self.head, value); self.head = node
    pub fn push(&mut self, value: impl Into<Val>) {
        let node = alloc_node(value.into(), self.head);
        self.head = node;
        self.size += 1;
    }

    /// linkedlist.py: pop() → node = self.head; self.head = node.next
    pub fn pop(&mut self) -> Val {
        assert!(!self.head.is_null(), "stack underflow");
        let old = self.head;
        let value = unsafe { (*old).value };
        self.head = unsafe { (*old).next };
        self.size -= 1;
        free_node(old);
        value
    }

    /// linkedlist.py: dup() → self.push(self.head.value)
    pub fn dup(&mut self) {
        assert!(!self.head.is_null(), "stack underflow on dup");
        let top = unsafe { (*self.head).value };
        self.push(top);
    }

    /// linkedlist.py: swap() → node1.value, node2.value = node2.value, node1.value
    pub fn swap(&mut self) {
        assert!(self.size >= 2, "stack underflow on swap");
        unsafe {
            let a = &mut *self.head;
            let b = &mut *a.next;
            std::mem::swap(&mut a.value, &mut b.value);
        }
    }

    /// Pop r1 (top), peek r2 (new top), compute f(r2, r1), replace top with result.
    /// linkedlist.py: _get_2_values + _put_value
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.pop();
        assert!(!self.head.is_null(), "stack underflow on binop");
        let r2 = unsafe { (*self.head).value };
        let result = f(r2, r1);
        unsafe {
            (*self.head).value = result;
        }
    }

    /// Peek top value without popping.
    pub fn peek_top(&self) -> Val {
        assert!(!self.head.is_null(), "stack underflow on peek");
        unsafe { (*self.head).value }
    }

    pub fn len(&self) -> usize {
        self.size
    }
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn peek_at(&self, i: usize) -> i64 {
        let mut cur = self.head;
        for _ in 0..i {
            assert!(!cur.is_null(), "peek_at out of bounds");
            cur = unsafe { (*cur).next };
        }
        assert!(!cur.is_null(), "peek_at out of bounds");
        // Return raw bits — the JIT works with the raw Val representation
        // (tagged in bigint mode). val_to_i64 would panic on corrupted
        // tags produced by JIT IntAdd/IntSub on tagged values.
        unsafe { std::mem::transmute::<Val, i64>((*cur).value) }
    }

    /// Pop a value and return its raw i64 representation.
    /// JIT uses this for binop operands — avoids Val↔i64 transmute in source.
    pub fn pop_raw(&mut self) -> i64 {
        unsafe { std::mem::transmute(self.pop()) }
    }

    /// Push a raw i64 as a Val. JIT uses this for binop results.
    pub fn push_raw(&mut self, raw: i64) {
        self.push(unsafe { std::mem::transmute::<i64, Val>(raw) });
    }

    pub fn add(&mut self) {
        StorageOps::add(self)
    }
    pub fn sub(&mut self) {
        StorageOps::sub(self)
    }
    pub fn mul(&mut self) {
        StorageOps::mul(self)
    }
    pub fn div(&mut self) {
        StorageOps::div(self)
    }
    pub fn modulo(&mut self) {
        StorageOps::modulo(self)
    }
    pub fn cmp(&mut self) {
        StorageOps::cmp(self)
    }
}

impl StorageOps for AheuiStack {
    fn push(&mut self, value: Val) {
        AheuiStack::push(self, value)
    }
    fn pop(&mut self) -> Val {
        AheuiStack::pop(self)
    }
    fn dup(&mut self) {
        AheuiStack::dup(self)
    }
    fn swap(&mut self) {
        AheuiStack::swap(self)
    }
    fn len(&self) -> usize {
        self.size
    }
    fn peek_at(&self, i: usize) -> i64 {
        AheuiStack::peek_at(self, i)
    }
    fn add(&mut self) {
        self.binop(val_add)
    }
    fn sub(&mut self) {
        self.binop(val_sub)
    }
    fn mul(&mut self) {
        self.binop(val_mul)
    }
    fn div(&mut self) {
        self.binop(val_div)
    }
    fn modulo(&mut self) {
        self.binop(val_mod)
    }
    fn cmp(&mut self) {
        self.binop(|a, b| {
            if val_ge(&a, &b) {
                val_from_i32(1)
            } else {
                val_from_i32(0)
            }
        })
    }
}

/// Queue (FIFO). `rpaheui/aheui/storage/linkedlist.py::Queue(LinkedList)`.
///
/// Push to tail, pop from head. Slot `VAL_QUEUE` (21) only.
#[repr(C)]
pub struct AheuiQueue {
    pub head: *mut StackNode,
    pub size: usize,
    pub tail: *mut StackNode,
}

impl AheuiQueue {
    pub fn new() -> Self {
        // linkedlist.py:98-101: tail = Node(None); head = tail; size = 0
        let sentinel = alloc_node(val_from_i32(0), std::ptr::null_mut());
        AheuiQueue {
            head: sentinel,
            size: 0,
            tail: sentinel,
        }
    }

    /// linkedlist.py:103-110: push to tail
    pub fn push(&mut self, value: impl Into<Val>) {
        let tail = self.tail;
        unsafe {
            (*tail).value = value.into();
        }
        let new = alloc_node(val_from_i32(0), std::ptr::null_mut());
        unsafe {
            (*tail).next = new;
        }
        self.tail = new;
        self.size += 1;
    }

    /// linkedlist.py: pop from head (inherited from LinkedList)
    pub fn pop(&mut self) -> Val {
        assert!(
            !self.head.is_null() && self.head != self.tail,
            "queue underflow"
        );
        let old = self.head;
        let value = unsafe { (*old).value };
        self.head = unsafe { (*old).next };
        self.size -= 1;
        free_node(old);
        value
    }

    /// linkedlist.py:112-116: dup pushes head.value at front
    pub fn dup(&mut self) {
        assert!(self.head != self.tail, "queue underflow on dup");
        let head_val = unsafe { (*self.head).value };
        let node = alloc_node(head_val, self.head);
        self.head = node;
        self.size += 1;
    }

    /// linkedlist.py: swap head values
    pub fn swap(&mut self) {
        assert!(self.size >= 2, "queue underflow on swap");
        unsafe {
            let a = &mut *self.head;
            let b = &mut *a.next;
            std::mem::swap(&mut a.value, &mut b.value);
        }
    }

    /// linkedlist.py:118-122: _get_2_values = pop(), pop(); _put_value = push()
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.pop();
        let r2 = self.pop();
        let result = f(r2, r1);
        self.push(result);
    }
}

impl StorageOps for AheuiQueue {
    fn push(&mut self, value: Val) {
        AheuiQueue::push(self, value)
    }
    fn pop(&mut self) -> Val {
        AheuiQueue::pop(self)
    }
    fn dup(&mut self) {
        AheuiQueue::dup(self)
    }
    fn swap(&mut self) {
        AheuiQueue::swap(self)
    }
    fn len(&self) -> usize {
        self.size
    }
    fn peek_at(&self, i: usize) -> i64 {
        let mut cur = self.head;
        for _ in 0..i {
            cur = unsafe { (*cur).next };
        }
        val_to_i64(&unsafe { (*cur).value })
    }
    fn add(&mut self) {
        self.binop(val_add)
    }
    fn sub(&mut self) {
        self.binop(val_sub)
    }
    fn mul(&mut self) {
        self.binop(val_mul)
    }
    fn div(&mut self) {
        self.binop(val_div)
    }
    fn modulo(&mut self) {
        self.binop(val_mod)
    }
    fn cmp(&mut self) {
        self.binop(|a, b| {
            if val_ge(&a, &b) {
                val_from_i32(1)
            } else {
                val_from_i32(0)
            }
        })
    }
}

/// Port (unused stderr channel). `rpaheui/aheui/storage/linkedlist.py::Port`.
///
/// Like Stack but `dup` pushes `last_push` instead of the head value.
/// Slot `VAL_PORT` (27) only.
#[repr(C)]
pub struct AheuiPort {
    pub head: *mut StackNode,
    pub size: usize,
    pub last_push: Val,
}

impl AheuiPort {
    pub fn new() -> Self {
        AheuiPort {
            head: std::ptr::null_mut(),
            size: 0,
            last_push: val_from_i32(0),
        }
    }

    /// linkedlist.py:134-139: push with last_push tracking
    pub fn push(&mut self, value: impl Into<Val>) {
        let v = value.into();
        self.last_push = v;
        let node = alloc_node(v, self.head);
        self.head = node;
        self.size += 1;
    }

    /// linkedlist.py: pop (inherited from LinkedList)
    pub fn pop(&mut self) -> Val {
        assert!(!self.head.is_null(), "port underflow");
        let old = self.head;
        let value = unsafe { (*old).value };
        self.head = unsafe { (*old).next };
        self.size -= 1;
        free_node(old);
        value
    }

    /// linkedlist.py:141-142: dup pushes last_push
    pub fn dup(&mut self) {
        self.push(self.last_push);
    }

    /// linkedlist.py: swap (inherited from LinkedList)
    pub fn swap(&mut self) {
        assert!(self.size >= 2, "port underflow on swap");
        unsafe {
            let a = &mut *self.head;
            let b = &mut *a.next;
            std::mem::swap(&mut a.value, &mut b.value);
        }
    }

    /// linkedlist.py:144-148: same as Stack binop
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.pop();
        assert!(!self.head.is_null(), "port underflow on binop");
        let r2 = unsafe { (*self.head).value };
        let result = f(r2, r1);
        unsafe {
            (*self.head).value = result;
        }
    }
}

impl StorageOps for AheuiPort {
    fn push(&mut self, value: Val) {
        AheuiPort::push(self, value)
    }
    fn pop(&mut self) -> Val {
        AheuiPort::pop(self)
    }
    fn dup(&mut self) {
        AheuiPort::dup(self)
    }
    fn swap(&mut self) {
        AheuiPort::swap(self)
    }
    fn len(&self) -> usize {
        self.size
    }
    fn peek_at(&self, i: usize) -> i64 {
        let mut cur = self.head;
        for _ in 0..i {
            cur = unsafe { (*cur).next };
        }
        val_to_i64(&unsafe { (*cur).value })
    }
    fn add(&mut self) {
        self.binop(val_add)
    }
    fn sub(&mut self) {
        self.binop(val_sub)
    }
    fn mul(&mut self) {
        self.binop(val_mul)
    }
    fn div(&mut self) {
        self.binop(val_div)
    }
    fn modulo(&mut self) {
        self.binop(val_mod)
    }
    fn cmp(&mut self) {
        self.binop(|a, b| {
            if val_ge(&a, &b) {
                val_from_i32(1)
            } else {
                val_from_i32(0)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_basic() {
        let mut s = AheuiStack::new();
        s.push(10_i64);
        s.push(20_i64);
        assert_eq!(val_to_i64(&s.pop()), 20);
        assert_eq!(val_to_i64(&s.pop()), 10);
    }

    #[test]
    fn test_stack_binop() {
        let mut s = AheuiStack::new();
        s.push(10_i64);
        s.push(3_i64);
        // add: r1=3, r2=10, result=13, replaces top
        s.binop(val_add);
        assert_eq!(s.size, 1);
        assert_eq!(val_to_i64(&s.pop()), 13);
    }

    #[test]
    fn test_stack_dup_swap() {
        let mut s = AheuiStack::new();
        s.push(5_i64);
        s.dup();
        assert_eq!(s.size, 2);
        s.push(7_i64);
        s.swap();
        assert_eq!(val_to_i64(&s.pop()), 5); // was second-to-top
        assert_eq!(val_to_i64(&s.pop()), 7); // was top, now swapped
    }

    #[test]
    fn test_queue_basic() {
        let mut q = AheuiQueue::new();
        q.push(1_i64);
        q.push(2_i64);
        q.push(3_i64);
        assert_eq!(val_to_i64(&q.pop()), 1); // FIFO
        assert_eq!(val_to_i64(&q.pop()), 2);
    }

    #[test]
    fn test_queue_binop() {
        let mut q = AheuiQueue::new();
        q.push(10_i64);
        q.push(3_i64);
        // add: r1=10(front), r2=3(front), result=13, pushed to back
        q.binop(val_add);
        assert_eq!(q.size, 1);
        assert_eq!(val_to_i64(&q.pop()), 13);
    }

    #[test]
    fn test_port_dup() {
        let mut p = AheuiPort::new();
        p.push(5_i64);
        p.push(10_i64);
        p.dup(); // duplicates last_push=10
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 5);
    }
}
