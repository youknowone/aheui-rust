//! Line-by-line port of `rpaheui/aheui/storage/linkedlist.py`.
//!
//! In RPython this is the only storage backend (CPython-only `array.py`
//! is excluded by the `aheui._compat.PYR` guard). The Python source has
//! a shared `LinkedList` base class and three subclasses `Stack`,
//! `Queue`, `Port` that inherit from it. Rust emulates the inheritance
//! via the [`LinkedList`] trait: each subclass provides the `__slots__`
//! fields and overrides `push` / `dup` / `_get_2_values` / `_put_value`,
//! while the trait supplies the shared `pop` / `swap` / `add` / `sub` /
//! `mul` / `div` / `modulo` / `cmp` implementations.
use super::{alloc_node, free_node};
use crate::value::*;

// linkedlist.py:4-11
// class Node(object):
//     """Element unit for stack and queue."""
//     __slots__ = ('value', 'next')
//     def __init__(self, next, value=bigint.MINUS1):
//         self.value = value
//         self.next = next
#[repr(C)]
pub struct Node {
    pub value: Val,
    pub next: *mut Node,
}

/// Layout constants for JIT access to `Node` fields. RPython derives these
/// via `symbolic.get_size` / `symbolic.get_field_token`; we hard-code them
/// next to the `#[repr(C)]` struct.
pub const NODE_SIZE: usize = std::mem::size_of::<Node>();
pub const NODE_VALUE_OFFSET: usize = 0;
pub const NODE_NEXT_OFFSET: usize = 8;

// linkedlist.py:14-64
// class LinkedList(object):
//     """Common linked list for storages"""
//     __slots__ = ('head', 'size')
//
// Rust emulates Python inheritance by requiring subclasses to expose
// `head`/`size` accessors and the two `_get_2_values`/`_put_value`
// hooks. The arithmetic methods (`add` ... `cmp`), `pop`, `swap` and
// `__len__` follow the Python default implementations.
pub trait LinkedList {
    // __slots__ accessors
    fn head(&self) -> *mut Node;
    fn set_head(&mut self, h: *mut Node);
    fn size(&self) -> usize;
    fn set_size(&mut self, s: usize);

    // Subclass-supplied (Python: overridden in Stack/Queue/Port).
    fn push(&mut self, value: Val);
    fn dup(&mut self);
    fn _get_2_values(&mut self) -> (Val, Val);
    fn _put_value(&mut self, value: Val);

    // linkedlist.py:19-20
    // def __len__(self): return self.size
    fn __len__(&self) -> usize {
        self.size()
    }

    // linkedlist.py:22-28
    // def pop(self):
    //     node = self.head
    //     self.head = node.next
    //     value = node.value
    //     del node
    //     self.size -= 1
    //     return value
    fn pop(&mut self) -> Val {
        let node = self.head();
        assert!(!node.is_null(), "pop from empty linked list");
        let next = unsafe { (*node).next };
        let value = unsafe { (*node).value };
        self.set_head(next);
        free_node(node); // del node — return to nursery free list
        self.set_size(self.size() - 1);
        value
    }

    // linkedlist.py:30-33
    // def swap(self):
    //     node1 = self.head
    //     node2 = node1.next
    //     node1.value, node2.value = node2.value, node1.value
    fn swap(&mut self) {
        let node1 = self.head();
        assert!(!node1.is_null(), "swap on empty linked list");
        let node2 = unsafe { (*node1).next };
        assert!(!node2.is_null(), "swap on <2 elements");
        unsafe {
            let v = (*node1).value;
            (*node1).value = (*node2).value;
            (*node2).value = v;
        }
    }

    // linkedlist.py:35-38
    // def add(self): r1, r2 = self._get_2_values(); r = bigint.add(r2, r1); self._put_value(r)
    fn add(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_add(r2, r1));
    }

    // linkedlist.py:40-43
    fn sub(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_sub(r2, r1));
    }

    // linkedlist.py:45-48
    fn mul(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mul(r2, r1));
    }

    // linkedlist.py:50-53
    fn div(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_div(r2, r1));
    }

    // linkedlist.py:55-58
    // Python name is `mod`; Rust reserves that keyword, so use `modulo`.
    fn modulo(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mod(r2, r1));
    }

    // linkedlist.py:60-64
    // def cmp(self):
    //     r1, r2 = self._get_2_values()
    //     r = int(bigint.ge(r2, r1))
    //     big_r = bigint.fromint(r)
    //     self._put_value(big_r)
    fn cmp(&mut self) {
        let (r1, r2) = self._get_2_values();
        let r = if val_ge(&r2, &r1) {
            val_from_i32(1)
        } else {
            val_from_i32(0)
        };
        self._put_value(r);
    }
}

// linkedlist.py:67-91
// class Stack(LinkedList):
//     """Base data storage for Aheui, except for ieung and hieuh."""
//     __slots__ = ('head', 'size')
#[repr(C)]
pub struct Stack {
    pub head: *mut Node,
    /// The element count.  `u32`, not `usize`, so the JIT's field descr is
    /// sub-word and `intbounds` can bound a load of it: with no upper bound a
    /// depth `+ 1` may overflow, the sum goes rangeless, and every re-check of
    /// the depth has to be guarded again.  A list is a chain of 16-byte nodes,
    /// so 2^32 elements is 64 GiB of nodes.
    pub size: u32,
}

impl Stack {
    // linkedlist.py:72-74
    // def __init__(self):
    //     self.head = None
    //     self.size = 0
    pub fn new() -> Self {
        Stack {
            head: std::ptr::null_mut(),
            size: 0,
        }
    }
}

impl LinkedList for Stack {
    fn head(&self) -> *mut Node {
        self.head
    }
    fn set_head(&mut self, h: *mut Node) {
        self.head = h;
    }
    fn size(&self) -> usize {
        self.size as usize
    }
    fn set_size(&mut self, s: usize) {
        self.size = s as u32;
    }

    // linkedlist.py:76-80
    // def push(self, value):
    //     node = Node(self.head, value)
    //     self.head = node
    //     self.size += 1
    fn push(&mut self, value: Val) {
        let rooted = value;
        let mut root = rooted;
        with_bigint_transient_root(&mut root, || {
            let node = alloc_node(rooted, self.head);
            self.head = node;
            self.size += 1;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:82-83
    // def dup(self): self.push(self.head.value)
    fn dup(&mut self) {
        assert!(!self.head.is_null(), "dup on empty stack");
        let top = unsafe { (*self.head).value };
        self.push(top);
    }

    // linkedlist.py:87-88
    // def _get_2_values(self): return self.pop(), self.head.value
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        assert!(!self.head.is_null(), "_get_2_values on <2 elements");
        let r2 = unsafe { (*self.head).value };
        (r1, r2)
    }

    // linkedlist.py:90-91
    // def _put_value(self, value): self.head.value = value
    fn _put_value(&mut self, value: Val) {
        let rooted = value;
        let mut root = rooted;
        with_bigint_transient_root(&mut root, || {
            unsafe {
                (*self.head).value = rooted;
            }
            maybe_collect_bigints();
        });
    }
}

// linkedlist.py:94-122
// class Queue(LinkedList):
//     __slots__ = ('head', 'tail', 'size')
#[repr(C)]
pub struct Queue {
    pub head: *mut Node,
    /// The element count, narrowed for the same reason as [`Stack::size`].
    pub size: u32,
    pub tail: *mut Node,
}

impl Queue {
    // linkedlist.py:98-101
    // def __init__(self):
    //     self.tail = Node(None)
    //     self.head = self.tail
    //     self.size = 0
    pub fn new() -> Self {
        let sentinel = alloc_node(val_from_i32(0), std::ptr::null_mut());
        Queue {
            head: sentinel,
            size: 0,
            tail: sentinel,
        }
    }
}

impl LinkedList for Queue {
    fn head(&self) -> *mut Node {
        self.head
    }
    fn set_head(&mut self, h: *mut Node) {
        self.head = h;
    }
    fn size(&self) -> usize {
        self.size as usize
    }
    fn set_size(&mut self, s: usize) {
        self.size = s as u32;
    }

    // linkedlist.py:103-110
    // def push(self, value):
    //     tail = self.tail
    //     tail.value = value
    //     new = Node(None)
    //     tail.next = new
    //     self.tail = new
    //     self.size += 1
    fn push(&mut self, value: Val) {
        let rooted = value;
        let mut root = rooted;
        with_bigint_transient_root(&mut root, || {
            let tail = self.tail;
            unsafe {
                (*tail).value = rooted;
            }
            // Use the new sentinel's temporary `next` field as an explicit keep
            // root for the old tail. If allocation collects, this is rewritten to
            // the forwarded tail before `alloc_node` returns.
            let new = alloc_node(val_from_i32(0), tail);
            let tail = unsafe { (*new).next };
            unsafe {
                (*new).next = std::ptr::null_mut();
                (*tail).next = new;
            }
            self.tail = new;
            self.size += 1;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:112-116
    // def dup(self):
    //     head = self.head
    //     node = Node(head, head.value)
    //     self.head = node
    //     self.size += 1
    fn dup(&mut self) {
        let head = self.head;
        assert!(!head.is_null(), "dup on empty queue");
        let head_value = unsafe { (*head).value };
        let node = alloc_node(head_value, head);
        self.head = node;
        self.size += 1;
    }

    // linkedlist.py:118-119
    // def _get_2_values(self): return self.pop(), self.pop()
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.pop();
        (r1, r2)
    }

    // linkedlist.py:121-122
    // def _put_value(self, value): self.push(value)
    fn _put_value(&mut self, value: Val) {
        self.push(value);
    }
}

// linkedlist.py:125-148
// class Port(LinkedList):
//     __slots__ = ('head', 'size', 'last_push')
#[repr(C)]
pub struct Port {
    pub head: *mut Node,
    /// The element count, narrowed for the same reason as [`Stack::size`].
    pub size: u32,
    pub last_push: Val,
}

/// The three storages share a `{ head, size }` prefix, and the JIT depends on
/// it: `pools[selected]` is declared to load a `*mut Stack`, so a selected
/// queue or port is read at `Stack`'s offsets for as long as it stays in the
/// trace.  `#[repr(C)]` does not promise this by itself — it fixes each layout
/// independently, and nothing relates one to another.  Widening `Stack::size`,
/// or inserting a field ahead of `Queue::size`, would compile clean here and
/// leave the JIT reading a queue's `head` word as its element count.
///
/// Stated as one assert per shared field rather than a size comparison: two
/// layouts can agree on total size and disagree on where a field sits.
const _: () = assert!(
    core::mem::offset_of!(Stack, head) == core::mem::offset_of!(Queue, head)
        && core::mem::offset_of!(Stack, head) == core::mem::offset_of!(Port, head),
);
const _: () = assert!(
    core::mem::offset_of!(Stack, size) == core::mem::offset_of!(Queue, size)
        && core::mem::offset_of!(Stack, size) == core::mem::offset_of!(Port, size),
);

/// The other half of the same invariant: where the field starts is only half of
/// which bytes it owns.
///
/// A field descr carries an offset AND a width, and the JIT registers one of
/// each per field under `Stack`'s type id.  Equal offsets with unequal widths
/// still read a queue's next word as part of its count — the failure the
/// offsets above are asserted against, arriving from the other side.
///
/// Written against the fields rather than the types they are declared as, so
/// that renaming `Stack::size`'s type does not quietly retire the check.
const _: () = {
    const fn width<S, T>(_: fn(&S) -> T) -> usize {
        core::mem::size_of::<T>()
    }
    assert!(
        width(|s: &Stack| s.head) == width(|q: &Queue| q.head)
            && width(|s: &Stack| s.head) == width(|p: &Port| p.head),
    );
    assert!(
        width(|s: &Stack| s.size) == width(|q: &Queue| q.size)
            && width(|s: &Stack| s.size) == width(|p: &Port| p.size),
    );
};

impl Port {
    // linkedlist.py:129-132
    // def __init__(self):
    //     self.head = None
    //     self.size = 0
    //     self.last_push = bigint.fromint(0)
    pub fn new() -> Self {
        Port {
            head: std::ptr::null_mut(),
            size: 0,
            last_push: val_from_i32(0),
        }
    }
}

impl LinkedList for Port {
    fn head(&self) -> *mut Node {
        self.head
    }
    fn set_head(&mut self, h: *mut Node) {
        self.head = h;
    }
    fn size(&self) -> usize {
        self.size as usize
    }
    fn set_size(&mut self, s: usize) {
        self.size = s as u32;
    }

    // linkedlist.py:134-139
    // def push(self, value):
    //     node = Node(self.head, value)
    //     self.head = node
    //     self.size += 1
    //     self.last_push = value
    fn push(&mut self, value: Val) {
        let rooted = value;
        let mut root = rooted;
        with_bigint_transient_root(&mut root, || {
            let node = alloc_node(rooted, self.head);
            self.head = node;
            self.size += 1;
            self.last_push = rooted;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:141-142
    // def dup(self): self.push(self.last_push)
    fn dup(&mut self) {
        self.push(self.last_push);
    }

    // linkedlist.py:144-145
    // def _get_2_values(self): return self.pop(), self.head.value
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        assert!(!self.head.is_null(), "_get_2_values on <2 elements");
        let r2 = unsafe { (*self.head).value };
        (r1, r2)
    }

    // linkedlist.py:147-148
    // def _put_value(self, value): self.head.value = value
    fn _put_value(&mut self, value: Val) {
        unsafe {
            (*self.head).value = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_basic() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        s.push(val_from_i32(10));
        s.push(val_from_i32(20));
        assert_eq!(val_to_i64(&s.pop()), 20);
        assert_eq!(val_to_i64(&s.pop()), 10);
    }

    #[test]
    fn test_stack_add() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        s.push(val_from_i32(10));
        s.push(val_from_i32(3));
        // add: r1=3, r2=10, result=13, replaces top
        s.add();
        assert_eq!(s.size, 1);
        assert_eq!(val_to_i64(&s.pop()), 13);
    }

    #[test]
    fn test_stack_dup_swap() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        s.push(val_from_i32(5));
        s.dup();
        assert_eq!(s.size, 2);
        s.push(val_from_i32(7));
        s.swap();
        assert_eq!(val_to_i64(&s.pop()), 5);
        assert_eq!(val_to_i64(&s.pop()), 7);
    }

    #[test]
    fn test_queue_basic() {
        let _lock = crate::storage::nursery_test_lock();
        let mut q = Queue::new();
        q.push(val_from_i32(1));
        q.push(val_from_i32(2));
        q.push(val_from_i32(3));
        assert_eq!(val_to_i64(&q.pop()), 1);
        assert_eq!(val_to_i64(&q.pop()), 2);
    }

    #[test]
    fn test_queue_add() {
        let _lock = crate::storage::nursery_test_lock();
        let mut q = Queue::new();
        q.push(val_from_i32(10));
        q.push(val_from_i32(3));
        // add: r1=10 (front), r2=3 (front), result=13 pushed to back
        q.add();
        assert_eq!(q.size, 1);
        assert_eq!(val_to_i64(&q.pop()), 13);
    }

    #[test]
    fn test_port_dup() {
        let _lock = crate::storage::nursery_test_lock();
        let mut p = Port::new();
        p.push(val_from_i32(5));
        p.push(val_from_i32(10));
        p.dup(); // duplicates last_push=10
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 5);
    }
}
