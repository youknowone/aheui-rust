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

/// Swap the values of the first two nodes in a linked-list chain.
///
/// `Stack`, `Queue`, and `Port` share this operation; keeping the pointer
/// mutation here gives the interpreter and generated JIT one implementation.
pub fn swap_nodes(node1: *mut Node) {
    assert!(!node1.is_null(), "swap on empty linked list");
    let node2 = unsafe { (*node1).next };
    assert!(!node2.is_null(), "swap on <2 elements");
    swap_nodes_known_two(node1);
}

/// Swap two nodes after the storage's length guard has proved they exist.
pub(super) fn swap_nodes_known_two(node1: *mut Node) {
    let node2 = unsafe { (*node1).next };
    // Read-then-write through the raw pointers rather than `std::mem::swap`,
    // which would need a `&mut` to each node at once and is undefined the
    // moment they are the same node. Nothing here can rule that out — a chain
    // whose head links to itself is exactly the shape the collector's
    // forwarding is capable of producing when it goes wrong, and this runs on
    // the recovery paths that would be diagnosing it. The `allow` is what
    // keeps a lint from rewriting this back into that `&mut` pair.
    #[allow(clippy::manual_swap)]
    unsafe {
        let value = (*node1).value;
        (*node1).value = (*node2).value;
        (*node2).value = value;
    }
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
/// The fields `LinkedList` declares, as a struct the subclasses embed.
///
/// `linkedlist.py:15-17` puts `head` and `size` on `LinkedList` itself, and
/// `rclass.py:548` lays a subclass out as `MkStruct(name, ('super',
/// rbase.object_type), *own_fields)` — the base is a real type inlined as the
/// leading field, not a layout the subclasses each happen to repeat.  Spelling
/// it that way here is what lets one physical `head` word carry one field
/// descriptor for all three storages: `rclass.py:987-1001` resolves a field
/// against the struct that DECLARES it, so `Stack`, `Queue` and `Port` accesses
/// all name this type.
///
/// Named `ListBase` rather than `LinkedList` because that name is taken by the
/// trait below, which carries the other half of the Python class — its methods.
#[repr(C)]
pub struct ListBase {
    pub head: *mut Node,
    /// The element count.  `u32`, not `usize`, so the JIT's field descr is
    /// sub-word and `intbounds` can bound a load of it: with no upper bound a
    /// depth `+ 1` may overflow, the sum goes rangeless, and every re-check of
    /// the depth has to be guarded again.  A list is a chain of 16-byte nodes,
    /// so 2^32 elements is 64 GiB of nodes.
    pub size: u32,
}

/// Remove and return the head element shared by all linked-list storages.
pub fn pop_base(list: &mut ListBase) -> Val {
    assert!(!list.head.is_null(), "pop from empty linked list");
    pop_base_known_nonempty(list)
}

/// Pop after the storage's length guard has proved the head exists.
pub(super) fn pop_base_known_nonempty(list: &mut ListBase) -> Val {
    let node = list.head;
    let next = unsafe { (*node).next };
    let value = unsafe { (*node).value };
    list.head = next;
    list.size -= 1;
    free_node(node);
    value
}

/// Swap the first two values shared by all linked-list storages.
pub fn swap_base(list: &mut ListBase) {
    assert!(!list.head.is_null(), "swap on empty linked list");
    let node2 = unsafe { (*list.head).next };
    assert!(!node2.is_null(), "swap on <2 elements");
    swap_base_known_two(list);
}

/// Swap after the storage's length guard has proved two nodes exist.
pub(super) fn swap_base_known_two(list: &mut ListBase) {
    swap_nodes_known_two(list.head);
}

impl ListBase {
    pub fn new() -> Self {
        ListBase {
            head: std::ptr::null_mut(),
            size: 0,
        }
    }
}

impl Default for ListBase {
    fn default() -> Self {
        Self::new()
    }
}

pub trait LinkedList {
    /// The inlined base every storage embeds as its leading field.  The
    /// accessors below are derived from it once, where each subclass used to
    /// repeat them over its own copies of the two fields.
    fn base(&self) -> &ListBase;
    fn base_mut(&mut self) -> &mut ListBase;

    // __slots__ accessors
    fn head(&self) -> *mut Node {
        self.base().head
    }
    fn set_head(&mut self, h: *mut Node) {
        self.base_mut().head = h;
    }
    fn size(&self) -> usize {
        self.base().size as usize
    }
    fn set_size(&mut self, s: usize) {
        self.base_mut().size = s as u32;
    }

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
        pop_base(self.base_mut())
    }

    // linkedlist.py:30-33
    // def swap(self):
    //     node1 = self.head
    //     node2 = node1.next
    //     node1.value, node2.value = node2.value, node1.value
    fn swap(&mut self) {
        swap_base(self.base_mut());
    }

    // linkedlist.py:35-38
    // def add(self): r1, r2 = self._get_2_values(); r = bigint.add(r2, r1); self._put_value(r)
    //
    // The combining half lives in `crate::band`, which is also where a caller
    // holding its operands outside a chain reaches it. Same fast paths, one
    // implementation.
    fn add(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_add(r2, r1));
    }

    // linkedlist.py:40-43
    fn sub(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_sub(r2, r1));
    }

    // linkedlist.py:45-48
    fn mul(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_mul(r2, r1));
    }

    // linkedlist.py:50-53
    fn div(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_div(r2, r1));
    }

    // linkedlist.py:55-58
    // Python name is `mod`; Rust reserves that keyword, so use `modulo`.
    fn modulo(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_mod(r2, r1));
    }

    // linkedlist.py:60-64
    // def cmp(self):
    //     r1, r2 = self._get_2_values()
    //     r = int(bigint.ge(r2, r1))
    //     big_r = bigint.fromint(r)
    //     self._put_value(big_r)
    fn cmp(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(crate::band::band_val_cmp(r2, r1));
    }
}

// linkedlist.py:67-91
// class Stack(LinkedList):
//     """Base data storage for Aheui, except for ieung and hieuh."""
//     __slots__ = ('head', 'size')
#[repr(C)]
pub struct Stack {
    pub base: ListBase,
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Stack {
    // linkedlist.py:72-74
    // def __init__(self):
    //     self.head = None
    //     self.size = 0
    pub fn new() -> Self {
        Stack {
            base: ListBase::new(),
        }
    }
}

impl LinkedList for Stack {
    fn base(&self) -> &ListBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut ListBase {
        &mut self.base
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
            let node = alloc_node(rooted, self.base.head);
            self.base.head = node;
            self.base.size += 1;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:82-83
    // def dup(self): self.push(self.head.value)
    fn dup(&mut self) {
        assert!(!self.base.head.is_null(), "dup on empty stack");
        let top = unsafe { (*self.base.head).value };
        self.push(top);
    }

    // linkedlist.py:87-88
    // def _get_2_values(self): return self.pop(), self.head.value
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        assert!(!self.base.head.is_null(), "_get_2_values on <2 elements");
        let r2 = unsafe { (*self.base.head).value };
        (r1, r2)
    }

    // linkedlist.py:90-91
    // def _put_value(self, value): self.head.value = value
    fn _put_value(&mut self, value: Val) {
        let rooted = value;
        let mut root = rooted;
        with_bigint_transient_root(&mut root, || {
            unsafe {
                (*self.base.head).value = rooted;
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
    pub base: ListBase,
    pub tail: *mut Node,
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
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
            base: ListBase {
                head: sentinel,
                size: 0,
            },
            tail: sentinel,
        }
    }
}

impl LinkedList for Queue {
    fn base(&self) -> &ListBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut ListBase {
        &mut self.base
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
            self.base.size += 1;
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
        let head = self.base.head;
        assert!(!head.is_null(), "dup on empty queue");
        let head_value = unsafe { (*head).value };
        let node = alloc_node(head_value, head);
        self.base.head = node;
        self.base.size += 1;
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
    pub base: ListBase,
    pub last_push: Val,
}

/// The three storages embed `ListBase` as their leading field, and the JIT
/// depends on it: an access spelled `queue.head` is resolved against the struct
/// that declares `head`, and the resulting offset is applied to the pointer the
/// access already had.  That names the right word only if the base starts at
/// zero.  `lltype.py:296-305` admits an inlined substructure on the same terms.
///
/// Sharing the fields by declaring one type is stronger than the offset and
/// width asserts this replaced.  Those compared three independently-written
/// layouts and could only report a drift after the fact; there is now one
/// declaration of `head` and `size`, so there is nothing left to drift.
const _: () = assert!(
    core::mem::offset_of!(Stack, base) == 0
        && core::mem::offset_of!(Queue, base) == 0
        && core::mem::offset_of!(Port, base) == 0,
);

impl Default for Port {
    fn default() -> Self {
        Self::new()
    }
}

impl Port {
    // linkedlist.py:129-132
    // def __init__(self):
    //     self.head = None
    //     self.size = 0
    //     self.last_push = bigint.fromint(0)
    pub fn new() -> Self {
        Port {
            base: ListBase::new(),
            last_push: val_from_i32(0),
        }
    }
}

impl LinkedList for Port {
    fn base(&self) -> &ListBase {
        &self.base
    }
    fn base_mut(&mut self) -> &mut ListBase {
        &mut self.base
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
            let node = alloc_node(rooted, self.base.head);
            self.base.head = node;
            self.base.size += 1;
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
        assert!(!self.base.head.is_null(), "_get_2_values on <2 elements");
        let r2 = unsafe { (*self.base.head).value };
        (r1, r2)
    }

    // linkedlist.py:147-148
    // def _put_value(self, value): self.head.value = value
    fn _put_value(&mut self, value: Val) {
        unsafe {
            (*self.base.head).value = value;
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
        assert_eq!(s.base.size, 1);
        assert_eq!(val_to_i64(&s.pop()), 13);
    }

    #[test]
    fn test_stack_dup_swap() {
        let _lock = crate::storage::nursery_test_lock();
        let mut s = Stack::new();
        s.push(val_from_i32(5));
        s.dup();
        assert_eq!(s.base.size, 2);
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
        assert_eq!(q.base.size, 1);
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
