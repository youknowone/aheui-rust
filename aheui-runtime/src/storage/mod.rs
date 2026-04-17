//! Storage system for Aheui: 28 storage spaces (Stacks, Queue, Port).
//!
//! Directory layout mirrors `rpaheui/aheui/storage/`:
//!   * [`linkedlist`] — `linkedlist.py` (Node / LinkedList / Stack / Queue / Port).
//!   * [`array`] — parity stub for `array.py` (CPython-only backend).
//!
//! The aggregate [`Storage`] below mirrors `rpaheui/aheui/aheui.py::class Storage`.
//! RPython relies on Python duck typing and the GC to carry Stack/Queue/
//! Port references inside `pools`. Rust can not store mixed subclass
//! instances in a flat array, so we split them: a per-index `Stack`
//! array plus a dedicated `Queue` (slot `VAL_QUEUE`) and `Port`
//! (slot `VAL_PORT`). The `pools` indirection array gives the JIT and
//! the interpreter a uniform `*mut Stack` handle compatible with the
//! Stack/Queue head/size layout; polymorphic dispatch for push / dup /
//! _get_2_values / _put_value still goes through the [`LinkedList`]
//! trait via [`Storage::dispatch_mut`].

pub mod array;
pub mod linkedlist;

pub use linkedlist::{
    LinkedList, NODE_NEXT_OFFSET, NODE_SIZE, NODE_VALUE_OFFSET, Node, Port, Queue, Stack,
};

use crate::aheui::{STORAGE_COUNT, VAL_PORT, VAL_QUEUE};
use crate::value::*;

// ── Nursery bump allocator for `Node` ────────────────────────────────
//
// RPython-style nursery adapted for Rust: a large contiguous buffer
// where alloc() is a pointer bump and free() returns the node to a
// singly-linked free list. Python relies on the RPython GC for
// reclamation; we need an explicit pool because there is no tracing GC
// in aheui-runtime.

/// Number of `Node` slots per nursery chunk (256K nodes ≈ 4MB at 16 bytes each).
const NURSERY_SIZE: usize = 256 * 1024;

/// Max nursery chunks: 64 chunks × 4MB = 256MB.
const MAX_NURSERY_CHUNKS: usize = 64;

struct Nursery {
    free: *mut linkedlist::Node,      // bump pointer — next slot to allocate
    end: *mut linkedlist::Node,       // one past the last slot in current chunk
    free_list: *mut linkedlist::Node, // singly-linked free list of popped nodes
    chunk_count: usize,               // number of allocated chunks (safety limit)
}

impl Nursery {
    const fn uninit() -> Self {
        Nursery {
            free: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            free_list: std::ptr::null_mut(),
            chunk_count: 0,
        }
    }

    fn init(&mut self) {
        self.grow();
    }

    #[inline(always)]
    fn alloc(&mut self, value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
        if !self.free_list.is_null() {
            let node = self.free_list;
            unsafe {
                self.free_list = (*node).next;
                (*node).value = value;
                (*node).next = next;
            }
            return node;
        }
        if self.free >= self.end {
            self.grow();
        }
        let node = self.free;
        unsafe {
            (*node).value = value;
            (*node).next = next;
            self.free = node.add(1);
        }
        node
    }

    #[inline(always)]
    fn free_node(&mut self, node: *mut linkedlist::Node) {
        if node.is_null() {
            return;
        }
        unsafe {
            (*node).next = self.free_list;
        }
        self.free_list = node;
    }

    #[cold]
    fn grow(&mut self) {
        self.chunk_count += 1;
        if self.chunk_count > MAX_NURSERY_CHUNKS {
            eprintln!(
                "[nursery] allocation limit reached ({} chunks × {}KB = {}MB)",
                MAX_NURSERY_CHUNKS,
                NURSERY_SIZE * std::mem::size_of::<linkedlist::Node>() / 1024,
                MAX_NURSERY_CHUNKS * NURSERY_SIZE * std::mem::size_of::<linkedlist::Node>()
                    / (1024 * 1024),
            );
            std::process::exit(99);
        }
        let layout = std::alloc::Layout::array::<linkedlist::Node>(NURSERY_SIZE).unwrap();
        let base = unsafe { std::alloc::alloc(layout) as *mut linkedlist::Node };
        assert!(!base.is_null(), "nursery allocation failed");
        self.free = base;
        self.end = unsafe { base.add(NURSERY_SIZE) };
    }
}

static mut NURSERY: Nursery = Nursery::uninit();

#[inline(always)]
pub(crate) fn alloc_node(value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).alloc(value, next)
    }
}

/// Return a node to the free list. Called by `LinkedList::pop` to reclaim
/// memory.
#[inline(always)]
pub fn free_node(node: *mut linkedlist::Node) {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).free_node(node);
    }
}

/// Allocate a zeroed `Node` without initializing fields.
/// Used by JIT's GcAllocator to get a node from the shared nursery.
#[inline(always)]
pub fn alloc_node_raw() -> *mut linkedlist::Node {
    alloc_node(val_from_i32(0), std::ptr::null_mut())
}

fn init_nursery() {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).init();
    }
}

// rpaheui/aheui/aheui.py:37-53
// class Storage(object):
//     _immutable_fields_ = ['pools']
//     def __init__(self):
//         pools = []
//         for i in range(0, c.STORAGE_COUNT):
//             if i == c.VAL_QUEUE: pools.append(Queue())
//             elif i == c.VAL_PORT: pools.append(Port())
//             else: pools.append(Stack())
//         self.pools = pools
//     @jit.elidable
//     def __getitem__(self, idx):
//         return self.pools[idx]
//
// Python stores mixed subclass instances in a flat `pools` list.
// Rust splits them: a flat `Stack` array plus dedicated `Queue` / `Port`
// fields. The `pools` indirection array gives the JIT a uniform
// `*mut Stack`-compatible pointer for all 28 slots (Queue/Port share
// the `head`/`size` prefix thanks to `#[repr(C)]`), while polymorphic
// method dispatch still goes through the [`LinkedList`] trait.

/// Size of `Stack` for JIT stride calculation.
pub const STACK_SIZE: usize = std::mem::size_of::<Stack>();
/// `Stack.head` offset (== 0 for `#[repr(C)]`).
pub const STACK_HEAD_OFFSET: usize = 0;
/// `Stack.size` offset (== 8).
pub const STACK_SIZE_OFFSET: usize = 8;
/// `Queue.tail` offset (== 16). Queue push appends to tail.
pub const QUEUE_TAIL_OFFSET: usize = 16;

/// Byte offset of the `pools` indirection array from `Storage` base.
/// JIT reads: `base + STORAGE_POOLS_OFFSET + selected * 8` → `*mut Stack`.
pub const STORAGE_POOLS_OFFSET: usize = 0;

#[repr(C)]
pub struct Storage {
    /// JIT indirection: `pools[idx]` returns a `*mut Stack`-compatible pointer.
    /// Initialized to `&mut stacks[idx]` or `&mut queue`.
    pub pools: [*mut Stack; STORAGE_COUNT],
    /// Flat array of `Stack` (one per index).
    pub stacks: [Stack; STORAGE_COUNT],
    /// Queue storage (slot `VAL_QUEUE`).
    pub queue: Queue,
    /// Port storage (slot `VAL_PORT`).
    pub port: Port,
}

// SAFETY: raw pointers are only to self.stacks / self.queue, not shared across threads.
unsafe impl Send for Storage {}

impl Storage {
    // aheui.py:40-49
    pub fn new() -> Self {
        init_nursery();

        let mut storage = Storage {
            pools: [std::ptr::null_mut(); STORAGE_COUNT],
            stacks: std::array::from_fn(|_| Stack::new()),
            queue: Queue::new(),
            port: Port::new(),
        };
        storage.refresh_pools();
        storage
    }

    /// Sync the `pools` indirection array to point at `stacks[i]` /
    /// `queue`. Must be called after `Storage` is moved, because the
    /// pointers are self-referencing.
    pub fn refresh_pools(&mut self) {
        for i in 0..STORAGE_COUNT {
            self.pools[i] = &mut self.stacks[i] as *mut Stack;
        }
        // VAL_QUEUE: alias &mut queue (head/size share `#[repr(C)]` layout with Stack).
        self.pools[VAL_QUEUE] = &mut self.queue as *mut Queue as *mut Stack;
    }

    // aheui.py:51-53
    // @jit.elidable
    // def __getitem__(self, idx):
    //     return self.pools[idx]
    //
    // Rust needs two entry points: a thin raw-pointer version used by
    // the JIT (`get_stack_ptr`) and a polymorphic trait-object version
    // used by the interpreter for correct `Queue` / `Port` dispatch.
    #[inline(always)]
    pub fn get_stack_ptr(&mut self, idx: usize) -> *mut Stack {
        debug_assert!(idx < STORAGE_COUNT);
        self.pools[idx]
    }

    /// Polymorphic `storage[idx]` — returns `&dyn LinkedList`.
    pub fn dispatch(&self, idx: usize) -> &dyn LinkedList {
        if idx == VAL_QUEUE {
            &self.queue
        } else if idx == VAL_PORT {
            &self.port
        } else {
            &self.stacks[idx]
        }
    }

    /// Polymorphic `storage[idx]` — returns `&mut dyn LinkedList`.
    pub fn dispatch_mut(&mut self, idx: usize) -> &mut dyn LinkedList {
        if idx == VAL_QUEUE {
            &mut self.queue
        } else if idx == VAL_PORT {
            &mut self.port
        } else {
            &mut self.stacks[idx]
        }
    }

    // Convenience accessors for tests / I/O paths that need a concrete
    // subclass rather than polymorphic dispatch.
    pub fn stack(&self, idx: usize) -> &Stack {
        &self.stacks[idx]
    }
    pub fn stack_mut(&mut self, idx: usize) -> &mut Stack {
        &mut self.stacks[idx]
    }
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    pub fn queue_mut(&mut self) -> &mut Queue {
        &mut self.queue
    }
    pub fn port(&self) -> &Port {
        &self.port
    }
    pub fn port_mut(&mut self) -> &mut Port {
        &mut self.port
    }

    /// Length of the storage at `idx` — `len(storage[idx])` in Python.
    pub fn len_at(&self, idx: usize) -> usize {
        self.dispatch(idx).__len__()
    }

    /// Restore linked-list heads from JIT guard failure values. Rust-only
    /// — the JIT materializes virtualized `Node` chains back into the
    /// real storage on bridge entry.
    pub fn restore_heads(&mut self, storage_layout: &[(usize, usize)], heads: &[i64]) {
        for (i, &(sidx, _)) in storage_layout.iter().enumerate() {
            if let Some(&head_raw) = heads.get(i) {
                let stack = unsafe { &mut *self.pools[sidx] };
                stack.head = head_raw as *mut linkedlist::Node;
                let mut count = 0usize;
                let mut cur = stack.head;
                while !cur.is_null() {
                    count += 1;
                    cur = unsafe { (*cur).next };
                }
                stack.size = count;
            }
        }
    }

    /// JIT capability checks — always true for linked-list backends.
    pub fn all_values_small(&self) -> bool {
        true
    }

    #[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
    #[inline(always)]
    pub fn all_jit_compatible(&self) -> bool {
        true
    }

    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    #[inline(always)]
    pub fn all_jit_compatible(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_init() {
        let storage = Storage::new();
        assert_eq!(storage.stacks.len(), STORAGE_COUNT);
        assert_eq!(storage.stack(0).__len__(), 0);
        assert_eq!(storage.queue().__len__(), 0);
        assert_eq!(storage.port().__len__(), 0);
    }

    #[test]
    fn test_storage_queue_dispatch() {
        // Queue polymorphic dispatch goes through LinkedList trait.
        let mut storage = Storage::new();
        let q = storage.dispatch_mut(VAL_QUEUE);
        q.push(val_from_i32(1));
        q.push(val_from_i32(2));
        q.push(val_from_i32(3));
        // Queue.add pops twice from front (r1=1, r2=2), pushes r2+r1=3 to back.
        // Remaining queue is [3 (original), 3 (computed)].
        q.add();
        assert_eq!(q.__len__(), 2);
        assert_eq!(val_to_i64(&q.pop()), 3);
        assert_eq!(val_to_i64(&q.pop()), 3);
    }
}
