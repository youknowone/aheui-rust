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
pub mod linkedlist_jit;

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
    chunks: Vec<*mut linkedlist::Node>, // base pointer of every allocated chunk
}

impl Nursery {
    const fn uninit() -> Self {
        Nursery {
            free: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            free_list: std::ptr::null_mut(),
            chunk_count: 0,
            chunks: Vec::new(),
        }
    }

    fn init(&mut self) {
        self.grow();
    }

    #[inline(always)]
    fn alloc(&mut self, value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
        // When the free list is empty and the current chunk is exhausted,
        // reclaim dead nodes before growing. `collect` refills `free_list`
        // from swept slots; only if it recovers nothing do we grow.
        if self.free_list.is_null() && self.free >= self.end {
            // `next` is the node being linked as the new node's successor — the
            // current top of the selected chain. It is live but may be held
            // only in a register (its `setfield(head)` not yet committed to
            // memory), so pass it as an extra root so `collect` never sweeps
            // it or the chain hanging off it.
            self.collect(next);
            if self.free_list.is_null() {
                self.grow();
            }
        }
        if !self.free_list.is_null() {
            let node = self.free_list;
            unsafe {
                self.free_list = (*node).next;
                (*node).value = value;
                (*node).next = next;
            }
            return node;
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
        self.chunks.push(base);
        self.free = base;
        self.end = unsafe { base.add(NURSERY_SIZE) };
    }

    /// Non-moving mark-sweep collection of the node chunks.
    ///
    /// Runs from `alloc` when the free list is empty and the current chunk's
    /// bump region is exhausted, before growing. Marks every node reachable
    /// from the registered roots (the 28 `pools[*]` head chains plus the
    /// `Port` chain), then sweeps every unmarked chunk slot onto the free
    /// list. Nodes are never moved, so pointers held in registers / on the
    /// stacks stay valid.
    ///
    /// Precondition: the caller guarantees `free_list` is empty, so no slot
    /// is double-linked. Every chunk is fully bumped at this point (collect
    /// only fires when the current chunk is exhausted and all prior chunks
    /// were exhausted before it), so there is no uninitialized tail to skip.
    ///
    /// Soundness relies on the roots being current: `jit_alloc_node` is a
    /// `residual_ref` (general heap effect) call, so the optimizer forces all
    /// pending `setfield(head)` lazy stores to memory before it — every live
    /// node is therefore reachable from a `pools[*]` / `port` head at collect
    /// time. A dead node whose head-store is still pending is marked from the
    /// stale root and merely survives one extra cycle (never a dangling live
    /// pointer). A node outside every chunk (oversized / foreign origin) is
    /// treated as always-live: it is walked through but never swept.
    #[cold]
    fn collect(&mut self, keep: *mut linkedlist::Node) {
        let storage_addr = GC_ROOTS.load(std::sync::atomic::Ordering::Relaxed);
        if storage_addr == 0 || self.chunks.is_empty() {
            return;
        }
        let storage = unsafe { &*(storage_addr as *const Storage) };

        const WORDS: usize = NURSERY_SIZE / 64;
        let mut marks: Vec<Box<[u64]>> = (0..self.chunks.len())
            .map(|_| vec![0u64; WORDS].into_boxed_slice())
            .collect();

        // Register-resident root: the successor node passed to the allocation
        // that triggered this collect (its head-store may still be pending).
        Self::mark_chain(&self.chunks, &mut marks, keep);
        for i in 0..STORAGE_COUNT {
            let stackp = storage.pools[i];
            if !stackp.is_null() {
                Self::mark_chain(&self.chunks, &mut marks, unsafe { (*stackp).head });
            }
        }
        Self::mark_chain(&self.chunks, &mut marks, storage.port.head);

        for ci in 0..self.chunks.len() {
            let base = self.chunks[ci];
            let bitmap = &marks[ci];
            for slot in 0..NURSERY_SIZE {
                if bitmap[slot / 64] & (1u64 << (slot % 64)) == 0 {
                    let node = unsafe { base.add(slot) };
                    unsafe { (*node).next = self.free_list };
                    self.free_list = node;
                }
            }
        }
    }

    /// Mark every chunk-resident node reachable through `next` from `head`.
    /// Stops early when it re-reaches an already-marked node (chains never
    /// share nodes in aheui, so this only guards against re-walking). A node
    /// outside every chunk is walked through without marking.
    fn mark_chain(chunks: &[*mut linkedlist::Node], marks: &mut [Box<[u64]>], head: *mut linkedlist::Node) {
        let mut node = head;
        while !node.is_null() {
            for (ci, &base) in chunks.iter().enumerate() {
                let end = (base as usize) + NURSERY_SIZE * NODE_SIZE;
                if node >= base && (node as usize) < end {
                    let slot = unsafe { node.offset_from(base) } as usize;
                    let word = &mut marks[ci][slot / 64];
                    let bit = 1u64 << (slot % 64);
                    if *word & bit != 0 {
                        return;
                    }
                    *word |= bit;
                    break;
                }
            }
            node = unsafe { (*node).next };
        }
    }
}

static mut NURSERY: Nursery = Nursery::uninit();

/// Root set for `Nursery::collect`: raw `*mut Storage` whose `pools[*]` /
/// `port` head chains enumerate every live node. Registered by the mainloop
/// via [`set_gc_roots`] after `Storage::refresh_pools`. Zero until set, which
/// disables collection (the allocator then only bumps / grows).
static GC_ROOTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Register the live `Storage` as the collection root set. Called once from
/// the mainloop after the storage pointers are refreshed.
pub fn set_gc_roots(storage: *mut Storage) {
    GC_ROOTS.store(storage as usize, std::sync::atomic::Ordering::Relaxed);
}

#[inline(always)]
pub fn alloc_node(value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
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

/// Stable addresses of the nursery bump pointers (`free`, `end`) for the
/// JIT-emitted inline allocator. The `static NURSERY` never moves, so the
/// field slot addresses are stable; `grow()` mutates only the contents,
/// which the inline code re-reads through these addresses each allocation.
/// `end` is one-past-the-last slot (the nursery_top limit used by `alloc`'s
/// own `free >= end` guard).
pub fn nursery_bump_addrs() -> (usize, usize) {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (
            std::ptr::addr_of!((*p).free) as usize,
            std::ptr::addr_of!((*p).end) as usize,
        )
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
/// The array is length-prefixed (a `usize` `pools_len` header at offset 0),
/// matching the GC-array layout the JIT's `getarrayitem_gc_r` descriptor
/// models (`base_size = 8`, `lendescr.offset = 0`); the items therefore start
/// one word in.
pub const STORAGE_POOLS_OFFSET: usize = 8;

#[repr(C)]
pub struct Storage {
    /// Length header for the `pools` array (always `STORAGE_COUNT`).  Placed
    /// first so `Storage` has the GC-array shape `{ len, items.. }` the JIT's
    /// `ARRAYLEN_GC` reads at offset 0 when it re-establishes the
    /// `len > selected` bound for the `pools[selected]` `getarrayitem_gc_r`.
    /// Immutable after construction (`pools` is fixed-size).
    pub pools_len: usize,
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
            pools_len: STORAGE_COUNT,
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
