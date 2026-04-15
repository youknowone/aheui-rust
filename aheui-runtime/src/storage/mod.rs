//! Storage system for Aheui: 28 storage spaces (Stacks, Queue, Port).
//!
//! Directory layout mirrors `rpaheui/aheui/storage/`:
//!   * [`linkedlist`] — `linkedlist.py` (Node / Stack / Queue / Port).
//!   * [`array`] — parity stub for `array.py` (CPython-only backend).
//!
//! This module keeps the Rust-only glue: the nursery bump allocator
//! that backs `alloc_node`, the `StorageOps` trait used in place of
//! Python's duck typing, the `StorageKind` wrapper, and the
//! `StoragePool` aggregate that corresponds to rpaheui's
//! `aheui.aheui.Storage`.

pub mod array;
pub mod linkedlist;

pub use linkedlist::{
    AheuiPort, AheuiQueue, AheuiStack, STACKNODE_NEXT_OFFSET, STACKNODE_SIZE,
    STACKNODE_VALUE_OFFSET, StackNode,
};

use crate::aheui::{STORAGE_COUNT, VAL_PORT, VAL_QUEUE};
use crate::value::*;

// ── Nursery bump allocator for StackNode ─────────────────────────────
//
// RPython-style nursery: a large contiguous buffer where alloc() is just
// a pointer bump (~1 cycle).  free() is a no-op — memory is never
// reclaimed per-node.  When the current chunk is full, a new chunk is
// allocated (old chunk is leaked since nodes may still be referenced).
//
// For aheui programs with bounded stack depth this is ideal: the nursery
// never grows beyond a few MB, and allocation is maximally fast.
//
// SAFETY: aheui interpreter is single-threaded.  The static mut is only
// accessed from the main thread.

/// Number of StackNode slots per nursery chunk (256K nodes = 4MB at 16 bytes each).
const NURSERY_SIZE: usize = 256 * 1024;

/// Max nursery chunks: 64 chunks × 4MB = 256MB.
const MAX_NURSERY_CHUNKS: usize = 64;

struct Nursery {
    free: *mut StackNode,      // bump pointer — next slot to allocate
    end: *mut StackNode,       // one past the last slot in current chunk
    free_list: *mut StackNode, // singly-linked free list of popped nodes
    chunk_count: usize,        // number of allocated chunks (safety limit)
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

    /// Allocate the initial nursery chunk.  Called lazily on first alloc.
    fn init(&mut self) {
        self.grow();
    }

    /// Allocate a single StackNode: try free list first, then bump.
    #[inline(always)]
    fn alloc(&mut self, value: Val, next: *mut StackNode) -> *mut StackNode {
        // Reuse a previously freed node if available.
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

    /// Return a popped node to the free list for reuse.
    #[inline(always)]
    fn free_node(&mut self, node: *mut StackNode) {
        if node.is_null() {
            return;
        }
        unsafe {
            (*node).next = self.free_list;
        }
        self.free_list = node;
    }

    /// Allocate a fresh nursery chunk.  The old chunk is intentionally
    /// leaked — live nodes may still be referenced by stack/queue heads.
    #[cold]
    fn grow(&mut self) {
        self.chunk_count += 1;
        if self.chunk_count > MAX_NURSERY_CHUNKS {
            eprintln!(
                "[nursery] allocation limit reached ({} chunks × {}KB = {}MB)",
                MAX_NURSERY_CHUNKS,
                NURSERY_SIZE * std::mem::size_of::<StackNode>() / 1024,
                MAX_NURSERY_CHUNKS * NURSERY_SIZE * std::mem::size_of::<StackNode>()
                    / (1024 * 1024),
            );
            std::process::exit(99);
        }
        let layout = std::alloc::Layout::array::<StackNode>(NURSERY_SIZE).unwrap();
        let base = unsafe { std::alloc::alloc(layout) as *mut StackNode };
        assert!(!base.is_null(), "nursery allocation failed");
        self.free = base;
        self.end = unsafe { base.add(NURSERY_SIZE) };
    }
}

static mut NURSERY: Nursery = Nursery::uninit();

#[inline(always)]
pub(crate) fn alloc_node(value: Val, next: *mut StackNode) -> *mut StackNode {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).alloc(value, next)
    }
}

/// Return a node to the free list. Called by pop() to reclaim memory.
#[inline(always)]
pub fn free_node(node: *mut StackNode) {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).free_node(node);
    }
}

/// Allocate a zeroed StackNode without initializing fields.
/// Used by JIT's GcAllocator to get a node from the shared nursery.
#[inline(always)]
pub fn alloc_node_raw() -> *mut StackNode {
    alloc_node(val_from_i32(0), std::ptr::null_mut())
}

fn init_nursery() {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).init();
    }
}

/// rpaheui LinkedList common interface.
/// Stack, Queue, Port all implement these.
pub trait StorageOps {
    fn push(&mut self, value: Val);
    fn pop(&mut self) -> Val;
    fn push_raw(&mut self, raw: i64) {
        self.push(unsafe { std::mem::transmute(raw) });
    }
    fn pop_raw(&mut self) -> i64 {
        unsafe { std::mem::transmute(self.pop()) }
    }
    fn dup(&mut self);
    fn swap(&mut self);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn peek_at(&self, i: usize) -> i64;
    fn add(&mut self);
    fn sub(&mut self);
    fn mul(&mut self);
    fn div(&mut self);
    fn modulo(&mut self);
    fn cmp(&mut self);
}

/// Common storage operations.
pub enum StorageKind {
    Stack(AheuiStack),
    Queue(AheuiQueue),
    Port(AheuiPort),
}

impl StorageKind {
    pub fn push(&mut self, value: impl Into<Val>) {
        match self {
            StorageKind::Stack(s) => s.push(value),
            StorageKind::Queue(q) => q.push(value),
            StorageKind::Port(p) => p.push(value),
        }
    }

    pub fn pop(&mut self) -> Val {
        match self {
            StorageKind::Stack(s) => s.pop(),
            StorageKind::Queue(q) => q.pop(),
            StorageKind::Port(p) => p.pop(),
        }
    }

    pub fn dup(&mut self) {
        match self {
            StorageKind::Stack(s) => s.dup(),
            StorageKind::Queue(q) => q.dup(),
            StorageKind::Port(p) => p.dup(),
        }
    }

    pub fn swap(&mut self) {
        match self {
            StorageKind::Stack(s) => s.swap(),
            StorageKind::Queue(q) => q.swap(),
            StorageKind::Port(p) => p.swap(),
        }
    }

    pub fn add(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_add),
            StorageKind::Queue(q) => q.binop(val_add),
            StorageKind::Port(p) => p.binop(val_add),
        }
    }

    pub fn sub(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_sub),
            StorageKind::Queue(q) => q.binop(val_sub),
            StorageKind::Port(p) => p.binop(val_sub),
        }
    }

    pub fn mul(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_mul),
            StorageKind::Queue(q) => q.binop(val_mul),
            StorageKind::Port(p) => p.binop(val_mul),
        }
    }

    pub fn div(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_div),
            StorageKind::Queue(q) => q.binop(val_div),
            StorageKind::Port(p) => p.binop(val_div),
        }
    }

    pub fn modulo(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_mod),
            StorageKind::Queue(q) => q.binop(val_mod),
            StorageKind::Port(p) => p.binop(val_mod),
        }
    }

    pub fn cmp(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
            StorageKind::Queue(q) => q.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
            StorageKind::Port(p) => p.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            StorageKind::Stack(s) => s.size,
            StorageKind::Queue(q) => q.size,
            StorageKind::Port(p) => p.size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn peek_at(&self, i: usize) -> i64 {
        match self {
            StorageKind::Stack(s) => s.peek_at(i),
            StorageKind::Queue(q) => StorageOps::peek_at(q, i),
            StorageKind::Port(p) => StorageOps::peek_at(p, i),
        }
    }
}

// ── RPython parity: Storage layout ───────────────────────────────────
//
// rpaheui/aheui/aheui.py:
//   Storage.pools = [Stack(), Queue(), Port(), Stack(), ...]
//   selected = storage[idx]  → Stack object (head, size)
//   JIT accesses selected.head (offset 0) and selected.size (offset 8)
//
// In Rust, we place AheuiStack objects in a flat #[repr(C)] array
// so JIT can access pool_ptr + idx * sizeof(AheuiStack) directly.

/// Size of AheuiStack for JIT stride calculation.
pub const AHEUI_STACK_SIZE: usize = std::mem::size_of::<AheuiStack>();
/// AheuiStack.head offset (== 0 for #[repr(C)]).
pub const AHEUI_STACK_HEAD_OFFSET: usize = 0;
/// AheuiStack.size offset (== 8).
pub const AHEUI_STACK_SIZE_OFFSET: usize = 8;
/// AheuiQueue.tail offset (== 16). Queue push appends to tail.
pub const AHEUI_QUEUE_TAIL_OFFSET: usize = 16;

/// Pool of 28 storage spaces.
///
/// RPython: `aheui.aheui.Storage.pools` — fixed-size list of
/// Stack/Queue/Port.
/// JIT: `selected = storage[idx]` via @jit.elidable.
///
/// Layout: pointer indirection array first (JIT reads pool_ptr + idx * 8),
/// then flat stacks array, then Queue/Port.
#[repr(C)]
pub struct StoragePool {
    /// JIT indirection: pool_ptr + STACKS_OFFSET + idx * 8 → *mut AheuiStack.
    /// Initialized to point at stacks[idx]. JIT does GetfieldGcR here.
    pub stack_ptrs: [*mut AheuiStack; STORAGE_COUNT],
    /// Flat array of AheuiStack (actual data).
    pub stacks: [AheuiStack; STORAGE_COUNT],
    /// Queue storage (slot VAL_QUEUE).
    pub queue: AheuiQueue,
    /// Port storage (slot VAL_PORT).
    pub port: AheuiPort,
}

// SAFETY: raw pointers are only to self.stacks, not shared across threads.
unsafe impl Send for StoragePool {}

/// Byte offset of stack_ptrs array from StoragePool base.
/// JIT reads: pool_ptr + STORAGEPOOL_STACKS_OFFSET + selected * 8 → *mut AheuiStack.
pub const STORAGEPOOL_STACKS_OFFSET: usize = 0;

impl StoragePool {
    pub fn new() -> Self {
        // Initialize nursery bump allocator (first chunk).
        init_nursery();

        let mut pool = StoragePool {
            stack_ptrs: [std::ptr::null_mut(); STORAGE_COUNT],
            stacks: std::array::from_fn(|_| AheuiStack::new()),
            queue: AheuiQueue::new(),
            port: AheuiPort::new(),
        };
        pool.refresh_stack_ptrs();
        pool
    }

    /// Sync stack_ptrs indirection array to point at stacks[i].
    /// Must be called after StoragePool is moved.
    /// VAL_QUEUE is aliased to &queue (head/size fields at same offsets as AheuiStack).
    pub fn refresh_stack_ptrs(&mut self) {
        for i in 0..STORAGE_COUNT {
            self.stack_ptrs[i] = &mut self.stacks[i] as *mut AheuiStack;
        }
        // Alias queue: AheuiQueue.head (offset 0) and .size (offset 8) match AheuiStack layout.
        // JIT can read/write head and size through the same field descriptors.
        self.stack_ptrs[VAL_QUEUE] = &mut self.queue as *mut AheuiQueue as *mut AheuiStack;
    }

    /// RPython: storage[idx] (macro uses this as `.get(idx)`)
    /// Uses stack_ptrs indirection so VAL_QUEUE returns the queue (head/size compatible).
    pub fn get(&self, idx: usize) -> &AheuiStack {
        unsafe { &*self.stack_ptrs[idx] }
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut AheuiStack {
        unsafe { &mut *self.stack_ptrs[idx] }
    }

    pub fn get_stack(&self, idx: usize) -> &AheuiStack {
        self.get(idx)
    }
    pub fn get_stack_mut(&mut self, idx: usize) -> &mut AheuiStack {
        self.get_mut(idx)
    }

    pub fn get_queue(&self) -> &AheuiQueue {
        &self.queue
    }
    pub fn get_queue_mut(&mut self) -> &mut AheuiQueue {
        &mut self.queue
    }
    pub fn get_port(&self) -> &AheuiPort {
        &self.port
    }

    pub fn get_port_mut(&mut self) -> &mut AheuiPort {
        &mut self.port
    }

    /// RPython: storage[idx] with polymorphic dispatch.
    /// Returns &dyn StorageOps for Stack/Queue/Port.
    pub fn dispatch(&self, idx: usize) -> &dyn StorageOps {
        if idx == VAL_QUEUE {
            &self.queue
        } else if idx == VAL_PORT {
            &self.port
        } else {
            &self.stacks[idx]
        }
    }

    pub fn dispatch_mut(&mut self, idx: usize) -> &mut dyn StorageOps {
        if idx == VAL_QUEUE {
            &mut self.queue
        } else if idx == VAL_PORT {
            &mut self.port
        } else {
            &mut self.stacks[idx]
        }
    }

    /// Pointer to AheuiStack at idx — JIT's selected_ref points here.
    /// JIT reads head at offset 0, size at offset 8.
    #[inline(always)]
    pub fn get_stack_ptr(&mut self, idx: usize) -> *mut AheuiStack {
        debug_assert!(idx < STORAGE_COUNT);
        self.stack_ptrs[idx]
    }

    /// Raw mutable pointer to AheuiStack at idx (for cross-storage MOV).
    #[inline(always)]
    pub fn get_mut_ptr(&mut self, idx: usize) -> *mut AheuiStack {
        self.get_stack_ptr(idx)
    }

    /// Storage length for any slot.
    pub fn len_at(&self, idx: usize) -> usize {
        if idx == VAL_QUEUE {
            self.queue.size
        } else if idx == VAL_PORT {
            self.port.size
        } else {
            self.stacks[idx].len()
        }
    }

    /// Restore linked-list heads from JIT guard failure values.
    pub fn restore_heads(&mut self, storage_layout: &[(usize, usize)], heads: &[i64]) {
        for (i, &(sidx, _)) in storage_layout.iter().enumerate() {
            if let Some(&head_raw) = heads.get(i) {
                // Use stack_ptrs indirection to handle VAL_QUEUE correctly.
                let stack = unsafe { &mut *self.stack_ptrs[sidx] };
                stack.head = head_raw as *mut StackNode;
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
        true // linked list stacks use i64 values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_pool() {
        let pool = StoragePool::new();
        assert_eq!(pool.stacks.len(), STORAGE_COUNT);
        // Stack slots are AheuiStack
        assert_eq!(pool.get_stack(0).len(), 0);
        // Queue and Port are separate fields
        assert_eq!(pool.get_queue().size, 0);
        assert_eq!(pool.get_port().size, 0);
    }
}
