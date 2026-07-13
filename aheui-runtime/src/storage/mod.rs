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
use std::sync::atomic::{AtomicUsize, Ordering};

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
    spare_chunk: *mut linkedlist::Node, // to-space for the next copying collect
}

impl Nursery {
    const fn uninit() -> Self {
        Nursery {
            free: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            free_list: std::ptr::null_mut(),
            chunk_count: 0,
            chunks: Vec::new(),
            spare_chunk: std::ptr::null_mut(),
        }
    }

    fn init(&mut self) {
        self.grow();
    }

    #[inline(always)]
    fn alloc(&mut self, value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
        let mut next = next;
        // When the free list is empty and the current chunk is exhausted,
        // evacuate live nodes into to-space before growing. Interpreter
        // pops still reuse `free_list`; collection discards it because the
        // moving collector builds a compact current chunk set instead.
        if self.free_list.is_null() && self.free >= self.end {
            // `next` is the node being linked as the new node's successor — the
            // current top of the selected chain. It is live but may be held
            // only in a register (its `setfield(head)` not yet committed to
            // memory), so pass it as an extra root so `collect` forwards it
            // and the chain hanging off it.
            self.collect(&mut next);
            if self.free_list.is_null() && self.free >= self.end {
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
        let base = self.allocate_chunk();
        self.chunks.push(base);
        self.free = base;
        self.end = unsafe { base.add(NURSERY_SIZE) };
    }

    fn allocate_chunk(&mut self) -> *mut linkedlist::Node {
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
        base
    }

    fn release_chunk(&mut self, base: *mut linkedlist::Node) {
        if base.is_null() {
            return;
        }
        let layout = std::alloc::Layout::array::<linkedlist::Node>(NURSERY_SIZE).unwrap();
        unsafe { std::alloc::dealloc(base as *mut u8, layout) };
        self.chunk_count -= 1;
    }

    #[cfg(test)]
    fn reset_for_test(&mut self) {
        let layout = std::alloc::Layout::array::<linkedlist::Node>(NURSERY_SIZE).unwrap();
        for chunk in self.chunks.drain(..) {
            unsafe { std::alloc::dealloc(chunk as *mut u8, layout) };
        }
        if !self.spare_chunk.is_null() {
            unsafe { std::alloc::dealloc(self.spare_chunk as *mut u8, layout) };
        }
        *self = Nursery::uninit();
    }

    /// Copying collection of nursery-resident `Node` chains.
    ///
    /// Runs from `alloc` when the interpreter's eager-free list is empty and
    /// the current bump chunk is exhausted. The collector follows RPython
    /// incminimark's nursery evacuation shape, anchored at
    /// `rpython/memory/gc/incminimark.py:2108`
    /// `trace_and_drag_out_of_nursery`: copy a nursery object, install a
    /// forwarding pointer in from-space, then trace copied objects until the
    /// worklist is drained. `Node` is headerless (`value,next`), so the
    /// forwarding mark lives in a side bitmap per from-space chunk and the
    /// forwarded address is stored in the old node's `next` field after
    /// copying.
    ///
    /// Root set:
    /// - every `pools[*].head` chain registered through [`set_gc_roots`],
    /// - `queue.tail`, because the queue sentinel may be reachable only from
    ///   the tail field,
    /// - `port.head`,
    /// - the allocating call's `&mut next`, which may still be register-held
    ///   before the caller stores the new head, and
    /// - every slot reported by [`NODE_ROOT_WALK_HOOK`] for JIT-held node
    ///   references.
    ///
    /// Soundness contract: at can-collect calls, `SAVE_GCREF_REGS` spills all
    /// JIT-held `Ref`s to `jf_gcmap`-marked jitframe slots, and the hook walks
    /// those slots before from-space is released. Interpreter helpers never
    /// hold a node pointer across allocation except for the forwarded `next`
    /// root passed here and `Queue::push`'s tail, which is passed through
    /// allocation as an explicit keep root because collection may move it.
    #[cold]
    fn collect(&mut self, keep: &mut *mut linkedlist::Node) {
        let storage_addr = GC_ROOTS.load(Ordering::Relaxed);
        if storage_addr == 0 || self.chunks.is_empty() {
            return;
        }
        #[cfg(test)]
        COPYING_COLLECT_COUNT_FOR_TESTS.fetch_add(1, Ordering::Relaxed);

        let storage = unsafe { &mut *(storage_addr as *mut Storage) };
        let from_chunks = self.chunks.clone();
        let mut copy = CopyState::new(from_chunks);

        let first_to = if self.spare_chunk.is_null() {
            self.allocate_chunk()
        } else {
            let chunk = self.spare_chunk;
            self.spare_chunk = std::ptr::null_mut();
            chunk
        };
        copy.add_to_chunk(first_to);

        self.forward_root(&mut copy, keep as *mut *mut linkedlist::Node);
        for i in 0..STORAGE_COUNT {
            let stackp = storage.pools[i];
            if !stackp.is_null() {
                self.forward_root(&mut copy, unsafe { std::ptr::addr_of_mut!((*stackp).head) });
            }
        }
        self.forward_root(&mut copy, std::ptr::addr_of_mut!(storage.queue.tail));
        self.forward_root(&mut copy, std::ptr::addr_of_mut!(storage.port.head));

        let hook_addr = NODE_ROOT_WALK_HOOK.load(Ordering::Relaxed);
        if hook_addr != 0 {
            let hook: NodeRootWalkHook = unsafe { std::mem::transmute(hook_addr) };
            let mut visit = |slot: *mut *mut linkedlist::Node| {
                self.forward_root(&mut copy, slot);
            };
            hook(&mut visit);
        }

        self.trace_copied_nodes(&mut copy);

        let old_chunks = std::mem::replace(&mut self.chunks, copy.to_chunks);
        self.free = copy.to_free;
        self.end = copy.to_end;
        self.free_list = std::ptr::null_mut();

        let mut spare = old_chunks.into_iter();
        self.spare_chunk = spare.next().unwrap_or(std::ptr::null_mut());
        for chunk in spare {
            self.release_chunk(chunk);
        }
    }

    fn trace_copied_nodes(&mut self, copy: &mut CopyState) {
        let mut scan_chunk = 0usize;
        let mut scan = copy.to_chunks[0];
        loop {
            if scan_chunk >= copy.to_chunks.len() {
                break;
            }

            if scan_chunk + 1 == copy.to_chunks.len() {
                if scan == copy.to_free {
                    break;
                }
            } else {
                let chunk_end = unsafe { copy.to_chunks[scan_chunk].add(NURSERY_SIZE) };
                if scan == chunk_end {
                    scan_chunk += 1;
                    scan = copy.to_chunks[scan_chunk];
                    continue;
                }
            }

            self.forward_root(copy, unsafe { std::ptr::addr_of_mut!((*scan).next) });
            scan = unsafe { scan.add(1) };
        }
    }

    fn forward_root(&mut self, copy: &mut CopyState, slot: *mut *mut linkedlist::Node) {
        if slot.is_null() {
            return;
        }
        let old = unsafe { *slot };
        if old.is_null() {
            return;
        }
        let Some((ci, index)) = Self::chunk_slot(&copy.from_chunks, old) else {
            return;
        };

        let bit = 1u64 << (index % 64);
        if copy.forwarded[ci][index / 64] & bit != 0 {
            unsafe {
                *slot = (*old).next;
            }
            return;
        }

        let new = self.copy_alloc(copy);
        unsafe {
            (*new).value = (*old).value;
            (*new).next = (*old).next;
            (*old).next = new;
            *slot = new;
        }
        copy.forwarded[ci][index / 64] |= bit;
    }

    fn copy_alloc(&mut self, copy: &mut CopyState) -> *mut linkedlist::Node {
        if copy.to_free >= copy.to_end {
            let chunk = self.allocate_chunk();
            copy.add_to_chunk(chunk);
        }
        let node = copy.to_free;
        copy.to_free = unsafe { copy.to_free.add(1) };
        node
    }

    fn chunk_slot(
        chunks: &[*mut linkedlist::Node],
        node: *mut linkedlist::Node,
    ) -> Option<(usize, usize)> {
        let node_addr = node as usize;
        for (ci, &base) in chunks.iter().enumerate() {
            let base_addr = base as usize;
            let end = base_addr + NURSERY_SIZE * NODE_SIZE;
            if node_addr >= base_addr && node_addr < end {
                let slot = unsafe { node.offset_from(base) } as usize;
                return Some((ci, slot));
            }
        }
        None
    }
}

struct CopyState {
    from_chunks: Vec<*mut linkedlist::Node>,
    forwarded: Vec<Box<[u64]>>,
    to_chunks: Vec<*mut linkedlist::Node>,
    to_free: *mut linkedlist::Node,
    to_end: *mut linkedlist::Node,
}

impl CopyState {
    fn new(from_chunks: Vec<*mut linkedlist::Node>) -> Self {
        const WORDS: usize = NURSERY_SIZE / 64;
        let forwarded = (0..from_chunks.len())
            .map(|_| vec![0u64; WORDS].into_boxed_slice())
            .collect();
        CopyState {
            from_chunks,
            forwarded,
            to_chunks: Vec::new(),
            to_free: std::ptr::null_mut(),
            to_end: std::ptr::null_mut(),
        }
    }

    fn add_to_chunk(&mut self, chunk: *mut linkedlist::Node) {
        self.to_chunks.push(chunk);
        self.to_free = chunk;
        self.to_end = unsafe { chunk.add(NURSERY_SIZE) };
    }
}

static mut NURSERY: Nursery = Nursery::uninit();

/// Root set for `Nursery::collect`: raw `*mut Storage` whose pool heads,
/// queue tail, and port head enumerate interpreter-visible live nodes.
/// Registered by the mainloop via [`set_gc_roots`] after
/// `Storage::refresh_pools`. Zero until set, which disables collection (the
/// allocator then only bumps / grows).
static GC_ROOTS: AtomicUsize = AtomicUsize::new(0);

pub type NodeRootWalkHook = fn(&mut dyn FnMut(*mut *mut Node));

/// Optional JIT-frame root walker for moving `Node` pointers. The JIT
/// registers a function that visits every mutable jitframe slot holding a node
/// reference before copying collection swaps spaces.
pub static NODE_ROOT_WALK_HOOK: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
static COPYING_COLLECT_COUNT_FOR_TESTS: AtomicUsize = AtomicUsize::new(0);

/// Register the live `Storage` as the collection root set. Called once from
/// the mainloop after the storage pointers are refreshed.
pub fn set_gc_roots(storage: *mut Storage) {
    GC_ROOTS.store(storage as usize, Ordering::Relaxed);
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
    use std::sync::{Mutex, MutexGuard, OnceLock};

    struct NurseryTestGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl NurseryTestGuard {
        fn new() -> Self {
            static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
            let lock = LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            GC_ROOTS.store(0, Ordering::Relaxed);
            NODE_ROOT_WALK_HOOK.store(0, Ordering::Relaxed);
            COPYING_COLLECT_COUNT_FOR_TESTS.store(0, Ordering::Relaxed);
            unsafe {
                (*std::ptr::addr_of_mut!(NURSERY)).reset_for_test();
            }

            NurseryTestGuard { _lock: lock }
        }
    }

    impl Drop for NurseryTestGuard {
        fn drop(&mut self) {
            GC_ROOTS.store(0, Ordering::Relaxed);
            NODE_ROOT_WALK_HOOK.store(0, Ordering::Relaxed);
            unsafe {
                (*std::ptr::addr_of_mut!(NURSERY)).reset_for_test();
            }
        }
    }

    fn assert_chain_in_current_chunks(mut node: *mut Node, expected_len: usize, label: &str) {
        let chunks = unsafe { &(*std::ptr::addr_of!(NURSERY)).chunks };
        let mut seen = 0usize;
        while !node.is_null() {
            assert!(
                Nursery::chunk_slot(chunks, node).is_some(),
                "{label} node {seen} points outside current to-space chunks"
            );
            seen += 1;
            assert!(
                seen <= expected_len,
                "{label} chain is longer than expected or cyclic"
            );
            node = unsafe { (*node).next };
        }
        assert_eq!(seen, expected_len, "{label} chain length changed");
    }

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

    #[test]
    fn test_copying_collect_preserves_deep_stack_and_queue_chains() {
        let _guard = NurseryTestGuard::new();

        let mut storage = Storage::new();
        storage.refresh_pools();
        set_gc_roots(&mut storage as *mut Storage);

        let stack_idx = (0..STORAGE_COUNT)
            .find(|&idx| idx != VAL_QUEUE && idx != VAL_PORT)
            .unwrap();
        const STACK_VALUES: i32 = 300_000;
        const QUEUE_VALUES: i32 = 270_000;

        for value in 0..STACK_VALUES {
            storage.stack_mut(stack_idx).push(val_from_i32(value));
        }
        assert!(
            COPYING_COLLECT_COUNT_FOR_TESTS.load(Ordering::Relaxed) > 0,
            "stack pushes did not trigger copying collection"
        );
        assert_chain_in_current_chunks(
            storage.stack(stack_idx).head,
            STACK_VALUES as usize,
            "stack",
        );

        for value in 0..QUEUE_VALUES {
            storage.queue_mut().push(val_from_i32(value));
        }
        assert!(
            COPYING_COLLECT_COUNT_FOR_TESTS.load(Ordering::Relaxed) > 1,
            "queue pushes did not trigger copying collection"
        );
        assert_chain_in_current_chunks(storage.queue().head, QUEUE_VALUES as usize + 1, "queue");

        assert_eq!(storage.stack(stack_idx).__len__(), STACK_VALUES as usize);
        for expected in (0..STACK_VALUES).rev() {
            assert_eq!(
                val_to_i64(&storage.stack_mut(stack_idx).pop()),
                expected as i64
            );
        }
        assert_eq!(storage.stack(stack_idx).__len__(), 0);

        assert_eq!(storage.queue().__len__(), QUEUE_VALUES as usize);
        for expected in 0..QUEUE_VALUES {
            assert_eq!(val_to_i64(&storage.queue_mut().pop()), expected as i64);
        }
        assert_eq!(storage.queue().__len__(), 0);
        assert_chain_in_current_chunks(storage.queue().head, 1, "queue sentinel");
    }
}
