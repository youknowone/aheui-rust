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
#[cfg(feature = "jit")]
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

/// Chunk limit, overridable via `AHEUI_NURSERY_CHUNKS` (analog of
/// `PYPY_GC_NURSERY` sizing) to run programs that outgrow the 256MB default.
/// Read once; defaults to `MAX_NURSERY_CHUNKS`.
fn max_nursery_chunks() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("AHEUI_NURSERY_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(MAX_NURSERY_CHUNKS)
    })
}

/// Whether the `AHEUI_GC_LOG` diagnostics are on. Read once: the allocation
/// path consults this per node, and `std::env::var_os` scans the environment
/// on every call.
fn gc_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("AHEUI_GC_LOG").is_some())
}

/// Perform at most this many copying collections, then fall back to growing.
/// Bisecting it names the first collection after which a run diverges, which
/// separates "the collector corrupts state" from "collection timing exposes
/// something else". Unset = unlimited.
fn max_collects() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("AHEUI_GC_MAX_COLLECTS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(usize::MAX)
    })
}

/// When `AHEUI_GC_DISABLE` is set, the nursery grows monotonically instead of
/// running the moving collection (analog of `gc.disable()`); the bump allocator
/// never evacuates, so `Node` addresses stay stable for the whole run. Read
/// once; defaults to collection enabled.
fn gc_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var_os("AHEUI_GC_DISABLE").is_some())
}

struct Nursery {
    free: *mut linkedlist::Node,      // bump pointer — next slot to allocate
    end: *mut linkedlist::Node,       // one past the last slot in current chunk
    free_list: *mut linkedlist::Node, // singly-linked free list of popped nodes
    chunk_count: usize,               // number of allocated chunks (safety limit)
    chunks: Vec<*mut linkedlist::Node>, // base pointer of every allocated chunk
    spare_chunk: *mut linkedlist::Node, // to-space for the next copying collect
    quarantined: Vec<*mut linkedlist::Node>, // diagnostic: retained from-space
    collected: bool,                  // a copying collection has run at least once
    collect_count: usize,             // collections performed (see `max_collects`)
    /// `chunks` as `(start, end)` byte ranges, sorted by start, so membership
    /// is a binary search. `free_node` asks on every pop, and under
    /// `--no-jit` / `AHEUI_GC_DISABLE` nothing is ever retired so `chunks`
    /// grows to dozens — a linear scan there is a hot-path regression.
    chunk_ranges: Vec<(usize, usize)>,
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
            quarantined: Vec::new(),
            collected: false,
            collect_count: 0,
            chunk_ranges: Vec::new(),
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
            if !gc_disabled() {
                gc_log_check_chains("before");
                self.collect(&mut next);
                gc_log_check_chains("after");
            }
            if self.free_list.is_null() && self.free >= self.end {
                self.grow();
            }
        }
        self.take_node(value, next)
    }

    /// [`Self::alloc`] for a caller that cannot survive a collection: grow
    /// instead of evacuating.
    ///
    /// The metainterp's jitcode tracer runs `BC_NEW` for a `Node` while raw
    /// node pointers sit in its own register bank — the chain head an earlier
    /// `getfield` read, which the `setfield` after the `new` links up. That
    /// bank is in no root set, and unlike [`Self::alloc`] there is no `next`
    /// to pass as a keep root, so a moving collection here would leave the
    /// tracer holding from-space addresses.
    #[inline(always)]
    fn alloc_no_collect(
        &mut self,
        value: Val,
        next: *mut linkedlist::Node,
    ) -> *mut linkedlist::Node {
        if self.free_list.is_null() && self.free >= self.end {
            self.grow();
        }
        self.take_node(value, next)
    }

    /// Hand out a node from the free list, else from the bump region. The
    /// caller guarantees one of the two has room.
    #[inline(always)]
    fn take_node(&mut self, value: Val, next: *mut linkedlist::Node) -> *mut linkedlist::Node {
        if !self.free_list.is_null() {
            let node = self.free_list;
            unsafe {
                self.free_list = (*node).next;
                (*node).value = value;
                (*node).next = next;
            }
            self.debug_check_allocated(node, "free_list");
            return node;
        }
        let node = self.free;
        unsafe {
            (*node).value = value;
            (*node).next = next;
            self.free = node.add(1);
        }
        self.debug_check_allocated(node, "bump");
        node
    }

    /// Report an allocation that fell outside every live chunk. The collector
    /// only recognizes nodes inside `chunks`, so a node handed out from
    /// anywhere else is invisible to it: `forward_root` refuses to trace
    /// through it and the link past it survives a collection unforwarded.
    #[inline(always)]
    fn debug_check_allocated(&self, node: *mut linkedlist::Node, source: &str) {
        if !gc_log_enabled() {
            return;
        }
        if !self.owns(node) {
            let spare = Self::chunk_byte_offset(std::slice::from_ref(&self.spare_chunk), node);
            eprintln!(
                "@@@GC   ALLOC-OUTSIDE-CHUNKS source={source} node={node:?} spare={}",
                spare.is_some()
            );
        }
    }

    #[inline(always)]
    fn free_node(&mut self, node: *mut linkedlist::Node) {
        if node.is_null() {
            return;
        }
        // Only a node in the live chunk set may be recycled. A collection
        // evacuates every live node into to-space and retires from-space as
        // `spare_chunk`, but a caller can still be holding the pre-collection
        // address of a node it popped; freeing that puts a dead from-space
        // address on the free list, and the next `alloc` hands it back as a
        // live node. The collector then cannot see it — `forward_root`
        // range-checks against the chunk set — so the link past it survives
        // the next collection unforwarded, and if the address is in
        // `spare_chunk` the following collection allocates its copies right on
        // top of it. A node outside the chunk set is already dead, so dropping
        // it here loses nothing.
        //
        // The test is unconditional. Nodes outside the chunk set also reach
        // here before the first collection has run — the jitcode tracer's
        // `BC_NEW` allocates a `Node` with a plain host `alloc_zeroed`
        // (`pyjitpl/dispatch.rs` `BC_NEW`) rather than through the nursery —
        // so gating it on having collected lets those through. Membership is a
        // binary search over `chunk_ranges` because this runs on every pop,
        // and with `--no-jit` or `AHEUI_GC_DISABLE` nothing is ever retired,
        // so `chunks` grows to dozens.
        if !self.owns(node) {
            if gc_log_enabled() {
                static ONCE: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !ONCE.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "@@@GC   FREE-OUTSIDE-CHUNKS node={node:?} collected={}\n{}",
                        self.collected,
                        std::backtrace::Backtrace::force_capture()
                    );
                }
            }
            return;
        }
        unsafe {
            (*node).next = self.free_list;
        }
        self.free_list = node;
    }

    /// Re-derive [`Self::chunk_ranges`] after any change to `chunks`.
    fn rebuild_chunk_ranges(&mut self) {
        self.chunk_ranges.clear();
        self.chunk_ranges.extend(
            self.chunks
                .iter()
                .map(|&b| (b as usize, b as usize + NURSERY_SIZE * NODE_SIZE)),
        );
        self.chunk_ranges.sort_unstable();
    }

    /// Whether `node` lies in a chunk the collector currently owns.
    #[inline(always)]
    fn owns(&self, node: *mut linkedlist::Node) -> bool {
        let addr = node as usize;
        let at = self
            .chunk_ranges
            .partition_point(|&(start, _)| start <= addr);
        at > 0 && addr < self.chunk_ranges[at - 1].1
    }

    #[cold]
    fn grow(&mut self) {
        let base = self.allocate_chunk();
        self.chunks.push(base);
        self.rebuild_chunk_ranges();
        self.free = base;
        self.end = unsafe { base.add(NURSERY_SIZE) };
    }

    fn allocate_chunk(&mut self) -> *mut linkedlist::Node {
        let limit = max_nursery_chunks();
        self.chunk_count += 1;
        if self.chunk_count > limit {
            eprintln!(
                "[nursery] allocation limit reached ({} chunks × {}KB = {}MB)",
                limit,
                NURSERY_SIZE * std::mem::size_of::<linkedlist::Node>() / 1024,
                limit * NURSERY_SIZE * std::mem::size_of::<linkedlist::Node>() / (1024 * 1024),
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

        if self.collect_count >= max_collects() {
            return;
        }
        self.collect_count += 1;
        self.collected = true;
        let storage = unsafe { &mut *(storage_addr as *mut Storage) };
        let from_chunks = self.chunks.clone();
        let mut copy = CopyState::new(from_chunks);
        let gc_log = gc_log_enabled();
        let mut jit_slots = 0usize;
        let mut jit_node_slots = 0usize;
        if gc_log {
            static N: AtomicUsize = AtomicUsize::new(0);
            eprintln!(
                "@@@GC collect#{} chunks={}",
                N.fetch_add(1, Ordering::Relaxed),
                self.chunks.len()
            );
        }

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
                jit_slots += 1;
                if gc_log && !slot.is_null() {
                    let v = unsafe { *slot };
                    if let Some(off) = Self::chunk_byte_offset(&copy.from_chunks, v) {
                        jit_node_slots += 1;
                        // `Node` is headerless, so a slot is treated as a node
                        // purely because its address lands in a nursery chunk.
                        // Report the offset: a value that is not a multiple of
                        // `NODE_SIZE` is not a node start, and forwarding it
                        // would write a forwarding pointer into the middle of
                        // some other node.
                        eprintln!(
                            "@@@GC   jit_node_slot slot={slot:?} node={v:?} off={off} misalign={}",
                            off % NODE_SIZE
                        );
                    }
                }
                self.forward_root(&mut copy, slot);
            };
            hook(&mut visit);
        }
        if gc_log {
            eprintln!("@@@GC   jit_root_slots={jit_slots} of_which_nursery={jit_node_slots}");
        }

        self.trace_copied_nodes(&mut copy);

        let old_chunks = std::mem::replace(&mut self.chunks, copy.to_chunks);
        self.rebuild_chunk_ranges();
        self.free = copy.to_free;
        self.end = copy.to_end;
        self.free_list = std::ptr::null_mut();

        // Diagnostic: quarantine the N most recent from-space chunks instead of
        // releasing/reusing them, so a stale pointer missed by the root walk
        // reads its pre-copy contents rather than recycled memory. Separates
        // "stale pointer into recycled memory" from other evacuation faults.
        let quarantine = std::env::var("AHEUI_GC_QUARANTINE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if quarantine > 0 {
            self.spare_chunk = std::ptr::null_mut();
            // Optional: stamp every evacuated node's `value` with a sentinel so a
            // stale read of from-space is identifiable at its consumption point.
            // `next` keeps its forwarding pointer (the bitmap is per-collect, so
            // it is dead once this returns) to leave chain walks reachable.
            let poison = std::env::var_os("AHEUI_GC_POISON").is_some();
            for chunk in &old_chunks {
                if poison {
                    for i in 0..NURSERY_SIZE {
                        unsafe { (*chunk.add(i)).value = crate::value::val_from_i32(0x5EEDDEA) };
                    }
                }
            }
            for chunk in old_chunks {
                self.quarantined.push(chunk);
            }
            while self.quarantined.len() > quarantine {
                let old = self.quarantined.remove(0);
                self.release_chunk(old);
            }
            return;
        }
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

    /// Byte offset of `node` within whichever chunk contains it, or `None` if
    /// it lies outside every chunk. Unlike [`Self::chunk_slot`] this does not
    /// assume node alignment, so it can report a pointer into the middle of a
    /// node.
    fn chunk_byte_offset(
        chunks: &[*mut linkedlist::Node],
        node: *mut linkedlist::Node,
    ) -> Option<usize> {
        if node.is_null() {
            return None;
        }
        let node_addr = node as usize;
        for &base in chunks {
            let base_addr = base as usize;
            if node_addr >= base_addr && node_addr < base_addr + NURSERY_SIZE * NODE_SIZE {
                return Some(node_addr - base_addr);
            }
        }
        None
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

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
pub fn walk_bigint_root_values(visit: &mut dyn FnMut(&mut Val)) {
    fn walk_node_chain(mut node: *mut linkedlist::Node, visit: &mut dyn FnMut(&mut Val)) {
        while !node.is_null() {
            unsafe {
                visit(&mut (*node).value);
                node = (*node).next;
            }
        }
    }

    let storage_addr = GC_ROOTS.load(Ordering::Relaxed);
    if storage_addr != 0 {
        let storage = unsafe { &mut *(storage_addr as *mut Storage) };
        for i in 0..STORAGE_COUNT {
            if i != VAL_QUEUE && i != VAL_PORT {
                walk_node_chain(storage.stacks[i].head, visit);
            }
        }
        walk_node_chain(storage.queue.head, visit);
        walk_node_chain(storage.port.head, visit);
        visit(&mut storage.port.last_push);
    }

    // Bignum roots come only from Storage. The collection safepoint is the
    // bignum allocation itself (`alloc_bigint_oldgen`), reached at an aheui op
    // boundary where every live bignum is in a committed Storage node; the
    // just-computed result is not yet a GC object. Do NOT reuse the untyped JIT
    // shadow-stack node hook here: its slots can hold Stack*/Storage* pointers,
    // and blindly following `.next` from those addresses walks arbitrary memory
    // forever. `Nursery::collect` may use the hook because `forward_root`
    // range-checks each slot against node chunks before treating it as a Node.

    crate::value::walk_bigint_transient_roots(visit);
}

/// Convert every reachable mode-0 raw value into its mode-1 tagged form, once.
///
/// This is the dual-mode flip. It runs on the same root set and under the same
/// safepoint discipline as the bignum collection above: an aheui op boundary
/// where every live value sits in a committed node. Two things are outside
/// that set and are the caller's job:
///
/// * the operands of the op that overflowed, which it has already popped;
/// * the queue's sentinel node, whose value is walked here and so must have
///   been built raw in mode 0 — it is never read, but promoting a tagged word
///   a second time would double it.
///
/// Returns whether a storage was registered. `false` means nothing was
/// converted, and the caller must not treat the flip as done: with no storage
/// the walk reaches no value at all yet reports no error, so a mode switch on
/// top of it leaves every live raw word to be reread as a tagged one.
#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
#[must_use]
pub fn promote_all_root_values() -> bool {
    if GC_ROOTS.load(Ordering::Relaxed) == 0 {
        return false;
    }
    // Not merely an optimisation: a half-promoted storage cannot be walked by
    // the collector, because an unvisited raw even value is bit-identical to a
    // bigint pointer. See `value::with_no_collect`.
    crate::value::with_no_collect(|| {
        walk_bigint_root_values(&mut |v: &mut Val| crate::value::promote_raw_value(v));
    });
    true
}

/// Report any storage whose node chain disagrees with its recorded `size`,
/// bracketing a collection so a malformed chain can be attributed to the
/// collector or to the mutator that ran before it. Gated on `AHEUI_GC_LOG`
/// so it is available in a release build.
///
/// `rpython/memory/gc/base.py` runs the analogous `debug_check_consistency`
/// walk around each collection under `debug_gc`.
#[cold]
fn gc_log_check_chains(when: &str) {
    if !gc_log_enabled() {
        return;
    }
    let storage_addr = GC_ROOTS.load(Ordering::Relaxed);
    if storage_addr == 0 {
        return;
    }
    let storage = unsafe { &*(storage_addr as *const Storage) };
    let digest = storage.chain_digest();
    let violation = storage.check_chains();
    let violation = violation.as_deref().unwrap_or("ok");
    eprintln!("@@@GC chains {when} collect: {violation}");

    // Node addresses change on every collection but the values they carry
    // must not, so a storage whose digest moves across a collection has been
    // altered by the collector itself.
    thread_local! {
        static BEFORE: std::cell::RefCell<([u64; STORAGE_COUNT], Vec<Vec<u64>>)> =
            const { std::cell::RefCell::new(([0; STORAGE_COUNT], Vec::new())) };
    }
    if when == "before" {
        let values = (0..STORAGE_COUNT)
            .map(|i| storage.chain_values(i))
            .collect();
        BEFORE.with(|b| *b.borrow_mut() = (digest, values));
        return;
    }
    BEFORE.with(|b| {
        let (before_digest, before_values) = &*b.borrow();
        for (i, before) in before_digest.iter().enumerate() {
            if *before == digest[i] {
                continue;
            }
            let after = storage.chain_values(i);
            let before = &before_values[i];
            let at = (0..before.len().max(after.len()))
                .find(|&n| before.get(n) != after.get(n))
                .unwrap_or(0);
            let lo = at.saturating_sub(2);
            eprintln!(
                "@@@GC   CONTENT-CHANGED storage[{i}] len {} -> {} first differing index {at}",
                before.len(),
                after.len(),
            );
            eprintln!(
                "@@@GC     before[{lo}..]: {:x?}",
                &before[lo..(lo + 5).min(before.len())]
            );
            eprintln!(
                "@@@GC     after [{lo}..]: {:x?}",
                &after[lo..(lo + 5).min(after.len())]
            );
        }
    });
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

/// [`alloc_node_raw`] for a caller that must not collect — the nursery grows
/// instead of evacuating. Serves the metainterp's jitcode tracer, which holds
/// raw node pointers outside every root set while it runs `BC_NEW`.
#[inline(always)]
pub fn alloc_node_raw_no_collect() -> *mut linkedlist::Node {
    unsafe {
        let p = std::ptr::addr_of_mut!(NURSERY);
        (*p).alloc_no_collect(val_from_i32(0), std::ptr::null_mut())
    }
}

/// Serializes every test that touches the process-global `NURSERY`.
///
/// `Nursery::reset_for_test` frees every chunk, so a test holding this lock
/// deallocates nodes any concurrently running test still points at. Tests run
/// on parallel threads by default, so each one that allocates a `Node` — even
/// indirectly through `Stack` / `Queue` / `Port` / `Storage` — has to hold it.
#[cfg(test)]
pub(crate) fn nursery_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    /// `queue` / `port`. Must be called after `Storage` is moved, because
    /// the pointers are self-referencing.
    ///
    /// Every slot `aheui.py:40-49` fills with a non-`Stack` instance has to
    /// be aliased here, or `get_stack_ptr` hands the JIT a decoy
    /// `stacks[idx]` that no other path ever writes: `dispatch_mut` reaches
    /// the real object, so the two views of one storage index drift apart
    /// silently and `OP_SEL`'s `stacksize = len(selected)` reads 0.
    pub fn refresh_pools(&mut self) {
        for i in 0..STORAGE_COUNT {
            self.pools[i] = &mut self.stacks[i] as *mut Stack;
        }
        // VAL_QUEUE / VAL_PORT: alias &mut queue / &mut port (head/size share
        // `#[repr(C)]` layout with Stack).
        self.pools[VAL_QUEUE] = &mut self.queue as *mut Queue as *mut Stack;
        self.pools[VAL_PORT] = &mut self.port as *mut Port as *mut Stack;
    }

    /// Walk every storage's node chain and compare its length with the
    /// recorded `size`, returning a description of the first storage that
    /// disagrees.
    ///
    /// `linkedlist.py:98-110` keeps the queue exactly one node longer than
    /// `size`: `__init__` seeds a dummy `tail` node and `push` appends a fresh
    /// one, so `size` counts real elements and the chain ends at `tail`.
    /// Stacks and the port hold exactly `size` nodes.
    ///
    /// The walk is bounded by the expected length, so a chain that loops back
    /// on itself is reported rather than followed forever.
    pub fn check_chains(&self) -> Option<String> {
        for i in 0..STORAGE_COUNT {
            let list = self.dispatch(i);
            let size = list.size();
            let expected = if i == VAL_QUEUE { size + 1 } else { size };
            let mut node = list.head();
            let mut walked = 0usize;
            let mut last = std::ptr::null_mut();
            while !node.is_null() {
                walked += 1;
                if walked > expected {
                    return Some(format!(
                        "storage[{i}] chain longer than size (size={size} expected={expected})"
                    ));
                }
                last = node;
                node = unsafe { (*node).next };
            }
            if walked != expected {
                return Some(format!(
                    "storage[{i}] chain length {walked} != expected {expected} (size={size})"
                ));
            }
            if i == VAL_QUEUE && last != self.queue.tail {
                return Some(format!(
                    "queue chain ends at {:?}, but tail is {:?} (size={size})",
                    last, self.queue.tail
                ));
            }
        }
        None
    }

    /// Order-sensitive digest of every storage's contents: the index, the
    /// recorded size, and each node's raw `value` word, in chain order.
    ///
    /// Node addresses change on every collection but the values they carry
    /// must not, so bracketing a collection with this digest asks whether the
    /// collector is observably the identity on storage. A bigint `Val` is a
    /// pointer into the majit heap, which the node collector does not move, so
    /// hashing the raw word stays stable for those too.
    pub fn chain_digest(&self) -> [u64; STORAGE_COUNT] {
        std::array::from_fn(|i| {
            let list = self.dispatch(i);
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            let mut mix = |word: u64| {
                hash ^= word;
                hash = hash.wrapping_mul(0x1000_0000_01b3);
            };
            mix(list.size() as u64);
            let mut node = list.head();
            // Digest only the live prefix. Nodes past `size` are stale tail
            // left over from a pop whose `head` store the JIT has not yet
            // committed; the collector is free to reuse them, so including
            // them would report a change that is not observable.
            let mut budget = if i == VAL_QUEUE {
                list.size() + 1
            } else {
                list.size()
            };
            while !node.is_null() && budget > 0 {
                mix(unsafe { *(node as *const u64) });
                node = unsafe { (*node).next };
                budget -= 1;
            }
            hash
        })
    }

    /// The live prefix of one storage's values, in chain order — the same
    /// range [`Self::chain_digest`] hashes, so a digest mismatch can be
    /// resolved to the exact index that changed.
    pub fn chain_values(&self, idx: usize) -> Vec<u64> {
        let list = self.dispatch(idx);
        let mut budget = list.size() + usize::from(idx == VAL_QUEUE);
        let mut node = list.head();
        let mut out = Vec::with_capacity(budget);
        while !node.is_null() && budget > 0 {
            out.push(unsafe { *(node as *const u64) });
            node = unsafe { (*node).next };
            budget -= 1;
        }
        out
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
                stack.size = count as u32;
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
    use std::sync::MutexGuard;

    struct NurseryTestGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl NurseryTestGuard {
        fn new() -> Self {
            let lock = super::nursery_test_lock();

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
        let _lock = super::nursery_test_lock();
        let storage = Storage::new();
        assert_eq!(storage.stacks.len(), STORAGE_COUNT);
        assert_eq!(storage.stack(0).__len__(), 0);
        assert_eq!(storage.queue().__len__(), 0);
        assert_eq!(storage.port().__len__(), 0);
    }

    #[test]
    fn test_storage_queue_dispatch() {
        // Queue polymorphic dispatch goes through LinkedList trait.
        let _lock = super::nursery_test_lock();
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

    /// The dual-mode flip has to reach every live value, and one of them —
    /// `port.last_push` — lives outside every node chain. Asserted through the
    /// walk itself, so a holder added to `Storage` later without being added
    /// to the walk fails here.
    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    #[test]
    fn test_promote_all_root_values_reaches_every_holder() {
        use crate::value::val_from_raw_i64;

        let _guard = NurseryTestGuard::new();

        let mut storage = Storage::new();
        storage.refresh_pools();
        set_gc_roots(&mut storage as *mut Storage);

        let stack_idx = (0..STORAGE_COUNT)
            .find(|&idx| idx != VAL_QUEUE && idx != VAL_PORT)
            .unwrap();
        // Past the small range, so the heap-bigint arm of the promotion is
        // exercised and not only the retag.
        const BIG: i64 = (1i64 << 62) + 7;

        storage.dispatch_mut(stack_idx).push(val_from_raw_i64(-5));
        storage.dispatch_mut(stack_idx).push(val_from_raw_i64(BIG));
        storage.dispatch_mut(VAL_QUEUE).push(val_from_raw_i64(11));
        storage.dispatch_mut(VAL_PORT).push(val_from_raw_i64(13));
        // After the port push, which writes `last_push` itself.
        storage.port.last_push = val_from_raw_i64(17);

        promote_all_root_values();

        let mut seen = Vec::new();
        walk_bigint_root_values(&mut |v: &mut Val| seen.push(v.try_to_i64()));
        for want in [BIG, -5, 11, 13, 17] {
            assert!(
                seen.contains(&Some(want)),
                "promotion did not reach {want}; walked {seen:?}"
            );
        }
    }

    /// The flip allocates — every value past ±2^62 boxes — and
    /// `alloc_bigint_oldgen` collects *before* each allocation. A collection
    /// there walks a storage whose unvisited values are still raw words, and a
    /// raw even number is bit-identical to a bigint pointer, so it would be
    /// followed. Pinned with a hook that mimics that allocator.
    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    #[test]
    fn test_promote_all_root_values_suppresses_collection() {
        use crate::value::{AheuiBigInt, val_from_raw_i64};

        static COLLECTS: AtomicUsize = AtomicUsize::new(0);

        fn counting_collect() {
            COLLECTS.fetch_add(1, Ordering::Relaxed);
        }

        // The shape of `alloc_bigint_oldgen`: collect, then allocate.
        fn collecting_alloc(v: AheuiBigInt) -> *mut AheuiBigInt {
            crate::value::maybe_collect_bigints();
            Box::into_raw(Box::new(v))
        }

        let _guard = NurseryTestGuard::new();
        COLLECTS.store(0, Ordering::Relaxed);
        crate::value::register_bigint_maybe_collect_hook(counting_collect);
        crate::value::register_bigint_alloc_hook(collecting_alloc);

        // Control first: a zero below has to mean "suppressed", not "hook was
        // never wired" — those two produce the same count.
        crate::value::maybe_collect_bigints();
        assert_eq!(
            COLLECTS.load(Ordering::Relaxed),
            1,
            "collect hook is not wired, so the count below would prove nothing"
        );

        let mut storage = Storage::new();
        storage.refresh_pools();
        set_gc_roots(&mut storage as *mut Storage);
        let stack_idx = (0..STORAGE_COUNT)
            .find(|&idx| idx != VAL_QUEUE && idx != VAL_PORT)
            .unwrap();
        // Past ±2^62, so promoting it must box and therefore must allocate.
        storage
            .dispatch_mut(stack_idx)
            .push(val_from_raw_i64((1i64 << 62) + 7));

        let before = COLLECTS.load(Ordering::Relaxed);
        promote_all_root_values();
        let fired = COLLECTS.load(Ordering::Relaxed) - before;

        crate::value::bigint::clear_bigint_hooks_for_test();
        assert_eq!(fired, 0, "collection fired while the storage was half promoted");
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
