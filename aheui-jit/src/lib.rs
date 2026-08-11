// JIT-enabled Aheui interpreter — graph pipeline + #[jit_interp] macro.
//
// RPython parity: rpaheui/aheui/aheui.py
//   greens = [pc, stackok, is_queue, program]
//   reds   = [stacksize, storage, selected]
//   storage = linked list stacks (no virtualizable arrays)
//
// `bm` is a fifth green with no counterpart upstream. It carries the dual-mode
// encoding — pyre runs values as raw machine words until one overflows — and
// being a green is what keeps the mode out of compiled code: each trace is
// keyed on one encoding, so the arithmetic arms below pick their helper once,
// at record time, instead of testing a global per operation. The flip changes
// the key, which retires the mode-0 traces and records mode-1 ones.
//
// stackok is a green (rpaheui parity): specialising the trace on it lets
// `jit_effective_stacksize_delta(op, stackok)` fold to a constant, so the
// per-op stacksize update carries no residual call. The green-key-explosion
// concern (each stackok flip generating a separate compiled loop) does not
// reproduce on logo.aheui — the distinct (pc, stackok) merge points that
// occur are few, so logo compiles a single bounded loop.

extern crate majit_ir;
extern crate majit_metainterp as majit_meta;
/// The metainterp crate, re-exported so the binary can reach `mc_diag_summary`
/// and the [`majit_meta::JitStats`] fields without its own dependency edge.
pub use majit_metainterp;

use majit_meta::jit::promote;

pub use aheui_runtime;
pub use aheui_runtime::aheui;
pub use aheui_runtime::io;
pub use aheui_runtime::storage;
pub use aheui_runtime::value;

pub mod jit;

/// Default JIT threshold. RPython default = 1039.
pub const JIT_THRESHOLD: u32 = 1039;

/// JIT threshold honoring the `MAJIT_THRESHOLD` env override (for testing
/// small hot loops without the production 1039-iteration warmup). RPython
/// exposes the threshold as a configurable jitdriver param (`warmspot.py`);
/// this is the same knob, read once at startup.
pub fn jit_threshold() -> u32 {
    std::env::var("MAJIT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(JIT_THRESHOLD)
}

/// The JIT trace budget, chosen by measurement.
///
/// rpaheui does not set one: `option.py:186` defaults `--trace-limit` /
/// `RPAHEUI_TRACE_LIMIT` to `-1`, and `aheui.py:360` calls
/// `jit.set_param(driver, 'trace_limit', ...)` only when it is `>= 0`, so the
/// RPython default is what runs. Its `jit-summary` on logo with no override is
/// identical to `RPAHEUI_TRACE_LIMIT=6000` (1 loop, 6 bridges, 2 too-long and 5
/// segmenting aborts, 37839 recorded ops) and differs from 30000 (1 loop, 1
/// bridge, no aborts, 25713 ops) — so the limit in force there is 6000, the
/// same default majit carries in `trace_ctx.rs` `DEFAULT_TRACE_LIMIT`.
///
/// 70000 keeps logo's whole-program trace — 33177 recorded ops — under the
/// budget, so it compiles one loop with no abort. That is not the fastest
/// setting. logo, min of 9 interleaved runs, CPU time, via the
/// `MAJIT_TRACE_LIMIT` override:
///
/// | limit  | CPU     | bridges | aborts | guard failures |
/// |--------|---------|---------|--------|----------------|
/// |   6000 | 0.430 s |       7 |      8 |           1601 |
/// |  10000 | 0.410 s |       5 |      6 |           1201 |
/// |  20000 | 0.410 s |       2 |      2 |            401 |
/// |  30000 | 0.390 s |       2 |      2 |            401 |
/// |  70000 | 0.470 s |       1 |      0 |            201 |
///
/// The single 33177-op loop costs 172ms to optimize plus 26ms to assemble,
/// which the run does not earn back; at 30000 that drops to 42ms plus 10ms and
/// the whole corpus stays byte-identical (75 programs; the 3 that differ are
/// non-terminating and match on any fixed output prefix). Lowering it is left
/// for after the remaining trace-quality work, since the trade shows up in the
/// `jitstats` gate as logo's `loops_aborted` 0 -> 2 and `guard_failures`
/// 201 -> 401.
pub const TRACE_LIMIT: u32 = 70000;

/// [`TRACE_LIMIT`] honoring a `MAJIT_TRACE_LIMIT` override. `trace_limit` is a
/// configurable jitdriver param upstream too (`warmspot.py`); this is the same
/// knob, read once at startup, so the budget can be swept without a rebuild.
pub fn trace_limit() -> u32 {
    std::env::var("MAJIT_TRACE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TRACE_LIMIT)
}

/// The last [`mainloop`] run's cumulative JIT counters.
///
/// `mainloop` owns its `JitDriver` for the length of the run and drops it on
/// return, and the binary `process::exit`s on the value `mainloop` returns —
/// so a caller that wants the counters has no live driver to ask. `mainloop`
/// publishes a snapshot here just before it returns, and
/// [`last_jit_stats`] reads it back.
static LAST_JIT_STATS: std::sync::Mutex<Option<majit_meta::JitStats>> = std::sync::Mutex::new(None);

/// The `Counters.ABORT_*` breakdown behind [`last_jit_stats`]'s
/// `loops_aborted`.
///
/// `JitStats` carries only the total, and the profiler's own
/// `print_stats` is behind `MAJIT_LOG` — which on a workload like
/// `pi.jinseo` (35s, 814 aborts) is unusably slow, so the counts it holds were
/// unreachable in practice. They are the statistic that says *why* a trace was
/// given up, so the snapshot rides along with the totals.
static LAST_ABORT_REASONS: std::sync::Mutex<Option<majit_meta::jitprof::JitProfilerSnapshot>> =
    std::sync::Mutex::new(None);

/// The counters the most recent [`mainloop`] finished with, or `None` if the
/// JIT interpreter has not run in this process.
pub fn last_jit_stats() -> Option<majit_meta::JitStats> {
    LAST_JIT_STATS.lock().unwrap().clone()
}

/// The profiler snapshot the most recent [`mainloop`] finished with.
pub fn last_abort_reasons() -> Option<majit_meta::jitprof::JitProfilerSnapshot> {
    LAST_ABORT_REASONS.lock().unwrap().clone()
}

fn publish_jit_stats(
    stats: majit_meta::JitStats,
    profiler: majit_meta::jitprof::JitProfilerSnapshot,
) {
    *LAST_JIT_STATS.lock().unwrap() = Some(stats);
    *LAST_ABORT_REASONS.lock().unwrap() = Some(profiler);
}

#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
mod bigint_gc {
    use aheui_runtime::value::bigint::AheuiBigInt;
    use majit_gc::GcAllocator;
    use majit_gc::collector::MiniMarkGC;
    use majit_gc::trace::TypeInfo;
    use majit_ir::GcRef;
    use std::cell::Cell;
    use std::sync::Once;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    const BIGINT_PAYLOAD_SIZE: usize = std::mem::size_of::<AheuiBigInt>();
    const BIGINT_COLLECT_THRESHOLD: usize = 8 * 1024 * 1024;

    static BIGINT_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);
    static BIGINT_BYTES_SINCE_COLLECT: AtomicUsize = AtomicUsize::new(0);
    static GC_GLOBAL_INIT: Once = Once::new();
    static BIGINT_HOOKS_INIT: Once = Once::new();

    thread_local! {
        static GC_THREAD_REGISTERED: Cell<bool> = const { Cell::new(false) };
    }

    pub fn init() {
        GC_GLOBAL_INIT.call_once(|| {
            let tid = if majit_gc::gc_sync::is_initialized() {
                // `gc_op` hands the closure a concrete `&mut MiniMarkGC`; the
                // unsizing to `&mut dyn GcAllocator` happens at the call.
                majit_gc::gc_sync::gc_op(|gc| register_bigint_type(gc))
            } else {
                let mut gc = MiniMarkGC::new();
                let tid = register_bigint_type(&mut gc);
                majit_gc::gc_sync::store_singleton(Box::new(gc));
                tid
            };
            BIGINT_GC_TYPE_ID.store(tid, Ordering::Release);
            majit_gc::shadow_stack::register_extra_root_walker(walk_aheui_bigint_roots);
        });

        GC_THREAD_REGISTERED.with(|registered| {
            if !registered.get() {
                majit_gc::gc_sync::register_thread();
                majit_gc::shadow_stack::register_mutator();
                registered.set(true);
            }
        });

        BIGINT_HOOKS_INIT.call_once(|| {
            aheui_runtime::value::register_bigint_alloc_hook(alloc_bigint_oldgen);
            aheui_runtime::value::register_bigint_maybe_collect_hook(maybe_collect_bigints);
        });
    }

    fn register_bigint_type(gc: &mut dyn GcAllocator) -> u32 {
        gc.register_type(TypeInfo::with_destructor(
            BIGINT_PAYLOAD_SIZE,
            bigint_destructor,
        ))
    }

    fn alloc_bigint_oldgen(value: AheuiBigInt) -> *mut AheuiBigInt {
        let tid = BIGINT_GC_TYPE_ID.load(Ordering::Acquire);
        if tid == u32::MAX {
            return Box::into_raw(Box::new(value));
        }

        // Collect BEFORE allocating the new bignum: `value` is a Rust-owned
        // AheuiBigInt (not yet a GC object) so it cannot be swept, and the
        // just-consumed operands are reclaimable. Bignum allocation is the GC
        // safepoint for compiled node-virt code — the interpreter push/pop
        // hooks and the loop merge point are bypassed once the trace is
        // compiled, so collection would otherwise never fire under --jit.
        maybe_collect_bigints();
        let external = bigint_external_bytes(&value);
        let raw = majit_gc::gc_sync::gc_op(|gc| gc.alloc_oldgen_typed(tid, BIGINT_PAYLOAD_SIZE));
        if raw.is_null() {
            return Box::into_raw(Box::new(value));
        }

        unsafe {
            std::ptr::write(raw.0 as *mut AheuiBigInt, value);
        }
        // The limb `Vec` lives outside the GC heap, so it does not enter the
        // collector's own major-collection threshold; `BIGINT_BYTES_SINCE_COLLECT`
        // is what accounts for it and drives `maybe_collect_bigints`.
        BIGINT_BYTES_SINCE_COLLECT.fetch_add(BIGINT_PAYLOAD_SIZE + external, Ordering::Relaxed);
        raw.0 as *mut AheuiBigInt
    }

    fn maybe_collect_bigints() {
        // `alloc_bigint_oldgen` reaches this directly rather than through
        // `value::maybe_collect_bigints`, so the suppression has to be read
        // here too — the dual-mode flip allocates while the storage is half
        // promoted, and a collection would read its unvisited raw words as
        // bigint pointers.
        if aheui_runtime::value::no_collect_active() {
            return;
        }
        if BIGINT_BYTES_SINCE_COLLECT.load(Ordering::Relaxed) < BIGINT_COLLECT_THRESHOLD {
            return;
        }
        if BIGINT_BYTES_SINCE_COLLECT
            .compare_exchange(
                BIGINT_BYTES_SINCE_COLLECT.load(Ordering::Relaxed),
                0,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            majit_gc::gc_sync::gc_op(|gc| gc.collect_oldgen_nonmoving());
        }
    }

    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    fn walk_aheui_bigint_roots(visit: &mut dyn FnMut(&mut GcRef)) {
        aheui_runtime::storage::walk_bigint_root_values(&mut |value| {
            if let Some(addr) = aheui_runtime::value::val_bigint_addr(value) {
                let mut root = GcRef(addr);
                visit(&mut root);
                if root.0 != addr {
                    // Today's collect_oldgen_nonmoving leaves these old-gen
                    // payloads in place. Write back anyway so a future moving
                    // visitor can forward the live Val instead of a temporary.
                    aheui_runtime::value::val_set_bigint_addr(value, root.0);
                }
            }
        });
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bigint_root_walker_writes_forwarded_address_back_to_val() {
            let mut value = aheui_runtime::value::val_from_str("9223372036854775808")
                .expect("value must parse as a heap bigint");
            let original = aheui_runtime::value::val_bigint_addr(&value)
                .expect("value must use the heap-bigint representation");
            let forwarded = 0x1000;
            assert_ne!(forwarded, original);
            assert_eq!(forwarded & 1, 0);
            assert_ne!(forwarded, 0);

            aheui_runtime::value::with_bigint_transient_root(&mut value, || {
                walk_aheui_bigint_roots(&mut |root| {
                    if root.0 == original {
                        *root = GcRef(forwarded);
                    }
                });
            });

            assert_eq!(
                aheui_runtime::value::val_bigint_addr(&value),
                Some(forwarded)
            );
        }
    }

    unsafe fn bigint_destructor(addr: usize) {
        unsafe { std::ptr::drop_in_place(addr as *mut AheuiBigInt) }
    }

    fn bigint_external_bytes(value: &AheuiBigInt) -> usize {
        let bits = value.bits();
        if bits <= 64 {
            0
        } else {
            ((bits + 63) / 64) as usize * 8
        }
    }
}

#[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
mod bigint_gc {
    pub fn init() {}
}

pub fn init_gc_subsystem() {
    bigint_gc::init();
}

include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

// ── Imports ──

use aheui_runtime::aheui::*;
use aheui_runtime::io as aheui_io;
use aheui_runtime::storage::linkedlist_jit as lj;
use aheui_runtime::storage::{LinkedList, Storage};
use ahsembler::compiler::Program;

use aheui_runtime::value::*;

// ── Diagnostic env-var helpers (cached once, safe for hot loops) ──

fn spdiag_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_SPDIAG").is_some())
}

fn check_chains_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("AHEUI_CHECK_CHAINS").is_some())
}

fn bh_debug_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_BH_DEBUG").is_some())
}

/// GC allocator for JIT-compiled New() ops.
/// Delegates to the global nursery so alloc/free share the same pool
/// as the interpreter path.
///
/// Node-sized allocations (`size <= NODE_SIZE`) go to the nursery, which
/// self-bounds via its own chunk cap (`grow()` exits at 64 chunks). The
/// cumulative 256 MB limit applies only to the oversized `alloc_zeroed`
/// fallback, which has no other bound — a running program allocates far
/// more than 256 MB of *nodes* over its lifetime (each freed and reused),
/// so a cumulative cap on the node path would spuriously fail.
const JIT_ALLOC_LIMIT: usize = 256 * 1024 * 1024;

struct AheuiBlackholeAllocator;

#[inline(always)]
fn raw_store(struct_ptr: i64, offset: usize, value: i64) {
    unsafe {
        *((struct_ptr as usize + offset) as *mut i64) = value;
    }
}

impl majit_metainterp::resume::BlackholeAllocator for AheuiBlackholeAllocator {
    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        let sd = typedescr
            .as_size_descr()
            .expect("aheui bh_new: not a SizeDescr");
        let size = sd.size();
        if size <= aheui_runtime::storage::NODE_SIZE {
            aheui_runtime::storage::alloc_node_raw() as i64
        } else {
            let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
            unsafe { std::alloc::alloc_zeroed(layout) as i64 }
        }
    }

    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        raw_store(struct_ptr, descr_info.offset, value);
    }

    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        raw_store(struct_ptr, descr_info.offset, value);
    }

    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        raw_store(struct_ptr, descr_info.offset, value);
    }
}

struct NurseryGcAllocator {
    oversized_allocated: usize,
}

impl NurseryGcAllocator {
    fn new() -> Self {
        Self {
            oversized_allocated: 0,
        }
    }
}

impl majit_gc::GcAllocator for NurseryGcAllocator {
    fn alloc_nursery(&mut self, size: usize) -> majit_ir::GcRef {
        if size <= aheui_runtime::storage::NODE_SIZE {
            let node = aheui_runtime::storage::alloc_node_raw();
            majit_ir::GcRef(node as usize)
        } else {
            self.oversized_allocated += size;
            if self.oversized_allocated > JIT_ALLOC_LIMIT {
                // Return NULL to signal allocation failure — compiled code
                // will hit a guard and fall back to the interpreter.
                return majit_ir::GcRef::NULL;
            }
            let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            majit_ir::GcRef(ptr as usize)
        }
    }
    fn alloc_nursery_headerless(&mut self, size: usize) -> majit_ir::GcRef {
        // Headerless-aware: aheui's node nursery is collected by its own
        // copying node GC and handles raw 16B nodes without MiniMark headers.
        self.alloc_nursery(size)
    }
    /// Serves the metainterp's jitcode tracer, which runs `BC_NEW` for a
    /// `Node` while holding raw node pointers in its own register bank. That
    /// bank is in no root set, so the nursery has to grow rather than evacuate.
    fn alloc_nursery_headerless_no_collect(&mut self, size: usize) -> majit_ir::GcRef {
        if size <= aheui_runtime::storage::NODE_SIZE {
            let node = aheui_runtime::storage::alloc_node_raw_no_collect();
            majit_ir::GcRef(node as usize)
        } else {
            // Oversized is a plain `alloc_zeroed`, which does not collect
            // either. It is also outside the node nursery and therefore
            // invisible to the copying collector, which is only sound while
            // every headerless struct fits a `Node`; a larger one would have to
            // grow the nursery's own allocator instead.
            self.alloc_nursery(size)
        }
    }
    fn alloc_nursery_no_collect(&mut self, size: usize) -> majit_ir::GcRef {
        self.alloc_nursery(size)
    }
    fn alloc_varsize(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> majit_ir::GcRef {
        self.alloc_nursery(base_size + item_size * length)
    }
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> majit_ir::GcRef {
        self.alloc_varsize(base_size, item_size, length)
    }
    fn write_barrier(&mut self, _obj: majit_ir::GcRef) {}
    fn jit_remember_young_pointer_from_array(&mut self, _obj: majit_ir::GcRef) {}
    fn remember_young_pointer_from_array2(
        &mut self,
        _obj: majit_ir::GcRef,
        _index: usize,
        _card_page_shift: u32,
    ) {
    }
    fn collect_nursery(&mut self) {}
    fn collect_full(&mut self) {}
    fn nursery_free(&self) -> *mut u8 {
        std::ptr::null_mut()
    }
    fn nursery_top(&self) -> *const u8 {
        std::ptr::null()
    }
    /// nursery_free_addr / nursery_top_addr expose the bump-pointer slot
    /// addresses to the JIT-emitted inline allocator so a compiled
    /// alloc-fast-path can bump `free` toward `end` inline and cond-call the
    /// slowpath only on nursery exhaustion.
    fn nursery_free_addr(&self) -> usize {
        aheui_runtime::storage::nursery_bump_addrs().0
    }
    fn nursery_top_addr(&self) -> usize {
        aheui_runtime::storage::nursery_bump_addrs().1
    }
    fn max_nursery_object_size(&self) -> usize {
        usize::MAX
    }
}

fn register_aheui_copying_gc_jit_roots() {
    majit_gc::shadow_stack::register_libc_jitframe_tracer(
        majit_backend::jitframe::jitframe_custom_trace,
    );
    let hook: aheui_runtime::storage::NodeRootWalkHook = walk_aheui_jit_node_roots;
    aheui_runtime::storage::NODE_ROOT_WALK_HOOK
        .store(hook as usize, std::sync::atomic::Ordering::Relaxed);
}

fn walk_aheui_jit_node_roots(
    visit_node_slot: &mut dyn FnMut(*mut *mut aheui_runtime::storage::linkedlist::Node),
) {
    let mut visit_gcref_slot = |slot: *mut majit_ir::GcRef| {
        visit_node_slot(slot as *mut *mut aheui_runtime::storage::linkedlist::Node);
    };

    majit_gc::shadow_stack::walk_roots(|gcref| {
        visit_gcref_slot(gcref as *mut majit_ir::GcRef);
    });

    majit_gc::shadow_stack::walk_jf_roots(|gcref| {
        visit_gcref_slot(gcref as *mut majit_ir::GcRef);
        if !gcref.is_null() && majit_gc::shadow_stack::is_libc_jitframe(gcref.0) {
            majit_gc::shadow_stack::trace_libc_jitframe(gcref.0, &mut visit_gcref_slot);
        }
    });

    majit_gc::shadow_stack::walk_bh_regs(|gcref| {
        visit_gcref_slot(gcref as *mut majit_ir::GcRef);
    });

    majit_gc::shadow_stack::walk_resume_ref_roots(|gcref| {
        visit_gcref_slot(gcref as *mut majit_ir::GcRef);
    });

    // This hook enumerates only shadow-stack node-ref slots; majit extra-roots hold aheui bignums, walked separately by MiniMarkGC.
}

/// Trace-time state for the Aheui JIT.
///
/// rpaheui/aheui/aheui.py:228-234 stores the reds as:
///   stacksize = 0
///   storage   = Storage()
///   selected  = storage[0]         # object reference
///
/// Rust adaptations — kept narrow and co-located here so reviewers can
/// see exactly which deviations are required by the borrow checker /
/// JIT raw-pointer ABI:
///
/// * `selected: usize` — RPython captures the polymorphic storage object
///   directly. Rust cannot hold a mutable borrow into `self.storage`
///   across subsequent `self.storage` mutations, so the index is kept
///   and the object is re-fetched via [`Storage::dispatch_mut`] on every
///   use. This is semantically identical to rpaheui's reference form.
///
/// * `selected_ref: GcRef` — The JIT backend reads `head` / `size` at
///   fixed byte offsets through a raw pointer. This mirrors rpaheui's
///   `selected` object reference at the machine level: `GcRef` is
///   literally the raw pointer to the selected `Stack` (Queue/Port
///   share the `head`/`size` prefix via `#[repr(C)]`). We keep it next
///   to `selected: usize` because `refresh_selected_ref` has to run any
///   time `selected` changes; treating them as a single logical field
///   avoided plumbing a dedicated getter through the `#[jit_interp]`
///   macro.
static SPDIAG_TRACE_OPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

thread_local! {
    // Latest pre-walk storage dump (updated while output_bytes is in the
    // trace-start window) so `recover` can print the S_k baseline to compare
    // against S_{k+1} — disambiguates "stale = state-field red" vs "stale =
    // folded heap write".
    static SK_SNAPSHOT: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
}

struct AheuiState {
    storage: Storage,
    selected: usize,
    stacksize: i64,
    /// `&mut state.storage.pools[selected]` packed as `usize`. Tracked as
    /// `ref(Stack)` in `state_fields` so it is carried in the ref register
    /// bank as a genuine `InputArgRef`: promoted with `ref_guard_value` and
    /// passed to monomorphic storage helpers (`stack_*` / `queue_*`) as a
    /// ref-kind arg. The `usize` carrier round-trips the raw pointer bits.
    selected_ref: usize,
    /// `&mut state.storage as *mut Storage` packed as `usize` — the base of
    /// the contiguous `pools: [*mut Stack; N]` array (offset 0 of `Storage`).
    /// Tracked as `ref(Storage)` and declared a `pool_arrays` base so the
    /// OP_SEL `selected_ref = pools[selected]` read lowers to a re-producible
    /// `getarrayitem_gc_r` on this base instead of an opaque residual call;
    /// the loaded stack ref then re-derives from `selected` each loop entry
    /// rather than being carried as an independent, divergence-prone red.
    /// The sole carrier of the storage pointer: `aheui.py:27` carries
    /// `storage` as one red, so the residual storage helpers take this same
    /// ref-kind field rather than an aliased `int` copy (an int/ref pair for
    /// one value makes the resume box kind of every `pools[N]` stack ref
    /// ambiguous, seeding Ref-typed loop-header slots with Int boxes).
    storage_ref: usize,
}

impl AheuiState {
    #[inline(always)]
    fn refresh_selected_ref(&mut self) {
        // rpaheui/aheui/aheui.py:233,282: selected = storage[idx].
        // Point directly at `Stack` in the flat stacks array so the JIT
        // can read head (offset 0) and size (offset 8) without going
        // through the indirection helper.
        self.selected_ref = self.storage.get_stack_ptr(self.selected) as usize;
    }

    /// rpaheui: selected.push/pop/add — polymorphic dispatch (Stack/Queue/Port).
    fn selected_dispatch_mut(&mut self) -> &mut dyn LinkedList {
        self.storage.dispatch_mut(self.selected)
    }

    fn selected_dispatch(&self) -> &dyn LinkedList {
        self.storage.dispatch(self.selected)
    }

    fn spdiag_dump_stacks(&self) -> String {
        // The queue and the port are two of the three chains the collector
        // forwards explicitly, so a dump that skips them cannot show whether a
        // root moved. Print every storage, and cap the walk high enough that a
        // one-node discrepancy is visible rather than truncated away.
        let limit = std::env::var("MAJIT_SPDIAG_NODES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(64);
        let mut dump = String::new();
        for i in 0..STORAGE_COUNT {
            let mut p = self.storage.dispatch(i).head() as *const u8;
            let mut vals: Vec<i64> = Vec::new();
            while !p.is_null() && vals.len() < limit {
                let raw = unsafe { *(p as *const i64) };
                // bigint Val: small = (v<<1)|1; smallint Val: raw == v.
                vals.push(if raw & 1 != 0 { raw >> 1 } else { raw });
                p = unsafe { *(p.add(8) as *const *const u8) };
            }
            dump.push_str(&format!(
                " stack[{i}](size={}){vals:?}",
                self.storage.len_at(i)
            ));
        }
        dump.push_str(&format!(
            " queue.tail={:?} port.head={:?}",
            self.storage.queue.tail, self.storage.port.head,
        ));
        dump
    }

    /// Every storage's `size` and its `head` chain as raw addresses, so a
    /// chain/size mismatch can be read as "which node is extra/missing"
    /// rather than just a count. `spdiag_dump_stacks` prints values only.
    fn dump_chain_addrs(&self) -> String {
        let mut dump = String::new();
        for i in 0..STORAGE_COUNT {
            let list = self.storage.dispatch(i);
            let mut node = list.head();
            let mut addrs: Vec<String> = Vec::new();
            while !node.is_null() && addrs.len() < 8 {
                addrs.push(format!("{node:?}"));
                node = unsafe { (*node).next };
            }
            dump.push_str(&format!(
                " storage[{i}]@{:#x}(size={}){addrs:?}\n",
                list as *const _ as *const u8 as usize,
                list.size()
            ));
        }
        dump
    }

    fn refresh_state_from_storage(&mut self) {
        if spdiag_enabled() {
            eprintln!(
                "@@@SPDIAG recover output_bytes={} selected={} old_stacksize={} new_stacksize={}{}",
                aheui_io::output_total_bytes(),
                self.selected,
                self.stacksize,
                self.storage.len_at(self.selected),
                self.spdiag_dump_stacks(),
            );
            SK_SNAPSHOT.with(|s| eprintln!("@@@SPDIAG S_k-snapshot{}", s.borrow()));
            SPDIAG_TRACE_OPS.store(300, std::sync::atomic::Ordering::Relaxed);
        }
        self.storage_ref = &mut self.storage as *mut Storage as usize;
        self.refresh_selected_ref();
        // Recover does not receive the green pc/program, so it cannot name
        // only the current OP_MOV operand. Check every storage; this
        // includes any OP_MOV target distinct from `selected`, and the
        // queue and port, whose chains the collector forwards as roots of
        // their own.
        //
        // A release build runs the same walk under `AHEUI_CHECK_CHAINS`: the
        // chains that break under the compiled path hold thousands of nodes,
        // which `spdiag_dump_stacks` truncates away, and a debug build is too
        // slow to reach the failure.
        if cfg!(debug_assertions) || check_chains_enabled() {
            if let Some(err) = self.storage.check_chains() {
                panic!("check_chains: {err}\n{}", self.dump_chain_addrs());
            }
        }
        self.stacksize = self.storage.len_at(self.selected) as i64;
    }
}

fn find_used_storages(_program: &Program, _header_pc: usize, initial: usize) -> Vec<usize> {
    let mut storages: Vec<usize> = (0..STORAGE_COUNT).filter(|&idx| idx != VAL_PORT).collect();
    if initial != VAL_PORT && !storages.contains(&initial) {
        storages.push(initial);
        storages.sort_unstable();
    }
    storages
}

thread_local! {
    /// W1 diag: raw `*const Storage` set by the mainloop before each
    /// `jit_merge_point!`, read by the walk's output shims so they can dump
    /// the walk's per-emission shared-storage state (the walk runs inside the
    /// JitCodeMachine, not the mainloop, so the resume-op log can't see it).
    static WALK_STORAGE_PTR: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn dump_storage_ptr(ptr: usize) -> String {
    if ptr == 0 {
        return String::new();
    }
    let storage = unsafe { &*(ptr as *const Storage) };
    let mut dump = String::new();
    for i in 0..STORAGE_COUNT {
        if storage.len_at(i) == 0 {
            continue;
        }
        let mut p = storage.dispatch(i).head() as *const u8;
        let mut vals: Vec<i64> = Vec::new();
        while !p.is_null() && vals.len() < 8 {
            let raw = unsafe { *(p as *const i64) };
            vals.push(if raw & 1 != 0 { raw >> 1 } else { raw });
            p = unsafe { *(p.add(8) as *const *const u8) };
        }
        dump.push_str(&format!(" stack[{i}]={vals:?}"));
    }
    dump
}

/// io-shim targets for `aheui_io::output_write_*(&r)`: the recorded call
/// carries the raw `Val` word (tagged smallint or boxed-bigint pointer),
/// so reconstruct the `Val` and route through the interpreter's own
/// decode + output buffer. Writing the raw word through majit's
/// `jit_write_number_i64` printed tagged payloads (289 for 144) into a
/// second buffer that interleaved wrongly with interpreter output.
extern "C" fn jit_write_number(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    if bh_debug_enabled() {
        eprintln!("[io-debug] jit_write_number raw={value}");
    }
    if spdiag_enabled() {
        let out = aheui_io::output_total_bytes();
        if (1000..=3000).contains(&out) {
            eprintln!(
                "@@@WALKEMIT num out={out} val={value}{}",
                dump_storage_ptr(WALK_STORAGE_PTR.with(|c| c.get()))
            );
        }
    }
    aheui_io::output_write_number(&v);
}

extern "C" fn jit_write_utf8(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    if spdiag_enabled() {
        let out = aheui_io::output_total_bytes();
        if (1000..=3000).contains(&out) {
            eprintln!(
                "@@@WALKEMIT utf8 out={out} val={value}{}",
                dump_storage_ptr(WALK_STORAGE_PTR.with(|c| c.get()))
            );
        }
    }
    aheui_io::output_write_utf8(&v);
}

// ── Input I/O shims for JIT tracing ──
//
// RPython parity: I/O input (os.read) is a residual call — the JIT traces
// through it as CallI, producing a new INT variable. We use a thread-local
// InputBuffer so the extern "C" shim can access it from compiled code.

use std::cell::RefCell;

thread_local! {
    static JIT_INPUT_BUFFER: RefCell<aheui_io::InputBuffer> = RefCell::new(aheui_io::InputBuffer::new());
}

extern "C" fn jit_read_utf8() -> i64 {
    JIT_INPUT_BUFFER.with(|cell| cell.borrow_mut().read_utf8())
}

extern "C" fn jit_read_number() -> i64 {
    JIT_INPUT_BUFFER.with(|cell| cell.borrow_mut().read_number())
}

fn jit_output_flush() {
    aheui_io::output_flush();
}

/// Convert raw i64 to tagged Val. extern "C" for JIT ABI compatibility.
/// Val is #[repr(transparent)] i64, so returning Val is ABI-safe.
/// Registered as elidable_int — pure function, result feeds into push.
extern "C" fn jit_tag_val(raw: i64) -> Val {
    val_from_i32(raw as i32)
}

/// [`jit_tag_val`]'s mode-0 twin: the word is the value, so this is identity.
///
/// Unlike its sibling it keeps the full `i64`. The truncation there predates
/// dual mode and only bites on an input number wider than 32 bits; mode 0 has
/// no reason to inherit it.
#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
extern "C" fn jit_tag_val_raw(raw: i64) -> Val {
    aheui_runtime::value::val_from_raw_i64(raw)
}

#[inline(always)]
#[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
fn jit_retag_small(untagged: i64) -> Val {
    aheui_runtime::value::val_retag_small(untagged)
}

/// The dual-mode encoding as a jitcode value: 1 while values are tagged, 0
/// while they are raw machine words.
///
/// Elidable although it reads a mutable global, because it is a green: the mode
/// is part of the merge-point key, so it is constant for the whole life of any
/// one trace, and the flip retires that trace by changing the key rather than
/// by invalidating a value inside it.
#[inline(always)]
fn jit_bigint_mode() -> i64 {
    aheui_runtime::value::bigint_mode() as i64
}

// ── Node alloc/free + val comparison — local JIT wrappers ──
//
// Local wrappers for functions in aheui_runtime whose `__majit_call_policy_*`
// probes the macro generates in the LOCAL scope.  Calling `lj::alloc_node_jit`
// directly would look for the probe in the `lj` module, which doesn't exist.

/// JIT-visible mirror of `aheui_runtime::storage::linkedlist::Node`.
/// Used as a struct literal in the `#[jit_interp]` mainloop so the lowerer
/// emits `New + SetfieldGc` ops that `OptVirtualize` can fold away when the
/// Node doesn't escape the current iteration. The `struct_allocs` config
/// rewrites the literal back to `jit_alloc_node(value, next)` on the
/// concrete (non-tracing) path.
#[repr(C)]
struct NodeJit {
    value: Val,
    next: usize,
}

#[inline(always)]
fn jit_alloc_node(value: Val, next: usize) -> usize {
    aheui_runtime::storage::linkedlist_jit::alloc_node_jit(value, next)
}

#[inline(always)]
fn jit_free_node(node: usize) {
    aheui_runtime::storage::linkedlist_jit::free_node_jit(node)
}

/// `val_ge` returning Val (tagged 0 or 1) for use with `_put_value`.
#[inline(always)]
fn jit_val_ge(a: Val, b: Val) -> Val {
    aheui_runtime::storage::linkedlist_jit::val_ge_jit(a, b)
}

/// Pipeline jitcode resolver for `inline_pipeline_*` call policies.
/// The `#[jit_interp]` macro's dispatch JitCode builder calls this to
/// resolve a function name (e.g. `"val_add"`) to the pipeline-built
/// sub-jitcode that the tracer will inline-call into.
#[allow(non_snake_case)]
fn __majit_pipeline_jitcode(name: &str) -> std::sync::Arc<majit_metainterp::JitCode> {
    jit::jitcode_runtime::pipeline_jitcode_by_name(name)
        .unwrap_or_else(|| panic!("pipeline jitcode for '{name}' not found"))
}

// ── OP_BRZ pop+is_zero shims ──
//
// rpaheui/aheui/aheui.py:299-301: `top = selected.pop(); jump =
// bigint.is_zero(top)`. The polymorphic `selected.pop()` plus the
// `bigint.is_zero` call are silent-skipped by the lowerer, so the
// JIT-compiled trace previously had no exit guard for OP_BRZ —
// loop.aheui infinite-looped in compiled mode.
//
// These shims bundle pop + is_zero into a single residual call that
// returns 1 (zero, take the branch) or 0 (non-zero, fall through).
// Registered as `residual_int` because pop has the side effect of
// shrinking the storage; trace records `CallI(jit_pop_is_zero_stack,
// selected_ref)` followed by an `IntNe(result, 0) + GuardFalse` pair
// for the branch decision.

extern "C" fn jit_pop_is_zero_stack(stack_ref: usize) -> i64 {
    let v = unsafe { (*(stack_ref as *mut aheui_runtime::storage::linkedlist::Stack)).pop() };
    if val_is_zero(&v) { 1 } else { 0 }
}

extern "C" fn jit_pop_is_zero_queue(queue_ref: usize) -> i64 {
    let v = unsafe { (*(queue_ref as *mut aheui_runtime::storage::linkedlist::Queue)).pop() };
    if val_is_zero(&v) { 1 } else { 0 }
}

/// Unified OP_BRZ: pop + is_zero for any selected type.
/// Returns 1 if zero (take branch), 0 otherwise.
extern "C" fn jit_pop_is_zero(pool_ptr: usize, selected: usize) -> i64 {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    let v = storage.dispatch_mut(selected).pop();
    if val_is_zero(&v) { 1 } else { 0 }
}

/// OP_MOV helper: push the popped value onto the move target
/// (aheui.py:277-279 `storage[val].push(r)`). The target index is a
/// per-opcode operand, not the promoted `selected`, so the polymorphic
/// `dispatch_mut(target)` is bundled into one residual call the same
/// way `jit_pop_is_zero` bundles pop + is_zero.
extern "C" fn jit_storage_push(pool_ptr: usize, target: usize, value: Val) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).push(value);
}

// Index-keyed dynamic storage dispatch — rpaheui parity for
// `selected.METHOD()` (aheui.py:260-389). `selected` is a RED (never
// promoted), so each storage op is one residual call that dispatches on
// the live `selected` index at run time (Stack / Queue / Port) instead of
// a JIT-green 3-way `is_port`/`is_queue` branch + `selected_ref`. This
// keeps the recorded trace structurally identical to rpaheui's single
// polymorphic call site, so the optimiser never emits a contradictory
// `guard_value(selected)` and the loop closes through the real back-edge.
extern "C" fn jit_storage_pop(pool_ptr: usize, target: usize) -> Val {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).pop()
}
extern "C" fn jit_storage_add(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).add();
}
extern "C" fn jit_storage_sub(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).sub();
}
extern "C" fn jit_storage_mul(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).mul();
}
extern "C" fn jit_storage_div(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).div();
}
extern "C" fn jit_storage_mod(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).modulo();
}
extern "C" fn jit_storage_dup(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).dup();
}
extern "C" fn jit_storage_swap(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).swap();
}
extern "C" fn jit_storage_cmp(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).cmp();
}

/// OP_SEL helper: return the raw pointer to the selected linked-list as
/// the new `selected_ref`. `storage[idx]` (aheui.py:280-284) is a list
/// getitem returning the Stack/Queue/Port object reference; the result is
/// carried in the ref register bank, so the call is recorded ref-returning
/// (`residual_ref_cannot_raise_wrapped`). `#[dont_look_inside_cannot_raise]`
/// emits the `__majit_call_policy_*` trace/concrete targets the wrapped-ref
/// lowering reads; the `usize` carrier round-trips the pointer bits.
#[majit_macros::dont_look_inside_cannot_raise]
fn jit_sel_get_ref(pool_ptr: usize, selected: usize) -> usize {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.get_stack_ptr(selected) as usize
}

fn jit_stacksize_delta(op: usize) -> i64 {
    (-OP_STACKDEL[op] + OP_STACKADD[op]) as i64
}

fn jit_op_gated_on_stackok(op: usize) -> bool {
    OP_STACKDEL[op] > 0 && op != OP_BRZ as usize
}

fn jit_effective_stacksize_delta(op: usize, stackok: i64) -> i64 {
    if stackok != 0 || !jit_op_gated_on_stackok(op) {
        jit_stacksize_delta(op)
    } else {
        0
    }
}

// Guard failure resume: handled by the RPython-standard JIT framework.
// can_enter_jit! / jit_merge_point! flow through JitDriver.back_edge_structured
// and JitDriver.merge_point, which restore state via JitState::restore.

// ── JIT mainloop ──
//
// RPython parity: rpaheui/aheui/aheui.py mainloop()
// - storage = linked list stacks (no compact arrays, no virtualizable arrays)
// - selected = red variable (promoted within trace)
// - push/pop = Node allocation/deallocation (OptVirtualize target)

#[majit_macros::jit_interp(
    state = AheuiState,
    env = Program,
    // RPython parity: rpaheui/aheui/aheui.py:30 reds=['stacksize','storage','selected'].
    // Storage is the polymorphic 28-slot pool that cannot be flattened
    // as ints — declared `opaque(Storage)` so the macro carries it on
    // the state struct without enumerating any inputarg/fail_arg/Sym slot
    // for it. Pool/selected raw-pointer handles are also opaque (single
    // GcRef word each); polymorphic dispatch into Stack/Queue/Port goes
    // through `selected_dispatch_mut()` and is handled at the codewriter
    // layer by Step 4b's handle_regular_indirect_call (-live- +
    // ref_guard_value + residual call).
    state_fields = {
        storage: opaque(aheui_runtime::storage::Storage),
        // RPython parity: AheuiState.selected is `usize` (slot index into
        // 28-slot pool); stacksize is `i64` (signed pop/push delta) — rpaheui
        // carries it as a full machine-word Python int, and `i64` matches the
        // IR's native Int width so no per-op `as i64` sign-extend is emitted
        // on every `stackok`/delta read. The macro carries them as Int in IR;
        // `int(<Type>)` keeps the user's natural Rust storage type and inserts
        // `as i64` / `as <Type>` casts at the JIT boundary (identity for i64).
        selected: int(usize),
        stacksize: int(i64),
        // The selected list's object reference (aheui.py:256
        // `selected = jit.promote(selected)`). Carried in the ref register
        // bank as a genuine `InputArgRef` so it can be promoted with
        // `ref_guard_value` and passed to the monomorphic `stack_*` /
        // `queue_*` helpers as a ref-kind arg (`JitCallArg::reference`). The
        // `usize` carrier round-trips the raw pointer bits; `Stack` is a
        // nominal tag (Queue/Port share the head/size prefix via repr(C)).
        selected_ref: ref(aheui_runtime::storage::linkedlist::Stack),
        // Base of the `pools: [*mut Stack; N]` array (offset 0 of `Storage`).
        // Declared `ref(Storage)` + listed in `pool_arrays` so OP_SEL's
        // `selected_ref = jit_sel_get_ref(storage_ref, selected)` lowers to a
        // re-producible `getarrayitem_gc_r` on this base. aheui.py:27 carries
        // `storage` as ONE red, so this ref is also the storage argument of the
        // residual `jit_storage_*` helpers: an aliased `int` copy of the same
        // pointer would leave the resume box kind of every `pools[N]` stack ref
        // ambiguous, seeding Ref-typed loop-header slots with Int boxes and
        // making the bridge unmatchable (VirtualStatesCantMatch).
        storage_ref: ref(aheui_runtime::storage::Storage),
    },
    // `storage_ref` is a raw-pointer-array base: the registered getter
    // `jit_sel_get_ref(state.storage_ref, state.selected)` indexes
    // `pools[selected]` and lowers to getarrayitem_gc_r instead of the opaque
    // residual call below.  Keyed on the getter identity so only this call —
    // not any other helper sharing the `(state.storage_ref, int)` shape — is
    // recognized as a pool read.
    pool_arrays = { storage_ref => jit_sel_get_ref -> aheui_runtime::storage::linkedlist::Stack },
    // Struct field type declarations for ref-kind field access.
    // Tells the lowerer to emit getfield_gc_r / setfield_gc_r (ref-kind)
    // instead of _gc_i (int-kind) when accessing these fields through a
    // ref(T) state scalar or a local ref binding.
    ref_fields = {
        aheui_runtime::storage::linkedlist::Stack::head => aheui_runtime::storage::linkedlist::Node,
        aheui_runtime::storage::linkedlist::Queue::head => aheui_runtime::storage::linkedlist::Node,
        aheui_runtime::storage::linkedlist::Port::head => aheui_runtime::storage::linkedlist::Node,
        aheui_runtime::storage::linkedlist::Node::next => aheui_runtime::storage::linkedlist::Node,
        NodeJit::next => aheui_runtime::storage::linkedlist::Node,
    },
    // The element count is `u32`, so the field descr is sub-word and
    // `intbounds` can bound a load of it. Without a bound the depth's `+ 1`
    // may overflow, the sum goes rangeless, and the `stackok` check has to be
    // guarded again at every opcode instead of following from the last one.
    // All three lists share the `head`/`size` prefix and the JIT reaches them
    // through the `Stack` tag, so every name is declared.
    int_fields = {
        aheui_runtime::storage::linkedlist::Stack::size => u32,
        aheui_runtime::storage::linkedlist::Queue::size => u32,
        aheui_runtime::storage::linkedlist::Port::size => u32,
    },
    struct_allocs = { NodeJit => jit_alloc_node },
    headerless_structs = { NodeJit },
    io_shims = {
        aheui_io::output_write_number => jit_write_number,
        aheui_io::output_write_utf8 => jit_write_utf8,
    },
    calls = {
        jit_read_utf8 => residual_int,
        jit_read_number => residual_int,
        jit_output_flush => residual_void_cannot_raise,
        jit_tag_val => elidable_int_cannot_raise,
        jit_tag_val_raw => elidable_int_cannot_raise,
        jit_retag_small => elidable_int_cannot_raise,
        jit_bigint_mode => elidable_int_cannot_raise,
        // First MethodCall RHS consumer; lowered via `lower_method_call_value`.
        Program::get_req_size => elidable_int_cannot_raise,
        Program::get_op => elidable_int_cannot_raise,
        Program::get_label => elidable_int_cannot_raise,
        Program::get_operand => elidable_int_cannot_raise,
        // Phase D-1 monomorphic storage helpers. The hot Stack ops and the
        // cold queue_pop/queue_swap are `#[jit_inline]` (inline_int /
        // inline_void), so the lowerer splices their field-level body into
        // the trace; the remaining ops stay residual — a concrete
        // `call_void_args` / `call_int_args` — rather than silent-skipping
        // the storage op.
        //
        // The registered path segments must match the call site verbatim
        // (the macro compares segment-by-segment); use the `lj::*` alias
        // here since the mainloop arms call `lj::stack_push(...)` etc.
        lj::stack_push => inline_void,
        lj::stack_pop => inline_int,
        lj::stack_add => inline_void,
        lj::stack_sub => inline_void,
        lj::stack_mul => inline_void,
        lj::stack_div => residual_void,
        lj::stack_mod => residual_void,
        lj::stack_dup => inline_void,
        lj::stack_swap => inline_void,
        lj::stack_cmp => inline_void,
        lj::queue_push => inline_void,
        lj::queue_pop => inline_int,
        lj::queue_add => inline_void,
        lj::queue_sub => inline_void,
        lj::queue_mul => inline_void,
        lj::queue_div => residual_void,
        lj::queue_mod => residual_void,
        lj::queue_dup => inline_void,
        lj::queue_swap => inline_void,
        lj::queue_cmp => inline_void,
        // Mode-0 twins of the arithmetic helpers, selected on the `bm` green.
        // All inline, including div and mod, which are residual above: their
        // mode-0 form carries a guard that only survives inside the trace.
        lj::stack_add_raw => inline_void,
        lj::stack_sub_raw => inline_void,
        lj::stack_mul_raw => inline_void,
        lj::stack_div_raw => inline_void,
        lj::stack_mod_raw => inline_void,
        lj::stack_cmp_raw => inline_void,
        lj::queue_add_raw => inline_void,
        lj::queue_sub_raw => inline_void,
        lj::queue_mul_raw => inline_void,
        lj::queue_div_raw => inline_void,
        lj::queue_mod_raw => inline_void,
        lj::queue_cmp_raw => inline_void,
        jit_pop_is_zero_stack => residual_int,
        jit_pop_is_zero_queue => residual_int,
        jit_pop_is_zero => residual_int,
        jit_storage_push => residual_void,
        jit_storage_pop => residual_int,
        jit_storage_add => residual_void,
        jit_storage_sub => residual_void,
        jit_storage_mul => residual_void,
        jit_storage_div => residual_void,
        jit_storage_mod => residual_void,
        jit_storage_dup => residual_void,
        jit_storage_swap => residual_void,
        jit_storage_cmp => residual_void,
        // `storage[idx]` returns the selected list's object reference; the
        // result lands in the ref register bank. Elidable + cannot-raise:
        // `pools` is an immutable array (`_immutable_fields_ = ['pools']`)
        // of stable `Stack` pointers, so `pools[selected]` is a pure,
        // exception-free function of (pool_ptr, selected). Elidable lowers
        // to CALL_PURE, which the optimizer can re-emit in the short
        // preamble — required because the stacksize is now a getfield on
        // `selected_ref.size`, and the short preamble can only re-produce
        // that getfield if its base (`selected_ref`) is itself re-producible.
        // A residual call result is not re-emittable, so the length-getfield
        // loop would fail to close (InvalidLoop). Elidable calls are exempt
        // from the observer replay queue (CALL_PURE is not recorded), so this
        // also keeps the observer/concrete walks in lockstep.
        jit_sel_get_ref => elidable_ref_cannot_raise_wrapped,
        jit_stacksize_delta => elidable_int_cannot_raise,
        jit_effective_stacksize_delta => elidable_int_cannot_raise,
        // Phase D-2: field-level IR for Stack ops — free and val arithmetic
        // registered so the lowerer emits concrete IR ops instead of
        // silent-skipping unregistered calls. Node allocation now goes through
        // `struct_allocs = { NodeJit => jit_alloc_node }`: concrete execution
        // calls `jit_alloc_node`, while tracing emits headerless `New(NodeJit)`
        // plus field stores.
        // jit_free_node is `concrete_only_void` — the free runs on the
        // concrete path only; the JIT trace omits it. The GNE after the
        // call is a store-scheduling fence for the preceding
        // setfield_gc_r(selected_ref.head) lazy set.
        jit_free_node => concrete_only_void,
        val_add => elidable_int,
        val_sub => elidable_int,
        val_mul => elidable_int,
        val_div => elidable_int,
        val_mod => elidable_int,
        jit_val_ge => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
    },
    // Residual storage mutators that change `Stack.size` or its `head` chain
    // pointer.  Traced push/pop methods emit in-trace `setfield_gc` stores,
    // which invalidate the matching heapcache entries directly.
    // The inline_int / inline_void helpers (stack_push/pop/add/sub/mul/dup/
    // swap/cmp and queue_pop/swap) splice that `self.size` setfield into the
    // trace directly, so they reproduce the barrier and need no entry here.
    // The remaining ops lower to opaque residual calls, which carry an empty
    // write-set by default.  A size-only declaration caused the pi #29 spin;
    // omitting `head` caused the fib #32 SIGSEGV when a cached pre-pop node was
    // reused after the residual call.  Declaring both fields restores the
    // invalidation performed by traced setfield barriers.  The lists are
    // conservative supersets because an extra reload is harmless while a
    // missing invalidation is not.
    residual_writes = {
        selected_ref.size => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        // `Storage::pools` type-puns its `*mut Stack` slots over Stack,
        // Queue, and Port.  All three share size@8, but the JIT's field
        // descr identity includes the nominal struct type.  Register every
        // compatible layout so a residual mutation invalidates whichever
        // getfield descriptor the specialized trace cached.
        selected_ref.size @ aheui_runtime::storage::linkedlist::Queue => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        selected_ref.size @ aheui_runtime::storage::linkedlist::Port => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        selected_ref.head => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        // See the size aliases above.  Missing one of these nominal descrs
        // lets a cached pre-call head survive an opaque Queue/Port mutation.
        selected_ref.head @ aheui_runtime::storage::linkedlist::Queue => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        selected_ref.head @ aheui_runtime::storage::linkedlist::Port => [
            lj::stack_div, lj::stack_mod,
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
        // `tail` exists only on Queue (the dummy-tail sentinel append target).
        // An opaque residual push/arith that appends at the tail must invalidate
        // a cached `queue.tail`, else a following inlined queue op reads the
        // stale sentinel and appends off the live chain, orphaning nodes
        // (chainlen < size + 1) until a later pop dereferences a null head.
        selected_ref.tail @ aheui_runtime::storage::linkedlist::Queue => [
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_pop, jit_storage_add, jit_storage_sub,
            jit_storage_mul, jit_storage_div, jit_storage_mod, jit_storage_dup,
            jit_storage_swap, jit_storage_cmp,
            jit_pop_is_zero_stack, jit_pop_is_zero_queue, jit_pop_is_zero,
        ],
    },
    // rpaheui/aheui/aheui.py:29: greens=['pc','stackok','is_queue','program'].
    //
    // A.3.7 closes the literal-parity gap that Phase D-3 reverted. The
    // earlier revert was forced by the macro's `lower_value_expr` not
    // registering greens as lowerable bindings; A.3.6.1
    // (jitcode_lower.rs:5605 `bind_pre_merge_point_stmts`) walks
    // pre-merge-point body-local `let` stmts and binds them, so a
    // synthesised `let is_queue = state.selected == 21usize;` flows
    // through `resolve_greens` / `emit_promote_greens` without panic.
    //
    // The dispatch arms keep their `state.selected == 21usize` /
    // `state.selected == 27usize` 3-way structure — pyre's discriminator
    // is finer-grained than rpaheui's 2-way `is_queue` (Port is split
    // out from the stack family), and within a trace specialised by
    // `guard_value(selected)` the comparison folds to a constant and
    // only the live branch reaches the optimised IR.
    // `bm` is pyre's own fifth green — see the module header. It binds through
    // the same pre-merge-point walker as `is_queue`.
    greens = [pc, stackok, is_queue, bm, program],
    recover = refresh_state_from_storage,
    switch_dispatch = true,
    native_tag_small = { jit_retag_small },
)]
pub fn mainloop(program: &Program, threshold: u32) -> Val {
    init_gc_subsystem();

    // Dual mode: run on raw machine words until one operation overflows one.
    // Ahead of `Storage::new()` below, because the queue builds a sentinel
    // value and the conversion walks it — it has to be written in the mode it
    // will be read in.
    aheui_runtime::value::start_in_raw_mode();

    let mut driver: majit_meta::JitDriver<AheuiState> = majit_meta::JitDriver::new(threshold);

    // Register the nursery-backed GC allocator for JIT-compiled New() ops.
    // Linked-list nodes (Node) share the interpreter's nursery pool, so
    // compiled `New` allocation must route through this allocator rather
    // than `libc::malloc`; the copying collector only knows how to forward
    // nodes allocated from nursery chunks.
    let gc = Box::new(NurseryGcAllocator::new());
    driver.meta_interp_mut().backend_mut().set_gc_allocator(gc);
    driver.meta_interp_mut().backend_mut().set_new_via_gc(true);
    // resume.py:1367 — materializes virtual headerless NodeJit nodes on guard-fail deopt; otherwise NullAllocator leaves a NULL head and size/chain desync SIGSEGV.
    driver.register_blackhole_allocator(AheuiBlackholeAllocator);

    // rpaheui/aheui/aheui.py:325 `jit.set_param(driver, 'trace_limit', 30000)`
    // — see [`TRACE_LIMIT`] for the scaling.
    driver.set_param("trace_limit", trace_limit() as i64);

    let mut pc: usize = 0;
    // rpaheui/aheui/aheui.py:30: reds=['stacksize','storage','selected']
    let mut state = AheuiState {
        storage: Storage::new(),
        selected: 0,
        stacksize: 0,
        selected_ref: 0,
        storage_ref: 0,
    };
    // Storage was moved into state — refresh self-referencing pointers.
    state.storage.refresh_pools();
    state.storage_ref = &mut state.storage as *mut Storage as usize;
    state.refresh_selected_ref();
    // Register the storage as the nursery collector's root set. `state` is a
    // stationary local for the rest of the mainloop, so the pointer stays
    // valid; the compiled traces omit `jit_free_node`, so leaked nodes are
    // reclaimed by `Nursery::collect` walking these roots.
    aheui_runtime::storage::set_gc_roots(&mut state.storage as *mut Storage);
    register_aheui_copying_gc_jit_roots();

    // RPython `warmspot.py:281-289` `make_jitcodes() →
    // finish_setup(codewriter)` parity for state-field JIT: register the
    // canonical `(live_i, live_r, live_f)` liveness slots and seed
    // `MetaInterpStaticData` so blackhole resume's
    // `BlackholeInterpreter::get_current_position_info` (which reads
    // `code[pc] == op_live`) recognises the macro-emitted `BC_LIVE`
    // markers. Without this hook `op_live` defaults to `-1` (= u8::MAX
    // post-conversion in `setup_cached_control_opcodes`) and the
    // resume path panics with `missing liveness[N] in JitCode`.
    {
        let meta = <AheuiState as majit_meta::JitState>::build_meta(&state, 0, program);
        meta.install_canonical_liveness(&mut driver);
    }

    if let Ok(range) = std::env::var("MAJIT_PROGDUMP") {
        if let Some((s, e)) = range.split_once(':') {
            let (s, e): (usize, usize) = (s.parse().unwrap_or(0), e.parse().unwrap_or(0));
            for p in s..e.min(program.size) {
                let op = program.get_op(p);
                let name = ahsembler::consts::OP_NAMES
                    .get(op as usize)
                    .copied()
                    .flatten()
                    .unwrap_or("?");
                eprintln!(
                    "@@@PROG pc={p} op={op}({name}) label={} operand={} req={}",
                    program.get_label(p),
                    program.get_operand(p),
                    program.get_req_size(p),
                );
            }
        }
    }
    while pc < program.size {
        // Handoff diagnostic (MAJIT_HANDOFF): log the pre-op state at the top
        // of every loop iteration — one line per opcode in both the naive
        // (MAJIT_THRESHOLD huge) and JIT paths, since both hit the loop top
        // once per opcode. Diff the two to find the FIRST divergent opcode at
        // the native-interp→walk seam. Windowed + count-capped so it cannot
        // flood.
        {
            // Env reads cached once (a per-iteration `env::var` makes the
            // naive-threshold oracle run ~100x slower than the interp).
            static HENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            if *HENABLED.get_or_init(|| std::env::var_os("MAJIT_HANDOFF").is_some()) {
                let out = aheui_io::output_total_bytes();
                // Latch on the first opcode at/after MAJIT_HANDOFF_AT (default
                // 2085 output bytes), then log the next MAJIT_HANDOFF_CAP (default
                // 1500) opcodes UNCONDITIONALLY — robust to `out` advancing
                // unevenly (heavy compute phases span 100s of opcodes per byte).
                static HLATCH: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                static HCOUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                static HAT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                static HCAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
                let at: u64 = *HAT.get_or_init(|| {
                    std::env::var("MAJIT_HANDOFF_AT")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(2085)
                });
                let cap: u32 = *HCAP.get_or_init(|| {
                    std::env::var("MAJIT_HANDOFF_CAP")
                        .ok()
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(1500)
                });
                if !HLATCH.load(std::sync::atomic::Ordering::Relaxed) && out >= at {
                    HLATCH.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                if HLATCH.load(std::sync::atomic::Ordering::Relaxed) {
                    let n = HCOUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < cap {
                        let op0 = program.get_op(pc);
                        eprintln!(
                            "@@@HANDOFF#{n} pc={pc} op={op0} out={out} ss={} sel={}{}",
                            state.stacksize,
                            state.selected,
                            state.spdiag_dump_stacks(),
                        );
                    }
                }
            }
        }
        // rpaheui/aheui/aheui.py:252
        let mut stackok = program.get_req_size(pc) as i64 <= state.stacksize;
        // rpaheui/aheui/aheui.py:284 sets `is_queue = (value == VAL_QUEUE)`
        // inside OP_SEL; pyre recomputes it pre-merge-point from the
        // canonical source (`state.selected == VAL_QUEUE`) so A.3.6.1's
        // body-local walker can bind it as a green for `resolve_greens`.
        let is_queue = state.selected == 21usize;
        // The dual-mode encoding, read once per dispatch so it reaches the
        // merge-point key. Bound here rather than hoisted out of the loop
        // because the flip has to change the key the moment it happens.
        let bm = jit_bigint_mode();

        WALK_STORAGE_PTR.with(|c| c.set(&state.storage as *const Storage as usize));
        // rpaheui/aheui/aheui.py:253-255: jit_merge_point
        // `; state` selects the single-pass close: the walk's final state is
        // transferred into `state` here (via the hook's `recover`) instead of
        // being replayed. Byte-identical to `jit_merge_point!()` until the walk
        // closes a loop.
        jit_merge_point!(driver, program, pc; state);
        // rpaheui parity: `selected` is a RED (reds=['stacksize','storage',
        // 'selected']) — it is NEVER promoted (aheui.py promotes only
        // `program`@326 and `storage`@332). Storage ops dispatch on the live
        // `selected` index via `jit_storage_*` residual calls (one call site,
        // mirroring rpaheui's polymorphic `selected.METHOD()`), so no
        // `guard_value(selected)` is emitted and the loop closes through the
        // real back-edge instead of being rejected as an invalid loop.
        let op = program.get_op(pc);
        if spdiag_enabled() {
            let out = aheui_io::output_total_bytes();
            if (1240..=1260).contains(&out) {
                let snap = format!(
                    " out={out} selected={}{}",
                    state.selected,
                    state.spdiag_dump_stacks()
                );
                SK_SNAPSHOT.with(|s| *s.borrow_mut() = snap);
            }
            if op == 19 || op == 20 {
                let out = aheui_io::output_total_bytes();
                if (1000..=3000).contains(&out) {
                    eprintln!(
                        "@@@MAINEMIT op={op} out={out} selected={}{}",
                        state.selected,
                        state.spdiag_dump_stacks(),
                    );
                }
            }
        }
        if SPDIAG_TRACE_OPS.load(std::sync::atomic::Ordering::Relaxed) > 0 {
            SPDIAG_TRACE_OPS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            eprintln!(
                "@@@SPDIAG resume-op pc={pc} op={op} stacksize={} selected={} stackok={stackok} out={}{}",
                state.stacksize,
                state.selected,
                aheui_io::output_total_bytes(),
                state.spdiag_dump_stacks(),
            );
        }
        // Per-op stack-size delta gated on stackok. When a guarded op
        // (OP_STACKDEL > 0) is skipped because the stack is too small,
        // its delta must also be skipped to keep stacksize in sync with
        // the real storage length. `op` is green so the call constant-
        // folds per trace arm.
        state.stacksize += jit_effective_stacksize_delta(op as usize, stackok as i64);
        // Phase D-1 §5: pre-advance `pc` so the interpreter's pc matches
        // the trace's `__jit_pc = op_pc + 1` convention. Operand reads in
        // the arms use `pc - 1` to recover the opcode row; the trailing
        // `pc += 1` at the end of the loop is dropped (replaced by this
        // pre-advance). Branch arms compute targets against `pc - 1`
        // (op_pc) so the back-edge check stays semantic.
        pc += 1;

        // rpaheui/aheui/aheui.py:294-311: branch ops at dispatch level.
        // lower_match_stmt lowers this into chained guards in the dispatch
        // JitCode. pc = target + continue update the JitCode pc register
        // and BC_GOTO loop_start.
        match op {
            OP_BRPOP1 | OP_BRPOP2 => {
                if stackok == false {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, program);
                    continue;
                }
            }
            OP_JMP => {
                pc = program.get_label(pc - 1);
                stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, program);
                continue;
            }
            OP_BRZ => {
                let zero = if is_queue {
                    jit_pop_is_zero_queue(state.selected_ref)
                } else {
                    // inline pop for Stack, then zero-check on the int value
                    let top_node = state.selected_ref.head;
                    let pop_val = top_node.value;
                    let next = top_node.next;
                    state.selected_ref.head = next;
                    state.selected_ref.size = state.selected_ref.size - 1u32;
                    jit_free_node(top_node);
                    // pop_val is Val (= i64 repr-transparent). val_is_zero
                    // checks `*v == 0` for smallint, or the tagged form
                    // `(0 << 1) | 1 = 1` for bigint. Use the raw int
                    // comparison `pop_val == jit_tag_val(0)` which the
                    // lowerer handles natively as IntEq.
                    let zero_val = if bm != 0 {
                        jit_tag_val(0i64)
                    } else {
                        jit_tag_val_raw(0i64)
                    };
                    if pop_val == zero_val { 1i64 } else { 0i64 }
                };
                if zero != 0 {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, program);
                    continue;
                }
            }
            _ => {}
        }

        match op {
            // rpaheui/aheui/aheui.py:260-389: `selected.<op>()`.
            // Phase D-1 §4: 2-way branch on the `is_queue` green.
            // The trace specializes per value so only one branch
            // survives in compiled code: concrete `stack_*` or
            // `queue_*` (Port falls through to polymorphic dispatch).
            // `selected_ref` is the `ref(Stack)` state scalar, which
            // points at the currently selected storage; `is_queue`
            // (= `selected == VAL_QUEUE`) is a green, so the branch
            // folds to a constant and the dead arm is eliminated.
            OP_ADD => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_add(state.selected_ref);
                        } else {
                            lj::queue_add_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        lj::stack_add(state.selected_ref);
                    } else {
                        lj::stack_add_raw(state.selected_ref);
                    }
                }
            }
            OP_SUB => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_sub(state.selected_ref);
                        } else {
                            lj::queue_sub_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        lj::stack_sub(state.selected_ref);
                    } else {
                        lj::stack_sub_raw(state.selected_ref);
                    }
                }
            }
            OP_MUL => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_mul(state.selected_ref);
                        } else {
                            lj::queue_mul_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        lj::stack_mul(state.selected_ref);
                    } else {
                        lj::stack_mul_raw(state.selected_ref);
                    }
                }
            }
            OP_DIV => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_div(state.selected_ref);
                        } else {
                            lj::queue_div_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_div(r2, r1);
                    } else {
                        lj::stack_div_raw(state.selected_ref);
                    }
                }
            }
            OP_MOD => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_mod(state.selected_ref);
                        } else {
                            lj::queue_mod_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_mod(r2, r1);
                    } else {
                        lj::stack_mod_raw(state.selected_ref);
                    }
                }
            }
            OP_POP => {
                if stackok {
                    // Bind the popped value (discarded) so the `inline_int`
                    // pop helpers lower in value position; a discarded
                    // statement-position `inline_int` call has no lowering and
                    // aborts the trace. Mirrors OP_POPNUM's pop shape.
                    let _popped = if is_queue {
                        lj::queue_pop(state.selected_ref)
                    } else {
                        let top_node = state.selected_ref.head;
                        let pop_val = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        pop_val
                    };
                }
            }
            OP_PUSH => {
                // rpaheui/aheui/aheui.py:272-275.
                let value = program.get_operand(pc - 1) as i64;
                let v = if bm != 0 {
                    jit_tag_val(value)
                } else {
                    jit_tag_val_raw(value)
                };
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else if state.selected == VAL_PORT {
                    // linkedlist.py:134-139 `Port.push` also records
                    // `last_push`, and linkedlist.py:141-142 `Port.dup`
                    // pushes that instead of the head value. `selected_ref`
                    // is typed `Stack`, so the inline push/dup below write
                    // head/size only and a `push; pop; dup` on the port
                    // duplicates the wrong value. The port takes the
                    // polymorphic residual, matching aheui.py:260-389
                    // `selected.METHOD()`. Only push and dup diverge —
                    // `pop`, `swap`, `_get_2_values` and `_put_value` are
                    // the shared `LinkedList` implementations.
                    jit_storage_push(state.storage_ref, state.selected, v);
                } else {
                    lj::stack_push(state.selected_ref, v);
                }
            }
            OP_DUP => {
                if stackok {
                    if is_queue {
                        lj::queue_dup(state.selected_ref);
                    } else if state.selected == VAL_PORT {
                        jit_storage_dup(state.storage_ref, state.selected);
                    } else {
                        lj::stack_dup(state.selected_ref);
                    }
                }
            }
            OP_SWAP => {
                if stackok {
                    if is_queue {
                        lj::queue_swap(state.selected_ref);
                    } else {
                        lj::stack_swap(state.selected_ref);
                    }
                }
            }
            OP_SEL => {
                // rpaheui/aheui/aheui.py:280-284: `selected = storage[value];
                // stacksize = len(selected)`. `len(selected)` is a getfield on
                // the selected list's mutable `.size` field — re-read each loop
                // entry and invalidated by stack mutation, so `stacksize` never
                // freezes to a loop-invariant constant (unlike a residual call
                // result, which the optimiser bakes once and drops). Mirror it:
                // rebind `selected_ref` to the new list, then read `.size`
                // through it as a `getfield_gc_i`.
                let value = program.get_operand(pc - 1) as usize;
                state.selected = value;
                // `jit_sel_get_ref(state.storage_ref, …)` indexes
                // `pools[selected]`; the `pool_arrays` recogniser lowers it to
                // `getarrayitem_gc_r` on the `storage_ref` base, so the loaded
                // stack ref re-derives from `selected` each loop entry instead
                // of being carried as an independent, divergence-prone red.
                state.selected_ref = jit_sel_get_ref(state.storage_ref, state.selected);
                state.stacksize = state.selected_ref.size as i64;
            }
            OP_MOV => {
                if stackok {
                    let r = if is_queue {
                        lj::queue_pop(state.selected_ref)
                    } else {
                        // inline pop for Stack
                        let top_node = state.selected_ref.head;
                        let pop_val = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        pop_val
                    };
                    let target = program.get_operand(pc - 1) as usize;
                    if target == VAL_QUEUE || target == VAL_PORT {
                        // Queue/Port keep the polymorphic residual (tail-append semantics).
                        jit_storage_push(state.storage_ref, target, r);
                    } else {
                        // Stack: orthodox inline push into pools[target],
                        // mirroring OP_PUSH into selected_ref.
                        let target_ref = jit_sel_get_ref(state.storage_ref, target);
                        let old_head = target_ref.head;
                        let new_node = NodeJit {
                            value: r,
                            next: old_head,
                        };
                        target_ref.head = new_node;
                        target_ref.size = target_ref.size + 1u32;
                    }
                    if state.selected == target {
                        state.stacksize += 1;
                    }
                }
            }
            OP_CMP => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_cmp(state.selected_ref);
                        } else {
                            lj::queue_cmp_raw(state.selected_ref);
                        }
                    } else if bm != 0 {
                        lj::stack_cmp(state.selected_ref);
                    } else {
                        lj::stack_cmp_raw(state.selected_ref);
                    }
                }
            }
            // Branch ops handled by dispatch-level if-chain above.
            OP_POPNUM => {
                if stackok {
                    let r = if is_queue {
                        lj::queue_pop(state.selected_ref)
                    } else {
                        let top_node = state.selected_ref.head;
                        let pop_val = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        pop_val
                    };
                    aheui_io::output_write_number(&r);
                }
            }
            OP_POPCHAR => {
                if stackok {
                    let r = if is_queue {
                        lj::queue_pop(state.selected_ref)
                    } else {
                        let top_node = state.selected_ref.head;
                        let pop_val = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1u32;
                        jit_free_node(top_node);
                        pop_val
                    };
                    aheui_io::output_write_utf8(&r);
                }
            }
            OP_PUSHNUM => {
                // rpaheui/aheui/aheui.py:318-321
                jit_output_flush();
                let num = jit_read_number();
                let v = if bm != 0 {
                    jit_tag_val(num)
                } else {
                    jit_tag_val_raw(num)
                };
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else if state.selected == VAL_PORT {
                    jit_storage_push(state.storage_ref, state.selected, v);
                } else {
                    let old_head = state.selected_ref.head;
                    let new_node = jit_alloc_node(v, old_head);
                    state.selected_ref.head = new_node;
                    state.selected_ref.size = state.selected_ref.size + 1u32;
                }
            }
            OP_PUSHCHAR => {
                // rpaheui/aheui/aheui.py:322-325
                jit_output_flush();
                let ch = jit_read_utf8();
                let v = if bm != 0 {
                    jit_tag_val(ch)
                } else {
                    jit_tag_val_raw(ch)
                };
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else if state.selected == VAL_PORT {
                    jit_storage_push(state.storage_ref, state.selected, v);
                } else {
                    let old_head = state.selected_ref.head;
                    let new_node = jit_alloc_node(v, old_head);
                    state.selected_ref.head = new_node;
                    state.selected_ref.size = state.selected_ref.size + 1u32;
                }
            }
            // Branch ops: concrete execution handled by the pre-dispatch
            // match (OP_JMP) or the runtime if-chain. These empty arms
            // ensure the JitCode dispatch chain has entries for them so
            // guard failures don't abort the trace.
            OP_BRPOP1 | OP_BRPOP2 | OP_BRZ | OP_JMP => {}
            OP_NONE => {}
            OP_HALT => break,
            _ => {}
        }
    }

    aheui_io::output_flush();

    // The driver dies with this frame and the caller `process::exit`s on the
    // returned value, so hand the counters over before either happens.
    publish_jit_stats(
        driver.get_stats(),
        driver.meta_interp().staticdata.profiler.snapshot(),
    );

    // rpaheui/aheui/aheui.py:363-366
    if state.selected_dispatch().__len__() > 0 {
        state.selected_dispatch_mut().pop()
    } else {
        val_from_i32(0)
    }
}
