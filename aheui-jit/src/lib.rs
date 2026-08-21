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

pub use aheui_runtime;
pub use aheui_runtime::aheui;
pub use aheui_runtime::io;
pub use aheui_runtime::storage;
pub use aheui_runtime::value;

pub mod jit;

/// Default JIT threshold, taken from the parameter table rather than
/// restated: a jitdriver that does not set the value gets `PARAMETERS`'
/// default (`rlib/jit.py:588`), so the number has one home.
pub const JIT_THRESHOLD: u32 = majit_metainterp::jit::PARAMETERS.threshold;

/// JIT threshold honoring the `MAJIT_THRESHOLD` env override (for testing
/// small hot loops without the production warmup). RPython exposes the
/// threshold as a configurable jitdriver param (`warmspot.py`); this is
/// the same knob, read once at startup.
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
/// the sampled terminating programs stay byte-identical. The default remains
/// 70000 because lowering it changes the committed `jitstats` floor: logo's
/// `loops_aborted` moves from 0 to 2 and `guard_failures` from 201 to 401.
/// [`trace_limit`] exposes an override for performance experiments.
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

#[cfg(feature = "bigint-backend")]
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

    #[cfg(feature = "bigint-backend")]
    fn walk_aheui_bigint_roots(visit: &mut dyn FnMut(&mut GcRef)) {
        aheui_runtime::storage::walk_bigint_root_values(&mut |value| {
            if let Some(addr) = aheui_runtime::value::val_bigint_addr(value) {
                let mut root = GcRef(addr);
                visit(&mut root);
                if root.0 != addr {
                    // `collect_oldgen_nonmoving` leaves old-generation payloads
                    // in place. Write back any changed address so this root
                    // walker also supports a moving visitor.
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
            bits.div_ceil(64) as usize * 8
        }
    }
}

#[cfg(not(feature = "bigint-backend"))]
mod bigint_gc {
    pub fn init() {}
}

pub fn init_gc_subsystem() {
    bigint_gc::init();
}

include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

// Imports required by generated JIT code.

use aheui_runtime::aheui::*;
use aheui_runtime::band as bd;
use aheui_runtime::io as aheui_io;
use aheui_runtime::storage::linkedlist_jit as lj;
use aheui_runtime::storage::{LinkedList, Storage};
use ahsembler::compiler::Program;

use aheui_runtime::value::*;

// Diagnostic environment variables, cached outside hot loops.

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

    /// `llmodel.py:717-721 bh_setfield_gc_i` — the store takes the field's
    /// width from its descriptor. A field narrower than a word is a real
    /// field here: `size` on the list base is a `u32` followed by another
    /// member, so a store that assumed a word would write over its
    /// neighbour.
    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        // SAFETY: `struct_ptr` is a virtual this allocator has just
        // materialized; offset and size come from the descriptor of the
        // store being replayed.
        unsafe {
            majit_backend::llmodel::write_int_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                descr_info.field_size,
                value,
            )
        };
    }

    /// `llmodel.py:723-727 bh_setfield_gc_r` — pointer width, no size.
    ///
    /// `write_ref_at_mem` documents that its caller owes the barrier
    /// upstream's `llop.raw_store` rewrite would have carried. This
    /// collector has no generational barrier to owe — its `write_barrier`
    /// is a no-op — so the bare store is the whole operation.
    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe {
            majit_backend::llmodel::write_ref_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                value as usize,
            )
        };
    }

    /// `llmodel.py:730-734 bh_setfield_gc_f` — storage width, no size.
    /// `value` carries the float's storage bits, the deadframe's untyped
    /// form.
    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe {
            majit_backend::llmodel::write_float_at_mem(
                struct_ptr as usize,
                descr_info.offset,
                f64::from_bits(value as u64),
            )
        };
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

    // This hook enumerates only shadow-stack node references. MiniMarkGC walks
    // Aheui bignums held by majit's extra-root set separately.
}

/// Trace-time state for the Aheui JIT.
///
/// rpaheui/aheui/aheui.py:228-234 stores the reds as:
///   stacksize = 0
///   storage   = Storage()
///   selected  = storage[0]         # object reference
///
/// Rust represents the red state differently where required by the borrow
/// checker and the JIT raw-pointer ABI:
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
///   literally the raw pointer to the base the selected storage embeds,
///   which is where those two words are declared. We keep it next
///   to `selected: usize` because `refresh_selected_ref` has to run any
///   time `selected` changes; treating them as a single logical field
///   avoided plumbing a dedicated getter through the `#[jit_interp]`
///   macro.
static SPDIAG_TRACE_OPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

thread_local! {
    // Storage snapshot captured before trace walking. Recovery prints it next
    // to the restored state to distinguish a stale state field from a stale
    // optimized heap write.
    static PRE_WALK_SNAPSHOT: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
}

/// Operand words each stack pool keeps out of its node chain.
///
/// A power of two: the slot for absolute height `h` is `h & (CAP - 1)`, so the
/// window slides without moving anything. 64 covers every depth logo reaches
/// (its deepest pool rests at 55 inside one straight-line body), and the two
/// arrays together declare 1820 elements, well inside the resume numbering's
/// live-box ceiling.
/// Operand words each banded stack pool keeps out of its node chain.
///
/// A pool's band is a ring holding its **top** `CAP` elements: element at
/// absolute height `h` lives at `pool * CAP + (h & CAP_MASK)`. Anything below
/// that stays in the node chain, so a push past `CAP` evicts one word to the
/// chain and a pop below it refills one word back — into the slot the pop just
/// vacated, since `h` and `h - CAP` share a ring slot. Holding the top rather
/// than the bottom is what keeps both operands of a binary op on the same tier
/// at every depth.
///
/// A power of two, so the ring index is a mask. 64 covers the depth programs
/// actually reach without paying for slots they never use — the declared length
/// of a virtualizable array is what its per-compile cost scales with.
pub const CAP: usize = 64;
const _: () = assert!(CAP.is_power_of_two());
/// `h & CAP_MASK` is `h`'s slot inside its pool's ring.
pub const CAP_MASK: usize = CAP - 1;

/// The live `AheuiState`, for [`walk_band_values`]. Zero until the mainloop
/// registers it, which is also when the first band slot can hold a value.
static BAND_STATE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Visit every operand word a band currently holds.
///
/// Registered as `storage::BAND_ROOT_WALK_HOOK`, so it runs as part of the one
/// enumeration that both the bignum collector and the mode flip walk. Bounding
/// by each pool's live count is not an optimisation: a slot above it holds a
/// word left by an element that has already been popped, and following that as
/// a value would read a freed heap value.
fn walk_band_values(visit: &mut dyn FnMut(&mut Val)) {
    let state_addr = BAND_STATE.load(std::sync::atomic::Ordering::Relaxed);
    if state_addr == 0 {
        return;
    }
    let state = unsafe { &mut *(state_addr as *mut AheuiState) };
    // The armed count, not `vals.len() / CAP`: a pool at or above it never
    // takes a band arm, so its slots hold whatever an earlier arming left
    // there while its words are all in the chain, which the collector walks
    // on its own.
    let bands = BAND_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    for pool in 0..bands {
        if pool == VAL_QUEUE || pool == VAL_PORT {
            continue;
        }
        // The selected pool's count lives in `stacksize`; every other pool
        // parked its own at the `OP_SEL` that left it.
        let depth = if pool == state.selected {
            state.stacksize
        } else {
            state.depths[pool]
        };
        let depth = if depth < 0 { 0usize } else { depth as usize };
        let held = depth.min(CAP);
        for step in 0..held {
            let slot = pool * CAP + ((depth - 1 - step) & CAP_MASK);
            // A `Val` is `#[repr(transparent)]` over the word the band stores,
            // so the slot is already the value; the reference just names it as
            // one.
            let word: &mut i64 = &mut state.vals[slot];
            visit(unsafe { &mut *(word as *mut i64 as *mut Val) });
        }
    }
}

/// The band count of the running program, for [`jit_band_count`].
static BAND_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The band count as a green.
///
/// Constant for a run, but a *program* property rather than a compile-time one,
/// so it enters the trace the way `bm` does: read once per dispatch, bound
/// before the merge point, and keyed on there.
extern "C" fn jit_band_count() -> i64 {
    BAND_COUNT.load(std::sync::atomic::Ordering::Relaxed) as i64
}

/// Highest stack pool index a program selects or moves into, plus one.
///
/// Only pools below this get a band in `vals`, so a program that stays in the
/// first few pools declares a short array. A band is addressed as `pool * CAP`
/// with no remapping table to read at run time, so the count is a ceiling, not
/// a population.
///
/// Capped at `VAL_QUEUE`, which is what makes `selected < bands` a complete
/// test: queue and port push and pop at opposite ends, so neither can hold a
/// band, and capping below them lets one comparison answer both questions.
/// Pools above the queue trade their band for that single comparison.
fn banded_pool_count(program: &Program) -> usize {
    // One band. Over logo that arm spends 146ms against the node chain's 421ms,
    // a second band 157ms and a fifth 200ms: a declared slot is paid for in
    // compile time and in per-iteration work whether or not the program ever
    // selects its pool, so reaching past the first pool loses more than the band
    // there wins.
    // `AHEUI_BANDS` selects the arm, which is also what an A/B needs — both arms
    // then come from one binary.
    let Ok(text) = std::env::var("AHEUI_BANDS") else {
        return 1;
    };
    match text.parse::<usize>() {
        Ok(count) => return count.min(VAL_QUEUE),
        Err(_) => {}
    }
    let mut highest: Option<usize> = None;
    for pc in 0..program.size {
        let op = program.get_op(pc);
        if op == OP_SEL || op == OP_MOV {
            let pool = program.get_operand(pc) as usize;
            if pool < STORAGE_COUNT && pool != VAL_QUEUE && pool != VAL_PORT {
                highest = Some(match highest {
                    Some(h) if h >= pool => h,
                    _ => pool,
                });
            }
        }
    }
    // Pool 0 is selected before the first instruction runs, so it is banded
    // whether or not the program ever names it.
    let count = highest.map_or(1, |h| h + 1);
    if count > VAL_QUEUE { VAL_QUEUE } else { count }
}

struct AheuiState {
    storage: Storage,
    /// The top operand words of every pool, `STORAGE_COUNT * CAP` of them.
    ///
    /// Index `i * CAP + (h & CAP_MASK)` is pool `i`'s element at absolute
    /// height `h`, for the heights the window currently owns. Declared
    /// `[int; virt]` so an access promotes its index and answers out of the
    /// virtualizable's boxes rather than memory.
    vals: majit_metainterp::virt_array::VirtArray<i64>,
    /// Total element count of each pool, window and node chain together.
    ///
    /// Authoritative for every pool except the selected one, whose live count
    /// is `sp` — `OP_SEL` writes the outgoing pool's `sp` back here before it
    /// reads the incoming pool's.
    depths: majit_metainterp::virt_array::VirtArray<i64>,
    selected: usize,
    stacksize: i64,
    /// `stacksize` as an unsigned ring index, refreshed once per dispatch.
    ///
    /// The dispatch pre-applies the opcode's declared delta, so inside an arm
    /// this is the depth the pool will have *after* the op; each arm reaches
    /// its operands at a fixed literal offset from it.
    sp: usize,
    /// `state.storage.pools[selected]` packed as `usize`. Tracked as
    /// `ref(ListBase)` in `state_fields` so it is carried in the ref register
    /// bank as a genuine `InputArgRef`: promoted with `ref_guard_value` and
    /// passed to the monomorphic storage helpers as a ref-kind arg. The
    /// `usize` carrier round-trips the raw pointer bits.
    selected_ref: usize,
    /// `&mut state.storage as *mut Storage` packed as `usize` — the base the
    /// contiguous `pools: [*mut ListBase; N]` array is read off.
    /// Tracked as `ref(Storage)` and declared a `pool_arrays` base so the
    /// OP_SEL `selected_ref = pools[selected]` read lowers to a re-producible
    /// `getarrayitem_gc_r` on this base instead of an opaque residual call;
    /// the loaded list ref then re-derives from `selected` each loop entry
    /// rather than being carried as an independent, divergence-prone red.
    /// The sole carrier of the storage pointer: `aheui.py:27` carries
    /// `storage` as one red, so the residual storage helpers take this same
    /// ref-kind field rather than an aliased `int` copy (an int/ref pair for
    /// one value makes the resume box kind of every `pools[N]` list ref
    /// ambiguous, seeding Ref-typed loop-header slots with Int boxes).
    storage_ref: usize,
}

impl AheuiState {
    #[inline(always)]
    fn refresh_selected_ref(&mut self) {
        // rpaheui/aheui/aheui.py:233,282: selected = storage[idx].
        // `pools[idx]` already holds the address of the base the storage at
        // that index embeds, so the JIT reads `head` and `size` straight off
        // it without a further step.
        self.selected_ref = self.storage.get_list_ptr(self.selected) as usize;
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
            self.storage.queue.tail, self.storage.port.base.head,
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

    /// Disagreement between a banded pool's depth and the chain under it.
    ///
    /// A band holds a pool's top `CAP` words, so its chain holds exactly the
    /// rest: `depth - CAP` of them, or none while the band is not yet full.
    /// [`Storage::check_chains`] cannot see a break in that split, because
    /// each chain stays internally consistent across it; the program only
    /// notices opcodes later, when a refill pops a chain the depth said was
    /// not empty.
    fn check_bands(&self) -> Option<String> {
        let bands = BAND_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        for pool in 0..bands {
            let depth = if pool == self.selected {
                self.stacksize
            } else {
                self.depths[pool]
            };
            let want = (depth - CAP as i64).max(0);
            let have = self.storage.len_at(pool) as i64;
            if want != have {
                return Some(format!(
                    "pool {pool} depth {depth} leaves {want} below the band, chain holds {have}"
                ));
            }
        }
        None
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
            PRE_WALK_SNAPSHOT.with(|s| eprintln!("@@@SPDIAG pre-walk-snapshot{}", s.borrow()));
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
            if let Some(err) = self.check_bands() {
                panic!("check_bands: {err}\n{}", self.dump_chain_addrs());
            }
        }
        // `len_at` counts the node chain, which is the pool's whole content
        // only while the pool is unbanded. A banded pool keeps its top `CAP`
        // words in `vals`, which the chain cannot see, and the number of them
        // it holds is `min(stacksize, CAP)` — so the chain under-reports by
        // exactly the quantity that would be needed to correct it, and no
        // count derived from storage can name the pool's depth. `stacksize` is
        // a scalar state field written back from the walk immediately before
        // this hook runs, so for a banded pool it is already the answer.
        if self.selected >= BAND_COUNT.load(std::sync::atomic::Ordering::Relaxed) {
            self.stacksize = self.storage.len_at(self.selected) as i64;
        }
    }
}

thread_local! {
    /// Raw `*const Storage` set before each `jit_merge_point!`. Output shims
    /// use it to inspect the walk's shared-storage state while execution is
    /// inside `JitCodeMachine`, where mainloop recovery diagnostics cannot
    /// observe it.
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
/// The `out` byte range `@@@WALKEMIT` reports, from `MAJIT_SPDIAG_FROM` /
/// `MAJIT_SPDIAG_TO`.
///
/// A window rather than the whole run, because a program emitting hundreds of
/// kilobytes buries the one write being chased. Which window is interesting is
/// a property of the failure — the byte offset at which an A/B says the two
/// runs first differed — so it is a knob rather than a constant.
fn spdiag_window() -> (u64, u64) {
    static WINDOW: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| {
        let read = |name: &str, fallback: u64| {
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(fallback)
        };
        (
            read("MAJIT_SPDIAG_FROM", 1000),
            read("MAJIT_SPDIAG_TO", 3000),
        )
    })
}

/// Report one value emitted from COMPILED code.
///
/// Only the JIT shims call this; the interpreter arms write through
/// `aheui_io` directly. That asymmetry is the point — a diverging byte that
/// appears here was produced by a trace, and one that does not was produced by
/// the interpreter after a guard sent it back.
fn walk_emit_log(kind: &str, value: i64) {
    if !spdiag_enabled() {
        return;
    }
    let out = aheui_io::output_total_bytes();
    let (from, to) = spdiag_window();
    if !(from..=to).contains(&out) {
        return;
    }
    eprintln!(
        "@@@WALKEMIT {kind} out={out} val={value}{}",
        dump_storage_ptr(WALK_STORAGE_PTR.with(|c| c.get()))
    );
}

extern "C" fn jit_write_number(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    if bh_debug_enabled() {
        eprintln!("[io-debug] jit_write_number raw={value}");
    }
    walk_emit_log("num", value);
    aheui_io::output_write_number(&v);
}

extern "C" fn jit_write_utf8(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    walk_emit_log("utf8", value);
    aheui_io::output_write_utf8(&v);
}

// Input I/O shims for JIT tracing.
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
/// Unlike [`jit_tag_val`], which constructs a tagged value through an `i32`,
/// this helper preserves the full `i64` range used by raw-word mode.
#[cfg(feature = "bigint-backend")]
extern "C" fn jit_tag_val_raw(raw: i64) -> Val {
    aheui_runtime::value::val_from_raw_i64(raw)
}

/// [`jit_tag_val`] as the packed word, for a value entering a band slot.
#[cfg(feature = "bigint-backend")]
extern "C" fn jit_tag_word(raw: i64) -> i64 {
    aheui_runtime::value::val_as_raw_i64(val_from_i32(raw as i32))
}

/// [`jit_tag_val_raw`] as the packed word. Mode 0 keeps the word as the value,
/// so this is identity; it exists so the two modes read alike at a call site.
#[cfg(feature = "bigint-backend")]
extern "C" fn jit_tag_word_raw(raw: i64) -> i64 {
    raw
}

/// [`jit_tag_val_raw`]'s inverse: the packed word behind a `Val`, for a store
/// into a band slot.
///
/// The band is an int-kind array, so a value entering it is spelled as its
/// word. The word carries the mode's encoding, not the mode, which is why the
/// band is only ever read back through the twin of the helper that filled it.
#[cfg(feature = "bigint-backend")]
extern "C" fn jit_win_store(v: Val) -> i64 {
    aheui_runtime::value::val_as_raw_i64(v)
}

/// `val_ge` as the 1-or-0 int `cmp` pushes (linkedlist.py:60-64).
///
/// The comparison itself is a value operation; taking it as an int here keeps
/// the arm's result a plain word that goes straight into a band slot.
#[cfg(feature = "bigint-backend")]
extern "C" fn jit_val_ge_i(a: Val, b: Val) -> i64 {
    i64::from(aheui_runtime::value::val_ge(&a, &b))
}

#[inline(always)]
#[cfg(feature = "bigint-backend")]
// Referenced only from the `jit_interp` attribute above (`native_tag_small`)
// and the call-policy table, neither of which the dead-code pass reads.
#[allow(dead_code)]
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

// Local JIT wrappers for node allocation and value comparison.
// Local wrappers for functions in aheui_runtime whose `__majit_call_policy_*`
// probes the macro generates in the LOCAL scope.  Calling `lj::alloc_node_jit`
// directly would look for the probe in the `lj` module, which doesn't exist.

#[inline(always)]
fn jit_alloc_node(value: Val, next: usize) -> usize {
    aheui_runtime::storage::linkedlist_jit::alloc_node_jit(value, next)
}

#[inline(always)]
fn jit_free_node(node: usize) {
    aheui_runtime::storage::linkedlist_jit::free_node_jit(node)
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

fn __majit_pipeline_liveness_prebuild(assembler: &mut majit_metainterp::Assembler) {
    jit::jitcode_runtime::prebuild_pipeline_liveness(assembler);
}

// Index-keyed dynamic storage dispatch — rpaheui parity for
// `selected.METHOD()` (aheui.py:260-389). The target index is a per-opcode
// operand rather than the promoted `selected_ref`, so the polymorphic
// `dispatch_mut(target)` is bundled into one residual call that dispatches on
// the live index at run time (Stack / Queue / Port) instead of a JIT-green
// 3-way `is_port`/`is_queue` branch. This keeps the recorded trace
// structurally identical to rpaheui's single polymorphic call site, so the
// optimiser never emits a contradictory `guard_value(selected)` and the loop
// closes through the real back-edge.
//
// Only push and dup survive, and for two independent reasons. OP_MOV names its
// target with a per-opcode operand rather than `selected` (aheui.py:277-279
// `storage[val].push(r)`), so nothing has resolved it to a concrete list. And
// `Port::push`/`Port::dup` are the only two storage methods Port does not
// share with the `LinkedList` implementation (`last_push`), so the port takes
// this path for them while the `is_queue` / `selected == VAL_PORT` split hands
// every other op a monomorphic `selected_ref` helper.
extern "C" fn jit_storage_push(pool_ptr: usize, target: usize, value: Val) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).push(value);
}
extern "C" fn jit_storage_dup(pool_ptr: usize, target: usize) {
    let storage = unsafe { &mut *(pool_ptr as *mut Storage) };
    storage.dispatch_mut(target).dup();
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
    storage.get_list_ptr(selected) as usize
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

// JIT mainloop.
//
// RPython parity: rpaheui/aheui/aheui.py mainloop()
// - storage = linked list stacks (no compact arrays, no virtualizable arrays)
// - selected = red variable dispatched through the live storage index
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
    // through `selected_dispatch_mut()`; `handle_regular_indirect_call`
    // preserves the live receiver, guards its concrete value, and emits the
    // residual call.
    state_fields = {
        storage: opaque(aheui_runtime::storage::Storage),
        // The operand words and depths, as virtualizable arrays.
        // `pyjitpl.py:1201-1216 _get_arrayitem_vable_index` promotes the index
        // of a virtualizable array access and then answers from
        // `virtualizable_boxes`, so an element access indexed by a running
        // depth still lowers to boxes rather than to memory. Declared before
        // the scalars because the flat vable layout follows declaration order.
        vals: [int; virt],
        depths: [int; virt],
        // RPython parity: AheuiState.selected is `usize` (slot index into
        // 28-slot pool); stacksize is `i64` (signed pop/push delta) — rpaheui
        // carries it as a full machine-word Python int, and `i64` matches the
        // IR's native Int width so no per-op `as i64` sign-extend is emitted
        // on every `stackok`/delta read. The macro carries them as Int in IR;
        // `int(<Type>)` keeps the user's natural Rust storage type and inserts
        // `as i64` / `as <Type>` casts at the JIT boundary (identity for i64).
        selected: int(usize),
        stacksize: int(i64),
        sp: int(usize),
        // The selected list's object reference (aheui.py:256
        // `selected = jit.promote(selected)`). Carried in the ref register
        // bank as a genuine `InputArgRef` so it can be promoted with
        // `ref_guard_value` and passed to the monomorphic storage helpers as a
        // ref-kind arg (`JitCallArg::reference`). The `usize` carrier
        // round-trips the raw pointer bits.
        //
        // The type is the base every storage embeds, matching
        // `aheui.py:251 selected = storage[value]`, where the list holds a
        // Stack, a Queue or a Port and the binding's type is what they have in
        // common. Naming one of the three here instead would make every access
        // through this field a reinterpretation of one storage as another.
        selected_ref: ref(aheui_runtime::storage::linkedlist::ListBase),
        // Base the `pools: [*mut ListBase; N]` array is read off.
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
    pool_arrays = { storage_ref.pools[pools_len] => jit_sel_get_ref -> aheui_runtime::storage::linkedlist::ListBase },
    // Struct field type declarations for ref-kind field access.
    // Tells the lowerer to emit getfield_gc_r / setfield_gc_r (ref-kind)
    // instead of _gc_i (int-kind) when accessing these fields through a
    // ref(T) state scalar or a local ref binding.
    ref_fields = {
        // Declared once, on the type that owns it.  A `Stack`, `Queue` or
        // `Port` access resolves onto `ListBase` and so mints one descriptor
        // for the one word, whichever of the three it was spelled through.
        aheui_runtime::storage::linkedlist::ListBase::head => aheui_runtime::storage::linkedlist::Node,
        aheui_runtime::storage::linkedlist::Node::next => aheui_runtime::storage::linkedlist::Node,
        // Declared for the same reason as the `Queue`/`Port` entries in
        // `int_fields`: the `residual_writes` group below mints this word from
        // here, and the field's kind comes from whichever producer describes
        // it. The linked-list helpers declare it a pointer; left undeclared
        // here it minted as a signed 8-byte integer instead, so a ref-kind
        // access read a descr that called the field an int — one word with two
        // descriptions, and a wrong width on a 32-bit pointer target.
        aheui_runtime::storage::linkedlist::Queue::tail => aheui_runtime::storage::linkedlist::Node,
    },
    // The element count is `u32`, so the field descr is sub-word and
    // `intbounds` can bound a load of it. Without a bound the depth's `+ 1`
    // may overflow, the sum goes rangeless, and the `stackok` check has to be
    // guarded again at every opcode instead of following from the last one.
    int_fields = {
        aheui_runtime::storage::linkedlist::ListBase::size => u32,
    },
    // The three storages embed `ListBase` as their leading field, so a field
    // they do not declare themselves is resolved against it.
    inlined_prefix = {
        aheui_runtime::storage::linkedlist::Stack::base => aheui_runtime::storage::linkedlist::ListBase,
        aheui_runtime::storage::linkedlist::Queue::base => aheui_runtime::storage::linkedlist::ListBase,
        aheui_runtime::storage::linkedlist::Port::base => aheui_runtime::storage::linkedlist::ListBase,
    },
    struct_allocs = { aheui_runtime::storage::linkedlist::Node => jit_alloc_node },
    headerless_structs = { aheui_runtime::storage::linkedlist::Node },
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
        jit_win_store => elidable_int_cannot_raise,
        jit_tag_word => elidable_int_cannot_raise,
        jit_tag_word_raw => elidable_int_cannot_raise,
        // The band arms hold the value as its word, which is what the output
        // shims already take, so they name the shim rather than going back
        // through a `Val` to reach it.
        jit_write_number => residual_void,
        jit_write_utf8 => residual_void,
        jit_val_ge_i => elidable_int_cannot_raise,
        jit_retag_small => elidable_int_cannot_raise,
        jit_bigint_mode => elidable_int_cannot_raise,
        jit_band_count => elidable_int_cannot_raise,
        // Method-call results consumed as values are lowered through
        // `lower_method_call_value`.
        Program::get_req_size => elidable_int_cannot_raise,
        Program::get_op => elidable_int_cannot_raise,
        Program::get_label => elidable_int_cannot_raise,
        Program::get_operand => elidable_int_cannot_raise,
        // Monomorphic storage helpers. The hot Stack ops are `#[jit_inline]`,
        // while the storage-independent pop / swap come from the graph
        // pipeline's shared `LinkedList` implementation. Queue div/mod stay
        // residual — a concrete `call_void_args` / `call_int_args` — rather
        // than silent-skipping the storage op. Their stack twins are not
        // registered because the arms hand-inline the pop and call
        // `val_div`/`val_mod` on the operands directly, so there is no
        // `lj::stack_div` call site left to classify.
        //
        // The registered path segments must match the call site verbatim
        // (the macro compares segment-by-segment); use the `lj::*` alias
        // here since the mainloop arms call `lj::stack_push(...)` etc.
        lj::stack_push => inline_void,
        lj::stack_add => inline_void,
        lj::stack_sub => inline_void,
        lj::stack_mul => inline_void,
        lj::stack_dup => inline_void,
        lj::stack_cmp => inline_void,
        lj::queue_push => inline_void,
        lj::queue_add => inline_void,
        lj::queue_sub => inline_void,
        lj::queue_mul => inline_void,
        lj::queue_div => residual_void,
        lj::queue_mod => residual_void,
        lj::queue_dup => inline_void,
        lj::queue_cmp => inline_void,
        // Named by neither family: their parameter is the base the three
        // storages embed, so one registration covers every selection.
        lj::pop_base_known_nonempty => inline_pipeline_int,
        // Band arithmetic. Value-returning, so they reach the trace through
        // the graph pipeline rather than as macro-inlined bodies, and they
        // take the packed word both ways so a band access needs no conversion.
        bd::band_add => inline_pipeline_int,
        bd::band_sub => inline_pipeline_int,
        bd::band_mul => inline_pipeline_int,
        bd::band_div => inline_pipeline_int,
        bd::band_mod => inline_pipeline_int,
        bd::band_cmp => inline_pipeline_int,
        bd::band_add_raw => inline_pipeline_int,
        bd::band_sub_raw => inline_pipeline_int,
        bd::band_mul_raw => inline_pipeline_int,
        bd::band_div_raw => inline_pipeline_int,
        bd::band_mod_raw => inline_pipeline_int,
        bd::band_cmp_raw => inline_pipeline_int,
        lj::swap_base_known_two => inline_pipeline_void,
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
        jit_storage_push => residual_void,
        jit_storage_dup => residual_void,
        // `storage[idx]` returns the selected list's object reference; the
        // result lands in the ref register bank. Elidable + cannot-raise:
        // `pools` is an immutable array (`_immutable_fields_ = ['pools']`)
        // of stable base pointers, so `pools[selected]` is a pure,
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
        // Register node frees and value arithmetic so field-level Stack
        // operations emit concrete IR instead of
        // silent-skipping unregistered calls. Node allocation now goes through
        // `struct_allocs` makes concrete execution call `jit_alloc_node`, while
        // tracing emits a headerless `New(Node)` plus field stores with the
        // descriptor identity used by graph-pipeline storage helpers.
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
        val_from_i32 => elidable_int_cannot_raise,
    },
    // Residual storage mutators that change `Stack.size` or its `head` chain
    // pointer.  Traced push/pop methods emit in-trace `setfield_gc` stores,
    // which invalidate the matching heapcache entries directly.
    // The macro-inlined Stack helpers and graph-pipeline pop/swap jitcodes
    // splice that `self.size` setfield into the trace directly, so they
    // reproduce the barrier and need no entry here. The same holds for pops
    // that arithmetic arms inline as head/next/size stores.
    // The remaining ops lower to opaque residual calls, which carry an empty
    // write-set by default. A size-only declaration can leave a stale head
    // cached after a residual pop, while a head-only declaration can leave a
    // stale size. Declaring both fields restores the invalidation performed by
    // traced setfield barriers. The lists are
    // conservative supersets because an extra reload is harmless while a
    // missing invalidation is not.
    // The `@ Struct` alias groups this used to carry are gone: `head` and
    // `size` are declared once, on the type that owns them, so there is one
    // descriptor per word to invalidate rather than one per nominal struct
    // an access might be spelled through.
    residual_writes = {
        selected_ref.size => [
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_dup,
        ],
        selected_ref.head => [
            lj::queue_push, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_cmp,
            jit_storage_push, jit_storage_dup,
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
            jit_storage_push, jit_storage_dup,
        ],
    },
    // rpaheui/aheui/aheui.py:29: greens=['pc','stackok','is_queue','program'].
    //
    // `bind_pre_merge_point_stmts` registers body-local bindings before green
    // resolution. Therefore a synthesized
    // `let is_queue = state.selected == 21usize;` flows through
    // `resolve_greens` and `emit_promote_greens` as an ordinary green.
    //
    // The dispatch arms keep their `state.selected == 21usize` /
    // `state.selected == 27usize` 3-way structure — pyre's discriminator
    // is finer-grained than rpaheui's 2-way `is_queue` (Port is split
    // out from the stack family), and within a trace specialised by
    // `guard_value(selected)` the comparison folds to a constant and
    // only the live branch reaches the optimised IR.
    // `bm` is pyre's own fifth green — see the module header. It binds through
    // the same pre-merge-point walker as `is_queue`.
    greens = [pc, stackok, is_queue, bm, bands, program],
    recover = refresh_state_from_storage,
    switch_dispatch = true,
    native_tag_small = { jit_retag_small },
)]
// This body is the `jit_interp` macro's INPUT, so its control flow is lowered,
// not merely read. Both of these lints are about the shape of a conditional and
// applying either one changed what came out: `cargo clippy --fix` rewrote
// `stackok == false` to `!stackok` and folded a nested `if let` into a let
// chain, and the emitted jitcode came out with a label that no block ever
// marked — every program then panicked at trace-install time. The crate still
// compiled cleanly; only running a program showed it.
//
// Do not apply a style fix inside this function without re-running a real
// program end to end. Anything the lints suggest here is a change to the
// generated code.
#[allow(clippy::bool_comparison, clippy::collapsible_if)]
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
    // `resume.py:1367` materializes virtual headerless `Node` values during
    // guard-failure deoptimization. The blackhole allocator must do the same or
    // a virtual node becomes a null head while the recorded size stays nonzero.
    driver.register_blackhole_allocator(AheuiBlackholeAllocator);

    // rpaheui/aheui/aheui.py:325 `jit.set_param(driver, 'trace_limit', 30000)`
    // — see [`TRACE_LIMIT`] for the scaling.
    driver.set_param("trace_limit", trace_limit() as i64);

    // `ALL_OPTS` minus `unroll` (warmspot.py:73 passes the list per driver).
    // rpaheui leaves the default, so this is a pyre-side choice for this
    // frontend only; nothing else on the process is affected.
    //
    // Peeling the preamble costs this driver more than it returns. On logo it
    // spends ~68ms of the ~108ms warmup, and the peeled body is no faster:
    // wall 271ms -> 209ms with the steady-state phase moving 163.5ms ->
    // 169.0ms, i.e. within noise. The aheui loop body carries almost nothing
    // loop-invariant to hoist — every operation reads or writes the mutable
    // selected stack — so the second copy buys the optimizer no new facts.
    // `AHEUI_ENABLE_OPTS` overrides the list, which is how a suspect pass gets
    // taken out of one arm without a second binary.
    const ENABLE_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap";
    match std::env::var("AHEUI_ENABLE_OPTS") {
        Ok(text) => driver.set_param_enable_opts(&text),
        Err(_) => driver.set_param_enable_opts(ENABLE_OPTS),
    }

    let mut pc: usize = 0;
    // rpaheui/aheui/aheui.py:30: reds=['stacksize','storage','selected']
    let mut state = AheuiState {
        storage: Storage::new(),
        vals: majit_metainterp::virt_array::VirtArray::filled(
            0i64,
            banded_pool_count(program) * CAP,
        ),
        depths: majit_metainterp::virt_array::VirtArray::filled(0i64, STORAGE_COUNT),
        selected: 0,
        stacksize: 0,
        sp: 0,
        selected_ref: 0,
        storage_ref: 0,
    };
    BAND_STATE.store(
        &mut state as *mut AheuiState as usize,
        std::sync::atomic::Ordering::Relaxed,
    );
    aheui_runtime::storage::BAND_ROOT_WALK_HOOK.store(
        walk_band_values as aheui_runtime::storage::BandRootWalkHook as usize,
        std::sync::atomic::Ordering::Relaxed,
    );
    // `AHEUI_BAND_ARMS` clamps the green below the declared array length, which
    // leaves the arrays exactly as long as the banded arm would make them while
    // taking every band arm out of reach. That separates the cost of declaring
    // the slots from the cost of using them.
    let arms = match std::env::var("AHEUI_BAND_ARMS")
        .ok()
        .and_then(|t| t.parse::<usize>().ok())
    {
        Some(limit) => limit.min(state.vals.len() / CAP),
        None => state.vals.len() / CAP,
    };
    BAND_COUNT.store(arms, std::sync::atomic::Ordering::Relaxed);

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
        // canonical source (`state.selected == VAL_QUEUE`) so the body-local
        // binding pass can expose it to `resolve_greens` as a green.
        let is_queue = state.selected == 21usize;
        // The dual-mode encoding, read once per dispatch so it reaches the
        // merge-point key. Bound here rather than hoisted out of the loop
        // because the flip has to change the key the moment it happens.
        let bm = jit_bigint_mode();
        // The declared band count, bound alongside `bm` so an arm's
        // `state.selected < bands` test folds inside a specialised trace.
        let bands = jit_band_count() as usize;

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
                PRE_WALK_SNAPSHOT.with(|s| *s.borrow_mut() = snap);
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
        state.sp = state.stacksize as usize;
        // Pre-advance `pc` so the interpreter's pc matches
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
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, bands, program);
                    continue;
                }
            }
            OP_JMP => {
                pc = program.get_label(pc - 1);
                stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, bands, program);
                continue;
            }
            OP_BRZ => {
                // The pop is the same for every storage, and the zero test is
                // one comparison on the popped `Val` either way.
                let pop_word = if state.selected < bands {
                    let top_slot = state.selected * CAP + (state.sp & CAP_MASK);
                    let word = state.vals[top_slot];
                    if state.sp >= CAP {
                        state.vals[top_slot] =
                            jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                    }
                    word
                } else {
                    jit_win_store(lj::pop_base_known_nonempty(state.selected_ref))
                };
                // pop_val is Val (= i64 repr-transparent). val_is_zero
                // checks `*v == 0` for smallint, or the tagged form
                // `(0 << 1) | 1 = 1` for bigint. Use the raw int
                // comparison `pop_val == jit_tag_val(0)` which the
                // lowerer handles natively as IntEq.
                let zero_word = if bm != 0 {
                    jit_tag_word(0i64)
                } else {
                    jit_tag_word_raw(0i64)
                };
                let zero = if pop_word == zero_word { 1i64 } else { 0i64 };
                if zero != 0 {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, bands, program);
                    continue;
                }
            }
            _ => {}
        }

        match op {
            // rpaheui/aheui/aheui.py:260-389: `selected.<op>()`.
            // Branch on the `is_queue` green. The trace specializes per value
            // so only one branch
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_add(r2, r1)
                        } else {
                            bd::band_add_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_sub(r2, r1)
                        } else {
                            bd::band_sub_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_mul(r2, r1)
                        } else {
                            bd::band_mul_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_div(r2, r1)
                        } else {
                            bd::band_div_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_mod(r2, r1)
                        } else {
                            bd::band_mod_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    // pop helper lowers in value position; a discarded
                    // statement-position `inline_int` call has no lowering and
                    // aborts the trace. Mirrors OP_POPNUM's pop shape.
                    if state.selected < bands {
                        let top_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        if state.sp >= CAP {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                    } else {
                        let _popped = lj::pop_base_known_nonempty(state.selected_ref);
                    }
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
                } else if state.selected < bands {
                    // `sp` is the depth after the push, so the new word takes
                    // the slot of height `sp - 1`.
                    let free_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                    if state.sp > CAP {
                        // The ring is full: the word this slot holds is the
                        // band's oldest and leaves for the chain.
                        lj::stack_push(state.selected_ref, jit_tag_val_raw(state.vals[free_slot]));
                    }
                    let __band_word = if bm != 0 {
                        jit_tag_word(value)
                    } else {
                        jit_tag_word_raw(value)
                    };
                    state.vals[free_slot] = __band_word;
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
                    } else if state.selected < bands {
                        let src_slot = state.selected * CAP + ((state.sp - 2) & CAP_MASK);
                        let free_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let top = state.vals[src_slot];
                        if state.sp > CAP {
                            lj::stack_push(
                                state.selected_ref,
                                jit_tag_val_raw(state.vals[free_slot]),
                            );
                        }
                        state.vals[free_slot] = top;
                    } else {
                        lj::stack_dup(state.selected_ref);
                    }
                }
            }
            OP_SWAP => {
                if stackok {
                    // linkedlist.py:30-33: one `swap` on `LinkedList`, which
                    // every storage inherits unchanged. The split this replaces
                    // was not dispatch -- the two helpers were identical -- it
                    // was what kept a queue-specialized trace touching `head`
                    // only under descriptors minted for `Queue`. `head` is
                    // declared once now, so the arm can say what rpaheui says.
                    if state.selected < bands {
                        let top_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let second_slot = state.selected * CAP + ((state.sp - 2) & CAP_MASK);
                        let top = state.vals[top_slot];
                        state.vals[top_slot] = state.vals[second_slot];
                        state.vals[second_slot] = top;
                    } else {
                        lj::swap_base_known_two(state.selected_ref);
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
                // A banded pool's count is not its chain's `.size` — the band
                // holds the top of it — so the count lives in `depths`, and the
                // selected pool's copy lives in `stacksize`. Park the outgoing
                // pool's before reading the incoming pool's back.
                if state.selected < bands {
                    state.depths[state.selected] = state.stacksize;
                }
                state.selected = value;
                // `jit_sel_get_ref(state.storage_ref, …)` indexes
                // `pools[selected]`; the `pool_arrays` recogniser lowers it to
                // `getarrayitem_gc_r` on the `storage_ref` base, so the loaded
                // stack ref re-derives from `selected` each loop entry instead
                // of being carried as an independent, divergence-prone red.
                state.selected_ref = jit_sel_get_ref(state.storage_ref, state.selected);
                // `depths` is authoritative only where a band exists; every
                // other pool keeps its whole count in its chain's `.size`.
                if state.selected < bands {
                    state.stacksize = state.depths[state.selected];
                } else {
                    state.stacksize = state.selected_ref.size as i64;
                }
                state.sp = state.stacksize as usize;
            }
            OP_MOV => {
                if stackok {
                    // The moved word, taken off whichever tier holds the
                    // source pool's top.
                    let moved = if state.selected < bands {
                        let top_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let word = state.vals[top_slot];
                        if state.sp >= CAP {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                        word
                    } else {
                        jit_win_store(lj::pop_base_known_nonempty(state.selected_ref))
                    };
                    let target = program.get_operand(pc - 1) as usize;
                    if target == VAL_QUEUE || target == VAL_PORT {
                        // Queue/Port keep the polymorphic residual (tail-append semantics).
                        jit_storage_push(state.storage_ref, target, jit_tag_val_raw(moved));
                    } else if target < bands {
                        // A move into the selected pool lands one below where
                        // the pop left it, which is the slot the pop vacated;
                        // any other pool's count comes out of `depths`.
                        let depth = if target == state.selected {
                            state.sp
                        } else {
                            state.depths[target] as usize
                        };
                        let target_ref = jit_sel_get_ref(state.storage_ref, target);
                        let free_slot = target * CAP + (depth & CAP_MASK);
                        if depth >= CAP {
                            lj::stack_push(target_ref, jit_tag_val_raw(state.vals[free_slot]));
                        }
                        state.vals[free_slot] = moved;
                        if target != state.selected {
                            state.depths[target] = (depth + 1) as i64;
                        }
                    } else {
                        // Stack: orthodox inline push into pools[target],
                        // mirroring OP_PUSH into selected_ref.
                        let target_ref = jit_sel_get_ref(state.storage_ref, target);
                        let old_head = target_ref.head;
                        let new_node = aheui_runtime::storage::linkedlist::Node {
                            value: jit_tag_val_raw(moved),
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
                    } else if state.selected < bands {
                        // `sp` is the depth after the op, so the two operands
                        // are at `sp` and `sp - 1` and the result takes the
                        // lower slot — the shape linkedlist.py:35-38 has over
                        // the chain, over the band instead.
                        let r1_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let r2_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_cmp(r2, r1)
                        } else {
                            bd::band_cmp_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= CAP {
                            // The pop dropped the band's bottom element out of
                            // range, so the chain hands the next one back — into
                            // the slot the popped operand just vacated, which is
                            // the same ring slot.
                            state.vals[r1_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
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
                    if state.selected < bands {
                        let top_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let word = state.vals[top_slot];
                        if state.sp >= CAP {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                        jit_write_number(word);
                    } else {
                        let r = lj::pop_base_known_nonempty(state.selected_ref);
                        aheui_io::output_write_number(&r);
                    }
                }
            }
            OP_POPCHAR => {
                if stackok {
                    if state.selected < bands {
                        let top_slot = state.selected * CAP + (state.sp & CAP_MASK);
                        let word = state.vals[top_slot];
                        if state.sp >= CAP {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                        jit_write_utf8(word);
                    } else {
                        let r = lj::pop_base_known_nonempty(state.selected_ref);
                        aheui_io::output_write_utf8(&r);
                    }
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
                } else if state.selected < bands {
                    let free_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                    if state.sp > CAP {
                        lj::stack_push(state.selected_ref, jit_tag_val_raw(state.vals[free_slot]));
                    }
                    let __band_word = if bm != 0 {
                        jit_tag_word(num)
                    } else {
                        jit_tag_word_raw(num)
                    };
                    state.vals[free_slot] = __band_word;
                } else {
                    lj::stack_push(state.selected_ref, v);
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
                } else if state.selected < bands {
                    let free_slot = state.selected * CAP + ((state.sp - 1) & CAP_MASK);
                    if state.sp > CAP {
                        lj::stack_push(state.selected_ref, jit_tag_val_raw(state.vals[free_slot]));
                    }
                    let __band_word = if bm != 0 {
                        jit_tag_word(ch)
                    } else {
                        jit_tag_word_raw(ch)
                    };
                    state.vals[free_slot] = __band_word;
                } else {
                    lj::stack_push(state.selected_ref, v);
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
    // The armed count for the same reason `walk_band_values` takes it: a pool
    // at or above it holds every word in its chain, and its band slots are
    // whatever an earlier arming left there.
    let bands = jit_band_count() as usize;
    let selected = state.selected;
    if selected != VAL_QUEUE && selected != VAL_PORT && selected < bands {
        if state.stacksize > 0 {
            let depth = state.stacksize as usize;
            aheui_runtime::value::val_from_raw_i64(
                state.vals[selected * CAP + ((depth - 1) & CAP_MASK)],
            )
        } else {
            val_from_i32(0)
        }
    } else if state.selected_dispatch().__len__() > 0 {
        state.selected_dispatch_mut().pop()
    } else {
        val_from_i32(0)
    }
}
