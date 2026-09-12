// Canonical Aheui interpreter: one dispatch body with optional JIT translation.
//
// RPython parity: rpaheui/aheui/aheui.py
//   greens = [pc, stackok, is_queue, program]
//   reds   = [stacksize, storage, selected]
//   storage = linked list stacks
//
// `bm` is a fifth green with no counterpart upstream. It carries the dual-mode
// encoding — Aheui runs values as raw machine words until one overflows — and
// being a green is what keeps the mode out of compiled code: each trace is
// keyed on one encoding, so the arithmetic arms below pick their helper once,
// at record time, instead of testing a global per operation. The flip changes
// the key, which retires the mode-0 traces and records mode-1 ones.
//
// stackok is a green (rpaheui parity): specialising the trace on it lets
// `jit_effective_stacksize_delta(op, stackok)` fold to a constant, so the
// per-op stacksize update carries no residual call. It costs a green key per
// flip, but the distinct (pc, stackok) merge points a program reaches are few.

extern crate majit_ir;
#[cfg(feature = "jit")]
extern crate majit_metainterp as majit_meta;
/// The metainterp crate, re-exported so the binary can reach `mc_diag_summary`
/// and the [`majit_meta::JitStats`] fields without its own dependency edge.
#[cfg(feature = "jit")]
pub use majit_metainterp;

pub use aheui_runtime;
pub use aheui_runtime::aheui;
pub use aheui_runtime::io;
pub use aheui_runtime::storage;
pub use aheui_runtime::value;

#[cfg(feature = "jit")]
pub mod jit;
pub use aheui_runtime as runtime;
pub use aheui_runtime::{Val, ahsembler};

/// Ordinary execution of the same portal, with tracing disabled.
pub mod interp {
    pub fn mainloop(program: &super::aheui::Program) -> super::Val {
        super::mainloop(program, None)
    }
}
#[cfg(feature = "jit")]
use majit_meta::{bh_debug_enabled, spdiag_enabled};
#[cfg(not(feature = "jit"))]
fn spdiag_enabled() -> bool {
    false
}
#[cfg(not(feature = "jit"))]
fn bh_debug_enabled() -> bool {
    false
}
#[cfg(not(feature = "jit"))]
macro_rules! jit_merge_point {
    ($($tokens:tt)*) => {};
}
#[cfg(not(feature = "jit"))]
macro_rules! can_enter_jit {
    ($($tokens:tt)*) => {};
}
#[cfg(feature = "jit")]
type BandArray = majit_metainterp::virt_array::VirtArray<i64>;
#[cfg(not(feature = "jit"))]
type BandArray = Vec<i64>;
fn band_array(value: i64, len: usize) -> BandArray {
    #[cfg(feature = "jit")]
    {
        majit_metainterp::virt_array::VirtArray::filled(value, len)
    }
    #[cfg(not(feature = "jit"))]
    {
        vec![value; len]
    }
}

/// JIT threshold, with a MAJIT_THRESHOLD startup override.
#[cfg(feature = "jit")]
pub fn jit_threshold() -> u32 {
    std::env::var("MAJIT_THRESHOLD")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(majit_metainterp::jit::PARAMETERS.threshold)
}

/// The JIT trace budget, in recorded ops.
///
/// Well above majit's `DEFAULT_TRACE_LIMIT`, and above what the corpus's
/// largest whole-program trace records, so those programs compile one loop
/// with no abort. It is not the fastest setting — a trace that large costs
/// more to optimize and assemble than the run earns back — but it is the one
/// the committed `jitstats` baselines were recorded under, and lowering it
/// moves their `loops_aborted` and `guard_failures` floors.
///
/// [`trace_limit`] exposes an override for sweeping it.
pub const TRACE_LIMIT: u32 = 70000;

/// Trace budget, with a MAJIT_TRACE_LIMIT startup override.
#[cfg(feature = "jit")]
pub fn trace_limit() -> u32 {
    std::env::var("MAJIT_TRACE_LIMIT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(TRACE_LIMIT)
}

/// Optional startup override for the driver's trace_eagerness parameter.
#[cfg(feature = "jit")]
fn trace_eagerness_override() -> Option<i64> {
    std::env::var("AHEUI_TRACE_EAGERNESS")
        .ok()
        .and_then(|v| v.parse().ok())
}

/// `--jit name=value,...` pairs, staged by [`set_user_jit_params`] before
/// [`mainloop`] builds the driver that consumes them.
#[cfg(feature = "jit")]
static USER_JIT_PARAMS: std::sync::Mutex<UserJitParams> = std::sync::Mutex::new(UserJitParams {
    driver: Vec::new(),
    stack_cap: None,
});

#[cfg(feature = "jit")]
#[derive(Default)]
struct UserJitParams {
    driver: Vec<String>,
    stack_cap: Option<usize>,
}

/// Stage the tunable JIT parameters from a user-supplied string in the
/// `rlib/jit.py set_user_param` format, parsed by majit. The Aheui extension
/// `stack_cap` sizes the band ring before state allocation; its power-of-two
/// constraint belongs here. The CLI may handle `off` by selecting the naive
/// interpreter; the shared parser also supports disabling a live JIT driver.
#[cfg(feature = "jit")]
pub fn set_user_jit_params(text: &str) -> Result<(), String> {
    let staged = parse_user_jit_params(text)?;
    let mut params = USER_JIT_PARAMS.lock().unwrap();
    params.driver.extend(staged.driver);
    if staged.stack_cap.is_some() {
        params.stack_cap = staged.stack_cap;
    }
    Ok(())
}

#[cfg(feature = "jit")]
fn parse_user_jit_params(text: &str) -> Result<UserJitParams, String> {
    let mut staged = UserJitParams::default();
    let mut params = majit_metainterp::jit::PARAMETERS;
    if text == "off" || text == "default" {
        majit_metainterp::jit::set_user_param(&mut params, text).map_err(|err| err.to_string())?;
        staged.driver.push(text.to_string());
        return Ok(staged);
    }
    for part in text.split(',') {
        let part = part.trim_matches(' ');
        if let Some(value) = part.strip_prefix("stack_cap=") {
            let cap = value
                .trim()
                .parse::<usize>()
                .map_err(|_| format!("stack_cap is not a number: `{value}`"))?;
            if !cap.is_power_of_two() || cap < 2 {
                return Err(format!("stack_cap must be a power of two >= 2: {cap}"));
            }
            staged.stack_cap = Some(cap);
        } else {
            // off/default are whole-string commands, never list entries.
            if !part.contains('=') {
                return Err(format!("--jit parameter without '=': `{part}`"));
            }
            majit_metainterp::jit::set_user_param(&mut params, part)
                .map_err(|err| err.to_string())?;
            staged.driver.push(part.to_string());
        }
    }
    Ok(staged)
}

#[cfg(feature = "jit")]
#[cfg(test)]
mod user_jit_param_tests {
    use super::parse_user_jit_params;

    #[test]
    fn driver_order_and_last_stack_cap_are_independent() {
        let params = parse_user_jit_params(
            "threshold=10,stack_cap=4,enable_opts=all,stack_cap=8,threshold=20",
        )
        .unwrap();
        assert_eq!(
            params.driver,
            ["threshold=10", "enable_opts=all", "threshold=20"]
        );
        assert_eq!(params.stack_cap, Some(8));
    }

    #[test]
    fn driver_commands_do_not_set_aheui_capacity() {
        for text in ["off", "default"] {
            let params = parse_user_jit_params(text).unwrap();
            assert_eq!(params.driver, [text]);
            assert_eq!(params.stack_cap, None);
        }
        assert!(parse_user_jit_params("stack_cap=8,unknown=1").is_err());
    }
}

/// The last [`mainloop`] run's cumulative JIT counters.
///
/// `mainloop` owns its `JitDriver` for the length of the run and drops it on
/// return, and the binary `process::exit`s on the value `mainloop` returns —
/// so a caller that wants the counters has no live driver to ask. `mainloop`
/// publishes a snapshot here just before it returns, and
/// [`last_jit_stats`] reads it back.
#[cfg(feature = "jit")]
static LAST_JIT_STATS: std::sync::Mutex<Option<majit_meta::JitStats>> = std::sync::Mutex::new(None);

/// The `Counters.ABORT_*` breakdown behind [`last_jit_stats`]'s
/// `loops_aborted`.
///
/// `JitStats` carries only the total, and the profiler's own
/// `print_stats` is behind `MAJIT_LOG` — which on a workload like
/// `pi.jinseo` (35s, 814 aborts) is unusably slow, so the counts it holds were
/// unreachable in practice. They are the statistic that says *why* a trace was
/// given up, so the snapshot rides along with the totals.
#[cfg(feature = "jit")]
static LAST_ABORT_REASONS: std::sync::Mutex<Option<majit_meta::jitprof::JitProfilerSnapshot>> =
    std::sync::Mutex::new(None);

/// The counters the most recent [`mainloop`] finished with, or `None` if the
/// JIT interpreter has not run in this process.
#[cfg(feature = "jit")]
pub fn last_jit_stats() -> Option<majit_meta::JitStats> {
    LAST_JIT_STATS.lock().unwrap().clone()
}

/// The profiler snapshot the most recent [`mainloop`] finished with.
#[cfg(feature = "jit")]
pub fn last_abort_reasons() -> Option<majit_meta::jitprof::JitProfilerSnapshot> {
    LAST_ABORT_REASONS.lock().unwrap().clone()
}

#[cfg(feature = "jit")]
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

/// Trace entries, host trace-module materializations, and materializations
/// served from the byte-identical module cache — all counted inside the guest.
///
/// A wasm host cannot report the first of these: a published trace is called
/// straight through the shared function table (its handle IS the table slot),
/// so entering one crosses no import the host could count.
#[cfg(all(feature = "jit", target_arch = "wasm32"))]
pub fn wasm_jit_counts() -> (u64, u64, u64) {
    (
        majit_backend_wasm::jit_execute_count(),
        majit_backend_wasm::jit_compile_count(),
        majit_backend_wasm::jit_compile_cache_hits(),
    )
}

/// The wasm backend's own bridge/inline decline census, as `label=value` for
/// every tally that fired.
///
/// The backend counts these in a static array and the labels are its own, so
/// joining them here keeps a decline readable without a second place to keep
/// the names in step. Zero rows are dropped: the array is a census of rare
/// declines, and printing 57 zeroes hides the one that is not.
#[cfg(all(feature = "jit", target_arch = "wasm32"))]
pub fn wasm_bridge_diag_summary() -> String {
    majit_backend_wasm::BRIDGE_DIAG_LABELS
        .iter()
        .enumerate()
        .filter_map(|(i, label)| match majit_backend_wasm::bridge_diag(i) {
            0 => None,
            v => Some(format!("{label}={v}")),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn init_gc_subsystem() {
    bigint_gc::init();
    #[cfg(all(feature = "jit", target_arch = "wasm32", feature = "wasm-host"))]
    majit_backend_wasm::install_residual_host_call();
    #[cfg(all(feature = "jit", target_arch = "wasm32"))]
    {
        // The direct-residual-call lowering picks a callee's wasm type from the
        // IR: a word-typed argument becomes `i64`. Not every helper registered
        // in `calls` below answers to that — some still take pool pointers,
        // node pointers and opcode indices as `usize`, and a `usize` is `i32`
        // on wasm32. `call_indirect` type-checks its callee, so a call lowered
        // that way would trap rather than quietly pass the wrong width.
        //
        // So name the ones that are word-spelled and let the rest keep the
        // trampoline, which reads each callee's declared type first. Vouching
        // for too few costs a host round trip per call; vouching for one that
        // is not word-spelled costs a trap, so a helper joins the list only
        // once its own signature says `i64`.
        majit_backend_wasm::set_faithful_residual_call_addrs(&word_abi_residual_addrs());
        majit_backend_wasm::codegen::set_residual_call_abi(
            majit_backend_wasm::codegen::ResidualCallAbi::Vouched,
        );
    }
}

/// Residual callees whose parameters and result are every one of them a
/// machine word spelled `i64`, so a compiled trace can call them directly.
///
/// The static assertions below are the check: each names the exact signature
/// the trace will emit a call for, so a helper that grows a `usize` parameter
/// stops compiling instead of starting to trap.
#[cfg(all(feature = "jit", target_arch = "wasm32"))]
fn word_abi_residual_addrs() -> Vec<i64> {
    const _: extern "C" fn(i64) = jit_write_number;
    const _: extern "C" fn(i64) = jit_write_utf8;
    const _: extern "C" fn() -> i64 = jit_read_utf8;
    const _: extern "C" fn() -> i64 = jit_read_number;
    const _: fn() = jit_output_flush;
    const _: extern "C" fn(i64) -> Val = jit_tag_val;
    const _: extern "C" fn(i64) -> i64 = jit_tag_word;
    const _: fn() -> i64 = jit_bigint_mode;
    const _: extern "C" fn() -> i64 = jit_band_count;
    const _: extern "C" fn() -> i64 = jit_cap;

    let mut addrs: Vec<i64> = vec![
        jit_write_number as *const () as usize as i64,
        jit_write_utf8 as *const () as usize as i64,
        jit_read_utf8 as *const () as usize as i64,
        jit_read_number as *const () as usize as i64,
        jit_output_flush as *const () as usize as i64,
        jit_tag_val as *const () as usize as i64,
        jit_tag_word as *const () as usize as i64,
        jit_bigint_mode as *const () as usize as i64,
        jit_band_count as *const () as usize as i64,
        jit_cap as *const () as usize as i64,
    ];
    addrs.extend((jit::artifacts().word_abi_fnaddrs)());
    addrs
}

// Imports required by generated JIT code.

use aheui_runtime::aheui::*;
use aheui_runtime::band as bd;
use aheui_runtime::io as aheui_io;
use aheui_runtime::storage::linkedlist_jit as lj;
use aheui_runtime::storage::{LinkedList, Storage};
use ahsembler::compiler::Program;

use aheui_runtime::value::*;

// `MAJIT_SPDIAG` and `MAJIT_BH_DEBUG` are majit's own gates and are read
// through `majit_meta`, so a run cannot have one half of either enabled.
// This one is aheui's, cached outside hot loops the same way.

#[cfg(feature = "jit")]
fn check_chains_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("AHEUI_CHECK_CHAINS").is_some())
}

/// Cumulative ceiling on the oversized `alloc_zeroed` fallback.
///
/// It applies only there. Node-sized allocations go to the nursery, which
/// self-bounds through its own chunk cap, and a running program allocates far
/// more than this in nodes over its lifetime — each freed and reused — so a
/// cumulative cap on the node path would fail spuriously.
#[cfg(feature = "jit")]
const JIT_ALLOC_LIMIT: usize = 256 * 1024 * 1024;

/// GC allocator for JIT-compiled `New()` ops, delegating to the global nursery
/// so alloc and free share the pool the interpreter path uses.
#[cfg(feature = "jit")]
struct AheuiBlackholeAllocator;

#[cfg(feature = "jit")]
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

    /// `llmodel.py bh_setfield_gc_i` — the store takes the field's
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

    /// `llmodel.py bh_setfield_gc_r` — pointer width, no size.
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

    /// `llmodel.py bh_setfield_gc_f` — storage width, no size.
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

#[cfg(feature = "jit")]
struct NurseryGcAllocator {
    oversized_allocated: usize,
}

#[cfg(feature = "jit")]
impl NurseryGcAllocator {
    fn new() -> Self {
        Self {
            oversized_allocated: 0,
        }
    }
}

#[cfg(feature = "jit")]
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
    fn nursery_recycle_list_addr(&self) -> usize {
        aheui_runtime::storage::nursery_free_list_addr()
    }
    fn nursery_recycle_window_addr(&self) -> usize {
        aheui_runtime::storage::nursery_recycle_window_addr()
    }
    fn max_nursery_object_size(&self) -> usize {
        usize::MAX
    }
}

#[cfg(feature = "jit")]
fn register_aheui_copying_gc_jit_roots() {
    majit_gc::shadow_stack::register_libc_jitframe_tracer(
        majit_backend::jitframe::jitframe_custom_trace,
    );
    let hook: aheui_runtime::storage::NodeRootWalkHook = walk_aheui_jit_node_roots;
    aheui_runtime::storage::NODE_ROOT_WALK_HOOK
        .store(hook as usize, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "jit")]
fn walk_aheui_jit_node_roots(
    visit_node_slot: &mut dyn FnMut(*mut *mut aheui_runtime::storage::linkedlist::Node),
) {
    majit_gc::shadow_stack::walk_jit_roots(|slot| {
        visit_node_slot(
            slot as *mut majit_ir::GcRef as *mut *mut aheui_runtime::storage::linkedlist::Node,
        );
    });

    // This hook enumerates JIT stack and deadframe node references. MiniMarkGC walks
    // Aheui bignums held by majit's extra-root set separately.
}

static SPDIAG_TRACE_OPS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

thread_local! {
    // Storage snapshot captured before trace walking. Recovery prints it next
    // to the restored state to distinguish a stale state field from a stale
    // optimized heap write.
    static PRE_WALK_SNAPSHOT: std::cell::RefCell<String> =
        const { std::cell::RefCell::new(String::new()) };
}

/// Operand words each banded stack pool keeps out of its node chain.
///
/// A pool's band is a ring holding its **top** `cap` elements: element at
/// absolute height `h` lives at `pool * cap + (h & cap_mask)`. Anything below
/// that stays in the node chain, so a push past `cap` evicts one word to the
/// chain and a pop below it refills one word back — into the slot the pop just
/// vacated, since `h` and `h - cap` share a ring slot. Holding the top rather
/// than the bottom is what keeps both operands of a binary op on the same tier
/// at every depth.
///
/// A power of two, so the ring index is a mask. It is the size a program gets
/// when neither the static bound nor the measurement names one — an
/// input-dependent program, whose depth is a property of the input as much as
/// of the code. Running `aheui.aheui` is that case, and its own scratch pool
/// stays eight deep whichever guest it interprets, so `--jit=stack_cap=8` is
/// worth naming there; the default has to cover the programs that go deeper
/// instead. 64 covers the depth programs actually reach without paying for
/// slots they never use — the declared length
/// of a virtualizable array is what its per-compile cost scales with, and every
/// deopt decodes and writes the whole array back. Programs whose stacks stay
/// shallow pay that full length per guard failure for slots they never fill,
/// which is why the size is selectable per run rather than fixed.
pub const CAP_DEFAULT: usize = 64;
/// The ring size of the running program: [`CAP_DEFAULT`], or the power of two
/// `--jit=stack_cap=N` or `AHEUI_CAP` named — the argument outranks the
/// variable. Stored once before the state arrays are sized and read
/// through [`cap_selected`] / [`jit_cap`] everywhere else, so the array length,
/// the traced index arithmetic and the root walk cannot disagree.
static CAP_SELECTED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(CAP_DEFAULT);
fn cap_selected() -> usize {
    CAP_SELECTED.load(std::sync::atomic::Ordering::Relaxed)
}
/// The ring size as a green — same footing as [`jit_band_count`].
extern "C" fn jit_cap() -> i64 {
    CAP_SELECTED.load(std::sync::atomic::Ordering::Relaxed) as i64
}

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
    // The armed count, not `vals.len() / cap`: a pool at or above it never
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
        let cap = cap_selected();
        let held = depth.min(cap);
        for step in 0..held {
            let slot = pool * cap + ((depth - 1 - step) & (cap - 1));
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

/// Select a smaller ring from static bounds, exact measurement or a bounded
/// prefix estimate. Estimates may spill to the node chain later in the run.
fn proven_cap(program: &Program) -> Option<usize> {
    let bands = banded_pool_count(program);
    // Widening at the largest ring this will ever pick keeps the lattice as
    // short as the answer needs to be: a pool deeper than the default gets
    // the default anyway, so proving how much deeper is work with no
    // consumer.
    let bounds = ahsembler::depth::max_pool_depths_up_to(program, CAP_DEFAULT as u32);
    let mut deepest: usize = 0;
    let mut measured: Option<[u32; STORAGE_COUNT]> = None;
    for (pool, bound) in bounds.iter().enumerate().take(bands) {
        if pool == VAL_QUEUE || pool == VAL_PORT {
            continue;
        }
        let bound = match bound {
            ahsembler::depth::DepthBound::Bounded(b) => *b as usize,
            // Measurement handles bounds the static lattice cannot prove.
            // A bounded prefix is only an estimate; later pushes may spill.
            ahsembler::depth::DepthBound::Unbounded => {
                if measured.is_none() {
                    measured = Some(depths_by_running(program)?);
                }
                measured.unwrap()[pool] as usize
            }
        };
        deepest = deepest.max(bound);
    }
    let cap = deepest.next_power_of_two().max(2);
    (cap < CAP_DEFAULT).then_some(cap)
}

/// Measure an input-free run, then try a bounded prefix using buffered stdin.
/// Prefix peaks are estimates; ring overflow must retain the spill path.
/// Do not read interactive input ahead of the real interpreter.
fn depths_by_running(program: &Program) -> Option<[u32; STORAGE_COUNT]> {
    const STEP_BUDGET: u64 = 250_000;
    if let Some(exact) = ahsembler::depth::measured_pool_depths(program, STEP_BUDGET) {
        return Some(exact);
    }
    if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
        return None;
    }
    // The interpreter's own buffer, over the same bytes the run will read:
    // observing through a second decoder would walk branches the real run
    // does not.
    let mut input = aheui_runtime::io::InputBuffer::new();
    let mut read = |as_number: bool| {
        if as_number {
            input.read_number()
        } else {
            input.read_utf8()
        }
    };
    ahsembler::depth::observed_pool_depths(program, STEP_BUDGET, &mut read)
}

/// Band one pool by default. A numeric AHEUI_BANDS selects a clamped count;
/// a nonnumeric value derives the count from the program's selected pools.
fn banded_pool_count(program: &Program) -> usize {
    // One band by default: a declared slot is paid for in compile time and in
    // per-iteration work whether or not the program ever selects its pool, so
    // banding past the first pool costs more than it wins. `AHEUI_BANDS`
    // overrides the count, so both arms come from one binary.
    let Ok(text) = std::env::var("AHEUI_BANDS") else {
        return 1;
    };
    match text.parse::<usize>() {
        // Floored at one for the same reason the derived count below is: pool
        // 0 is banded whether or not the program names it, and a zero-length
        // band array trips the metainterp's `vable_size > 0`. The knob that
        // takes banding out of the picture is `AHEUI_BAND_ARMS`, which leaves
        // the array declared and every band arm unreachable.
        Ok(count) => return count.clamp(1, VAL_QUEUE),
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

/// Trace-time state for the Aheui JIT: the reds `mainloop` carries.
///
/// `aheui.py` keeps them as `stacksize`, `storage` and `selected`, where
/// `selected` is the polymorphic storage object itself. Two of the three are
/// spelled differently here:
///
/// * `selected: usize` is the index, not the object. Rust cannot hold a
///   mutable borrow into `self.storage` across subsequent `self.storage`
///   mutations, so the object is re-fetched through [`Storage::dispatch_mut`]
///   on every use.
///
/// * `selected_ref` is that object at the machine level: the raw pointer to
///   the base the selected storage embeds, which is where `head` and `size`
///   are declared and where the backend reads them at fixed byte offsets. It
///   sits beside `selected` because `refresh_selected_ref` has to run whenever
///   `selected` changes.
struct AheuiState {
    storage: Storage,
    /// The top operand words of every pool, `bands * cap` of them.
    ///
    /// Index `i * cap + (h & cap_mask)` is pool `i`'s element at absolute
    /// height `h`, for the heights the window currently owns. Declared
    /// `[int; virt]` so an access promotes its index and answers out of the
    /// virtualizable's boxes rather than memory.
    vals: BandArray,
    /// Total element count of each pool, window and node chain together.
    ///
    /// Authoritative for every pool except the selected one, whose live count
    /// is `sp` — `OP_SEL` writes the outgoing pool's `sp` back here before it
    /// reads the incoming pool's.
    depths: BandArray,
    selected: usize,
    stacksize: i64,
    /// `stacksize` as an unsigned ring index, refreshed once per dispatch.
    ///
    /// The dispatch pre-applies the opcode's declared delta, so inside an arm
    /// this is the depth the pool will have *after* the op; each arm reaches
    /// its operands at a fixed literal offset from it.
    sp: usize,
    /// `state.storage.pools[selected]` packed as `usize`; the `state_fields`
    /// block below tracks it as `ref(ListBase)`.
    selected_ref: usize,
    /// `&mut state.storage as *mut Storage` packed as `usize` — the base the
    /// contiguous `pools: [*mut ListBase; N]` array is read off, and the sole
    /// carrier of the storage pointer.
    storage_ref: usize,
}

impl AheuiState {
    #[inline(always)]
    fn refresh_selected_ref(&mut self) {
        // rpaheui/aheui/aheui.py: selected = storage[idx].
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
    #[cfg(feature = "jit")]
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
    /// A band holds a pool's top `cap` words, so its chain holds exactly the
    /// rest: `depth - cap` of them, or none while the band is not yet full.
    /// [`Storage::check_chains`] cannot see a break in that split, because
    /// each chain stays internally consistent across it; the program only
    /// notices opcodes later, when a refill pops a chain the depth said was
    /// not empty.
    #[cfg(feature = "jit")]
    fn check_bands(&self) -> Option<String> {
        let bands = BAND_COUNT.load(std::sync::atomic::Ordering::Relaxed);
        for pool in 0..bands {
            let depth = if pool == self.selected {
                self.stacksize
            } else {
                self.depths[pool]
            };
            let want = (depth - cap_selected() as i64).max(0);
            let have = self.storage.len_at(pool) as i64;
            if want != have {
                return Some(format!(
                    "pool {pool} depth {depth} leaves {want} below the band, chain holds {have}"
                ));
            }
        }
        None
    }

    #[cfg(feature = "jit")]
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
        // only while the pool is unbanded. A banded pool keeps its top `cap`
        // words in `vals`, which the chain cannot see, and the number of them
        // it holds is `min(stacksize, cap)` — so the chain under-reports by
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

/// The `out` byte range `@@@WALKEMIT` reports, from `MAJIT_SPDIAG_FROM` /
/// `MAJIT_SPDIAG_TO`.
///
/// A window rather than the whole run, because a program emitting hundreds of
/// kilobytes buries the one write being chased. Which window is interesting
/// depends on where the run being diagnosed first diverged, so it is a knob
/// rather than a constant.
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

/// io-shim target for `aheui_io::output_write_number(&r)`.
///
/// The recorded call carries the raw `Val` word — a tagged smallint or a
/// boxed-bigint pointer — so reconstruct the `Val` and route it through the
/// interpreter's own decode and output buffer. Writing the raw word through
/// majit's own writer would print the tagged payload into a second buffer that
/// interleaves wrongly with interpreter output.
extern "C" fn jit_write_number(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    if bh_debug_enabled() {
        eprintln!("[io-debug] jit_write_number raw={value}");
    }
    walk_emit_log("num", value);
    aheui_io::output_write_number(&v);
}

/// io-shim target for `aheui_io::output_write_utf8(&r)`. Same reconstruction
/// as [`jit_write_number`].
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
#[cfg(feature = "jit")]
fn __majit_pipeline_jitcode(name: &str) -> std::sync::Arc<majit_metainterp::JitCode> {
    (jit::artifacts().jitcode)(name)
        .unwrap_or_else(|| panic!("pipeline jitcode for '{name}' not found"))
}

#[cfg(feature = "jit")]
fn __majit_pipeline_liveness_prebuild(assembler: &mut majit_metainterp::Assembler) {
    (jit::artifacts().prebuild_liveness)(assembler);
}

// Index-keyed dynamic storage dispatch — rpaheui parity for
// `selected.METHOD()` (aheui.py). The target index is a per-opcode
// operand rather than the promoted `selected_ref`, so the polymorphic
// `dispatch_mut(target)` is bundled into one residual call that dispatches on
// the live index at run time (Stack / Queue / Port) instead of a JIT-green
// 3-way `is_port`/`is_queue` branch. This keeps the recorded trace
// structurally identical to rpaheui's single polymorphic call site, so the
// optimiser never emits a contradictory `guard_value(selected)` and the loop
// closes through the real back-edge.
//
// Only push and dup survive, and for two independent reasons. OP_MOV names its
// target with a per-opcode operand rather than `selected` (aheui.py
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
/// the new `selected_ref`. `storage[idx]` (aheui.py) is a list
/// getitem returning the Stack/Queue/Port object reference; the result is
/// carried in the ref register bank, so the call is recorded ref-returning
/// (`residual_ref_cannot_raise_wrapped`). `#[dont_look_inside_cannot_raise]`
/// emits the `__majit_call_policy_*` trace/concrete targets the wrapped-ref
/// lowering reads; the `usize` carrier round-trips the pointer bits.
/// Cached `MAJIT_HANDOFF` flag. Elidable so a false result folds the
/// in-loop diagnostic away instead of recording OnceLock/atomics every
/// opcode.
#[cfg_attr(feature = "jit", majit_macros::dont_look_inside_cannot_raise)]
fn handoff_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("MAJIT_HANDOFF").is_some())
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn maybe_log_handoff(program: &Program, pc: usize, state: &AheuiState) {
    if !handoff_enabled() {
        return;
    }
    let out = aheui_io::output_total_bytes();
    static HLATCH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static HCOUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static HAT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static HCAP: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    let at: u64 = *HAT.get_or_init(|| {
        std::env::var("MAJIT_HANDOFF_AT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2085)
    });
    let hcap: u32 = *HCAP.get_or_init(|| {
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
        if n < hcap {
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

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn maybe_spdiag_pre_op(state: &AheuiState, op: u8) {
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
        if (1000..=3000).contains(&out) {
            eprintln!(
                "@@@MAINEMIT op={op} out={out} selected={}{}",
                state.selected,
                state.spdiag_dump_stacks(),
            );
        }
    }
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside)]
fn maybe_spdiag_resume_op(pc: usize, op: u8, stackok: bool, state: &AheuiState) {
    if SPDIAG_TRACE_OPS.load(std::sync::atomic::Ordering::Relaxed) == 0 {
        return;
    }
    SPDIAG_TRACE_OPS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    eprintln!(
        "@@@SPDIAG resume-op pc={pc} op={op} stacksize={} selected={} stackok={stackok} out={}{}",
        state.stacksize,
        state.selected,
        aheui_io::output_total_bytes(),
        state.spdiag_dump_stacks(),
    );
}

#[cfg_attr(feature = "jit", majit_macros::dont_look_inside_cannot_raise)]
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
// - storage = linked list stacks with virtualizable top-of-stack bands
// - selected = red variable dispatched through the live storage index
// - push/pop = Node allocation/deallocation (OptVirtualize target)

#[majit_macros::jit_interp(
    trace_cfg = (feature = "jit"),
    state = AheuiState,
    env = Program,
    // RPython parity: rpaheui/aheui/aheui.py reds=['stacksize','storage','selected'].
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
        // `pyjitpl.py _get_arrayitem_vable_index` promotes the index
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
        // The selected list's object reference. Carried in the ref register
        // bank as a genuine `InputArgRef` so it can be promoted with
        // `ref_guard_value` and passed to the monomorphic storage helpers as a
        // ref-kind arg (`JitCallArg::reference`).
        //
        // The type is the base every storage embeds — what a Stack, a Queue
        // and a Port have in common. Naming one of the three instead would
        // make every access through this field a reinterpretation of one
        // storage as another.
        selected_ref: ref(aheui_runtime::storage::linkedlist::ListBase),
        // Base the `pools: [*mut ListBase; N]` array is read off.
        // Declared `ref(Storage)` + listed in `pool_arrays` so OP_SEL's
        // `selected_ref = jit_sel_get_ref(storage_ref, selected)` lowers to a
        // re-producible `getarrayitem_gc_r` on this base. aheui.py carries
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
        jit_tag_word => elidable_int_cannot_raise,
        // The band arms hold the value as its word, which is what the output
        // shims already take, so they name the shim rather than going back
        // through a `Val` to reach it.
        jit_write_number => residual_void,
        jit_write_utf8 => residual_void,
        jit_bigint_mode => elidable_int_cannot_raise,
        jit_band_count => elidable_int_cannot_raise,
        jit_cap => elidable_int_cannot_raise,
        handoff_enabled => elidable_int_cannot_raise,
        maybe_log_handoff => residual_void_cannot_raise,
        maybe_spdiag_pre_op => residual_void_cannot_raise,
        maybe_spdiag_resume_op => residual_void_cannot_raise,
        spdiag_enabled => elidable_int_cannot_raise,
        // Method-call results consumed as values are lowered through
        // `lower_method_call_value`.
        Program::get_req_size => elidable_int_cannot_raise,
        Program::get_op => elidable_int_cannot_raise,
        Program::get_label => elidable_int_cannot_raise,
        Program::get_operand => elidable_int_cannot_raise,
        // Storage access specializes on the selected kind.
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
        jit_free_node => concrete_only_void,
        val_add => elidable_int,
        val_sub => elidable_int,
        val_mul => elidable_int,
        val_div => elidable_int,
        val_mod => elidable_int,
        val_from_i32 => elidable_int_cannot_raise,
    },
    // Residual storage mutators that change `size` or the `head` chain pointer.
    //
    // A helper whose stores reach the trace needs no entry: the in-trace
    // `setfield_gc` invalidates the matching heapcache entry itself, which
    // covers the macro-inlined Stack helpers, the graph-pipeline pop/swap
    // jitcodes, and the pops the arithmetic arms inline as head/next/size
    // stores. The rest lower to opaque residual calls, which carry an empty
    // write-set by default: a size-only declaration can leave a stale head
    // cached after a residual pop, and a head-only declaration a stale size,
    // so both fields are declared. The lists are conservative supersets —
    // an extra reload is harmless, a missing invalidation is not.
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
    // rpaheui/aheui/aheui.py: greens=['pc','stackok','is_queue','program'].
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
    // the same pre-merge-point walker as `is_queue`. `bands` and `cap` stay
    // as no-arg `elidable_int_cannot_raise` calls (`jit_band_count` /
    // `jit_cap`); promoting them at every merge only re-guards values that
    // cannot change for the life of the run.
    //
    // `stackok` must stay green. `jit_effective_stacksize_delta(op, stackok)`
    // folds only when the polarity is a merge-point constant; dropping it
    // left logo's body at ~21k residual ops (~10s) and aheui.aheui(99dan)
    // about 3× slower. `is_queue` is already a function of the red
    // `selected`, but promoting it does not change the aheui.aheui
    // guard-failure count (still ~11k), so it stays with rpaheui.
    greens = [pc, stackok, is_queue, bm, program],
    recover = refresh_state_from_storage,
    switch_dispatch = true,
    native_tag_small = { jit_retag_small },
    // The band holds a value as its word, and a `Val` IS that word under every
    // mode, so the two directions of the band boundary are register-level
    // identities. Left as calls they are three `blr`s an iteration for a
    // reinterpretation the trace does not need to see.
    native_identity = { jit_win_store, jit_tag_val_raw, jit_tag_word_raw },
)]
// This body is the `jit_interp` macro's INPUT, so its control flow is lowered,
// not merely read: a change to a conditional here is a change to the generated
// code, and the crate still compiles cleanly when the result is a jitcode the
// trace installer rejects. Run a real program end to end after reshaping one.
#[cfg_attr(not(feature = "jit"), allow(unused_variables, unused_assignments))]
pub fn mainloop(program: &Program, threshold: Option<u32>) -> Val {
    bigint_gc::init();
    #[cfg(feature = "jit")]
    if threshold.is_some() {
        init_gc_subsystem();
    }

    // Dual mode: run on raw machine words until one operation overflows one.
    // Ahead of `Storage::new()` below, because the queue builds a sentinel
    // value and the conversion walks it — it has to be written in the mode it
    // will be read in.
    aheui_runtime::value::start_in_raw_mode();

    #[cfg(feature = "jit")]
    let mut driver = {
        let mut driver: majit_meta::JitDriver<AheuiState> =
            majit_meta::JitDriver::new(threshold.unwrap_or(0));

        // Register the nursery-backed GC allocator for JIT-compiled New() ops.
        // Linked-list nodes (Node) share the interpreter's nursery pool, so
        // compiled `New` allocation must route through this allocator rather
        // than `libc::malloc`; the copying collector only knows how to forward
        // nodes allocated from nursery chunks.
        let gc = Box::new(NurseryGcAllocator::new());
        driver.meta_interp_mut().backend_mut().set_gc_allocator(gc);
        driver.meta_interp_mut().backend_mut().set_new_via_gc(true);
        // `resume.py` materializes virtual headerless `Node` values during
        // guard-failure deoptimization. The blackhole allocator must do the same or
        // a virtual node becomes a null head while the recorded size stays nonzero.
        driver.register_blackhole_allocator(AheuiBlackholeAllocator);

        driver.set_param("trace_limit", trace_limit() as i64);
        if let Some(eagerness) = trace_eagerness_override() {
            driver.set_param("trace_eagerness", eagerness);
        }

        // ALL_OPTS minus `unroll`. Peeling the preamble buys this driver
        // little — the loop body carries almost nothing loop-invariant to
        // hoist, because every operation reads or writes the mutable selected
        // stack — and the two-phase walk of a ~27k-op logo recording already
        // costs more than the 120 ms wall this host has recorded. Unread
        // JUMP constants that would specialize the single label are rewritten
        // back onto the LABEL boxes (`despecialize_unread_closing_jump_slots`),
        // so the simple-loop entry stays general without a peel.
        // `AHEUI_ENABLE_OPTS` overrides the list.
        const ENABLE_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap";
        match std::env::var("AHEUI_ENABLE_OPTS") {
            Ok(text) => driver.set_param_enable_opts(&text),
            Err(_) => driver.set_param_enable_opts(ENABLE_OPTS),
        }

        // `--jit` pairs land after the environment-derived settings above: an
        // argument on the command line outranks an ambient variable. `stack_cap`
        // is not a driver parameter — it is consumed below, before the state
        // arrays are sized.
        {
            let params = USER_JIT_PARAMS.lock().unwrap();
            for text in &params.driver {
                majit_metainterp::jit::set_user_param(&mut driver, text)
                    .expect("JIT parameters were validated before staging");
            }
        }
        if threshold.is_none() {
            majit_metainterp::jit::set_user_param(&mut driver, "off")
                .expect("off is a valid JIT command");
        }
        driver
    };
    #[cfg(feature = "jit")]
    let user_cap = USER_JIT_PARAMS.lock().unwrap().stack_cap;
    #[cfg(not(feature = "jit"))]
    let user_cap: Option<usize> = None;

    let mut pc: usize = 0;
    // Resolved before the arrays are sized: everything downstream — the
    // traced index arithmetic, the deopt writeback length, the root walk —
    // reads the stored value, so this store is the single decision point.
    // A value that is not a power of two would break the `& (cap - 1)` ring
    // indexing, so it is refused rather than rounded. 1 is refused too: a
    // one-slot ring evicts on every push and refills on every pop, which
    // multiplies runtime past any measured budget.
    if let Some(cap) = user_cap.or_else(|| {
        std::env::var("AHEUI_CAP")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|c| c.is_power_of_two() && *c >= 2)
    }) {
        CAP_SELECTED.store(cap, std::sync::atomic::Ordering::Relaxed);
    } else if let Some(cap) = proven_cap(program) {
        CAP_SELECTED.store(cap, std::sync::atomic::Ordering::Relaxed);
    }
    // rpaheui/aheui/aheui.py: reds=['stacksize','storage','selected']
    let mut state = AheuiState {
        storage: Storage::new(),
        vals: band_array(0i64, banded_pool_count(program) * cap_selected()),
        // Sized like `vals`, not `STORAGE_COUNT`: every read and write of a
        // depth is gated on `< bands`, and `bands` never exceeds the banded
        // pool count, so a slot past it is unreachable. A declared slot is a
        // loop-carried value the optimizer carries and the trace prologue
        // reloads whether or not any opcode can index it.
        depths: band_array(0i64, banded_pool_count(program)),
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
        Some(limit) => limit.min(state.vals.len() / cap_selected()),
        None => state.vals.len() / cap_selected(),
    };
    BAND_COUNT.store(arms, std::sync::atomic::Ordering::Relaxed);

    // Storage was moved into state — refresh self-referencing pointers.
    state.storage.refresh_pools();
    state.storage_ref = &mut state.storage as *mut Storage as usize;
    state.refresh_selected_ref();
    // Register the storage as the nursery collector's root set. `state` is a
    // stationary local for the rest of the mainloop, so the pointer stays
    // valid; compiled traces do not recycle individual nodes; dead nodes are
    // reclaimed by `Nursery::collect` walking these roots.
    // SAFETY: `state` stays at this address for the rest of the mainloop.
    let _roots = unsafe { aheui_runtime::storage::GcRootsGuard::new(&mut state.storage) };
    #[cfg(feature = "jit")]
    register_aheui_copying_gc_jit_roots();
    // Publish the storage for the output shims' diagnostic dump. It is taken
    // from the same stationary `state` as the roots above, so the pointer does
    // not change for the life of the run: it is published here, once, next to
    // the other publications of it, rather than before every merge point.
    WALK_STORAGE_PTR.with(|c| c.set(&state.storage as *const Storage as usize));

    // RPython `warmspot.py` `make_jitcodes() →
    // finish_setup(codewriter)` parity for state-field JIT: register the
    // canonical `(live_i, live_r, live_f)` liveness slots and seed
    // `MetaInterpStaticData` so blackhole resume's
    // `BlackholeInterpreter::get_current_position_info` (which reads
    // `code[pc] == op_live`) recognises the macro-emitted `BC_LIVE`
    // markers. Without this hook `op_live` defaults to `-1` (= u8::MAX
    // post-conversion in `setup_cached_control_opcodes`) and the
    // resume path panics with `missing liveness[N] in JitCode`.
    #[cfg(feature = "jit")]
    if threshold.is_some() {
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
        // Handoff diagnostic (MAJIT_HANDOFF). The enabled flag is elidable
        // so a false result folds this call out of the recorded trace.
        if handoff_enabled() {
            maybe_log_handoff(program, pc, &state);
        }
        let mut stackok = program.get_req_size(pc) as i64 <= state.stacksize;
        // rpaheui/aheui/aheui.py sets `is_queue = (value == VAL_QUEUE)`
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
        // The ring size and its index mask, bound the same way: an arm's
        // `sp >= cap` tier test and `& cap_mask` slot arithmetic fold inside
        // a trace specialised on them.
        let cap = jit_cap() as usize;
        let cap_mask = cap - 1;

        // rpaheui/aheui/aheui.py: jit_merge_point
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
            maybe_spdiag_pre_op(&state, op);
            maybe_spdiag_resume_op(pc, op, stackok, &state);
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

        // rpaheui/aheui/aheui.py: branch ops at dispatch level.
        // lower_match_stmt lowers this into chained guards in the dispatch
        // JitCode. pc = target + continue update the JitCode pc register
        // and BC_GOTO loop_start.
        match op {
            OP_BRPOP1 | OP_BRPOP2 => {
                if !stackok {
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
                // The pop is the same for every storage, and the zero test is
                // one comparison on the popped `Val` either way.
                let pop_word = if state.selected < bands {
                    let top_slot = state.selected * cap + (state.sp & cap_mask);
                    let word = state.vals[top_slot];
                    if state.sp >= cap {
                        state.vals[top_slot] =
                            jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                    }
                    word
                } else {
                    jit_win_store(lj::pop_base_known_nonempty(state.selected_ref))
                };
                // Zero is 0 in raw mode and 1 in tagged mode.
                let zero_word = if bm != 0 {
                    jit_tag_word(0i64)
                } else {
                    jit_tag_word_raw(0i64)
                };
                if pop_word == zero_word {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, bm, program);
                    continue;
                }
            }
            _ => {}
        }

        match op {
            // `selected.<op>()`, branched on the `is_queue` green. The trace
            // specializes per value, so only one branch survives in compiled
            // code: a concrete `stack_*` or `queue_*` helper reached through
            // `selected_ref`, the `ref(Stack)` state scalar pointing at the
            // selected storage. Port falls through to polymorphic dispatch.
            //
            // The banded arm — `selected < bands` — runs the same binary op
            // over the ring instead of the chain. `sp` is the depth after the
            // op, so the operands sit at `sp` and `sp - 1` and the result takes
            // the lower slot, the shape `linkedlist.py` has over the chain. At
            // `sp >= cap` the pop dropped the band's bottom element out of
            // range, so the chain hands the next one back — into the slot the
            // popped operand just vacated, which is the same ring slot.
            OP_ADD => {
                if stackok {
                    if is_queue {
                        if bm != 0 {
                            lj::queue_add(state.selected_ref);
                        } else {
                            lj::queue_add_raw(state.selected_ref);
                        }
                    } else if state.selected < bands {
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_add(r2, r1)
                        } else {
                            bd::band_add_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_sub(r2, r1)
                        } else {
                            bd::band_sub_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_mul(r2, r1)
                        } else {
                            bd::band_mul_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_div(r2, r1)
                        } else {
                            bd::band_div_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_mod(r2, r1)
                        } else {
                            bd::band_mod_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let top_slot = state.selected * cap + (state.sp & cap_mask);
                        if state.sp >= cap {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                    } else {
                        let _popped = lj::pop_base_known_nonempty(state.selected_ref);
                    }
                }
            }
            OP_PUSH => {
                // rpaheui/aheui/aheui.py.
                let value = program.get_operand(pc - 1) as i64;
                let v = if bm != 0 {
                    jit_tag_val(value)
                } else {
                    jit_tag_val_raw(value)
                };
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else if state.selected == VAL_PORT {
                    // linkedlist.py `Port.push` also records
                    // `last_push`, and linkedlist.py `Port.dup`
                    // pushes that instead of the head value. `selected_ref`
                    // is typed `Stack`, so the inline push/dup below write
                    // head/size only and a `push; pop; dup` on the port
                    // duplicates the wrong value. The port takes the
                    // polymorphic residual, matching aheui.py
                    // `selected.METHOD()`. Only push and dup diverge —
                    // `pop`, `swap`, `_get_2_values` and `_put_value` are
                    // the shared `LinkedList` implementations.
                    jit_storage_push(state.storage_ref, state.selected, v);
                } else if state.selected < bands {
                    // `sp` is the depth after the push, so the new word takes
                    // the slot of height `sp - 1`.
                    let free_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                    if state.sp > cap {
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
                        let src_slot = state.selected * cap + ((state.sp - 2) & cap_mask);
                        let free_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let top = state.vals[src_slot];
                        if state.sp > cap {
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
                    // One `swap`, on `LinkedList`, which every storage
                    // inherits unchanged: `head` is declared once, on the base
                    // they embed, so there is no per-storage helper to pick
                    // between.
                    if state.selected < bands {
                        let top_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let second_slot = state.selected * cap + ((state.sp - 2) & cap_mask);
                        let top = state.vals[top_slot];
                        state.vals[top_slot] = state.vals[second_slot];
                        state.vals[second_slot] = top;
                    } else {
                        lj::swap_base_known_two(state.selected_ref);
                    }
                }
            }
            OP_SEL => {
                // rpaheui/aheui/aheui.py: `selected = storage[value];
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
                        let top_slot = state.selected * cap + (state.sp & cap_mask);
                        let word = state.vals[top_slot];
                        if state.sp >= cap {
                            state.vals[top_slot] =
                                jit_win_store(lj::pop_base_known_nonempty(state.selected_ref));
                        }
                        word
                    } else {
                        jit_win_store(lj::pop_base_known_nonempty(state.selected_ref))
                    };
                    let target = program.get_operand(pc - 1) as usize;
                    if target == VAL_QUEUE {
                        // The operand is green, so `jit_sel_get_ref` resolves it
                        // to one concrete list exactly as OP_SEL resolves
                        // `selected`, and the tail-append becomes the same
                        // monomorphic `queue_push` the OP_PUSH arm takes. The
                        // residual it replaces re-dispatched on a target the
                        // trace already knew, and its callee reached the
                        // uninlined `Queue::push`.
                        let target_ref = jit_sel_get_ref(state.storage_ref, target);
                        lj::queue_push(target_ref, jit_tag_val_raw(moved));
                    } else if target == VAL_PORT {
                        // `Port::push` records `last_push`, which no inlined
                        // helper writes, so the port keeps the polymorphic
                        // residual.
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
                        let free_slot = target * cap + (depth & cap_mask);
                        if depth >= cap {
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
                        let r1_slot = state.selected * cap + (state.sp & cap_mask);
                        let r2_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                        let r1 = state.vals[r1_slot];
                        let r2 = state.vals[r2_slot];
                        let __band_word = if bm != 0 {
                            bd::band_cmp(r2, r1)
                        } else {
                            bd::band_cmp_raw(r2, r1)
                        };
                        state.vals[r2_slot] = __band_word;
                        if state.sp >= cap {
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
                        let top_slot = state.selected * cap + (state.sp & cap_mask);
                        let word = state.vals[top_slot];
                        if state.sp >= cap {
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
                        let top_slot = state.selected * cap + (state.sp & cap_mask);
                        let word = state.vals[top_slot];
                        if state.sp >= cap {
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
                    let free_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                    if state.sp > cap {
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
                    let free_slot = state.selected * cap + ((state.sp - 1) & cap_mask);
                    if state.sp > cap {
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
    #[cfg(feature = "jit")]
    publish_jit_stats(
        driver.get_stats(),
        driver.meta_interp().staticdata.profiler.snapshot(),
    );

    // The armed count for the same reason `walk_band_values` takes it: a pool
    // at or above it holds every word in its chain, and its band slots are
    // whatever an earlier arming left there.
    let bands = jit_band_count() as usize;
    let cap = cap_selected();
    let cap_mask = cap - 1;
    let selected = state.selected;
    if selected != VAL_QUEUE && selected != VAL_PORT && selected < bands {
        if state.stacksize > 0 {
            let depth = state.stacksize as usize;
            aheui_runtime::value::val_from_raw_i64(
                state.vals[selected * cap + ((depth - 1) & cap_mask)],
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
