// JIT-enabled Aheui interpreter — graph pipeline + #[jit_interp] macro.
//
// RPython parity: rpaheui/aheui/aheui.py
//   greens = [pc, stackok, is_queue, program]
//   reds   = [stacksize, storage, selected]
//   storage = linked list stacks (no virtualizable arrays)
//
// stackok is a green (rpaheui parity): specialising the trace on it lets
// `jit_effective_stacksize_delta(op, stackok)` fold to a constant, so the
// per-op stacksize update carries no residual call. The green-key-explosion
// concern (each stackok flip generating a separate compiled loop) does not
// reproduce on logo.aheui — the distinct (pc, stackok) merge points that
// occur are few, so logo compiles a single bounded loop.

extern crate majit_ir;
extern crate majit_metainterp as majit_meta;

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
    pool_ptr: usize,
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
    /// Same value as `pool_ptr` (kept `int` for the residual storage helpers).
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
        let mut dump = String::new();
        for i in 0..STORAGE_COUNT {
            if self.storage.len_at(i) == 0 {
                continue;
            }
            let mut p = self.storage.dispatch(i).head() as *const u8;
            let mut vals: Vec<i64> = Vec::new();
            while !p.is_null() && vals.len() < 8 {
                let raw = unsafe { *(p as *const i64) };
                // bigint Val: small = (v<<1)|1; smallint Val: raw == v.
                vals.push(if raw & 1 != 0 { raw >> 1 } else { raw });
                p = unsafe { *(p.add(8) as *const *const u8) };
            }
            dump.push_str(&format!(" stack[{i}]={vals:?}"));
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
        self.pool_ptr = &mut self.storage as *mut Storage as usize;
        self.storage_ref = self.pool_ptr;
        self.refresh_selected_ref();
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
        pool_ptr: int(usize),
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
        // re-producible `getarrayitem_gc_r` on this base. Same value as
        // `pool_ptr`.
        storage_ref: ref(aheui_runtime::storage::Storage),
    },
    // `storage_ref` is a raw-pointer-array base: the registered getter
    // `jit_sel_get_ref(state.storage_ref, state.selected)` indexes
    // `pools[selected]` and lowers to getarrayitem_gc_r instead of the opaque
    // residual call below.  Keyed on the getter identity so only this call —
    // not any other helper sharing the `(state.storage_ref, int)` shape — is
    // recognized as a pool read.
    pool_arrays = { storage_ref => jit_sel_get_ref },
    // Struct field type declarations for ref-kind field access.
    // Tells the lowerer to emit getfield_gc_r / setfield_gc_r (ref-kind)
    // instead of _gc_i (int-kind) when accessing these fields through a
    // ref(T) state scalar or a local ref binding.
    ref_fields = {
        aheui_runtime::storage::linkedlist::Stack::head => aheui_runtime::storage::linkedlist::Node,
        aheui_runtime::storage::linkedlist::Node::next => aheui_runtime::storage::linkedlist::Node,
        NodeJit::next => aheui_runtime::storage::linkedlist::Node,
    },
    io_shims = {
        aheui_io::output_write_number => jit_write_number,
        aheui_io::output_write_utf8 => jit_write_utf8,
    },
    calls = {
        jit_read_utf8 => residual_int,
        jit_read_number => residual_int,
        jit_output_flush => residual_void_cannot_raise,
        jit_tag_val => elidable_int_cannot_raise,
        // First MethodCall RHS consumer; lowered via `lower_method_call_value`.
        Program::get_req_size => elidable_int_cannot_raise,
        Program::get_op => elidable_int_cannot_raise,
        Program::get_label => elidable_int_cannot_raise,
        Program::get_operand => elidable_int_cannot_raise,
        // Phase D-1 monomorphic helpers — registered as residual calls
        // so the lowerer emits a concrete `call_void_args` /
        // `call_int_args` in the trace IR (function-pointer call) rather
        // than silent-skipping the storage op. `#[jit_inline]` upgrade
        // for IR-level body inlining is the next slice.
        //
        // The registered path segments must match the call site verbatim
        // (the macro compares segment-by-segment); use the `lj::*` alias
        // here since the mainloop arms call `lj::stack_push(...)` etc.
        lj::stack_push => residual_void,
        lj::stack_pop => residual_int,
        lj::stack_add => residual_void,
        lj::stack_sub => residual_void,
        lj::stack_mul => residual_void,
        lj::stack_div => residual_void,
        lj::stack_mod => residual_void,
        lj::stack_dup => residual_void,
        lj::stack_swap => residual_void,
        lj::stack_cmp => residual_void,
        lj::queue_push => residual_void,
        lj::queue_pop => residual_int,
        lj::queue_add => residual_void,
        lj::queue_sub => residual_void,
        lj::queue_mul => residual_void,
        lj::queue_div => residual_void,
        lj::queue_mod => residual_void,
        lj::queue_dup => residual_void,
        lj::queue_swap => residual_void,
        lj::queue_cmp => residual_void,
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
        // Phase D-2: field-level IR for Stack ops — alloc/free and val
        // arithmetic registered so the lowerer emits concrete IR ops
        // instead of silent-skipping unregistered calls.
        //
        // Node allocation is a residual ref-returning call tagged
        // `nursery_alloc_ref`: identical to `residual_ref` (the node stored
        // into the concrete `selected_ref.head` escapes and is never
        // virtualized away, so `size` and `head` are always mutated together
        // on real memory), but the descr EffectInfo carries
        // `PyreHelperKind::NurseryAlloc` so the dynasm CallR genop emits an
        // inline nursery bump (RPython malloc_cond shape) with a slowpath
        // cond-call into `jit_alloc_node`, instead of a full residual `blr`.
        // Nodes are not freed on the JIT path (`concrete_only_void` drops the
        // free); the leak is reclaimed by `Nursery::collect`.
        jit_alloc_node => nursery_alloc_ref,
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
    // Residual storage mutators that advance a `Stack.size`.  rpaheui's
    // push/pop are traced methods, so the `self.size` store is an in-trace
    // `setfield_gc` that the optimizer's heapcache uses to invalidate the
    // `len(selected)` getfield (`aheui.py:382 stacksize = len(selected)`).
    // pyre lowers these mutators to opaque residual calls, which carry an
    // empty write-set by default — so the `selected_ref.size` getfield that
    // OP_SEL re-reads is CSE-folded to a single loop-invariant load and a
    // loop whose exit derives from the stack length never observes the
    // mutation (spins).  Declaring the `(Stack, size)` field write here
    // restores that invalidation: `OptHeap::force_from_effectinfo` drops the
    // cached getfield after each call, the residual analogue of the traced
    // setfield barrier.  The list is a conservative superset of the
    // size-changing ops (it includes zero-delta `swap` and the OP_MOV
    // `storage[target]` family, whose target may alias `selected`); an extra
    // reload is harmless, a missing invalidation is the spin.
    residual_writes = {
        selected_ref.size => [
            lj::stack_push, lj::stack_pop, lj::stack_add, lj::stack_sub,
            lj::stack_mul, lj::stack_div, lj::stack_mod, lj::stack_dup,
            lj::stack_swap, lj::stack_cmp,
            lj::queue_push, lj::queue_pop, lj::queue_add, lj::queue_sub,
            lj::queue_mul, lj::queue_div, lj::queue_mod, lj::queue_dup,
            lj::queue_swap, lj::queue_cmp,
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
    greens = [pc, stackok, is_queue, program],
    recover = refresh_state_from_storage,
    switch_dispatch = true,
)]
pub fn mainloop(program: &Program, threshold: u32) -> Val {
    let mut driver: majit_meta::JitDriver<AheuiState> = majit_meta::JitDriver::new(threshold);

    // Register the nursery-backed GC allocator for JIT-compiled New() ops.
    // Linked-list nodes (Node) share the interpreter's nursery pool, so
    // compiled `New` allocation must route through this allocator rather
    // than `libc::malloc` — otherwise nodes freed to the nursery free-list
    // and nodes malloc'd by compiled code come from different pools.
    let gc = Box::new(NurseryGcAllocator::new());
    driver.meta_interp_mut().backend_mut().set_gc_allocator(gc);
    driver.meta_interp_mut().backend_mut().set_new_via_gc(true);

    // rpaheui/aheui/aheui.py:325: jit.set_param(driver, 'trace_limit', 30000).
    // Scaled for majit's denser IR: the sub-JitCode dispatch + pre-dispatch
    // branch match emit ~17 IR ops per aheui opcode (measured: logo's ~3751
    // opcodes -> ~64675 IR ops) vs rpaheui's ~8, so rpaheui's 30000 budget
    // maps to ~64500 here. 70000 keeps logo's whole-program trace compilable,
    // matching rpaheui (which compiles logo's loop rather than aborting it).
    driver.set_param("trace_limit", 70000);

    let mut pc: usize = 0;
    // rpaheui/aheui/aheui.py:30: reds=['stacksize','storage','selected']
    let mut state = AheuiState {
        storage: Storage::new(),
        selected: 0,
        stacksize: 0,
        pool_ptr: 0,
        selected_ref: 0,
        storage_ref: 0,
    };
    // Storage was moved into state — refresh self-referencing pointers.
    state.storage.refresh_pools();
    state.pool_ptr = &mut state.storage as *mut Storage as usize;
    state.storage_ref = state.pool_ptr;
    state.refresh_selected_ref();
    // Register the storage as the nursery collector's root set. `state` is a
    // stationary local for the rest of the mainloop, so the pointer stays
    // valid; the compiled traces omit `jit_free_node`, so leaked nodes are
    // reclaimed by `Nursery::collect` walking these roots.
    aheui_runtime::storage::set_gc_roots(&mut state.storage as *mut Storage);

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
            static HCOUNT: std::sync::atomic::AtomicU32 =
                std::sync::atomic::AtomicU32::new(0);
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
                let snap = format!(" out={out} selected={}{}", state.selected, state.spdiag_dump_stacks());
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
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
                    continue;
                }
            }
            OP_JMP => {
                pc = program.get_label(pc - 1);
                stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
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
                    state.selected_ref.size = state.selected_ref.size - 1usize;
                    jit_free_node(top_node);
                    // pop_val is Val (= i64 repr-transparent). val_is_zero
                    // checks `*v == 0` for smallint, or the tagged form
                    // `(0 << 1) | 1 = 1` for bigint. Use the raw int
                    // comparison `pop_val == jit_tag_val(0)` which the
                    // lowerer handles natively as IntEq.
                    let zero_val = jit_tag_val(0i64);
                    if pop_val == zero_val { 1i64 } else { 0i64 }
                };
                if zero != 0 {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i64 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
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
                        lj::queue_add(state.selected_ref);
                    } else {
                        // _get_2_values + _put_value (rpaheui linkedlist.py)
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_add(r2, r1);
                    }
                }
            }
            OP_SUB => {
                if stackok {
                    if is_queue {
                        lj::queue_sub(state.selected_ref);
                    } else {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_sub(r2, r1);
                    }
                }
            }
            OP_MUL => {
                if stackok {
                    if is_queue {
                        lj::queue_mul(state.selected_ref);
                    } else {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_mul(r2, r1);
                    }
                }
            }
            OP_DIV => {
                if stackok {
                    if is_queue {
                        lj::queue_div(state.selected_ref);
                    } else {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_div(r2, r1);
                    }
                }
            }
            OP_MOD => {
                if stackok {
                    if is_queue {
                        lj::queue_mod(state.selected_ref);
                    } else {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = val_mod(r2, r1);
                    }
                }
            }
            OP_POP => {
                if stackok {
                    if is_queue {
                        lj::queue_pop(state.selected_ref);
                    } else {
                        // pop: read top, advance head, free node
                        let top_node = state.selected_ref.head;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                    }
                }
            }
            OP_PUSH => {
                // rpaheui/aheui/aheui.py:272-275.
                let value = program.get_operand(pc - 1) as i64;
                let v = jit_tag_val(value);
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else {
                    // push: alloc node, link to current head
                    let old_head = state.selected_ref.head;
                    let new_node = jit_alloc_node(v, old_head);
                    state.selected_ref.head = new_node;
                    state.selected_ref.size = state.selected_ref.size + 1usize;
                }
            }
            OP_DUP => {
                if stackok {
                    if is_queue {
                        lj::queue_dup(state.selected_ref);
                    } else {
                        // dup: push(head.value) — rpaheui linkedlist.py
                        let head = state.selected_ref.head;
                        let top_val = head.value;
                        let new_node = jit_alloc_node(top_val, head);
                        state.selected_ref.head = new_node;
                        state.selected_ref.size = state.selected_ref.size + 1usize;
                    }
                }
            }
            OP_SWAP => {
                if stackok {
                    if is_queue {
                        lj::queue_swap(state.selected_ref);
                    } else {
                        // swap: exchange head.value and head.next.value
                        let node1 = state.selected_ref.head;
                        let node2 = node1.next;
                        let v1 = node1.value;
                        let v2 = node2.value;
                        node1.value = v2;
                        node2.value = v1;
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
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        pop_val
                    };
                    let target = program.get_operand(pc - 1) as usize;
                    // target may differ from selected → polymorphic push
                    jit_storage_push(state.pool_ptr, target, r);
                    if state.selected == target {
                        state.stacksize += 1;
                    }
                }
            }
            OP_CMP => {
                if stackok {
                    if is_queue {
                        lj::queue_cmp(state.selected_ref);
                    } else {
                        let top_node = state.selected_ref.head;
                        let r1 = top_node.value;
                        let next = top_node.next;
                        state.selected_ref.head = next;
                        state.selected_ref.size = state.selected_ref.size - 1usize;
                        jit_free_node(top_node);
                        let r2 = next.value;
                        next.value = jit_val_ge(r2, r1);
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
                        state.selected_ref.size = state.selected_ref.size - 1usize;
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
                        state.selected_ref.size = state.selected_ref.size - 1usize;
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
                let v = jit_tag_val(num);
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else {
                    let old_head = state.selected_ref.head;
                    let new_node = jit_alloc_node(v, old_head);
                    state.selected_ref.head = new_node;
                    state.selected_ref.size = state.selected_ref.size + 1usize;
                }
            }
            OP_PUSHCHAR => {
                // rpaheui/aheui/aheui.py:322-325
                jit_output_flush();
                let ch = jit_read_utf8();
                let v = jit_tag_val(ch);
                if is_queue {
                    lj::queue_push(state.selected_ref, v);
                } else {
                    let old_head = state.selected_ref.head;
                    let new_node = jit_alloc_node(v, old_head);
                    state.selected_ref.head = new_node;
                    state.selected_ref.size = state.selected_ref.size + 1usize;
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

    // rpaheui/aheui/aheui.py:363-366
    if state.selected_dispatch().__len__() > 0 {
        state.selected_dispatch_mut().pop()
    } else {
        val_from_i32(0)
    }
}
