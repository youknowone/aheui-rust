// JIT-enabled Aheui interpreter — graph pipeline + #[jit_interp] macro.
//
// RPython parity: rpaheui/aheui/aheui.py
//   greens = [pc, stackok, is_queue, program]
//   reds   = [stacksize, storage, selected]
//   storage = linked list stacks (no virtualizable arrays)

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

/// GC allocator for JIT-compiled New() ops.
/// Delegates to the global nursery so alloc/free share the same pool
/// as the interpreter path.  Prevents unbounded memory growth.
/// Max JIT heap allocation: 256 MB safety limit.
const JIT_ALLOC_LIMIT: usize = 256 * 1024 * 1024;

struct NurseryGcAllocator {
    total_allocated: usize,
}

impl NurseryGcAllocator {
    fn new() -> Self {
        Self { total_allocated: 0 }
    }
}

impl majit_gc::GcAllocator for NurseryGcAllocator {
    fn alloc_nursery(&mut self, size: usize) -> majit_ir::GcRef {
        self.total_allocated += size;
        if self.total_allocated > JIT_ALLOC_LIMIT {
            // Return NULL to signal allocation failure — compiled code
            // will hit a guard and fall back to the interpreter.
            return majit_ir::GcRef::NULL;
        }
        if size <= aheui_runtime::storage::NODE_SIZE {
            let node = aheui_runtime::storage::alloc_node_raw();
            majit_ir::GcRef(node as usize)
        } else {
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
    /// nursery_free_addr / nursery_top_addr expose the bump-pointer
    /// limits to the JIT-emitted inline allocator. aheui-jit allocates
    /// only `Node`-sized objects via the global node pool and never goes
    /// through the inline bump path; both addresses report 0 so any
    /// inline alloc-fast-path in compiled code immediately spills to
    /// alloc_nursery (which routes to alloc_node_raw).
    fn nursery_free_addr(&self) -> usize {
        0
    }
    fn nursery_top_addr(&self) -> usize {
        0
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
struct AheuiState {
    storage: Storage,
    selected: usize,
    stacksize: i32,
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

    fn refresh_state_from_storage(&mut self) {
        self.pool_ptr = &mut self.storage as *mut Storage as usize;
        self.storage_ref = self.pool_ptr;
        self.refresh_selected_ref();
        self.stacksize = self.storage.len_at(self.selected) as i32;
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

/// io-shim targets for `aheui_io::output_write_*(&r)`: the recorded call
/// carries the raw `Val` word (tagged smallint or boxed-bigint pointer),
/// so reconstruct the `Val` and route through the interpreter's own
/// decode + output buffer. Writing the raw word through majit's
/// `jit_write_number_i64` printed tagged payloads (289 for 144) into a
/// second buffer that interleaved wrongly with interpreter output.
extern "C" fn jit_write_number(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
    if std::env::var_os("MAJIT_BH_DEBUG").is_some() {
        eprintln!("[io-debug] jit_write_number raw={value}");
    }
    aheui_io::output_write_number(&v);
}

extern "C" fn jit_write_utf8(value: i64) {
    let v: Val = unsafe { std::mem::transmute(value) };
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
        // 28-slot pool); stacksize is `i32` (signed pop/push delta). The
        // macro carries them as Int in IR; `int(<Type>)` keeps the user's
        // natural Rust storage type and inserts `as i64` / `as <Type>`
        // casts at the JIT boundary.
        selected: int(usize),
        stacksize: int(i32),
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
    io_shims = {
        aheui_io::output_write_number => jit_write_number,
        aheui_io::output_write_utf8 => jit_write_utf8,
    },
    calls = {
        jit_read_utf8 => residual_int,
        jit_read_number => residual_int,
        jit_output_flush => residual_void,
        jit_tag_val => elidable_int,
        // First MethodCall RHS consumer; lowered via `lower_method_call_value`.
        Program::get_req_size => elidable_int,
        Program::get_op => elidable_int,
        Program::get_label => elidable_int,
        Program::get_operand => elidable_int,
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
        // Phase D-4: bundled pop + is_zero used by OP_BRZ. Residual
        // because the underlying pop mutates storage; the result is
        // immediately consumed as a branch decision so a CallI lands
        // in the trace IR followed by an IntNe + GuardFalse pair.
        jit_pop_is_zero_stack => residual_int,
        jit_pop_is_zero_queue => residual_int,
        jit_pop_is_zero => residual_int,
        // OP_MOV bundled `storage[target].push(r)` (target is a
        // per-opcode operand, not the promoted selected).
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
        jit_stacksize_delta => elidable_int,
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
)]
pub fn mainloop(program: &Program, threshold: u32) -> Val {
    let mut driver: majit_meta::JitDriver<AheuiState> = majit_meta::JitDriver::new(threshold);

    // Register a simple malloc-based GC allocator for JIT-compiled New() ops.
    // Linked list nodes (Node) are allocated here during compiled code execution.
    let gc = Box::new(NurseryGcAllocator::new());
    driver.meta_interp_mut().backend_mut().set_gc_allocator(gc);

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

    while pc < program.size {
        // rpaheui/aheui/aheui.py:252
        let mut stackok = program.get_req_size(pc) as i32 <= state.stacksize;
        // rpaheui/aheui/aheui.py:284 sets `is_queue = (value == VAL_QUEUE)`
        // inside OP_SEL; pyre recomputes it pre-merge-point from the
        // canonical source (`state.selected == VAL_QUEUE`) so A.3.6.1's
        // body-local walker can bind it as a green for `resolve_greens`.
        let is_queue = state.selected == 21usize;

        // rpaheui/aheui/aheui.py:253-255: jit_merge_point
        jit_merge_point!();
        // rpaheui parity: `selected` is a RED (reds=['stacksize','storage',
        // 'selected']) — it is NEVER promoted (aheui.py promotes only
        // `program`@326 and `storage`@332). Storage ops dispatch on the live
        // `selected` index via `jit_storage_*` residual calls (one call site,
        // mirroring rpaheui's polymorphic `selected.METHOD()`), so no
        // `guard_value(selected)` is emitted and the loop closes through the
        // real back-edge instead of being rejected as an invalid loop.
        let op = program.get_op(pc);
        state.stacksize += jit_stacksize_delta(op as usize) as i32;
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
                    stackok = program.get_req_size(pc) as i32 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
                    continue;
                }
            }
            OP_JMP => {
                pc = program.get_label(pc - 1);
                stackok = program.get_req_size(pc) as i32 <= state.stacksize;
                can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
                continue;
            }
            OP_BRZ => {
                let zero = jit_pop_is_zero(state.pool_ptr, state.selected);
                if zero != 0 {
                    pc = program.get_label(pc - 1);
                    stackok = program.get_req_size(pc) as i32 <= state.stacksize;
                    can_enter_jit!(driver, pc, &mut state, program, || {}, pc, state.stacksize; pc, stackok, is_queue, program);
                    continue;
                }
            }
            _ => {}
        }

        match op {
            // rpaheui/aheui/aheui.py:260-269: selected.<binop>().
            // Phase D-1 §4: 3-way branch on the (is_port, is_queue)
            // greens. The trace specializes per combination so each
            // compiled trace contains only one of the three call sites
            // (concrete `stack_*` or `queue_*` function pointer for the
            // hot path; polymorphic `dispatch_mut()` only for the cold
            // I/O Port slot).
            // rpaheui/aheui/aheui.py:260-389: `selected.<op>()` — one
            // polymorphic call site dispatched on the live RED `selected`.
            OP_ADD => {
                if stackok {
                    jit_storage_add(state.pool_ptr, state.selected);
                }
            }
            OP_SUB => {
                if stackok {
                    jit_storage_sub(state.pool_ptr, state.selected);
                }
            }
            OP_MUL => {
                if stackok {
                    jit_storage_mul(state.pool_ptr, state.selected);
                }
            }
            OP_DIV => {
                if stackok {
                    jit_storage_div(state.pool_ptr, state.selected);
                }
            }
            OP_MOD => {
                if stackok {
                    jit_storage_mod(state.pool_ptr, state.selected);
                }
            }
            OP_POP => {
                if stackok {
                    jit_storage_pop(state.pool_ptr, state.selected);
                }
            }
            OP_PUSH => {
                // rpaheui/aheui/aheui.py:272-275.
                //
                // Use `jit_tag_val` (registered `elidable_int`) instead of
                // bare `val_from_i32` so the macro lowerer recognises the
                // value-conversion call and emits BC_CALL_PURE_INT into
                // jitcode. Without registration the lowerer's
                // `expr_references_unknown_local` rejects the let-binding,
                // silent-skipping the rest of the body and dropping the
                // push from the trace IR — which leaves OP_PUSH as
                // state-field guards only and breaks any compiled-trace
                // path that depends on the push reaching storage.
                let value = program.get_operand(pc - 1) as i64;
                let v = jit_tag_val(value);
                jit_storage_push(state.pool_ptr, state.selected, v);
            }
            OP_DUP => {
                if stackok {
                    jit_storage_dup(state.pool_ptr, state.selected);
                }
            }
            OP_SWAP => {
                if stackok {
                    jit_storage_swap(state.pool_ptr, state.selected);
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
                state.stacksize = state.selected_ref.size as i32;
            }
            OP_MOV => {
                if stackok {
                    let r = jit_storage_pop(state.pool_ptr, state.selected);
                    let target = program.get_operand(pc - 1) as usize;
                    jit_storage_push(state.pool_ptr, target, r);
                    if state.selected == target {
                        state.stacksize += 1;
                    }
                }
            }
            OP_CMP => {
                if stackok {
                    jit_storage_cmp(state.pool_ptr, state.selected);
                }
            }
            // Branch ops handled by dispatch-level if-chain above.
            OP_POPNUM => {
                if stackok {
                    let r = jit_storage_pop(state.pool_ptr, state.selected);
                    aheui_io::output_write_number(&r);
                }
            }
            OP_POPCHAR => {
                if stackok {
                    let r = jit_storage_pop(state.pool_ptr, state.selected);
                    aheui_io::output_write_utf8(&r);
                }
            }
            OP_PUSHNUM => {
                // rpaheui/aheui/aheui.py:318-321
                jit_output_flush();
                let num = jit_read_number();
                let v = jit_tag_val(num);
                jit_storage_push(state.pool_ptr, state.selected, v);
            }
            OP_PUSHCHAR => {
                // rpaheui/aheui/aheui.py:322-325
                jit_output_flush();
                let ch = jit_read_utf8();
                let v = jit_tag_val(ch);
                jit_storage_push(state.pool_ptr, state.selected, v);
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
