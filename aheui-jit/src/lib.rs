// JIT-enabled Aheui interpreter — graph pipeline + #[jit_interp] macro.
//
// RPython parity: rpaheui/aheui/aheui.py
//   greens = [pc, stackok, is_queue, program]
//   reds   = [stacksize, storage, selected]
//   storage = linked list stacks (no virtualizable arrays)

extern crate majit_ir;
extern crate majit_metainterp as majit_meta;

pub use aheui_runtime;
pub use aheui_runtime::aheui;
pub use aheui_runtime::io;
pub use aheui_runtime::storage;
pub use aheui_runtime::value;

pub mod jit;

/// Default JIT threshold. RPython default = 1039.
pub const JIT_THRESHOLD: u32 = 1039;

include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

// ── Imports ──

use aheui_runtime::aheui::*;
use aheui_runtime::io as aheui_io;
use aheui_runtime::storage::Storage;
use aheui_runtime::storage::LinkedList;
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
    pool_ptr: majit_ir::GcRef,
    /// `&mut state.storage.pools[selected]` packed as `usize`. Tracked as
    /// `int(usize)` in `state_fields` so monomorphic storage helpers
    /// (`stack_*` / `queue_*`) can read it as an Int operand from the
    /// trace IR. Phase D-1 design doc: `~/.claude/plans/2026-04-28-phase-d1-monomorphic-dispatch-design.md`.
    selected_ref: usize,
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
}

fn find_used_storages(_program: &Program, _header_pc: usize, initial: usize) -> Vec<usize> {
    let mut storages: Vec<usize> = (0..STORAGE_COUNT).filter(|&idx| idx != VAL_PORT).collect();
    if initial != VAL_PORT && !storages.contains(&initial) {
        storages.push(initial);
        storages.sort_unstable();
    }
    storages
}

extern "C" fn jit_write_number(value: i64) {
    majit_meta::jit_write_number_i64(value);
}

extern "C" fn jit_write_utf8(value: i64) {
    majit_meta::jit_write_utf8_codepoint(value);
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
        pool_ptr: opaque(majit_ir::GcRef),
        // Tracked as int(usize) so the lowerer can read it via
        // `lower_state_field_read` and pass it as an Int-kind arg to
        // monomorphic `stack_*` / `queue_*` helpers (Phase D-1 design,
        // ~/.claude/plans/2026-04-28-phase-d1-monomorphic-dispatch-design.md).
        selected_ref: int(usize),
    },
    io_shims = {
        aheui_io::output_write_number => jit_write_number,
        aheui_io::output_write_utf8 => jit_write_utf8,
    },
    calls = {
        jit_read_utf8 => residual_int,
        jit_read_number => residual_int,
        jit_output_flush => residual_void,
        jit_tag_val => elidable_int,
    },
    // rpaheui/aheui/aheui.py:29: greens=['pc','stackok','is_queue','program'].
    // Phase D-1 adds `is_port` so the trace specializes per storage type
    // (stack vs queue vs port) — match arms branch on these greens to
    // call concrete `stack_*` / `queue_*` helpers instead of going
    // through the polymorphic `&dyn LinkedList` dispatch which the
    // lowerer silent-skips.
    greens = [stackok, is_queue, is_port],
)]
pub fn mainloop(program: &Program, threshold: u32) -> Val {
    let mut driver: majit_meta::JitDriver<AheuiState> = majit_meta::JitDriver::new(threshold);

    // Register a simple malloc-based GC allocator for JIT-compiled New() ops.
    // Linked list nodes (Node) are allocated here during compiled code execution.
    let gc = Box::new(NurseryGcAllocator::new());
    driver.meta_interp_mut().backend_mut().set_gc_allocator(gc);

    // Register JitCode factory for BlackholeInterpreter resume.
    // RPython: metainterp_sd.jitcodes — pre-compiled JitCodes for each portal.
    // In majit, #[jit_interp] generates __jitcode_mainloop which produces
    // JitCode on-demand from the interpreter bytecode.
    driver.register_jitcode_factory(|program: &Program, pc: usize, op: u8| {
        __jitcode_mainloop(program, pc, op)
    });

    // rpaheui/aheui/aheui.py:422 — only trace_limit is set explicitly
    // when the user passes --trace-limit. All other parameters use
    // RPython defaults (warmstate.py PARAMETERS). No bridge_threshold
    // or retrace_limit overrides in rpaheui.

    let mut pc: usize = 0;
    // rpaheui/aheui/aheui.py:30: reds=['stacksize','storage','selected']
    let mut state = AheuiState {
        storage: Storage::new(),
        selected: 0,
        stacksize: 0,
        pool_ptr: majit_ir::GcRef::NULL,
        selected_ref: 0,
    };
    // Storage was moved into state — refresh self-referencing pointers.
    state.storage.refresh_pools();
    state.pool_ptr = majit_ir::GcRef(&mut state.storage as *mut Storage as usize);
    state.refresh_selected_ref();
    // rpaheui/aheui/aheui.py:29: is_queue is a green variable.
    // Phase D-1: is_port green added so storage-op match arms branch
    // monomorphically (stack vs queue vs port).
    let mut is_queue: bool = false;
    let mut is_port: bool = false;

    while pc < program.size {
        // rpaheui/aheui/aheui.py:252
        let mut stackok = program.get_req_size(pc) as i32 <= state.stacksize;

        // rpaheui/aheui/aheui.py:253-255: jit_merge_point
        jit_merge_point!();
        let op = program.get_op(pc);
        state.stacksize += -OP_STACKDEL[op as usize] + OP_STACKADD[op as usize];
        // Phase D-1 §5: pre-advance `pc` so the interpreter's pc matches
        // the trace's `__jit_pc = op_pc + 1` convention. Operand reads in
        // the arms use `pc - 1` to recover the opcode row; the trailing
        // `pc += 1` at the end of the loop is dropped (replaced by this
        // pre-advance). Branch arms compute targets against `pc - 1`
        // (op_pc) so the back-edge check stays semantic.
        pc += 1;

        match op {
            // rpaheui/aheui/aheui.py:260-269: selected.<binop>().
            // Polymorphic dispatch through the `LinkedList` trait matches
            // Python method dispatch on the Stack/Queue/Port subclass.
            OP_ADD => state.selected_dispatch_mut().add(),
            OP_SUB => state.selected_dispatch_mut().sub(),
            OP_MUL => state.selected_dispatch_mut().mul(),
            OP_DIV => state.selected_dispatch_mut().div(),
            OP_MOD => state.selected_dispatch_mut().modulo(),
            OP_POP => {
                state.selected_dispatch_mut().pop();
            }
            OP_PUSH => {
                // rpaheui/aheui/aheui.py:272-275
                let value = program.get_operand(pc - 1) as i32;
                state.selected_dispatch_mut().push(val_from_i32(value));
            }
            OP_DUP => state.selected_dispatch_mut().dup(),
            OP_SWAP => state.selected_dispatch_mut().swap(),
            OP_SEL => {
                // rpaheui/aheui/aheui.py:280-284
                let value = program.get_operand(pc - 1) as usize;
                state.selected = value;
                state.refresh_selected_ref();
                state.stacksize = state.storage.len_at(state.selected) as i32;
                is_queue = value == VAL_QUEUE;
                is_port = value == VAL_PORT;
            }
            OP_MOV => {
                // rpaheui/aheui/aheui.py:285-291
                let r = state.selected_dispatch_mut().pop();
                let target = program.get_operand(pc - 1) as usize;
                state.storage.dispatch_mut(target).push(r);
                if state.selected == target {
                    state.stacksize += 1;
                }
            }
            OP_CMP => state.selected_dispatch_mut().cmp(),
            OP_BRPOP1 | OP_BRPOP2 | OP_JMP | OP_BRZ => {
                // rpaheui/aheui/aheui.py:294-311
                let jump = match op {
                    OP_BRPOP1 | OP_BRPOP2 => !stackok,
                    OP_JMP => true,
                    OP_BRZ => {
                        let top = state.selected_dispatch_mut().pop();
                        val_is_zero(&top)
                    }
                    _ => unreachable!(),
                };
                if jump {
                    let target = program.get_label(pc - 1);
                    if target <= pc - 1 {
                        // Back-edge: can_enter_jit before pc assignment.
                        // If compiled code ran and finished (FINISH), the
                        // macro sets pc=target and continues. If guard
                        // failure occurred, back_edge returns false and
                        // we fall through to pc=target below (interpreter
                        // resumes at the loop header).
                        can_enter_jit!(
                            driver,
                            target,
                            &mut state,
                            program,
                            || {
                                aheui_io::output_flush();
                            },
                            pc,
                            state.stacksize
                        );
                    }
                    pc = target;
                    continue;
                }
            }
            OP_POPNUM => {
                // rpaheui/aheui/aheui.py:312-314
                let r = state.selected_dispatch_mut().pop();
                aheui_io::output_write_number(&r);
            }
            OP_POPCHAR => {
                // rpaheui/aheui/aheui.py:315-317
                let r = state.selected_dispatch_mut().pop();
                aheui_io::output_write_utf8(&r);
            }
            OP_PUSHNUM => {
                // rpaheui/aheui/aheui.py:318-321
                jit_output_flush();
                let num = jit_read_number();
                state.selected_dispatch_mut().push(jit_tag_val(num));
            }
            OP_PUSHCHAR => {
                // rpaheui/aheui/aheui.py:322-325
                jit_output_flush();
                let ch = jit_read_utf8();
                state.selected_dispatch_mut().push(jit_tag_val(ch));
            }
            OP_NONE => {}
            OP_HALT => break,
            _ => {}
        }
        // Phase D-1 §5: pc was pre-advanced after `program.get_op`; do
        // not advance again here. Branch arms own their own `pc = target`
        // assignment + `continue`.
    }

    aheui_io::output_flush();

    // rpaheui/aheui/aheui.py:363-366
    if state.selected_dispatch().__len__() > 0 {
        state.selected_dispatch_mut().pop()
    } else {
        val_from_i32(0)
    }
}
