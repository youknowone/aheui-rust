/// Aheui interpreter with an optional compile-time JIT.
///
/// With the `jit` feature enabled, the loop below is analyzed by
/// `#[jit_interp]`. Without it, the same loop compiles as a plain interpreter
/// with no-op JIT hooks.
use ahsembler::Program;

use std::io::{self, BufWriter, Write};

#[cfg(feature = "jit")]
use majit_meta::JitDriver;
#[cfg(not(feature = "jit"))]
use std::marker::PhantomData;

use crate::aheui::*;
use crate::io as aheui_io;
use crate::storage::StoragePool;
use crate::value::*;

#[cfg(feature = "jit")]
const DEFAULT_THRESHOLD: u32 = 0;

#[cfg(not(feature = "jit"))]
macro_rules! jit_merge_point {
    () => {};
}

#[cfg(not(feature = "jit"))]
macro_rules! can_enter_jit {
    ($driver:expr, $target:expr, $state:expr, $env:expr, $pre_run:expr) => {{
        let _ = (&$driver, &$target, &$state, &$env, &$pre_run);
    }};
}

#[cfg(not(feature = "jit"))]
struct JitDriver<T>(PhantomData<T>);

#[cfg(not(feature = "jit"))]
impl<T> JitDriver<T> {
    fn new(_threshold: u32) -> Self {
        Self(PhantomData)
    }
}

// ── I/O shims for JIT-compiled code ──────────────────────────────────

#[cfg(feature = "jit")]
#[allow(dead_code)]
extern "C" fn jit_write_number(value: i64) {
    majit_meta::io_buffer_write_fmt(format_args!("{}", value));
}

#[cfg(feature = "jit")]
#[allow(dead_code)]
extern "C" fn jit_write_utf8(value: i64) {
    if let Some(c) = char::from_u32(value as u32) {
        majit_meta::io_buffer_write_fmt(format_args!("{}", c));
    } else {
        majit_meta::io_buffer_write_fmt(format_args!("\u{FFFD}"));
    }
}

// ── Interpreter state for JitDriver ─────────────────────────────────

/// Interpreter state exposed to the JIT framework.
struct AheuiState {
    storage: StoragePool,
    selected: usize,
}

// ── Unified interpreter loop ─────────────────────────────────────────

/// Core interpreter loop shared by both `mainloop_jit` and `mainloop_interp`.
///
/// When `threshold == u32::MAX`, `can_enter_jit` never triggers tracing,
/// making this equivalent to a plain interpreter.
#[cfg_attr(
    all(feature = "jit", not(feature = "bigint")),
    majit_macros::jit_interp(
    state = AheuiState,
    env = Program,
    greens = [state.selected],
    storage = {
        pool: state.storage,
        pool_type: StoragePool,
        selector: state.selected,
        untraceable: [VAL_QUEUE, VAL_PORT],
        scan: find_used_storages,
    },
    binops = {
        add => IntAdd, sub => IntSub, mul => IntMul,
        div => IntFloorDiv, modulo => IntMod, cmp => IntGe,
    },
    io_shims = {
        aheui_io::write_number => jit_write_number,
        aheui_io::write_utf8 => jit_write_utf8,
    },
    )
)]
#[cfg_attr(
    all(feature = "jit", feature = "bigint"),
    majit_macros::jit_interp(
    state = AheuiState,
    env = Program,
    greens = [state.selected],
    storage = {
        pool: state.storage,
        pool_type: StoragePool,
        selector: state.selected,
        untraceable: [VAL_QUEUE, VAL_PORT],
        scan: find_used_storages,
        can_trace_guard: all_jit_compatible,
    },
    binops = {
        add => IntAdd, sub => IntSub, mul => IntMul,
        div => IntFloorDiv, modulo => IntMod, cmp => IntGe,
    },
    io_shims = {
        aheui_io::write_number => jit_write_number,
        aheui_io::write_utf8 => jit_write_utf8,
    },
    )
)]
pub fn mainloop(program: &Program, threshold: u32) -> Val {
    #[cfg(not(feature = "jit"))]
    let _ = threshold;

    #[cfg(feature = "jit")]
    let mut driver: JitDriver<AheuiState> = JitDriver::new(threshold);
    #[cfg(not(feature = "jit"))]
    let driver: JitDriver<AheuiState> = JitDriver::new(threshold);

    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut state = AheuiState {
        storage: StoragePool::new(),
        selected: 0,
    };

    let mut input = aheui_io::InputBuffer::new();
    let stdout = io::stdout();
    let mut writer = BufWriter::new(stdout.lock());

    while pc < program.size {
        jit_merge_point!();
        let stackok = program.get_req_size(pc) as i32 <= stacksize;
        let op = program.get_op(pc);
        stacksize += -OP_STACKDEL[op as usize] + OP_STACKADD[op as usize];

        match op {
            OP_ADD => state.storage.get_mut(state.selected).add(),
            OP_SUB => state.storage.get_mut(state.selected).sub(),
            OP_MUL => state.storage.get_mut(state.selected).mul(),
            OP_DIV => state.storage.get_mut(state.selected).div(),
            OP_MOD => state.storage.get_mut(state.selected).modulo(),
            OP_POP => {
                state.storage.get_mut(state.selected).pop();
            }
            OP_PUSH => {
                let value = program.get_operand(pc) as i64;
                state.storage.get_mut(state.selected).push(value);
            }
            OP_DUP => state.storage.get_mut(state.selected).dup(),
            OP_SWAP => state.storage.get_mut(state.selected).swap(),
            OP_SEL => {
                let value = program.get_operand(pc) as usize;
                state.selected = value;
                stacksize = state.storage.get(state.selected).len() as i32;
            }
            OP_MOV => {
                let r = state.storage.get_mut(state.selected).pop();
                let target = program.get_operand(pc) as usize;
                state.storage.get_mut(target).push(r);
                if state.selected == target {
                    stacksize += 1;
                }
            }
            OP_CMP => state.storage.get_mut(state.selected).cmp(),
            OP_BRPOP1 | OP_BRPOP2 | OP_JMP | OP_BRZ => {
                let jump = match op {
                    OP_BRPOP1 | OP_BRPOP2 => !stackok,
                    OP_JMP => true,
                    OP_BRZ => {
                        let top = state.storage.get_mut(state.selected).pop();
                        val_is_zero(&top)
                    }
                    _ => unreachable!(),
                };
                if jump {
                    let target = program.get_label(pc);
                    if target <= pc {
                        can_enter_jit!(driver, target, &mut state, program, || {
                            let _ = writer.flush();
                        });
                    }
                    pc = target;
                    continue;
                }
            }
            OP_POPNUM => {
                let r = state.storage.get_mut(state.selected).pop();
                aheui_io::write_number(&r, &mut writer);
            }
            OP_POPCHAR => {
                let r = state.storage.get_mut(state.selected).pop();
                aheui_io::write_utf8(&r, &mut writer);
            }
            OP_PUSHNUM => {
                let _ = writer.flush();
                let num = input.read_number();
                state.storage.get_mut(state.selected).push(num);
            }
            OP_PUSHCHAR => {
                let _ = writer.flush();
                let ch = input.read_utf8();
                state.storage.get_mut(state.selected).push(ch);
            }
            OP_NONE => {}
            OP_HALT => break,
            _ => {}
        }
        pc += 1;
    }

    let _ = writer.flush();

    if !state.storage.get(state.selected).is_empty() {
        state.storage.get_mut(state.selected).pop()
    } else {
        val_from_i32(0)
    }
}

pub const NO_JIT: u32 = u32::MAX;
#[cfg(feature = "jit")]
pub const JIT_THRESHOLD: u32 = DEFAULT_THRESHOLD;
#[cfg(not(feature = "jit"))]
pub const JIT_THRESHOLD: u32 = NO_JIT;

// ── Storage helpers ──────────────────────────────────────────────────

/// Pre-scan bytecodes from header_pc to find all storage indices referenced via SEL/MOV.
#[cfg(feature = "jit")]
fn find_used_storages(program: &Program, header_pc: usize, initial: usize) -> Vec<usize> {
    let mut storages = vec![initial];
    let mut seen = std::collections::HashSet::new();
    seen.insert(initial);
    for pc in header_pc..program.size {
        let op = program.get_op(pc);
        let val = match op {
            OP_SEL | OP_MOV => program.get_operand(pc) as usize,
            _ => continue,
        };
        if seen.insert(val) {
            storages.push(val);
        }
    }
    storages
}
