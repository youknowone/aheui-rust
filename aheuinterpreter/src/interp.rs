//! Pure Aheui interpreter (no JIT).
//!
//! Mirrors `rpaheui/aheui/aheui.py::mainloop`. Variables are named after
//! the rpaheui locals so the structure stays line-comparable:
//!   * greens — `pc`, `stackok`, `is_queue`, `program`
//!   * reds   — `stacksize`, `storage`, `selected`
//!
//! In RPython `selected = storage[value]` captures a direct reference to the
//! polymorphic `Stack` / `Queue` / `Port` object. Rust cannot hold a mutable
//! borrow into `storage` across subsequent storage mutations, so `selected` is
//! kept as a `usize` index and the object is re-fetched through
//! [`Storage::dispatch_mut`] on every use — the minimum adaptation the borrow
//! checker needs, with an identical semantic result.
use crate::aheui::Program;

use crate::aheui::*;
use crate::io as aheui_io;
use crate::storage::Storage;
use crate::value::*;

// `is_queue` is kept for structural parity with the rpaheui mainloop:
// the JIT mirror in `aheui-jit` promotes it as a green, and the naive
// path here just maintains it. Suppress unused-write warnings.
#[allow(unused_assignments, unused_variables)]
pub fn mainloop(program: &Program) -> Val {
    // Dual mode: run on raw machine words until an operation overflows one.
    // Must precede `Storage::new()` — the queue allocates a sentinel value,
    // and the flip walks it, so it has to be built in the mode it is read in.
    start_in_raw_mode();

    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut is_queue: bool = false;
    let mut storage = Storage::new();
    let mut selected: usize = 0;

    // The registered storage is what enumerates the live values, both for the
    // node collection and for the dual-mode flip, which cannot run without it.
    crate::storage::set_gc_roots(&mut storage as *mut Storage);

    let mut input = aheui_io::InputBuffer::new();
    while pc < program.size {
        let stackok = program.get_req_size(pc) as i32 <= stacksize;
        let op = program.get_op(pc);
        stacksize += -OP_STACKDEL[op as usize] + OP_STACKADD[op as usize];

        match op {
            OP_ADD => storage.dispatch_mut(selected).add(),
            OP_SUB => storage.dispatch_mut(selected).sub(),
            OP_MUL => storage.dispatch_mut(selected).mul(),
            OP_DIV => storage.dispatch_mut(selected).div(),
            OP_MOD => storage.dispatch_mut(selected).modulo(),
            OP_POP => {
                storage.dispatch_mut(selected).pop();
            }
            OP_PUSH => {
                let value = program.get_operand(pc) as i32;
                storage.dispatch_mut(selected).push(val_from_i32(value));
            }
            OP_DUP => storage.dispatch_mut(selected).dup(),
            OP_SWAP => storage.dispatch_mut(selected).swap(),
            OP_SEL => {
                let value = program.get_operand(pc) as usize;
                selected = value;
                stacksize = storage.len_at(selected) as i32;
                is_queue = value == VAL_QUEUE;
            }
            OP_MOV => {
                let r = storage.dispatch_mut(selected).pop();
                let target = program.get_operand(pc) as usize;
                storage.dispatch_mut(target).push(r);
                if selected == target {
                    stacksize += 1;
                }
            }
            OP_CMP => storage.dispatch_mut(selected).cmp(),
            OP_BRPOP1 | OP_BRPOP2 | OP_JMP | OP_BRZ => {
                let jump = match op {
                    OP_BRPOP1 | OP_BRPOP2 => !stackok,
                    OP_JMP => true,
                    OP_BRZ => {
                        let top = storage.dispatch_mut(selected).pop();
                        val_is_zero(&top)
                    }
                    _ => unreachable!(),
                };
                if jump {
                    pc = program.get_label(pc);
                    continue;
                }
            }
            OP_POPNUM => {
                let r = storage.dispatch_mut(selected).pop();
                aheui_io::output_write_number(&r);
            }
            OP_POPCHAR => {
                let r = storage.dispatch_mut(selected).pop();
                aheui_io::output_write_utf8(&r);
            }
            OP_PUSHNUM => {
                aheui_io::output_flush();
                let num = input.read_number();
                storage.dispatch_mut(selected).push(num.into());
            }
            OP_PUSHCHAR => {
                aheui_io::output_flush();
                let ch = input.read_utf8();
                storage.dispatch_mut(selected).push(ch.into());
            }
            OP_NONE => {}
            OP_HALT => break,
            _ => {}
        }
        pc += 1;
    }

    aheui_io::output_flush();

    if storage.len_at(selected) > 0 {
        storage.dispatch_mut(selected).pop()
    } else {
        val_from_i32(0)
    }
}
