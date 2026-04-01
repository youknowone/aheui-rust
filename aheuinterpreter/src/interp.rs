/// Pure Aheui interpreter (no JIT).
use crate::aheui::Program;

use crate::aheui::*;
use crate::io as aheui_io;
use crate::storage::StoragePool;
use crate::value::*;

pub fn mainloop(program: &Program) -> Val {
    let mut pc: usize = 0;
    let mut stacksize: i32 = 0;
    let mut storage = StoragePool::new();
    let mut selected: usize = 0;

    let mut input = aheui_io::InputBuffer::new();
    while pc < program.size {
        let stackok = program.get_req_size(pc) as i32 <= stacksize;
        let op = program.get_op(pc);
        stacksize += -OP_STACKDEL[op as usize] + OP_STACKADD[op as usize];

        match op {
            OP_ADD => storage.get_mut(selected).add(),
            OP_SUB => storage.get_mut(selected).sub(),
            OP_MUL => storage.get_mut(selected).mul(),
            OP_DIV => storage.get_mut(selected).div(),
            OP_MOD => storage.get_mut(selected).modulo(),
            OP_POP => {
                storage.get_mut(selected).pop();
            }
            OP_PUSH => {
                let value = program.get_operand(pc) as i64;
                storage.get_mut(selected).push(value);
            }
            OP_DUP => storage.get_mut(selected).dup(),
            OP_SWAP => storage.get_mut(selected).swap(),
            OP_SEL => {
                let value = program.get_operand(pc) as usize;
                selected = value;
                stacksize = storage.get(selected).len() as i32;
            }
            OP_MOV => {
                let r = storage.get_mut(selected).pop();
                let target = program.get_operand(pc) as usize;
                storage.get_mut(target).push(r);
                if selected == target {
                    stacksize += 1;
                }
            }
            OP_CMP => storage.get_mut(selected).cmp(),
            OP_BRPOP1 | OP_BRPOP2 | OP_JMP | OP_BRZ => {
                let jump = match op {
                    OP_BRPOP1 | OP_BRPOP2 => !stackok,
                    OP_JMP => true,
                    OP_BRZ => {
                        let top = storage.get_mut(selected).pop();
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
                let r = storage.get_mut(selected).pop();
                aheui_io::output_write_number(&r);
            }
            OP_POPCHAR => {
                let r = storage.get_mut(selected).pop();
                aheui_io::output_write_utf8(&r);
            }
            OP_PUSHNUM => {
                aheui_io::output_flush();
                let num = input.read_number();
                storage.get_mut(selected).push(num);
            }
            OP_PUSHCHAR => {
                aheui_io::output_flush();
                let ch = input.read_utf8();
                storage.get_mut(selected).push(ch);
            }
            OP_NONE => {}
            OP_HALT => break,
            _ => {}
        }
        pc += 1;
    }

    aheui_io::output_flush();

    if !storage.get(selected).is_empty() {
        storage.get_mut(selected).pop()
    } else {
        val_from_i32(0)
    }
}
