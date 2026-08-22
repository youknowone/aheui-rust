//! Build a CFG from flat Aheui bytecode (Compiler's lines + label_map).

use std::collections::{HashMap, HashSet};

use crate::cfg::*;
use crate::compiler::Compiler;
use crate::consts::*;

/// Convert a Compiler's flat bytecode into a CFG.
pub fn build_cfg(compiler: &Compiler) -> Cfg {
    let lines = &compiler.lines;
    let label_map = &compiler.label_map;

    if lines.is_empty() {
        let mut cfg = Cfg::new();
        cfg.add_block(vec![], Terminator::Halt);
        return cfg;
    }

    // Identify block boundaries: every jump target and instruction after a
    // jump/halt starts a new block.
    let label_targets: HashSet<usize> = label_map.values().copied().collect();
    let mut block_starts: Vec<usize> = vec![0];

    for (pc, &(op, _)) in lines.iter().enumerate() {
        if (is_jump_op(op) || op == OP_HALT) && pc + 1 < lines.len() {
            block_starts.push(pc + 1);
        }
        // SEL does NOT create a block boundary. The codegen handles
        // intra-block SEL by switching the active storage Variable.
        if label_targets.contains(&pc) && pc > 0 {
            block_starts.push(pc);
        }
    }

    block_starts.sort_unstable();
    block_starts.dedup();

    // Map PC → block index (pre-allocated block IDs)
    let mut pc_to_block: HashMap<usize, BlockId> = HashMap::new();
    for (i, &start) in block_starts.iter().enumerate() {
        pc_to_block.insert(start, i as BlockId);
    }

    let num_blocks = block_starts.len();
    let mut cfg = Cfg::new();

    for (block_idx, &start) in block_starts.iter().enumerate() {
        let end = if block_idx + 1 < num_blocks {
            block_starts[block_idx + 1]
        } else {
            lines.len()
        };

        let mut instructions = Vec::new();
        let mut terminator = None;

        for pc in start..end {
            let (op, val) = lines[pc];

            match op {
                OP_BRPOP1 | OP_BRPOP2 => {
                    let min_depth = OP_REQSIZE[op as usize];
                    let ok = block_id_for_pc(pc + 1, &pc_to_block);
                    let fail = resolve_label(val, label_map, &pc_to_block);
                    terminator = Some(Terminator::StackGuard {
                        min_depth,
                        ok,
                        fail,
                    });
                    break;
                }
                OP_BRZ => {
                    let on_zero = resolve_label(val, label_map, &pc_to_block);
                    let on_nonzero = block_id_for_pc(pc + 1, &pc_to_block);
                    terminator = Some(Terminator::BranchZero {
                        on_zero,
                        on_nonzero,
                    });
                    break;
                }
                OP_JMP => {
                    let target = resolve_label(val, label_map, &pc_to_block);
                    terminator = Some(Terminator::Goto(target));
                    break;
                }
                OP_HALT => {
                    terminator = Some(Terminator::Halt);
                    break;
                }
                OP_SEL => {
                    instructions.push(Inst::Sel(val as usize));
                }
                _ => {
                    if let Some(inst) = opcode_to_inst(op, val) {
                        instructions.push(inst);
                    }
                }
            }
        }

        let terminator = terminator.unwrap_or_else(|| {
            // Fallthrough to next block
            if block_idx + 1 < num_blocks {
                Terminator::Goto((block_idx + 1) as BlockId)
            } else {
                Terminator::Halt
            }
        });

        let id = cfg.add_block(instructions, terminator);
        debug_assert_eq!(id, block_idx as BlockId);
    }

    cfg
}

fn resolve_label(
    val: i32,
    label_map: &HashMap<i32, usize>,
    pc_to_block: &HashMap<usize, BlockId>,
) -> BlockId {
    let target_pc = label_map.get(&val).copied().unwrap_or(0);
    block_id_for_pc(target_pc, pc_to_block)
}

fn block_id_for_pc(pc: usize, pc_to_block: &HashMap<usize, BlockId>) -> BlockId {
    // Find the block that contains this PC (the largest block start <= pc)
    if let Some(&id) = pc_to_block.get(&pc) {
        return id;
    }
    // PC is in the middle of a block — find the containing block
    let mut best_start = 0;
    let mut best_id = 0;
    for (&start, &id) in pc_to_block {
        if start <= pc && start >= best_start {
            best_start = start;
            best_id = id;
        }
    }
    best_id
}

fn opcode_to_inst(op: u8, val: i32) -> Option<Inst> {
    match op {
        OP_PUSH => Some(Inst::Push(val as i64)),
        OP_POP => Some(Inst::Pop),
        OP_DUP => Some(Inst::Dup),
        OP_SWAP => Some(Inst::Swap),
        OP_ADD => Some(Inst::BinOp(BinOpKind::Add)),
        OP_SUB => Some(Inst::BinOp(BinOpKind::Sub)),
        OP_MUL => Some(Inst::BinOp(BinOpKind::Mul)),
        OP_DIV => Some(Inst::BinOp(BinOpKind::Div)),
        OP_MOD => Some(Inst::BinOp(BinOpKind::Mod)),
        OP_CMP => Some(Inst::BinOp(BinOpKind::Cmp)),
        OP_SEL => Some(Inst::Sel(val as usize)),
        OP_MOV => Some(Inst::Mov(val as usize)),
        OP_POPNUM => Some(Inst::PopNum),
        OP_POPCHAR => Some(Inst::PopChar),
        OP_PUSHNUM => Some(Inst::PushNum),
        OP_PUSHCHAR => Some(Inst::PushChar),
        OP_NONE => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_empty() {
        let compiler = Compiler::new();
        let cfg = build_cfg(&compiler);
        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.block(0).terminator, Terminator::Halt);
    }

    #[test]
    fn test_build_push_halt() {
        let mut compiler = Compiler::new();
        compiler.compile("밤희");
        let cfg = build_cfg(&compiler);
        // PUSH 4; HALT → one block with [Push(4)] + Halt
        assert_eq!(cfg.num_blocks(), 1);
        assert_eq!(cfg.block(0).instructions, vec![Inst::Push(4)]);
        assert_eq!(cfg.block(0).terminator, Terminator::Halt);
    }

    #[test]
    fn test_build_push_add_halt() {
        let mut compiler = Compiler::new();
        compiler.compile("반반다희");
        let cfg = build_cfg(&compiler);
        // Should have blocks for: PUSH, PUSH, BRPOP2 → ADD, HALT path, fail paths
        assert!(cfg.num_blocks() >= 2);
        // Entry block should contain PUSH instructions
        let entry = cfg.block(0);
        assert!(
            entry
                .instructions
                .iter()
                .any(|i| matches!(i, Inst::Push(_)))
        );
    }

    #[test]
    fn test_build_preserves_structure() {
        let mut compiler = Compiler::new();
        compiler.compile("반반다희");
        let cfg = build_cfg(&compiler);

        // Verify all successor references are valid
        for block in &cfg.blocks {
            for &succ in &block.all_successors() {
                assert!(
                    (succ as usize) < cfg.num_blocks(),
                    "block {} has invalid successor {}",
                    block.id,
                    succ
                );
            }
        }
    }
}
