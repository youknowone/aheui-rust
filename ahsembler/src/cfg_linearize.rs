//! Linearize a CFG back into a flat Aheui Program.
//!
//! Block layout: reverse postorder for maximal fallthrough.

use std::collections::HashMap;

use crate::cfg::*;
use crate::compiler::Program;
use crate::consts::*;

/// Linearize a CFG into a flat Program.
pub fn linearize(cfg: &Cfg) -> Program {
    let order = compute_layout(cfg);
    let mut block_pc: HashMap<BlockId, usize> = HashMap::new();

    // First pass: compute the starting PC for each block.
    let mut pc = 0usize;
    for &block_id in &order {
        block_pc.insert(block_id, pc);
        let block = cfg.block(block_id);
        pc += block.instructions.len();
        pc += terminator_size(&block.terminator, block_id, &order, &block_pc);
    }

    let total_size = pc;
    let mut opcodes = Vec::with_capacity(total_size);
    let mut values = Vec::with_capacity(total_size);
    let mut labels: HashMap<i32, usize> = HashMap::new();
    let mut next_label_id = 0i32;

    // Second pass: emit bytecode.
    for (order_idx, &block_id) in order.iter().enumerate() {
        let block = cfg.block(block_id);
        let next_block = order.get(order_idx + 1).copied();

        // Emit instructions
        for inst in &block.instructions {
            let (op, val) = inst_to_opcode(inst);
            opcodes.push(op);
            values.push(val);
        }

        // Emit terminator
        match &block.terminator {
            Terminator::Goto(target) => {
                if next_block == Some(*target) {
                    // Fallthrough — no instruction needed
                } else {
                    let label = next_label_id;
                    next_label_id += 1;
                    labels.insert(label, block_pc[target]);
                    opcodes.push(OP_JMP);
                    values.push(label);
                }
            }
            Terminator::StackGuard {
                min_depth,
                ok,
                fail,
            } => {
                let brop = if *min_depth <= 1 {
                    OP_BRPOP1
                } else {
                    OP_BRPOP2
                };

                // BRPOP jumps to fail on underflow, falls through to ok
                let fail_label = next_label_id;
                next_label_id += 1;
                labels.insert(fail_label, block_pc[fail]);
                opcodes.push(brop);
                values.push(fail_label);

                // If ok is not the next block, emit a JMP
                if next_block != Some(*ok) {
                    let ok_label = next_label_id;
                    next_label_id += 1;
                    labels.insert(ok_label, block_pc[ok]);
                    opcodes.push(OP_JMP);
                    values.push(ok_label);
                }
            }
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } => {
                // BRZ jumps to on_zero, falls through to on_nonzero
                let zero_label = next_label_id;
                next_label_id += 1;
                labels.insert(zero_label, block_pc[on_zero]);
                opcodes.push(OP_BRZ);
                values.push(zero_label);

                // If on_nonzero is not the next block, emit a JMP
                if next_block != Some(*on_nonzero) {
                    let nz_label = next_label_id;
                    next_label_id += 1;
                    labels.insert(nz_label, block_pc[on_nonzero]);
                    opcodes.push(OP_JMP);
                    values.push(nz_label);
                }
            }
            Terminator::Halt => {
                opcodes.push(OP_HALT);
                values.push(-1);
            }
        }
    }

    let size = opcodes.len();
    Program {
        opcodes,
        values,
        labels,
        size,
    }
}

/// Number of opcodes the terminator will emit.
fn terminator_size(
    term: &Terminator,
    block_id: BlockId,
    order: &[BlockId],
    _block_pc: &HashMap<BlockId, usize>,
) -> usize {
    let order_idx = order.iter().position(|&b| b == block_id);
    let next_block = order_idx.and_then(|i| order.get(i + 1).copied());

    match term {
        Terminator::Goto(target) => {
            if next_block == Some(*target) {
                0 // Fallthrough
            } else {
                1 // JMP
            }
        }
        Terminator::StackGuard { ok, .. } => {
            if next_block == Some(*ok) {
                1 // BRPOP only
            } else {
                2 // BRPOP + JMP
            }
        }
        Terminator::BranchZero { on_nonzero, .. } => {
            if next_block == Some(*on_nonzero) {
                1 // BRZ only
            } else {
                2 // BRZ + JMP
            }
        }
        Terminator::Halt => 1,
    }
}

/// Compute block layout that maximizes fallthrough.
///
/// Greedy trace selection: starting from each unvisited block, follow the
/// "preferred" successor (the one that benefits most from fallthrough) to
/// build traces. Preferred successor priority:
/// - Goto target (always benefits from fallthrough — eliminates a JMP)
/// - StackGuard ok path (BRPOP falls through to ok)
/// - BranchZero on_nonzero path (BRZ falls through to nonzero)
fn compute_layout(cfg: &Cfg) -> Vec<BlockId> {
    let n = cfg.num_blocks();
    if n == 0 {
        return vec![];
    }

    let mut placed = vec![false; n];
    let mut order = Vec::with_capacity(n);

    // Compute in-degree for preferred edges to prioritize chains
    let rpo = cfg.reverse_postorder();

    // Start traces from RPO order (ensures entry block is first)
    for &seed in &rpo {
        if placed[seed as usize] {
            continue;
        }

        // Build a trace starting from seed
        let mut current = seed;
        loop {
            if placed[current as usize] {
                break;
            }
            placed[current as usize] = true;
            order.push(current);

            // Pick the preferred fallthrough successor
            let next = preferred_successor(cfg.block(current));
            match next {
                Some(succ) if !placed[succ as usize] => {
                    current = succ;
                }
                _ => break,
            }
        }
    }

    order
}

/// The successor that benefits most from being placed immediately after this block.
fn preferred_successor(block: &CfgBlock) -> Option<BlockId> {
    match &block.terminator {
        Terminator::Goto(target) => Some(*target),
        Terminator::StackGuard { ok, .. } => Some(*ok),
        Terminator::BranchZero { on_nonzero, .. } => Some(*on_nonzero),
        Terminator::Halt => None,
    }
}

fn inst_to_opcode(inst: &Inst) -> (u8, i32) {
    match inst {
        // `Program::values` is a `Vec<i32>`, so a wider `Push` cannot be
        // carried; truncating one silently changes what the program computes.
        // Source literals are small and `fold_constants` is bounded by
        // `ConstWidth::I32` on every path that reaches here, so this is an
        // invariant rather than a case to handle.
        Inst::Push(v) => (
            OP_PUSH,
            i32::try_from(*v)
                .unwrap_or_else(|_| panic!("Push({v}) does not fit the linearized i32 operand")),
        ),
        Inst::Pop => (OP_POP, -1),
        Inst::Dup => (OP_DUP, -1),
        Inst::Swap => (OP_SWAP, -1),
        Inst::BinOp(BinOpKind::Add) => (OP_ADD, -1),
        Inst::BinOp(BinOpKind::Sub) => (OP_SUB, -1),
        Inst::BinOp(BinOpKind::Mul) => (OP_MUL, -1),
        Inst::BinOp(BinOpKind::Div) => (OP_DIV, -1),
        Inst::BinOp(BinOpKind::Mod) => (OP_MOD, -1),
        Inst::BinOp(BinOpKind::Cmp) => (OP_CMP, -1),
        Inst::Sel(s) => (OP_SEL, *s as i32),
        Inst::Mov(s) => (OP_MOV, *s as i32),
        Inst::PopNum => (OP_POPNUM, -1),
        Inst::PopChar => (OP_POPCHAR, -1),
        Inst::PushNum => (OP_PUSHNUM, -1),
        Inst::PushChar => (OP_PUSHCHAR, -1),
        Inst::GuardDepth { .. } => unreachable!("GuardDepth should not appear in linearized code"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linearize_simple_halt() {
        let mut cfg = Cfg::new();
        cfg.add_block(vec![Inst::Push(42)], Terminator::Halt);

        let program = linearize(&cfg);
        assert_eq!(program.size, 2);
        assert_eq!(program.opcodes, vec![OP_PUSH, OP_HALT]);
        assert_eq!(program.values[0], 42);
    }

    #[test]
    fn test_linearize_fallthrough() {
        let mut cfg = Cfg::new();
        cfg.add_block(vec![Inst::Push(1)], Terminator::Goto(1));
        cfg.add_block(vec![Inst::Push(2)], Terminator::Halt);

        let program = linearize(&cfg);
        // B0 falls through to B1 — no JMP emitted
        assert_eq!(program.size, 3); // PUSH 1, PUSH 2, HALT
        assert_eq!(program.opcodes, vec![OP_PUSH, OP_PUSH, OP_HALT]);
    }

    #[test]
    fn test_linearize_branch() {
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(5)],
            Terminator::BranchZero {
                on_zero: 1,
                on_nonzero: 2,
            },
        );
        cfg.add_block(vec![Inst::Push(10)], Terminator::Halt); // on_zero
        cfg.add_block(vec![Inst::Push(20)], Terminator::Halt); // on_nonzero

        let program = linearize(&cfg);
        // PUSH 5, BRZ L_zero, PUSH 20, HALT, L_zero: PUSH 10, HALT
        // RPO order might vary, but structure should be valid
        assert!(program.size >= 4);

        // Verify all label targets are valid
        for (_, &target) in &program.labels {
            assert!(target < program.size, "label target {target} out of range");
        }
    }

    #[test]
    fn test_linearize_stack_guard() {
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(1)],
            Terminator::StackGuard {
                min_depth: 2,
                ok: 1,
                fail: 2,
            },
        );
        cfg.add_block(vec![Inst::BinOp(BinOpKind::Add)], Terminator::Halt);
        cfg.add_block(vec![], Terminator::Halt);

        let program = linearize(&cfg);
        assert!(program.size >= 3);

        // Should contain a BRPOP2
        assert!(program.opcodes.contains(&OP_BRPOP2));
    }

    #[test]
    fn test_roundtrip_push_add_halt() {
        // Build a CFG that represents PUSH 2, PUSH 3, ADD, HALT
        // After optimization, should produce PUSH 5, HALT
        use crate::cfg_optimize::optimize_cfg;

        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(2), Inst::Push(3)],
            Terminator::StackGuard {
                min_depth: 2,
                ok: 1,
                fail: 2,
            },
        );
        cfg.add_block(vec![Inst::BinOp(BinOpKind::Add)], Terminator::Halt);
        cfg.add_block(vec![], Terminator::Halt);

        let optimized = optimize_cfg(cfg, crate::cfg_optimize::ConstWidth::I32);
        let program = linearize(&optimized);

        assert_eq!(program.opcodes, vec![OP_PUSH, OP_HALT]);
        assert_eq!(program.values[0], 5);
    }
}
