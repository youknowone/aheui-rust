//! Reproduction harness for `get_parent_descr` failures when one trace contains
//! storage nodes with different descriptor identities.
//!
//! Builds a hot loop in one of seven storage-mix modes and runs it under a
//! caller-supplied threshold (so the JIT decision is deterministic regardless
//! of the `MAJIT_THRESHOLD` env var):
//!
//! - `queue` uses only queue `Node` allocation and reads.
//! - `stack` combines a stack `NodeJit` allocation with a `Node` read.
//! - `mixed` combines queue reads with stack allocation and reads.
//! - `movmix` combines `Node` and `NodeJit` allocations through `MOV`.
//! - `dupmov` combines both allocation types through `DUP` and `MOV`.
//! - `dup_only` is the single-descriptor control for `dupmov`.
//! - `mov_only` is the `NodeJit`-only control for `movmix`.
//!
//! Usage: `mixed_repro <mode> <threshold> [K]`

use std::collections::HashMap;

use ahsembler::compiler::Program;
use ahsembler::consts::{
    OP_BRPOP1, OP_DUP, OP_HALT, OP_JMP, OP_MOV, OP_POP, OP_POPCHAR, OP_POPNUM, OP_PUSH, OP_SEL,
    VAL_QUEUE,
};

const LABEL_LOOP: i32 = 1_000_001;
const LABEL_END: i32 = 1_000_002;

const STACK_SLOT: i32 = 1; // any non-queue storage id (a stack)

fn build(mode: &str, k: i32) -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    let push = |op: u8, v: i32, opc: &mut Vec<u8>, val: &mut Vec<i32>| {
        opc.push(op);
        val.push(v);
    };

    // Fill the storage consumed by the loop with K values.
    // `queue` and `mixed` drain the queue; every other mode drains stack 1.
    let drain_sel = match mode {
        "queue" | "mixed" => VAL_QUEUE as i32,
        _ => STACK_SLOT,
    };
    push(OP_SEL, drain_sel, &mut opcodes, &mut values);
    for i in 0..k {
        push(OP_PUSH, i, &mut opcodes, &mut values);
    }

    let loop_pc = opcodes.len();

    match mode {
        // Queue-only path: SEL 21; BRPOP1 END; POPNUM; JMP LOOP.
        "queue" => {
            push(OP_SEL, VAL_QUEUE as i32, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // Stack-only path: SEL 1; BRPOP1 END; POPNUM;
        // PUSH 42; POPCHAR (net-zero NodeJit push plus Node read); JMP LOOP.
        "stack" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
            push(OP_PUSH, 42, &mut opcodes, &mut values);
            push(OP_POPCHAR, -1, &mut opcodes, &mut values);
        }
        // Mixed path: drain the queue, then push and pop a stack `NodeJit`.
        "mixed" => {
            push(OP_SEL, VAL_QUEUE as i32, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_PUSH, 42, &mut opcodes, &mut values);
            push(OP_POPCHAR, -1, &mut opcodes, &mut values);
        }
        // `movmix` drains stack 1; each iteration does:
        //   PUSH 5    -> lj::stack_push  = aheui_runtime `Node` New  (type_id A)
        //   MOV 2     -> inline `NodeJit` literal = aheui-jit `NodeJit` New (type_id B)
        //   POPNUM    -> drain stack 1 by 1 (Node read)
        // This puts two Node-layout SizeDescr identities in one trace without
        // involving the queue.
        "movmix" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_PUSH, 5, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values); // pop stack1 -> push stack2 (NodeJit)
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // `dupmov` is the DUP variant of `movmix`:
        //   DUP    -> lj::stack_dup  = aheui_runtime `Node` New   (type_id A)
        //   MOV 2  -> inline `NodeJit` literal = aheui-jit `NodeJit` New (type_id B)
        //   POPNUM -> net drain (DUP +1, MOV -1, POPNUM -1 = -1/iter)
        "dupmov" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_DUP, -1, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // Single-descriptor control: DUP allocates `Node`, then POP drains it.
        "dup_only" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_DUP, -1, &mut opcodes, &mut values);
            push(OP_POP, -1, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // `NodeJit`-only control: move to stack 2, then drain stack 1.
        "mov_only" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        other => panic!("unknown mode {other:?}"),
    }

    // Common back-edge.
    push(OP_JMP, LABEL_LOOP, &mut opcodes, &mut values);
    let end_pc = opcodes.len();
    push(OP_HALT, -1, &mut opcodes, &mut values);

    let size = opcodes.len();

    let mut labels: HashMap<i32, usize> = HashMap::new();
    labels.insert(LABEL_LOOP, loop_pc);
    labels.insert(LABEL_END, end_pc);

    let mut program = Program {
        opcodes,
        values,
        labels,
        size,
    };
    program.resolve_jump_targets();
    program
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mode = args
        .next()
        .expect("usage: mixed_repro <mode> <threshold> [K]");
    let threshold: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .expect("usage: mixed_repro <mode> <threshold> [K]");
    let k: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    aheui_jit::init_gc_subsystem();
    let program = build(&mode, k);
    let _exit = aheui_jit::mainloop(&program, threshold);
}
