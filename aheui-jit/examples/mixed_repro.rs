//! Minimal reproduction harness for the optimizeopt:8350 `get_parent_descr`
//! None panic (aheui task #22 BUG 1).
//!
//! Builds a hot loop in one of three storage-mix modes and runs it under a
//! caller-supplied threshold (so the JIT decision is deterministic regardless
//! of the `MAJIT_THRESHOLD` env var):
//!
//!   mode "queue" : PURE queue  — every storage op touches SEL 21 (the queue).
//!                  Uses only aheui_runtime `Node` (queue_push/queue_pop).
//!   mode "stack" : PURE stack  — every storage op touches SEL 1 (a stack).
//!                  Loop pushes a const via the `NodeJit` literal arm and pops
//!                  it, so BOTH `NodeJit` (New+SetfieldGc) and `Node`
//!                  (GetfieldGc read) appear in ONE trace.
//!   mode "mixed" : queue POPNUM + stack PUSH/POPCHAR in one loop body — the
//!                  99dan print-loop shape (queue `Node` read + stack `NodeJit`
//!                  write + stack `Node` read all in one trace).
//!
//! Usage: `mixed_repro <queue|stack|mixed> <threshold> [K]`

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

    // Fill the DRAIN storage with K values so the loop runs K times.
    // queue/mixed drain the queue; stack/movmix/dupmov drain stack 1.
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
        // PURE queue: SEL 21; BRPOP1 END; POPNUM; JMP LOOP
        "queue" => {
            push(OP_SEL, VAL_QUEUE as i32, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // PURE stack: SEL 1; BRPOP1 END; POPNUM(drain);
        //             PUSH 42; POPCHAR(net-0 NodeJit push + Node read); JMP LOOP
        "stack" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
            push(OP_PUSH, 42, &mut opcodes, &mut values);
            push(OP_POPCHAR, -1, &mut opcodes, &mut values);
        }
        // MIXED: SEL 21; BRPOP1 END; POPNUM(queue drain);
        //        SEL 1; PUSH 42; POPCHAR(stack NodeJit push + Node read); JMP LOOP
        "mixed" => {
            push(OP_SEL, VAL_QUEUE as i32, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_PUSH, 42, &mut opcodes, &mut values);
            push(OP_POPCHAR, -1, &mut opcodes, &mut values);
        }
        // MOVMIX: PURE stack. Draws stack 1; per iter does
        //   PUSH 5    -> lj::stack_push  = aheui_runtime `Node` New  (type_id A)
        //   MOV 2     -> inline `NodeJit` literal = aheui-jit `NodeJit` New (type_id B)
        //   POPNUM    -> drain stack 1 by 1 (Node read)
        // => TWO distinct Node-layout SizeDescr type_ids in ONE trace, NO queue.
        // SEL 1; BRPOP1 END; PUSH 5; MOV 2; POPNUM; JMP LOOP
        "movmix" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_PUSH, 5, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values); // pop stack1 -> push stack2 (NodeJit)
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // DUPMOV: PURE stack, DUP variant matching 99dan's pc51-66 shape.
        //   DUP    -> lj::stack_dup  = aheui_runtime `Node` New   (type_id A)
        //   MOV 2  -> inline `NodeJit` literal = aheui-jit `NodeJit` New (type_id B)
        //   POPNUM -> net drain (DUP +1, MOV -1, POPNUM -1 = -1/iter)
        // SEL 1; BRPOP1 END; DUP; MOV 2; POPNUM; JMP LOOP
        "dupmov" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_DUP, -1, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // DUP_ONLY control: DUP (Node New) + POP drain. ONE type only.
        // SEL 1; BRPOP1 END; DUP; POP; POPNUM; JMP LOOP
        "dup_only" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_DUP, -1, &mut opcodes, &mut values);
            push(OP_POP, -1, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        // MOV_ONLY control: MOV to stack 2 (NodeJit New) + drain, NO Node New.
        // SEL 1; BRPOP1 END; MOV 2; POPNUM; JMP LOOP
        "mov_only" => {
            push(OP_SEL, STACK_SLOT, &mut opcodes, &mut values);
            push(OP_BRPOP1, LABEL_END, &mut opcodes, &mut values);
            push(OP_MOV, 2, &mut opcodes, &mut values);
            push(OP_POPNUM, -1, &mut opcodes, &mut values);
        }
        other => panic!("unknown mode {other:?}"),
    }

    // JMP LOOP back-edge.
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
        .expect("usage: mixed_repro <queue|stack|mixed> <threshold> [K]");
    let threshold: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .expect("usage: mixed_repro <queue|stack|mixed> <threshold> [K]");
    let k: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2000);

    aheui_jit::init_gc_subsystem();
    let program = build(&mode, k);
    let _exit = aheui_jit::mainloop(&program, threshold);
}
