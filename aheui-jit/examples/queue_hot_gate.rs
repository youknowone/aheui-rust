//! Correctness gate for the JIT-compiled queue storage path.
//!
//! Builds a PURE-QUEUE program (exactly one `SEL 21`; every storage op
//! touches only the queue) whose hot loop pops the queue via `POPNUM`,
//! exercising the inlined `queue_pop`. Running the same program under a low
//! threshold (the pop loop compiles) and a huge threshold (nothing compiles,
//! plain interpretation) must yield byte-identical stdout — that equality is
//! the gate: compiled inlined `queue_pop` == interpreted `queue_pop`.
//!
//! Layout (K distinct values so FIFO order is discriminating):
//!   pc 0        : SEL 21          # select the queue (the only SEL)
//!   pc 1..=K    : PUSH i          # push 0,1,..,K-1 (FIFO front = 0)
//!   pc LOOP     : BRPOP1 END      # queue empty -> exit loop
//!   pc LOOP+1   : POPNUM          # pop front + print (inlined queue_pop)
//!   pc LOOP+2   : JMP  LOOP       # back-edge (the hot loop)
//!   pc END      : HALT            # mid-program HALT keeps the compiled
//!                                 # trace from ever running off the end
//!
//! Usage: `queue_hot_gate <threshold> [K]`
//!   threshold is passed straight to `aheui_jit::mainloop`, so behavior is
//!   deterministic regardless of the `MAJIT_THRESHOLD` env var.

use std::collections::HashMap;

use ahsembler::compiler::Program;
use ahsembler::consts::{OP_BRPOP1, OP_HALT, OP_JMP, OP_POPNUM, OP_PUSH, OP_SEL, VAL_QUEUE};

// Label ids kept well clear of any valid pc so `resolve_jump_targets`
// rewrites them via the `labels` map rather than colliding with a real pc.
const LABEL_LOOP: i32 = 1_000_001;
const LABEL_END: i32 = 1_000_002;

fn build_pure_queue_loop(k: i32) -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    // pc 0: SEL 21 — select the queue (the ONLY storage selection).
    opcodes.push(OP_SEL);
    values.push(VAL_QUEUE as i32);

    // pc 1..=K: push distinct values 0..K into the queue (tail-append).
    for i in 0..k {
        opcodes.push(OP_PUSH);
        values.push(i);
    }

    let loop_pc = opcodes.len(); // BRPOP1
    let end_pc = loop_pc + 3; // HALT

    // pc LOOP: BRPOP1 END — branch to HALT when the queue is empty.
    opcodes.push(OP_BRPOP1);
    values.push(LABEL_END);

    // pc LOOP+1: POPNUM — pop the front of the queue and print it.
    opcodes.push(OP_POPNUM);
    values.push(-1);

    // pc LOOP+2: JMP LOOP — hot back-edge.
    opcodes.push(OP_JMP);
    values.push(LABEL_LOOP);

    // pc END: HALT.
    opcodes.push(OP_HALT);
    values.push(-1);

    let size = opcodes.len();
    debug_assert_eq!(end_pc, size - 1);

    let mut labels: HashMap<i32, usize> = HashMap::new();
    labels.insert(LABEL_LOOP, loop_pc);
    labels.insert(LABEL_END, end_pc);

    let mut program = Program {
        opcodes,
        values,
        labels,
        size,
    };
    // Rewrite jump operands from label ids to target pcs.
    program.resolve_jump_targets();
    program
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threshold: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .expect("usage: queue_hot_gate <threshold> [K]");
    let k: i32 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5000);

    aheui_jit::init_gc_subsystem();
    let program = build_pure_queue_loop(k);
    // Prints the popped queue values to stdout; returns the exit Val.
    let _exit = aheui_jit::mainloop(&program, threshold);
}
