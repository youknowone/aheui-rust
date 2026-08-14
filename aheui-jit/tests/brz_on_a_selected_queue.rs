//! `OP_BRZ` must decide the branch from the value it popped, on a queue.
//!
//! The two storages take different paths through that arm: the stack pop is
//! hand-inlined (head/next/size stores plus `jit_free_node`), the queue pop is
//! `lj::queue_pop`. Only the pop differs — the zero test is one comparison on
//! the popped `Val` either way — but "only" is a claim about source, and the
//! two halves lower separately.
//!
//! Nothing else in this repo covers the queue half. The 76-program corpus was
//! instrumented directly (2026-08-15): seven of its programs statically contain
//! both a `SEL 21` and a `BRZ`, and **all seven execute this arm zero times**.
//! Containing an opcode pair is not executing it, so the corpus A/B is vacuous
//! here and stays vacuous however many programs are added to it — the coverage
//! has to be built on purpose, which is what this file is.
//!
//! The program below discriminates three ways through the value `mainloop`
//! returns (aheui-jit/src/lib.rs, the `selected_dispatch_mut().pop()` tail):
//!
//! * `7`  — correct: the zero test fired on the terminator and on nothing else.
//! * `2`  — fired too early, on the first nonzero element.
//! * `0`  — never fired; the loop drained the queue and left through the
//!          `BRPOP1` safety exit instead.
//!
//! A bare "did it terminate" assertion would pass in the second case and a
//! stdout comparison would pass in all three, since the program prints nothing.

use ahsembler::compiler::Program;
use ahsembler::consts::{OP_BRPOP1, OP_BRZ, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, VAL_QUEUE};
use std::collections::HashMap;

const LABEL_LOOP: i32 = 1_000_001;
const LABEL_DONE: i32 = 1_000_002;
const LABEL_DRAINED: i32 = 1_000_003;

/// The nonzero element. Distinct from the sentinel so an early branch is
/// distinguishable from a correct one by the returned value alone.
const FILLER: i32 = 2;
/// Pushed last, so it is still queued when the branch is taken. FIFO: the
/// pushes come out in the order they went in, so this is what a correct run
/// leaves behind.
const SENTINEL: i32 = 7;

/// `SEL 21` then `n` nonzero elements, a zero terminator and the sentinel,
/// drained by a loop whose only exit is `BRZ`.
///
/// The `BRPOP1` is not loop control — `BRZ`'s own arm pops unconditionally, so
/// the guard in front of it is what keeps a never-firing zero test from reading
/// an empty queue. Reaching its target means the zero test never fired, which
/// is why it lands on its own `HALT` with the queue empty rather than sharing
/// the exit.
fn zero_terminated_queue_drain(n: i32) -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    // `-1` is the unused operand slot; only OP_PUSH, OP_SEL and the branches
    // read one.
    fn emit(opcodes: &mut Vec<u8>, values: &mut Vec<i32>, op: u8, val: i32) {
        opcodes.push(op);
        values.push(val);
    }

    emit(&mut opcodes, &mut values, OP_SEL, VAL_QUEUE as i32);
    for _ in 0..n {
        emit(&mut opcodes, &mut values, OP_PUSH, FILLER);
    }
    emit(&mut opcodes, &mut values, OP_PUSH, 0);
    emit(&mut opcodes, &mut values, OP_PUSH, SENTINEL);

    let loop_pc = opcodes.len();
    emit(&mut opcodes, &mut values, OP_BRPOP1, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_BRZ, LABEL_DONE);
    emit(&mut opcodes, &mut values, OP_JMP, LABEL_LOOP);

    let drained_pc = opcodes.len();
    emit(&mut opcodes, &mut values, OP_HALT, -1);
    let done_pc = opcodes.len();
    emit(&mut opcodes, &mut values, OP_HALT, -1);

    let size = opcodes.len();
    let mut labels: HashMap<i32, usize> = HashMap::new();
    labels.insert(LABEL_LOOP, loop_pc);
    labels.insert(LABEL_DRAINED, drained_pc);
    labels.insert(LABEL_DONE, done_pc);

    let mut program = Program {
        opcodes,
        values,
        labels,
        size,
    };
    program.resolve_jump_targets();
    program
}

/// Low enough that the loop compiles far inside `FILLERS`, high enough to leave
/// room above it.
const THRESHOLD: u32 = 8;
/// Iterations of the drain loop before the terminator. Well past `THRESHOLD`,
/// so the fall-through side runs compiled and the taken side leaves through a
/// guard rather than through the interpreter.
const FILLERS: i32 = 200;

/// One test, not several: `mainloop` installs process-global state (the
/// nursery, the GC roots, the raw/tagged value mode), so two `#[test]`s calling
/// it are two threads racing over it.
#[test]
fn brz_takes_its_branch_on_the_zero_it_popped_from_a_queue() {
    aheui_jit::init_gc_subsystem();
    let exit = aheui_jit::mainloop(&zero_terminated_queue_drain(FILLERS), THRESHOLD);
    let exit = aheui_runtime::value::val_to_i64(&exit);

    // Order matters: the returned value is the subject, but it is only
    // evidence about *compiled* code if a loop compiled. Assert the arm first,
    // then the fact that it was reached the way this test intends.
    assert_eq!(
        exit, SENTINEL as i64,
        "OP_BRZ on a selected queue branched on the wrong element. \
         Expected {SENTINEL} (the branch was taken on the zero terminator and \
         the sentinel behind it survived); {FILLER} means it fired on the \
         first nonzero element, 0 means it never fired and the loop left \
         through the BRPOP1 drain exit"
    );

    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0 && stats.loops_aborted == 0,
        "the drain loop did not compile (loops_compiled={}, loops_aborted={}), \
         so the assertion above graded the interpreter and left the queue \
         arm's lowering untested",
        stats.loops_compiled,
        stats.loops_aborted,
    );
}
