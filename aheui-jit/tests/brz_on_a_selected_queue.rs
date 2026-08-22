//! `OP_BRZ` must decide the branch from the value it popped, on a queue.
//!
//! The two storages take different paths through that arm: the stack pop is
//! hand-inlined (head/next/size stores plus `jit_free_node`), the queue pop is
//! `lj::queue_pop`. Only the zero test is shared — one comparison on the popped
//! `Val` — and the two halves lower separately.
//!
//! The program discriminates three ways through the value `mainloop` returns:
//!
//! * `SENTINEL` — correct: the zero test fired on the terminator, and on
//!   nothing else.
//! * `FILLER`   — fired too early, on the first nonzero element.
//! * `0`        — never fired; the loop drained the queue and left through the
//!   `BRPOP1` safety exit instead.
//!
//! A bare "did it terminate" assertion passes in the second case, and a stdout
//! comparison passes in all three since the program prints nothing.

mod common;

use ahsembler::compiler::Program;
use ahsembler::consts::{OP_BRPOP1, OP_BRZ, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, VAL_QUEUE};
use common::{Asm, ITERATIONS, LABEL_ALT, LABEL_END, LABEL_LOOP, NONE};

/// The nonzero element. Distinct from the sentinel so an early branch is
/// distinguishable from a correct one by the returned value alone.
const FILLER: i32 = 2;
/// Pushed last, so it is still queued when the branch is taken. FIFO: the
/// pushes come out in the order they went in, so this is what a correct run
/// leaves behind.
const SENTINEL: i32 = 7;

/// `SEL 21` then `n` nonzero elements, a zero terminator and the sentinel,
/// drained by a loop whose only intended exit is `BRZ`.
///
/// The `BRPOP1` is not loop control — `BRZ`'s own arm pops unconditionally, so
/// the guard in front of it is what keeps a never-firing zero test from reading
/// an empty queue. Reaching its target means the zero test never fired, which
/// is why it lands on its own `HALT` with the queue empty rather than sharing
/// the exit.
fn zero_terminated_queue_drain(n: i32) -> Program {
    let mut asm = Asm::new();
    asm.emit(OP_SEL, VAL_QUEUE as i32)
        .emit_n(n, OP_PUSH, |_| FILLER)
        .emit_all([(OP_PUSH, 0), (OP_PUSH, SENTINEL)])
        .label(LABEL_LOOP)
        .emit_all([
            (OP_BRPOP1, LABEL_END),
            (OP_BRZ, LABEL_ALT),
            (OP_JMP, LABEL_LOOP),
        ])
        .label(LABEL_END)
        .emit(OP_HALT, NONE)
        .label(LABEL_ALT)
        .emit(OP_HALT, NONE);
    asm.build()
}

#[test]
fn brz_takes_its_branch_on_the_zero_it_popped_from_a_queue() {
    assert_eq!(
        common::run(&zero_terminated_queue_drain(ITERATIONS)),
        SENTINEL as i64,
        "OP_BRZ on a selected queue branched on the wrong element. \
         Expected {SENTINEL} (the branch was taken on the zero terminator and \
         the sentinel behind it survived); {FILLER} means it fired on the \
         first nonzero element, 0 means it never fired and the loop left \
         through the BRPOP1 drain exit"
    );
    common::assert_compiled("the queue arm's lowering");
}
