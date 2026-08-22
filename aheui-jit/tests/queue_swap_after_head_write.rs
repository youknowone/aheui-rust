//! `OP_SWAP` must read the `head` its own trace just wrote, on a queue.
//!
//! `Stack`, `Queue` and `Port` deliberately pun a `{ head, size }` prefix, and
//! a field descriptor carries the nominal struct its base was declared as. The
//! swap arm reaches `head` through the selected-storage reference, which is
//! declared `Stack`; a queue op spliced into the same trace writes `head` under
//! `Queue`. That is two descriptor identities for one word, and the optimizer's
//! field cache is keyed by identity: a read under one is neither served by, nor
//! forces, a pending store under the other. What that costs is a swap performed
//! at the node `head` pointed at *before* the write — one node too far down the
//! chain, exchanging the wrong pair of values.
//!
//! The loop swaps in three positions that differ in what wrote `head` last:
//! after nothing, after a duplicate, and after the two pops a fold performs.
//! Displacing any one of them by a single node moves the final value, and moves
//! it to a different wrong value in each case, so a mismatch says which
//! position failed rather than only that one did.

mod common;

use ahsembler::compiler::Program;
use ahsembler::consts::{
    OP_ADD, OP_BRPOP1, OP_BRPOP2, OP_DUP, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, OP_SWAP, VAL_QUEUE,
};
use common::{Asm, LABEL_END, LABEL_LOOP, NONE, QOp, SEED, seed_value};

/// `SEL 21; PUSH×SEED; loop { SWAP; DUP; SWAP; ADD; SWAP; ADD }; HALT`.
fn queue_swap_drain() -> Program {
    let mut asm = Asm::new();
    asm.emit(OP_SEL, VAL_QUEUE as i32)
        .emit_n(SEED, OP_PUSH, seed_value)
        .label(LABEL_LOOP)
        // A swap with no preceding head write in this trip.
        .emit_all([(OP_BRPOP2, LABEL_END), (OP_SWAP, NONE)])
        // A swap behind a duplicate, which links a new node ahead of `head`.
        .emit_all([
            (OP_BRPOP1, LABEL_END),
            (OP_DUP, NONE),
            (OP_BRPOP2, LABEL_END),
            (OP_SWAP, NONE),
        ])
        // A swap behind a fold, whose two pops advance `head` twice.
        .emit_all([
            (OP_BRPOP2, LABEL_END),
            (OP_ADD, NONE),
            (OP_BRPOP2, LABEL_END),
            (OP_SWAP, NONE),
            (OP_BRPOP2, LABEL_END),
            (OP_ADD, NONE),
        ])
        .emit(OP_JMP, LABEL_LOOP)
        .label(LABEL_END)
        .emit(OP_HALT, NONE);
    asm.build()
}

#[test]
fn a_queue_swap_reads_the_head_its_own_trace_wrote() {
    let expected = common::drain_queue_model(&[
        QOp::Swap,
        QOp::Dup,
        QOp::Swap,
        QOp::Fold,
        QOp::Swap,
        QOp::Fold,
    ]);

    assert_eq!(
        common::run(&queue_swap_drain()),
        expected,
        "the queue drain disagrees with the storage's own rules. A swap that \
         read `head` from before the duplicate or the pops ahead of it lands on \
         the wrong pair of nodes, and each of the loop's three swap positions \
         fails to a different value"
    );
    common::assert_compiled("the queue swap arm's lowering");
}
