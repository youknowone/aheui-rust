//! `OP_SEL` must read the `size` its own trace just wrote, on a queue.
//!
//! Selecting a storage re-reads its element count: `Storage.select` is
//! `selected = storage[value]; stacksize = len(selected)`, and `len` is a load
//! of the list's mutable `size`. The arm mirrors that as a `getarrayitem_gc_r`
//! on the storage base followed by a `getfield_gc_i` through the loaded
//! reference. Both halves are re-derived every trip rather than carried, which
//! is what keeps `stacksize` from freezing to a loop-invariant constant.
//!
//! That re-read is only correct if it observes the writes the same trip already
//! performed. `size` is written by the spliced bodies of the storage helpers —
//! a fold decrements it, a duplicate increments it — and the optimizer's field
//! cache decides per field descriptor whether a later load is served from an
//! entry, forced against a pending store, or emitted afresh. So the guarantee
//! holds exactly as long as the helper's write and the select's read name one
//! descriptor for one word.
//!
//! A displaced `size` does not crash: it moves the trip at which the
//! branch-on-underflow fires, so the program drains a trip early or late and
//! halts on a different value. The drain is sensitive in both directions. Read
//! one too low it ends in 399 trips on 43777 instead of 400 on 236890; read one
//! too high the guard admits a fold on a queue holding fewer than two elements
//! and the second pop walks off the end of the chain.
//!
//! The two `SEL 21`s re-select the storage that is already selected. That is a
//! no-op on the data and the point: it forces the count to be read again
//! through the reference, immediately behind a helper that wrote it.

mod common;

use ahsembler::compiler::Program;
use ahsembler::consts::{
    OP_ADD, OP_BRPOP1, OP_BRPOP2, OP_DUP, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, VAL_QUEUE,
};
use common::{Asm, LABEL_END, LABEL_LOOP, NONE, QOp, SEED, seed_value};

/// `SEL 21; PUSH×SEED; loop { ADD; SEL 21; DUP; SEL 21; ADD }; HALT`.
fn reselect_drain() -> Program {
    let mut asm = Asm::new();
    asm.emit(OP_SEL, VAL_QUEUE as i32)
        .emit_n(SEED, OP_PUSH, seed_value)
        .label(LABEL_LOOP)
        // A fold: two pops and a push at the back, so `size` ends one lower.
        // Then re-read the count behind that write.
        .emit_all([
            (OP_BRPOP2, LABEL_END),
            (OP_ADD, NONE),
            (OP_SEL, VAL_QUEUE as i32),
        ])
        // A duplicate: one node linked ahead of `head`, so `size` ends one
        // higher. Re-reading behind it moves the count the other way.
        .emit_all([
            (OP_BRPOP1, LABEL_END),
            (OP_DUP, NONE),
            (OP_SEL, VAL_QUEUE as i32),
        ])
        .emit_all([(OP_BRPOP2, LABEL_END), (OP_ADD, NONE)])
        .emit(OP_JMP, LABEL_LOOP)
        .label(LABEL_END)
        .emit(OP_HALT, NONE);
    asm.build()
}

#[test]
fn a_reselect_reads_the_size_its_own_trace_wrote() {
    let expected = common::drain_queue_model(&[QOp::Fold, QOp::Dup, QOp::Fold]);

    assert_eq!(
        common::run(&reselect_drain()),
        expected,
        "the queue drain disagrees with the storage's own rules. A select that \
         read the count from before the fold or the duplicate ahead of it fires \
         the branch-on-underflow at the wrong trip, so the loop drains to a \
         different depth and halts on a different value"
    );
    common::assert_compiled("the select arm's count re-read");
}
