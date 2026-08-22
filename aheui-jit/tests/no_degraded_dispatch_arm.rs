//! No arm of the aheui dispatch may lower to an abort stub.
//!
//! When `#[jit_interp]` cannot express an arm body it does not fail the build.
//! It emits an abort stub for that arm and records the fact through
//! `record_degraded_dispatch_arm`, so the interpreter still runs and every
//! trace reaching the arm aborts instead. The damage is scoped to whichever
//! programs execute that opcode: they compile nothing and re-abort once per
//! threshold, forever.
//!
//! No other gate here sees it. The corpus baselines all record
//! `loops_aborted=0` because no pinned program executes a degraded arm, and an
//! abort that never happens leaves no counter to compare against; the op census
//! is blind for the same reason, since a program that compiles nothing
//! contributes no ops.
//!
//! Read through `degraded_dispatch_arms()` rather than by matching the
//! `[jit] degraded dispatch arm:` line out of `MAJIT_LOG`: that line exists
//! only when the log is on, and capturing it takes a `2>&1 |` redirection whose
//! meaning differs between shells.

mod common;

use ahsembler::consts::{OP_POP, OP_PUSHNUM};
use common::{ITERATIONS, NONE};

/// The `state = T` name every aheui dispatch arm is recorded under. Other
/// `#[jit_interp]` machines in the process record under their own names and
/// are not this test's subject.
const AHEUI_INTERP: &str = "AheuiState";

#[test]
fn no_aheui_dispatch_arm_lowered_to_an_abort_stub() {
    // The degraded-arm recording happens when the dispatch JitCode is
    // installed, so any program reaching the install would prove the arm
    // lowers. Running the input op inside the loop proves what actually
    // matters instead: a trace *containing* it compiles.
    //
    // `PUSHNUM` pushes the EOF 0 and the first `POP` discards it, so the pair
    // is stack-neutral and the second `POP` drains one real element. Plain
    // `OP_POP` prints nothing; `OP_POPNUM` would write to stdout.
    let program = common::drain_loop(
        ITERATIONS,
        &[(OP_PUSHNUM, NONE), (OP_POP, NONE), (OP_POP, NONE)],
    );
    let _exit = common::run(&program);

    // The precondition comes first: an empty degraded list means either "no arm
    // is degraded" or "no arm was looked at", and only the first is a pass.
    // Because the loop holds the input op, this is also the end-to-end fact the
    // degraded-arm assertion below only implies — while that arm is an abort
    // stub, the loop compiles nothing at any threshold.
    common::assert_compiled("a loop whose body runs OP_PUSHNUM");

    // The assertion itself belongs to whoever owns the census, and it carries
    // what a filter here cannot: the arm count the degraded ones are out of,
    // and a panic when this machine has no census entry at all.
    majit_metainterp::assert_no_degraded_dispatch_arms(AHEUI_INTERP);
}
