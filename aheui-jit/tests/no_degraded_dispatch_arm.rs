//! No arm of the aheui dispatch may lower to an abort stub.
//!
//! When `#[jit_interp]` cannot express an arm body it does not fail the build.
//! It emits an abort stub for that arm and records the fact through
//! `record_degraded_dispatch_arm` (majit-metainterp/src/lib.rs:503), so the
//! interpreter still runs and every trace reaching the arm aborts instead. The
//! damage is scoped to whichever programs execute that opcode: they compile
//! nothing and re-abort once per threshold, forever.
//!
//! Nothing here noticed. The corpus baselines cannot — all 62 record
//! `loops_aborted=0`, because no pinned program executes the degraded arms, and
//! an abort that never happens leaves no counter to compare against. Neither
//! can the op census, for the same reason: a program that compiles nothing
//! contributes no ops. Both gates are blind in exactly the region a degraded
//! arm occupies, which is why `OP_PUSHNUM`/`OP_PUSHCHAR` — ㅂ with a ㅇ or ㅎ
//! 받침, the two input instructions — stayed degraded from `d04aa52` until this
//! test was written.
//!
//! Read through `degraded_dispatch_arms()`, never by matching the
//! `[jit] degraded dispatch arm:` line out of `MAJIT_LOG`. That line exists
//! only when the log is on, and capturing it takes a `2>&1 |` redirection whose
//! meaning differs between shells — under zsh's MULTIOS the same pipeline
//! reports a count bash does not. The API is the same fact with no shell in it.

use ahsembler::compiler::Program;
use ahsembler::consts::{OP_BRPOP1, OP_HALT, OP_JMP, OP_POP, OP_PUSH, OP_PUSHNUM};
use std::collections::HashMap;

/// The `state = T` name every aheui dispatch arm is recorded under. Other
/// `#[jit_interp]` machines in the process record under their own names and
/// are not this test's subject.
const AHEUI_INTERP: &str = "AheuiState";

const LABEL_LOOP: i32 = 1_000_001;
const LABEL_END: i32 = 1_000_002;

/// A hot loop that runs `OP_PUSHNUM` — an input instruction — `n` times.
///
/// The degraded-arm recording happens when the dispatch JitCode is *installed*,
/// so any program reaching the install would prove the arm lowers. Putting the
/// input op inside the loop proves the thing that actually matters instead: a
/// trace *containing* it compiles. While the arm was an abort stub the loop
/// below compiled nothing at any threshold, which is the whole cost of the
/// defect and is invisible to every other gate in this repo.
///
/// Reading stdin is deterministic here without a fixture: `InputBuffer
/// ::read_number` returns 0 at EOF and never halts (aheui-runtime/src/io.rs
/// :250-252), and `cargo test` gives the test binary no stdin. That is also why
/// the loop needs its own counter — an EOF-fed aheui program that branches on
/// what it read never terminates.
///
/// The loop drains a stack of `n` values pushed up front, so it terminates on
/// its own count rather than on what it read — an EOF-fed aheui program that
/// branches on its input never does.
///
/// Stack shape per iteration: `BRPOP1` leaves the loop once the stack is empty;
/// `PUSHNUM` pushes the EOF 0 and the first `POP` discards it, so the pair is
/// stack-neutral and the second `POP` is what drains one real element.
fn input_op_in_a_hot_loop(n: i32) -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    for i in 0..n {
        opcodes.push(OP_PUSH);
        values.push(i);
    }

    let loop_pc = opcodes.len();
    // `-1` is the unused operand slot: only OP_PUSH and the branches read one.
    // Plain `OP_POP` prints nothing; `OP_POPNUM` would write to stdout.
    for (op, val) in [
        (OP_BRPOP1, LABEL_END),
        (OP_PUSHNUM, -1),
        (OP_POP, -1),
        (OP_POP, -1),
        (OP_JMP, LABEL_LOOP),
    ] {
        opcodes.push(op);
        values.push(val);
    }
    let end_pc = opcodes.len();
    opcodes.push(OP_HALT);
    values.push(-1);

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

/// Low enough that the loop compiles well inside `ITERATIONS`, high enough to
/// leave room above it.
const THRESHOLD: u32 = 8;
const ITERATIONS: i32 = 200;

/// One test, not two, because `mainloop` installs process-global state — the
/// nursery, the GC roots, the raw/tagged value mode — keyed to a `state` local
/// that lives on its own frame. Two `#[test]`s calling it are two threads
/// racing over that, and the second one hangs.
#[test]
fn no_aheui_dispatch_arm_lowered_to_an_abort_stub() {
    aheui_jit::init_gc_subsystem();
    let _exit = aheui_jit::mainloop(&input_op_in_a_hot_loop(ITERATIONS), THRESHOLD);

    // The precondition comes first: an empty degraded list means either "no arm
    // is degraded" or "no arm was looked at", and only the first is a pass.
    // `degraded_dispatch_arms()` starts empty and stays empty when the dispatch
    // JitCode is never installed, which is the same shape as the failure this
    // test exists to catch, one level up. A compiled loop can only exist
    // downstream of the install, so `loops_compiled` is what separates them —
    // and because the loop holds the input op, it is also the end-to-end fact
    // the degraded-arm assertion below only implies.
    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0 && stats.loops_aborted == 0,
        "a loop whose body runs OP_PUSHNUM did not compile \
         (loops_compiled={}, loops_aborted={}). While that arm is an abort stub \
         this is exactly what happens, and no other gate in this repo sees it",
        stats.loops_compiled,
        stats.loops_aborted,
    );

    let all = majit_metainterp::degraded_dispatch_arms();
    let degraded: Vec<_> = all
        .iter()
        .filter(|arm| arm.interp == AHEUI_INTERP)
        .collect();

    assert!(
        degraded.is_empty(),
        "{} aheui dispatch arm(s) lowered to an abort stub. Every trace that \
         reaches one aborts, so programs executing that opcode compile \
         nothing:\n{}",
        degraded.len(),
        degraded
            .iter()
            .map(|arm| format!(
                "  {}::{}\n    because: {}\n",
                arm.interp, arm.arm, arm.reason
            ))
            .collect::<String>(),
    );
}
