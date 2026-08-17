//! `OP_SEL` must read the `size` its own trace just wrote, on a queue.
//!
//! Selecting a storage re-reads its element count: `aheui.py:280-284` is
//! `selected = storage[value]; stacksize = len(selected)`, and `len` is a load
//! of the list's mutable `size`. The arm mirrors that as a `getarrayitem_gc_r`
//! on the storage base followed by a `getfield_gc_i` through the loaded
//! reference. Both halves are re-derived every trip rather than carried, which
//! is what keeps `stacksize` from freezing to a loop-invariant constant.
//!
//! That re-read is only correct if it observes the writes the same trip already
//! performed. `size` is written by the spliced bodies of the storage helpers —
//! a fold decrements it, a duplicate increments it — and the optimizer's field
//! cache decides whether a later load is served from an entry, forced against a
//! pending store, or emitted afresh. It makes that decision per field
//! descriptor, so the guarantee holds exactly as long as the helper's write and
//! the select's read name one descriptor for one word. Re-selecting the storage
//! a helper just mutated is the shortest program that depends on it.
//!
//! Nothing else grades this. `logo` — the byte-exact acceptance program —
//! selects a storage 457 times and selects the *queue* zero times, so the whole
//! queue family is outside it; the corpus sweep's queue programs do not
//! re-select between mutations in a hot loop. The count is also the quietest
//! thing to get wrong: a displaced `size` does not crash, it moves the trip at
//! which the branch-on-underflow fires, so the program simply drains a trip
//! early or late and halts on a different value.
//!
//! The loop below re-selects in two positions that differ in what wrote `size`
//! last: after a fold, which pops twice and pushes once, and after a duplicate,
//! which links a node ahead of `head`. The expected value is derived from the
//! storage's operational rules rather than recorded from a run, so a wrong
//! answer stays a wrong answer instead of becoming a baseline.
//!
//! The drain is sensitive to the count in both directions, which is what makes
//! it an oracle rather than a program that happens to agree. Modelled against a
//! count read one too low, it drains in 399 trips and halts on 43777 instead of
//! 400 and 236890; against a count read one too high, the guard admits a fold
//! on a queue holding fewer than two elements, and the second pop walks off the
//! end of the chain.
//!
//! What this pins is the behaviour, not a mechanism: it cannot distinguish
//! *why* a re-read was correct. Its job is to fail if the select and the
//! helpers ever stop naming one `size` again.

use ahsembler::compiler::Program;
use ahsembler::consts::{
    OP_ADD, OP_BRPOP1, OP_BRPOP2, OP_DUP, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, VAL_QUEUE,
};
use std::collections::{HashMap, VecDeque};

const LABEL_LOOP: i32 = 1_000_001;
const LABEL_DRAINED: i32 = 1_000_002;

/// Elements the loop starts with. Each trip nets one off, so this is also the
/// trip count — far enough past `THRESHOLD` that the large majority run
/// compiled.
const SEED: i32 = 400;
/// Low enough that the loop compiles early in the drain.
const THRESHOLD: u32 = 8;

/// `SEL 21; PUSH×SEED; loop { ADD; SEL 21; DUP; SEL 21; ADD }; HALT`.
///
/// The two `SEL 21`s re-select the storage that is already selected. That is a
/// no-op on the data and the whole point: it forces the count to be read again
/// through the reference, immediately behind a helper that wrote it.
///
/// Every operand consumer is preceded by its branch-on-underflow guard, and all
/// of them name one drained label. Two-dimensional source cannot do that: an
/// underflow there reverses direction, which in a rectangular layout re-enters
/// the loop from the other side and never terminates.
fn reselect_drain() -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    // `-1` is the unused operand slot; only OP_PUSH, OP_SEL and the branches
    // read one.
    fn emit(opcodes: &mut Vec<u8>, values: &mut Vec<i32>, op: u8, val: i32) {
        opcodes.push(op);
        values.push(val);
    }

    emit(&mut opcodes, &mut values, OP_SEL, VAL_QUEUE as i32);
    for i in 0..SEED {
        emit(&mut opcodes, &mut values, OP_PUSH, seed_value(i));
    }

    let loop_pc = opcodes.len();
    // A fold: two pops and a push at the back, so `size` ends one lower.
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_ADD, -1);
    // Re-read the count behind that write.
    emit(&mut opcodes, &mut values, OP_SEL, VAL_QUEUE as i32);
    // A duplicate: one node linked ahead of `head`, so `size` ends one higher.
    emit(&mut opcodes, &mut values, OP_BRPOP1, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_DUP, -1);
    // Re-read the count behind that write, which moves it the other way.
    emit(&mut opcodes, &mut values, OP_SEL, VAL_QUEUE as i32);
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_ADD, -1);
    emit(&mut opcodes, &mut values, OP_JMP, LABEL_LOOP);

    let drained_pc = opcodes.len();
    emit(&mut opcodes, &mut values, OP_HALT, -1);

    let size = opcodes.len();
    let mut labels: HashMap<i32, usize> = HashMap::new();
    labels.insert(LABEL_LOOP, loop_pc);
    labels.insert(LABEL_DRAINED, drained_pc);

    let mut program = Program {
        opcodes,
        values,
        labels,
        size,
    };
    program.resolve_jump_targets();
    program
}

/// Distinct by position and never zero, so a drain that ends one trip early and
/// one that ends one trip late both land somewhere else.
fn seed_value(i: i32) -> i32 {
    2 + (i % 8)
}

/// The same program under the rules the storage documents.
///
/// A queue folds by popping two from the front and pushing the sum at the back
/// (`Queue::_get_2_values` is two pops, `Queue::_put_value` is a push, and `add`
/// computes `r2 + r1` — second popped first) and duplicates at the front
/// (`Queue::dup` links a new node ahead of `head`). Selecting a storage that is
/// already selected changes nothing here, which is why a disagreement is about
/// the count and not about the data. What the program halts with is the front
/// of what remains.
fn expected_result() -> i64 {
    let mut q: VecDeque<i64> = (0..SEED).map(|i| i64::from(seed_value(i))).collect();
    fn fold(q: &mut VecDeque<i64>) {
        let r1 = q.pop_front().expect("guarded by the caller");
        let r2 = q.pop_front().expect("guarded by the caller");
        q.push_back(r2 + r1);
    }
    loop {
        if q.len() < 2 {
            break;
        }
        fold(&mut q);
        if q.is_empty() {
            break;
        }
        q.push_front(q[0]);
        if q.len() < 2 {
            break;
        }
        fold(&mut q);
    }
    q.pop_front().unwrap_or(0)
}

/// One test, not several: `mainloop` installs process-global state (the
/// nursery, the GC roots, the raw/tagged value mode), so two `#[test]`s calling
/// it are two threads racing over it.
#[test]
fn a_reselect_reads_the_size_its_own_trace_wrote() {
    aheui_jit::init_gc_subsystem();

    let expected = expected_result();
    let exit = aheui_jit::mainloop(&reselect_drain(), THRESHOLD);
    let exit = aheui_runtime::value::val_to_i64(&exit);

    assert_eq!(
        exit, expected,
        "the queue drain disagrees with the storage's own rules. A select that \
         read the count from before the fold or the duplicate ahead of it fires \
         the branch-on-underflow at the wrong trip, so the loop drains to a \
         different depth and halts on a different value"
    );

    // The comparison above is only evidence about compiled code if a loop
    // compiled: interpreted, this program agrees with the model for reasons
    // that have nothing to do with how a count is cached.
    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0 && stats.loops_aborted == 0,
        "the drain loop did not compile (loops_compiled={}, loops_aborted={}), \
         so the assertion above graded the interpreter and left the select arm's \
         count re-read untested",
        stats.loops_compiled,
        stats.loops_aborted,
    );
}
