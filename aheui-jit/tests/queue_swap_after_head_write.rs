//! `OP_SWAP` must read the `head` its own trace just wrote, on a queue.
//!
//! `Stack`, `Queue` and `Port` deliberately pun a `{ head, size }` prefix, and a
//! field descriptor carries the nominal struct its base was declared as. The
//! swap arm reaches `head` through the selected-storage reference, which is
//! declared `Stack`; a queue op spliced into the same trace writes `head` under
//! `Queue`. That is two descriptor identities for one word, and the optimizer's
//! field cache is keyed by identity: a read under one is neither served by, nor
//! forces, a pending store under the other. What that costs is a swap performed
//! at the node `head` pointed at *before* the write — one node too far down the
//! chain, exchanging the wrong pair of values.
//!
//! Nothing in the snippet corpus grades this. Of its 78 programs only seven
//! contain both a swap and a queue selection, the largest at three of each, and
//! none puts them adjacent in a hot loop. `logo` — the byte-exact acceptance
//! program — has 266 swaps and selects the queue zero times, so it exercises the
//! stack family alone and would stay green through the whole failure.
//!
//! The loop below swaps in three positions that differ in what wrote `head`
//! last: after nothing (`A`), after a duplicate (`B`), and after the two pops a
//! fold performs (`C`). Displacing any one of them by a single node moves the
//! final value, and moves it to a different wrong value in each case, so a
//! mismatch says which position failed rather than only that one did.
//!
//! The expected value is derived here from the storage's own operational rules
//! rather than recorded from a run, so a wrong answer stays a wrong answer
//! instead of becoming a baseline.

use ahsembler::compiler::Program;
use ahsembler::consts::{
    OP_ADD, OP_BRPOP1, OP_BRPOP2, OP_DUP, OP_HALT, OP_JMP, OP_PUSH, OP_SEL, OP_SWAP, VAL_QUEUE,
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

/// `SEL 21; PUSH×SEED; loop { SWAP; DUP; SWAP; ADD; SWAP; ADD }; HALT`.
///
/// Every operand consumer is preceded by its branch-on-underflow guard, and all
/// of them name one drained label. Two-dimensional source cannot do that: an
/// underflow there reverses direction, which in a rectangular layout re-enters
/// the loop from the other side and never terminates. Naming the exit is what
/// gives this program one.
fn queue_swap_drain() -> Program {
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
    // A: a swap with no preceding head write in this trip.
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_SWAP, -1);
    // B: a swap behind a duplicate, which links a new node ahead of `head`.
    emit(&mut opcodes, &mut values, OP_BRPOP1, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_DUP, -1);
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_SWAP, -1);
    // C: a swap behind a fold, whose two pops advance `head` twice.
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_ADD, -1);
    emit(&mut opcodes, &mut values, OP_BRPOP2, LABEL_DRAINED);
    emit(&mut opcodes, &mut values, OP_SWAP, -1);
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

/// Distinct by position and never zero, so a swap that silently does nothing
/// and a swap displaced by one node both move the result.
fn seed_value(i: i32) -> i32 {
    2 + (i % 8)
}

/// The same program under the rules the storage documents.
///
/// A queue duplicates at the front (`Queue::dup` links a new node ahead of
/// `head`), swaps the two values at the front (`LinkedList::swap` exchanges
/// `head.value` and `head.next.value`), and folds by popping two from the front
/// and pushing the sum at the back (`Queue::_get_2_values` is two pops,
/// `Queue::_put_value` is a push, and `add` computes `r2 + r1` — second popped
/// first). What the program halts with is the front of what remains.
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
        q.swap(0, 1);
        if q.is_empty() {
            break;
        }
        q.push_front(q[0]);
        if q.len() < 2 {
            break;
        }
        q.swap(0, 1);
        if q.len() < 2 {
            break;
        }
        fold(&mut q);
        if q.len() < 2 {
            break;
        }
        q.swap(0, 1);
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
fn a_queue_swap_reads_the_head_its_own_trace_wrote() {
    aheui_jit::init_gc_subsystem();

    let expected = expected_result();
    let exit = aheui_jit::mainloop(&queue_swap_drain(), THRESHOLD);
    let exit = aheui_runtime::value::val_to_i64(&exit);

    assert_eq!(
        exit, expected,
        "the queue drain disagrees with the storage's own rules. A swap that \
         read `head` from before the duplicate or the pops ahead of it lands on \
         the wrong pair of nodes, and each of the loop's three swap positions \
         fails to a different value"
    );

    // The comparison above is only evidence about compiled code if a loop
    // compiled: interpreted, this program agrees with the model for reasons
    // that have nothing to do with descriptor identity.
    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0 && stats.loops_aborted == 0,
        "the drain loop did not compile (loops_compiled={}, loops_aborted={}), \
         so the assertion above graded the interpreter and left the queue swap \
         arm's lowering untested",
        stats.loops_compiled,
        stats.loops_aborted,
    );
}
