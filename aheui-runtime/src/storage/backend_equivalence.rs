//! Differential equivalence between the two storage backends.
//!
//! [`linkedlist`](super::linkedlist) is the live representation and the one
//! the pinned corpus grades; [`array`](super::array) is a second
//! implementation of the same three pools that no consumer selects yet.
//! `array.rs` was written against `linkedlist.py`'s *observable* semantics
//! rather than ported line-by-line from `array.py`, so nothing structural
//! ties the two together — only behaviour does, and behaviour is what this
//! module pins.
//!
//! Both backends' fields are `pub`, so the comparison is not limited to
//! return values: after every single operation this walks each pool's whole
//! contents and requires them equal element-for-element. A backend that
//! returned the right value while leaving the wrong state behind fails here
//! on the step that produced it, not later on whichever step happens to
//! observe the damage.
//!
//! Canonical order is *the order `pop` would return the elements in*: top
//! first for `Stack` and `Port`, front first for `Queue`. That is the order
//! the chain already walks, and it is the order the array must be read in
//! reverse (`Stack`/`Port`) or through the ring (`Queue`) to produce, so the
//! two layouts meet on the semantics instead of on the representation.
//!
//! Equality goes through [`val_ge`] in both directions rather than
//! `val_to_i64`, because a value promoted out of the raw-word range does not
//! survive the conversion; `val_to_i64` appears only in failure messages.

use super::array::{self, ArrayStorage};
use super::linkedlist::{self, LinkedList};
use super::nursery_test_lock;
use crate::value::*;

/// One operation of the surface both traits share.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Op {
    Push(i32),
    Pop,
    Dup,
    Swap,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Cmp,
}

/// Every variant, one representative each, for the coverage assertion. The
/// `Push` payload is a placeholder: [`Op::kind`] erases it.
const ALL_OPS: [Op; 10] = [
    Op::Push(0),
    Op::Pop,
    Op::Dup,
    Op::Swap,
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Mod,
    Op::Cmp,
];

impl Op {
    /// The smallest element count at which the operation is defined.
    ///
    /// Applied to *both* backends identically, so a skip can never be the
    /// source of a divergence — but it does mean a shape whose stream skips
    /// an operation every time never tests it, which is what the coverage
    /// histogram below exists to catch.
    fn min_len(self) -> usize {
        match self {
            Op::Push(_) => 0,
            // `Port::dup` is also defined at 0 — it republishes `last_push`
            // rather than reading the top — and that case has its own test.
            Op::Pop | Op::Dup => 1,
            Op::Swap | Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Mod | Op::Cmp => 2,
        }
    }

    /// Discriminant only, so `Push(3)` and `Push(-7)` count as one kind.
    fn kind(self) -> usize {
        match self {
            Op::Push(_) => 0,
            Op::Pop => 1,
            Op::Dup => 2,
            Op::Swap => 3,
            Op::Add => 4,
            Op::Sub => 5,
            Op::Mul => 6,
            Op::Div => 7,
            Op::Mod => 8,
            Op::Cmp => 9,
        }
    }

    /// True when the operation divides, so the driver can skip it on a zero
    /// divisor instead of tripping an assertion that says nothing about
    /// backend equivalence.
    fn divides(self) -> bool {
        matches!(self, Op::Div | Op::Mod)
    }
}

// The value both backends' `_get_2_values` name `r1` — the one `pop` would
// return — is the divisor for `div` and `mod`, which is why the driver's
// zero-divisor guard reads element 0 of the canonical order.

/// Deterministic operation-stream source.
///
/// Hand-rolled so the test needs no `rand` dependency and so a failure is
/// reproducible from the seed alone.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }

    /// A pushed value in `[-9, 9]`, small enough that a long run of `mul`
    /// stays inside the raw-word range for the shortest streams and, when it
    /// does not, promotes both backends identically because they share
    /// `val_mul`.
    fn value(&mut self) -> i32 {
        (self.next() % 19) as i32 - 9
    }

    fn op(&mut self) -> Op {
        match self.next() % 10 {
            0 | 1 | 2 => {
                let v = self.value();
                Op::Push(v)
            }
            3 => Op::Pop,
            4 => Op::Dup,
            5 => Op::Swap,
            6 => Op::Add,
            7 => Op::Sub,
            8 => Op::Mul,
            _ => {
                // Split the last slot three ways so `div`, `mod` and `cmp`
                // all appear; a stream missing one fails the coverage
                // assertion rather than passing quietly.
                match self.next() % 3 {
                    0 => Op::Div,
                    1 => Op::Mod,
                    _ => Op::Cmp,
                }
            }
        }
    }
}

/// Walk the chain and collect `size` values, head first.
///
/// Head-first is already the canonical order for all three linked-list
/// pools: `Stack`/`Port` push at the head, and `Queue`'s head is its front.
/// The queue's tail sentinel sits one past the last real element, so
/// stopping at `size` excludes it.
fn ll_contents<T: LinkedList + ?Sized>(pool: &T) -> Vec<Val> {
    let mut out = Vec::with_capacity(pool.size());
    let mut node = pool.head();
    for i in 0..pool.size() {
        assert!(!node.is_null(), "chain ended at {i} but size is {}", pool.size());
        out.push(unsafe { (*node).value });
        node = unsafe { (*node).next };
    }
    out
}

/// `data[0..size]` read back-to-front, so the top comes first.
fn arr_stackish_contents(data: *const Val, size: u32) -> Vec<Val> {
    (0..size as usize)
        .rev()
        .map(|i| unsafe { *data.add(i) })
        .collect()
}

/// The ring `[front, front + size)`, front first.
fn arr_queue_contents(q: &array::Queue) -> Vec<Val> {
    (0..q.size as usize)
        .map(|i| {
            let idx = (q.front as usize + i) % q.cap as usize;
            unsafe { *q.data.add(idx) }
        })
        .collect()
}

/// Exact equality, valid for a promoted value as well as a raw word.
fn val_equal(a: &Val, b: &Val) -> bool {
    val_ge(a, b) && val_ge(b, a)
}

fn render(vals: &[Val]) -> String {
    let parts: Vec<String> = vals.iter().map(|v| val_to_i64(v).to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// Compare one observation of the two backends.
///
/// Returns `Err` instead of asserting so the inversion test can feed it a
/// deliberately wrong reading and require it to complain.
fn compare(what: &str, reference: &[Val], candidate: &[Val]) -> Result<(), String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "{what}: linked list holds {} value(s) {}, array holds {} value(s) {}",
            reference.len(),
            render(reference),
            candidate.len(),
            render(candidate),
        ));
    }
    for (i, (r, c)) in reference.iter().zip(candidate).enumerate() {
        if !val_equal(r, c) {
            return Err(format!(
                "{what}: element {i} differs — linked list {} vs array {} (linked list {}, array {})",
                val_to_i64(r),
                val_to_i64(c),
                render(reference),
                render(candidate),
            ));
        }
    }
    Ok(())
}

/// Drive one operation on both backends and report what each returned.
///
/// Only `pop` yields a value directly; the arithmetic operations publish
/// their result into the pool, where the per-step contents comparison sees
/// it, and `_get_2_values` is exercised through them rather than on its own
/// because a bare `_get_2_values` leaves a state no interpreter reaches.
fn ll_apply<T: LinkedList + ?Sized>(pool: &mut T, op: Op) -> Vec<Val> {
    match op {
        Op::Push(v) => {
            pool.push(val_from_i32(v));
            vec![]
        }
        Op::Pop => vec![pool.pop()],
        Op::Dup => {
            pool.dup();
            vec![]
        }
        Op::Swap => {
            pool.swap();
            vec![]
        }
        Op::Add => {
            pool.add();
            vec![]
        }
        Op::Sub => {
            pool.sub();
            vec![]
        }
        Op::Mul => {
            pool.mul();
            vec![]
        }
        Op::Div => {
            pool.div();
            vec![]
        }
        Op::Mod => {
            pool.modulo();
            vec![]
        }
        Op::Cmp => {
            pool.cmp();
            vec![]
        }
    }
}

fn arr_apply<T: ArrayStorage + ?Sized>(pool: &mut T, op: Op) -> Vec<Val> {
    match op {
        Op::Push(v) => {
            pool.push(val_from_i32(v));
            vec![]
        }
        Op::Pop => vec![pool.pop()],
        Op::Dup => {
            pool.dup();
            vec![]
        }
        Op::Swap => {
            pool.swap();
            vec![]
        }
        Op::Add => {
            pool.add();
            vec![]
        }
        Op::Sub => {
            pool.sub();
            vec![]
        }
        Op::Mul => {
            pool.mul();
            vec![]
        }
        Op::Div => {
            pool.div();
            vec![]
        }
        Op::Mod => {
            pool.modulo();
            vec![]
        }
        Op::Cmp => {
            pool.cmp();
            vec![]
        }
    }
}

/// How many times each [`Op::kind`] actually ran, so a stream that skipped
/// an operation on every step cannot pass as coverage of it.
type Coverage = [usize; 10];

/// Run one operation stream through both backends, comparing after each step.
///
/// The linked list is the reference: every guard (`min_len`, zero divisor)
/// reads *its* state. That is sound because the two are proven equal at the
/// end of every step, so they are equal at the start of the next one.
fn drive<L, A>(
    ll: &mut L,
    arr: &mut A,
    arr_dump: impl Fn(&A) -> Vec<Val>,
    ops: &[Op],
) -> Result<Coverage, String>
where
    L: LinkedList,
    A: ArrayStorage,
{
    let mut coverage: Coverage = [0; 10];

    compare("initial contents", &ll_contents(ll), &arr_dump(arr))?;

    for (step, &op) in ops.iter().enumerate() {
        let before = ll_contents(ll);
        if before.len() < op.min_len() {
            continue;
        }
        if op.divides() && val_is_zero(&before[0]) {
            continue;
        }

        let ll_returned = ll_apply(ll, op);
        let arr_returned = arr_apply(arr, op);
        coverage[op.kind()] += 1;

        let what = format!("step {step} {op:?}");
        compare(&format!("{what} return value"), &ll_returned, &arr_returned)?;
        compare(&format!("{what} contents"), &ll_contents(ll), &arr_dump(arr))?;

        let ll_len = ll.__len__();
        let arr_len = arr.len();
        if ll_len != arr_len {
            return Err(format!(
                "{what}: linked list reports len {ll_len}, array reports len {arr_len}"
            ));
        }
    }

    // Drain whatever is left: a backend that keeps its contents in the right
    // order but its bookkeeping wrong (a stale `front`, a `cap` the ring
    // wraps against incorrectly) shows up here rather than never.
    while ll.__len__() > 0 {
        let a = ll.pop();
        let b = arr.pop();
        compare("final drain", &[a], &[b])?;
    }
    compare("after drain", &ll_contents(ll), &arr_dump(arr))?;
    if arr.len() != 0 {
        return Err(format!("after drain: array still reports len {}", arr.len()));
    }

    Ok(coverage)
}

/// Fail unless every operation ran at least once.
///
/// Without this the suite passes when a shape's guards reject an operation
/// on every step — a green that means "never tested", which is the failure
/// mode a differential oracle is most likely to have.
fn assert_full_coverage(shape: &str, coverage: Coverage) {
    for op in ALL_OPS {
        assert!(
            coverage[op.kind()] > 0,
            "{shape}: {op:?} never ran, so the stream does not cover it"
        );
    }
}

/// Long enough that each shape's guards still let every operation through.
const STREAM_LEN: usize = 4000;

fn stream(seed: u64) -> Vec<Op> {
    let mut lcg = Lcg(seed);
    (0..STREAM_LEN).map(|_| lcg.op()).collect()
}

#[test]
fn stack_backends_agree_step_for_step() {
    let _guard = nursery_test_lock();
    let ops = stream(0x5EED_0001);
    let mut ll = linkedlist::Stack::new();
    let mut arr = array::Stack::new();
    let coverage = drive(&mut ll, &mut arr, |a| arr_stackish_contents(a.data, a.size), &ops)
        .unwrap_or_else(|e| panic!("Stack: {e}"));
    assert_full_coverage("Stack", coverage);
}

#[test]
fn queue_backends_agree_step_for_step() {
    let _guard = nursery_test_lock();
    let ops = stream(0x5EED_0002);
    let mut ll = linkedlist::Queue::new();
    let mut arr = array::Queue::new();
    let coverage = drive(&mut ll, &mut arr, arr_queue_contents, &ops)
        .unwrap_or_else(|e| panic!("Queue: {e}"));
    assert_full_coverage("Queue", coverage);
}

#[test]
fn port_backends_agree_step_for_step() {
    let _guard = nursery_test_lock();
    let ops = stream(0x5EED_0003);
    let mut ll = linkedlist::Port::new();
    let mut arr = array::Port::new();
    let coverage = drive(&mut ll, &mut arr, |a| arr_stackish_contents(a.data, a.size), &ops)
        .unwrap_or_else(|e| panic!("Port: {e}"));
    assert_full_coverage("Port", coverage);
}

/// `Port::dup` is the one operation defined on an empty pool, because it
/// republishes `last_push` instead of reading the top. The driver's
/// `min_len` guard skips it there, so it is checked separately.
#[test]
fn port_dup_from_empty_agrees() {
    let _guard = nursery_test_lock();
    let mut ll = linkedlist::Port::new();
    let mut arr = array::Port::new();

    ll.dup();
    arr.dup();
    compare("empty port dup", &ll_contents(&ll), &arr_stackish_contents(arr.data, arr.size))
        .unwrap_or_else(|e| panic!("Port: {e}"));

    // `_put_value` deliberately leaves `last_push` alone, so after this the
    // top and the shadow differ and a second `dup` tells them apart.
    for v in [5, 10] {
        ll.push(val_from_i32(v));
        arr.push(val_from_i32(v));
    }
    ll.add();
    arr.add();
    ll.dup();
    arr.dup();
    let reference = ll_contents(&ll);
    compare("port dup after add", &reference, &arr_stackish_contents(arr.data, arr.size))
        .unwrap_or_else(|e| panic!("Port: {e}"));
    assert!(
        val_equal(&reference[0], &val_from_i32(10)),
        "dup must republish last_push (10), not the top (15); got {}",
        render(&reference)
    );
}

/// Prove the comparison can fail.
///
/// `array.py:51-52` spells `Queue.dup` as `appendleft(self[0])`, and in its
/// orientation `self[0]` is the BACK — so it duplicates the most recently
/// pushed element where `linkedlist.py:112-116` duplicates the front. That
/// is the one divergence `array.rs` deliberately does not copy, which makes
/// it the right thing to invert against: if the oracle above cannot see this
/// difference it cannot see anything.
#[test]
fn the_comparison_rejects_array_pys_queue_dup_orientation() {
    let _guard = nursery_test_lock();
    let mut ll = linkedlist::Queue::new();
    for v in [1, 2, 3] {
        ll.push(val_from_i32(v));
    }
    ll.dup();
    let reference = ll_contents(&ll);
    assert!(
        val_equal(&reference[0], &val_from_i32(1)) && reference.len() == 4,
        "the front is duplicated: expected [1, 1, 2, 3], got {}",
        render(&reference)
    );

    // What `array.py`'s orientation would have produced from the same three
    // pushes: the back duplicated instead of the front.
    let wrong: Vec<Val> = [1, 2, 3, 3].iter().map(|&v| val_from_i32(v)).collect();
    let verdict = compare("inverted queue dup", &reference, &wrong);
    assert!(
        verdict.is_err(),
        "the comparison accepted array.py's dup orientation, so it grades nothing"
    );
}
