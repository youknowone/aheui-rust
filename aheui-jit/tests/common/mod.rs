//! Scaffolding shared by the tests that drive `mainloop` over a hand-built
//! program.
//!
//! Each such test is its own binary holding one `#[test]`. `mainloop` installs
//! process-global state — the nursery, the GC roots, the raw/tagged value mode
//! — keyed to a `state` local on its own frame, so two `#[test]`s calling it
//! are two threads racing over it and the second hangs.
//!
//! Each binary uses the part of this it needs; the rest is dead there.
#![allow(dead_code)]

use ahsembler::compiler::Program;
use std::collections::HashMap;

/// Threshold low enough that a loop compiles early in a few-hundred-iteration
/// drain, high enough to leave interpreted trips ahead of it.
pub const THRESHOLD: u32 = 8;

/// Iterations that leave the large majority of a drain running compiled.
pub const ITERATIONS: i32 = 200;

/// The operand slot of an instruction that reads none — only `OP_PUSH`,
/// `OP_SEL` and the branches take one.
pub const NONE: i32 = -1;

/// Label ids, numbered past any pc so they cannot be mistaken for one.
pub const LABEL_LOOP: i32 = 1_000_001;
pub const LABEL_END: i32 = 1_000_002;
pub const LABEL_ALT: i32 = 1_000_003;

/// Assembles an instruction list and its labels into a [`Program`].
///
/// Straight-line by construction: two-dimensional aheui source cannot express
/// these programs, because an underflow there reverses direction and in a
/// rectangular layout re-enters the loop from the other side instead of
/// terminating.
#[derive(Default)]
pub struct Asm {
    opcodes: Vec<u8>,
    values: Vec<i32>,
    labels: HashMap<i32, usize>,
}

impl Asm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one instruction.
    pub fn emit(&mut self, op: u8, val: i32) -> &mut Self {
        self.opcodes.push(op);
        self.values.push(val);
        self
    }

    /// Append `(op, operand)` pairs in order.
    pub fn emit_all(&mut self, ops: impl IntoIterator<Item = (u8, i32)>) -> &mut Self {
        for (op, val) in ops {
            self.emit(op, val);
        }
        self
    }

    /// Repeat one instruction, its operand computed from the index.
    pub fn emit_n(&mut self, n: i32, op: u8, val: impl Fn(i32) -> i32) -> &mut Self {
        for i in 0..n {
            self.emit(op, val(i));
        }
        self
    }

    /// Point `label` at the instruction emitted next.
    pub fn label(&mut self, label: i32) -> &mut Self {
        self.labels.insert(label, self.opcodes.len());
        self
    }

    pub fn build(&mut self) -> Program {
        let size = self.opcodes.len();
        let mut program = Program {
            opcodes: std::mem::take(&mut self.opcodes),
            values: std::mem::take(&mut self.values),
            labels: std::mem::take(&mut self.labels),
            size,
        };
        program.resolve_jump_targets();
        program
    }
}

/// `PUSH 0..n` on the default stack, then a loop that runs `body` once per
/// element until `BRPOP1` finds the stack empty.
///
/// `body` must be stack-neutral apart from the one element the loop is meant
/// to drain, so the loop terminates on its own count. That matters for bodies
/// holding an input instruction: `InputBuffer::read_number` returns 0 at EOF
/// and never halts, and `cargo test` gives the test binary no stdin, so an
/// aheui program that branched on what it read would never terminate.
pub fn drain_loop(n: i32, body: &[(u8, i32)]) -> Program {
    use ahsembler::consts::{OP_BRPOP1, OP_HALT, OP_JMP, OP_PUSH};

    let mut asm = Asm::new();
    asm.emit_n(n, OP_PUSH, |i| i)
        .label(LABEL_LOOP)
        .emit(OP_BRPOP1, LABEL_END)
        .emit_all(body.iter().copied())
        .emit(OP_JMP, LABEL_LOOP)
        .label(LABEL_END)
        .emit(OP_HALT, NONE);
    asm.build()
}

/// Elements a queue drain starts with. Each trip nets at least one off, so
/// this also bounds the trip count.
pub const SEED: i32 = 400;

/// Seed element `i`: distinct by position and never zero, so a drain that ends
/// one trip early and one that ends one trip late both land somewhere else.
pub fn seed_value(i: i32) -> i32 {
    2 + (i % 8)
}

/// A queue operation as the storage defines it, for [`drain_queue_model`].
#[derive(Clone, Copy, Debug)]
pub enum QOp {
    /// `LinkedList::swap` exchanges `head.value` and `head.next.value`.
    Swap,
    /// `Queue::dup` links a new node ahead of `head`.
    Dup,
    /// `Queue::_get_2_values` pops two from the front, `add` computes
    /// `r2 + r1` — second popped first — and `Queue::_put_value` pushes it at
    /// the back.
    Fold,
}

impl QOp {
    /// The smallest element count at which the operation is defined, and so
    /// the `BRPOP` guard the program emits ahead of it.
    fn min_len(self) -> usize {
        match self {
            QOp::Dup => 1,
            QOp::Swap | QOp::Fold => 2,
        }
    }
}

/// The expected result of a guarded queue drain, from the storage's own rules.
///
/// Mirrors the shape every drain program below assembles: `body` repeats, each
/// operation preceded by its branch-on-underflow guard, and all the guards name
/// one drained label — so the first shortfall ends the run and the program
/// halts on the front of what remains.
///
/// Derived rather than recorded from a run, so a wrong answer stays a wrong
/// answer instead of becoming a baseline.
pub fn drain_queue_model(body: &[QOp]) -> i64 {
    let mut q: std::collections::VecDeque<i64> =
        (0..SEED).map(|i| i64::from(seed_value(i))).collect();
    'drain: loop {
        for &op in body {
            if q.len() < op.min_len() {
                break 'drain;
            }
            match op {
                QOp::Swap => q.swap(0, 1),
                QOp::Dup => q.push_front(q[0]),
                QOp::Fold => {
                    let r1 = q.pop_front().expect("guarded above");
                    let r2 = q.pop_front().expect("guarded above");
                    q.push_back(r2 + r1);
                }
            }
        }
    }
    q.pop_front().unwrap_or(0)
}

/// Run `program` under the JIT and return the value it halted with.
pub fn run(program: &Program) -> i64 {
    aheui_jit::init_gc_subsystem();
    let exit = aheui_jit::mainloop(program, THRESHOLD);
    aheui_runtime::value::val_to_i64(&exit)
}

/// Fail unless the last run compiled a loop and aborted none.
///
/// Every assertion these tests make is about compiled code. Interpreted, the
/// programs agree with their models for reasons unrelated to what is graded,
/// so a run that compiled nothing turns the test green while leaving
/// `subject` untested.
pub fn assert_compiled(subject: &str) {
    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0 && stats.loops_aborted == 0,
        "the loop did not compile (loops_compiled={}, loops_aborted={}), so the \
         run graded the interpreter and left {subject} untested",
        stats.loops_compiled,
        stats.loops_aborted,
    );
}
