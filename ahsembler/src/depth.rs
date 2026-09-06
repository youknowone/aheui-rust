//! Per-pool depth analysis of a linearized [`Program`].
//!
//! Static states are keyed by `(pc, selected)`; joins widen upward, and
//! `OP_BRPOP*` constrains the underflow edge. Unbounded or invalid control
//! flow never produces a finite proof. Run after `resolve_jump_targets`.
//!
//! Bounded evaluation separately measures input-free runs or observes an
//! input-dependent prefix. An observation is a sizing hint, not a proof.

use crate::compiler::Program;
use crate::consts::*;
use std::collections::{HashMap, VecDeque};

/// An upper bound on a pool's element count, or the admission that none was
/// proven.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DepthBound {
    Bounded(u32),
    Unbounded,
}

impl DepthBound {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bounded(a), Self::Bounded(b)) => Self::Bounded(a.max(b)),
            _ => Self::Unbounded,
        }
    }

    /// `delta` may be negative; the count cannot be. A bound past
    /// [`WIDEN_LIMIT`] stops growing by becoming unbounded, which is what
    /// makes the fixpoint finite: masks only gain bits and bounds only move
    /// up a finite lattice.
    fn add(self, delta: i32, ceiling: u32) -> Self {
        match self {
            Self::Bounded(b) => {
                let next = (b as i64 + delta as i64).max(0);
                if next > ceiling as i64 {
                    Self::Unbounded
                } else {
                    Self::Bounded(next as u32)
                }
            }
            Self::Unbounded => Self::Unbounded,
        }
    }

    /// Meet with a known upper bound. An unbounded value becomes `hi`: the
    /// caller states a limit that holds on every path it applies this to.
    fn at_most(self, hi: u32) -> Self {
        match self {
            Self::Bounded(b) => Self::Bounded(b.min(hi)),
            Self::Unbounded => Self::Bounded(hi),
        }
    }

    /// Whether this bound admits a depth of `n` or more.
    fn can_reach(self, n: u32) -> bool {
        match self {
            Self::Bounded(b) => b >= n,
            Self::Unbounded => true,
        }
    }
}

/// The default ceiling a bound widens at — see [`max_pool_depths_up_to`] for
/// the caller-supplied one. Bounds above this are as good as none: the
/// consumers size small per-pool buffers, and a four-figure depth already
/// exceeds every size worth declaring.
const WIDEN_LIMIT: u32 = 4096;

/// Depth bounds for every pool at one program point.
type Depths = [DepthBound; STORAGE_COUNT];

/// The sound direction for every ambiguity is "the pool got deeper", and the
/// one ambiguity the analysis refuses to carry is *which* pool is selected.
///
/// Joining states that reached a program counter under different `OP_SEL`
/// values would force every push to credit each candidate pool — a loop that
/// pushes onto one pool would then look like it grows every other one too, and
/// a pool the loop never touches comes out unbounded. So the analysis is
/// polyvariant in the selected pool instead: a state is keyed by the pair
/// `(pc, selected)`, every push and pop names exactly one pool, and joins only
/// ever merge states that agree about what is selected. `OP_SEL` operands are
/// program constants, so the number of keys is bounded by the program size
/// times the pool count.
fn transfer(depths: &mut Depths, selected: usize, op: u8, operand: i32, ceiling: u32) {
    match op {
        OP_SEL => {}
        OP_MOV => {
            depths[selected] = depths[selected].add(-1, ceiling);
            let target = operand as usize;
            depths[target] = depths[target].add(1, ceiling);
        }
        _ => {
            let delta = OP_STACKADD[op as usize] - OP_STACKDEL[op as usize];
            depths[selected] = depths[selected].add(delta, ceiling);
        }
    }
}

/// States expanded before the analysis gives up and answers unbounded.
///
/// The lattice is finite on its own, but a program whose bounds crawl toward
/// the ceiling one push at a time can visit a key once per value it passes
/// through. This bounds the price of an answer the way `measured_pool_depths`
/// bounds the price of the exact one.
const STATE_BUDGET: usize = 16_384;

/// Per-pool depth upper bounds for `program`, or [`DepthBound::Unbounded`]
/// where no bound was proven.
pub fn max_pool_depths(program: &Program) -> [DepthBound; STORAGE_COUNT] {
    max_pool_depths_up_to(program, WIDEN_LIMIT)
}

/// [`max_pool_depths`] widening at `ceiling` instead of [`WIDEN_LIMIT`].
///
/// A caller that will not use a bound above some size — one sizing a buffer it
/// caps anyway — pays for the precision it can use and no more: the lattice is
/// as tall as the ceiling, so a lower one converges in fewer steps. A pool
/// whose true depth is above the ceiling comes back unbounded.
pub fn max_pool_depths_up_to(program: &Program, ceiling: u32) -> [DepthBound; STORAGE_COUNT] {
    let mut result = [DepthBound::Bounded(0); STORAGE_COUNT];
    if program.size == 0 {
        return result;
    }

    // Keyed by `(pc, selected)` — see `transfer`.
    let mut states: HashMap<(usize, usize), Depths> = HashMap::new();
    let mut worklist: Vec<(usize, usize)> = Vec::new();

    fn propagate(
        states: &mut HashMap<(usize, usize), Depths>,
        worklist: &mut Vec<(usize, usize)>,
        key: (usize, usize),
        depths: &Depths,
    ) {
        match states.get_mut(&key) {
            Some(old) => {
                let mut changed = false;
                for (o, n) in old.iter_mut().zip(depths.iter()) {
                    let joined = o.join(*n);
                    if joined != *o {
                        *o = joined;
                        changed = true;
                    }
                }
                if changed {
                    worklist.push(key);
                }
            }
            None => {
                states.insert(key, *depths);
                worklist.push(key);
            }
        }
    }

    // `mainloop` starts on pool 0 with every pool empty.
    propagate(
        &mut states,
        &mut worklist,
        (0, 0),
        &[DepthBound::Bounded(0); STORAGE_COUNT],
    );

    let mut expanded = 0usize;
    while let Some(key) = worklist.pop() {
        expanded += 1;
        if expanded > STATE_BUDGET {
            return [DepthBound::Unbounded; STORAGE_COUNT];
        }
        let (pc, selected) = key;
        let Some(entry) = states.get(&key).copied() else {
            continue;
        };
        let op = program.get_op(pc);
        let operand = program.get_operand(pc);

        // A pool index outside the storage is not something the analysis can
        // name a bound for, and it cannot happen on a program the compiler
        // produced; answering unbounded keeps the surface total.
        if matches!(op, OP_SEL | OP_MOV) && operand as usize >= STORAGE_COUNT {
            return [DepthBound::Unbounded; STORAGE_COUNT];
        }
        let next_selected = if op == OP_SEL {
            operand as usize
        } else {
            selected
        };

        let mut depths = entry;
        transfer(&mut depths, selected, op, operand, ceiling);
        for (r, d) in result.iter_mut().zip(depths.iter()) {
            *r = r.join(*d);
        }

        match op {
            OP_HALT => {}
            OP_JMP => {
                let target = program.get_label(pc);
                if target >= program.size {
                    return [DepthBound::Unbounded; STORAGE_COUNT];
                }
                propagate(&mut states, &mut worklist, (target, next_selected), &depths);
            }
            // `mainloop` takes the branch when `!stackok`, that is when the
            // selected pool holds fewer than `OP_REQSIZE[op]` elements. So the
            // taken edge carries a state whose depth is at most one below that
            // requirement — the guard is where a bound comes from at all, since
            // nothing else in a stack machine states an upper bound — and the
            // fall-through carries one whose depth is at least it, which makes
            // the fall-through unreachable when the bound cannot reach it.
            OP_BRPOP1 | OP_BRPOP2 => {
                let need = OP_REQSIZE[op as usize] as u32;
                let target = program.get_label(pc);
                if target >= program.size {
                    return [DepthBound::Unbounded; STORAGE_COUNT];
                }
                let mut taken = depths;
                taken[selected] = taken[selected].at_most(need - 1);
                propagate(&mut states, &mut worklist, (target, next_selected), &taken);
                if depths[selected].can_reach(need) && pc + 1 < program.size {
                    propagate(&mut states, &mut worklist, (pc + 1, next_selected), &depths);
                }
            }
            OP_BRZ => {
                let target = program.get_label(pc);
                if target >= program.size {
                    return [DepthBound::Unbounded; STORAGE_COUNT];
                }
                propagate(&mut states, &mut worklist, (target, next_selected), &depths);
                if pc + 1 < program.size {
                    propagate(&mut states, &mut worklist, (pc + 1, next_selected), &depths);
                }
            }
            _ => {
                if pc + 1 < program.size {
                    propagate(&mut states, &mut worklist, (pc + 1, next_selected), &depths);
                }
            }
        }
    }

    result
}

/// Exact per-pool peak depths, measured by running the program.
///
/// The static bounds above cannot count loop trips, so a counted loop that
/// nets a push is Unbounded there even when the real peak is small. A program
/// with no input is deterministic, which makes the exact answer computable —
/// by paying the program's own runtime. This caps that price with a step
/// budget and answers only when the run both finished inside it and stayed
/// on ground the simulation models exactly:
///
/// * an input opcode, or any touch of the port pool, means the run is not
///   deterministic from the program alone;
/// * an `i64` overflow means the real run continues in bigint mode the
///   simulation does not model;
/// * a division by zero or an underflowing pop means the real run diverges
///   from anything worth measuring.
///
/// Every one of those answers `None` — never an approximation — so a `Some`
/// is the true peak of the real run and a ring sized to it can never evict.
pub fn measured_pool_depths(program: &Program, step_budget: u64) -> Option<[u32; STORAGE_COUNT]> {
    run_pool_depths(program, step_budget, None)
}

/// The peaks a bounded prefix of the run reaches, reading input through
/// `read`.
///
/// This is the answer for a program [`measured_pool_depths`] must decline: one
/// that reads input, so it is not deterministic from the program alone, or one
/// too long to finish inside a budget worth paying at startup. Both mean the
/// exact peak is out of reach, and what is left is an observation — the
/// deepest each pool got over the prefix, which the rest of the run may
/// exceed.
///
/// So this is not a proof and a consumer must not treat it as one. The consumer
/// this exists for sizes a ring that spills to a chain when it overflows, where
/// an observation that comes in low costs traffic rather than an answer.
///
/// `read` is handed `true` for `OP_PUSHNUM` and `false` for `OP_PUSHCHAR` and
/// must decode exactly as the interpreter's own input does, or the branches
/// this walks are not the ones the real run takes. Everything else keeps
/// [`measured_pool_depths`]'s rule: leaving ground the simulation models
/// exactly answers `None` rather than a guess.
pub fn observed_pool_depths(
    program: &Program,
    step_budget: u64,
    read: &mut dyn FnMut(bool) -> i64,
) -> Option<[u32; STORAGE_COUNT]> {
    run_pool_depths(program, step_budget, Some(read))
}

fn run_pool_depths(
    program: &Program,
    step_budget: u64,
    mut read: Option<&mut dyn FnMut(bool) -> i64>,
) -> Option<[u32; STORAGE_COUNT]> {
    for pc in 0..program.size {
        let op = program.get_op(pc);
        if matches!(op, OP_PUSHNUM | OP_PUSHCHAR) && read.is_none() {
            return None;
        }
        if matches!(op, OP_SEL | OP_MOV) && program.get_operand(pc) as usize == VAL_PORT {
            return None;
        }
    }

    let mut pools: [VecDeque<i64>; STORAGE_COUNT] = std::array::from_fn(|_| VecDeque::new());
    let mut maxd = [0u32; STORAGE_COUNT];
    let mut selected = 0usize;
    let mut pc = 0usize;
    let mut steps = 0u64;

    fn pop(pools: &mut [VecDeque<i64>], pool: usize) -> Option<i64> {
        if pool == VAL_QUEUE {
            pools[pool].pop_front()
        } else {
            pools[pool].pop_back()
        }
    }

    while pc < program.size {
        steps += 1;
        if steps > step_budget {
            // An exact answer needs the whole run; an observation is whatever
            // the prefix reached.
            return read.is_some().then_some(maxd);
        }
        let op = program.get_op(pc);
        let operand = program.get_operand(pc);
        let push =
            |pools: &mut [VecDeque<i64>], maxd: &mut [u32; STORAGE_COUNT], pool: usize, v: i64| {
                pools[pool].push_back(v);
                maxd[pool] = maxd[pool].max(pools[pool].len() as u32);
            };
        match op {
            OP_HALT => break,
            OP_JMP => {
                pc = program.get_label(pc);
                if pc >= program.size {
                    return None;
                }
                continue;
            }
            OP_BRZ => {
                let v = pop(&mut pools, selected)?;
                if v == 0 {
                    pc = program.get_label(pc);
                    if pc >= program.size {
                        return None;
                    }
                    continue;
                }
            }
            OP_BRPOP1 | OP_BRPOP2 => {
                let need = if op == OP_BRPOP1 { 1 } else { 2 };
                if pools[selected].len() < need {
                    pc = program.get_label(pc);
                    if pc >= program.size {
                        return None;
                    }
                    continue;
                }
            }
            OP_SEL => {
                selected = operand as usize;
                if selected >= STORAGE_COUNT {
                    return None;
                }
            }
            OP_MOV => {
                let v = pop(&mut pools, selected)?;
                let target = operand as usize;
                if target >= STORAGE_COUNT {
                    return None;
                }
                push(&mut pools, &mut maxd, target, v);
            }
            OP_PUSH => push(&mut pools, &mut maxd, selected, operand as i64),
            OP_PUSHNUM | OP_PUSHCHAR => {
                let v = read.as_mut()?(op == OP_PUSHNUM);
                push(&mut pools, &mut maxd, selected, v);
            }
            OP_DUP => {
                if pools[selected].is_empty() {
                    return None;
                }
                if selected == VAL_QUEUE {
                    let v = pools[selected][0];
                    pools[selected].push_front(v);
                    maxd[selected] = maxd[selected].max(pools[selected].len() as u32);
                } else {
                    let v = *pools[selected].back().unwrap();
                    push(&mut pools, &mut maxd, selected, v);
                }
            }
            OP_SWAP => {
                let n = pools[selected].len();
                if n < 2 {
                    return None;
                }
                if selected == VAL_QUEUE {
                    pools[selected].swap(0, 1);
                } else {
                    pools[selected].swap(n - 1, n - 2);
                }
            }
            OP_POP | OP_POPNUM | OP_POPCHAR => {
                pop(&mut pools, selected)?;
            }
            OP_ADD | OP_SUB | OP_MUL | OP_DIV | OP_MOD | OP_CMP => {
                let b = pop(&mut pools, selected)?;
                let a = pop(&mut pools, selected)?;
                if matches!(op, OP_DIV | OP_MOD) && (b == 0 || (a == i64::MIN && b == -1)) {
                    return None;
                }
                let r = checked_binary_i64(op, a, b)?;
                push(&mut pools, &mut maxd, selected, r);
            }
            _ => {}
        }
        pc += 1;
    }
    Some(maxd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OptimizationLevel;

    fn depths_of(source: &str) -> [DepthBound; STORAGE_COUNT] {
        let program = crate::compile(source, OptimizationLevel::O1);
        max_pool_depths(&program)
    }

    #[test]
    fn straight_line_pushes_bound_the_selected_pool() {
        // 반반반 wraps and pushes three more values each lap: unbounded.
        let d = depths_of("반반반");
        assert_eq!(d[0], DepthBound::Unbounded);
    }

    #[test]
    fn push_pop_loop_stays_bounded() {
        // 반망: push then pop-print, wrapping forever. Net zero per lap, so
        // pool 0 is provably at most one deep.
        let d = depths_of("반망");
        assert_eq!(d[0], DepthBound::Bounded(1));
    }

    #[test]
    fn terminated_pushes_are_bounded() {
        // 반반반희: three pushes then halt — depth exactly three.
        let d = depths_of("반반반희");
        assert_eq!(d[0], DepthBound::Bounded(3));
        assert_eq!(d[1], DepthBound::Bounded(0));
    }

    #[test]
    fn mov_transfers_credit_the_target_pool() {
        // push; push; MOV to pool 3; halt — pool 0 peaks at two, pool 3 at
        // one. Built directly: a MOV under a unique selected debits the
        // source and credits the target.
        let program = Program {
            opcodes: vec![OP_PUSH, OP_PUSH, OP_MOV, OP_HALT],
            values: vec![4, 4, 3, 0],
            labels: Default::default(),
            size: 4,
        };
        let d = max_pool_depths(&program);
        assert_eq!(d[0], DepthBound::Bounded(2));
        assert_eq!(d[3], DepthBound::Bounded(1));
    }

    #[test]
    fn two_selected_paths_that_join_keep_their_own_depths() {
        // Two SEL paths join, then a push and a pop: each path is its own
        // state, so the push credits the pool that path selected and nothing
        // else.
        //   0: BRZ -> 3    1: SEL 1    2: JMP -> 4
        //   3: SEL 2       4: PUSH     5: POP      6: HALT
        let mut labels = std::collections::HashMap::new();
        labels.insert(0, 3usize);
        labels.insert(1, 4usize);
        let mut program = Program {
            opcodes: vec![OP_BRZ, OP_SEL, OP_JMP, OP_SEL, OP_PUSH, OP_POP, OP_HALT],
            values: vec![0, 1, 1, 2, 4, 0, 0],
            labels,
            size: 7,
        };
        program.resolve_jump_targets();
        let d = max_pool_depths(&program);
        assert_eq!(d[1], DepthBound::Bounded(1));
        assert_eq!(d[2], DepthBound::Bounded(1));
    }

    #[test]
    fn a_loop_after_a_join_credits_only_the_pool_its_path_selected() {
        // Two paths select different pools and join on a push/pop loop. Both
        // pools stay one deep: joining the two selected values instead would
        // credit each pool for the other's push and debit neither on the
        // ambiguous pop, making both unbounded.
        //   0: PUSH   1: BRZ -> 4   2: SEL 1   3: JMP -> 5
        //   4: SEL 2  5: PUSH       6: POP     7: JMP -> 5
        let mut labels = std::collections::HashMap::new();
        labels.insert(0, 4usize);
        labels.insert(1, 5usize);
        labels.insert(2, 5usize);
        let mut program = Program {
            opcodes: vec![
                OP_PUSH, OP_BRZ, OP_SEL, OP_JMP, OP_SEL, OP_PUSH, OP_POP, OP_JMP,
            ],
            values: vec![4, 0, 1, 1, 2, 4, 0, 2],
            labels,
            size: 8,
        };
        program.resolve_jump_targets();
        let d = max_pool_depths(&program);
        assert_eq!(d[1], DepthBound::Bounded(1));
        assert_eq!(d[2], DepthBound::Bounded(1));
    }

    #[test]
    fn a_guard_that_cannot_pass_prunes_its_fall_through() {
        // BRPOP1 on an empty pool always jumps, so the push behind it never
        // runs and pool 0 never holds anything.
        //   0: BRPOP1 -> 2   1: PUSH   2: HALT
        let mut labels = std::collections::HashMap::new();
        labels.insert(0, 2usize);
        let mut program = Program {
            opcodes: vec![OP_BRPOP1, OP_PUSH, OP_HALT],
            values: vec![0, 4, 0],
            labels,
            size: 3,
        };
        program.resolve_jump_targets();
        let d = max_pool_depths(&program);
        assert_eq!(d[0], DepthBound::Bounded(0));
    }

    #[test]
    fn a_taken_guard_carries_the_depth_it_proves() {
        // Four pushes, then a guard whose taken edge means the pool holds
        // nothing: the push behind that edge lands on an empty pool, so the
        // peak stays the four from before the guard.
        //   0..3: PUSH   4: BRPOP1 -> 6   5: HALT   6: PUSH   7: HALT
        let mut labels = std::collections::HashMap::new();
        labels.insert(0, 6usize);
        let mut program = Program {
            opcodes: vec![
                OP_PUSH, OP_PUSH, OP_PUSH, OP_PUSH, OP_BRPOP1, OP_HALT, OP_PUSH, OP_HALT,
            ],
            values: vec![4, 4, 4, 4, 0, 0, 4, 0],
            labels,
            size: 8,
        };
        program.resolve_jump_targets();
        let d = max_pool_depths(&program);
        assert_eq!(d[0], DepthBound::Bounded(4));
    }

    #[test]
    fn a_ceiling_widens_a_bound_it_cannot_hold() {
        // Ten pushes then halt: proven at a ceiling that fits them, widened
        // to unbounded at one that does not.
        let program = crate::compile("반반반반반반반반반반희", OptimizationLevel::O1);
        assert_eq!(
            max_pool_depths_up_to(&program, 64)[0],
            DepthBound::Bounded(10)
        );
        assert_eq!(max_pool_depths_up_to(&program, 4)[0], DepthBound::Unbounded);
    }

    #[test]
    fn a_finished_run_measures_the_exact_peak() {
        let program = crate::compile("반반반희", OptimizationLevel::O1);
        let d = measured_pool_depths(&program, 1_000).unwrap();
        assert_eq!(d[0], 3);
        assert_eq!(d[1], 0);
    }

    #[test]
    fn an_input_opcode_declines_the_measurement() {
        // 방 reads a number from input.
        let program = crate::compile("방희", OptimizationLevel::O0);
        assert!(measured_pool_depths(&program, 1_000).is_none());
    }

    #[test]
    fn a_budget_overrun_declines_the_measurement() {
        // A bare wrapping row of pushes never halts.
        let program = crate::compile("반반반", OptimizationLevel::O1);
        assert!(measured_pool_depths(&program, 10_000).is_none());
    }

    #[test]
    fn an_observed_run_reads_its_input() {
        // 방방방희: three numbers read onto pool 0, then halt.
        let program = crate::compile("방방방희", OptimizationLevel::O1);
        let mut next = 0i64;
        let mut read = |as_number: bool| {
            assert!(as_number);
            next += 1;
            next
        };
        let d = observed_pool_depths(&program, 1_000, &mut read).unwrap();
        assert_eq!(d[0], 3);
    }

    #[test]
    fn a_budget_overrun_reports_the_prefix_peak_when_observing() {
        // A bare wrapping row of pushes never halts. There is no exact peak to
        // measure, but the prefix reached one worth reporting.
        let program = crate::compile("반반반", OptimizationLevel::O1);
        assert!(measured_pool_depths(&program, 10_000).is_none());
        let mut read = |_| 0;
        let d = observed_pool_depths(&program, 10_000, &mut read).unwrap();
        assert!(d[0] > 0);
    }

    #[test]
    fn observing_still_declines_ground_it_does_not_model() {
        // A zero divisor is not something the simulation follows, with or
        // without an input to read.
        let mut program = Program {
            opcodes: vec![OP_PUSH, OP_PUSH, OP_DIV, OP_HALT],
            values: vec![4, 0, 0, 0],
            labels: Default::default(),
            size: 4,
        };
        program.resolve_jump_targets();
        let mut read = |_| 0;
        assert!(observed_pool_depths(&program, 1_000, &mut read).is_none());
    }

    #[test]
    fn a_zero_divisor_declines_the_measurement() {
        let mut program = Program {
            opcodes: vec![OP_PUSH, OP_PUSH, OP_DIV, OP_HALT],
            values: vec![4, 0, 0, 0],
            labels: Default::default(),
            size: 4,
        };
        program.resolve_jump_targets();
        assert!(measured_pool_depths(&program, 1_000).is_none());
    }
}
