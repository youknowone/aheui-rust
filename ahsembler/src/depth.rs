//! Static per-pool depth upper bounds over a linearized [`Program`].
//!
//! The stack-machine analog of a code object's `co_stacksize`: the compiler
//! proves, per storage pool, how deep the pool can ever get, so a consumer
//! sizing a per-pool buffer at frame-creation time can size it to the proof
//! instead of to a worst-case constant. Aheui has no static stack discipline,
//! so unlike `co_stacksize` the answer is allowed to be "unbounded" — a loop
//! whose body nets a push grows without limit — and every imprecision in the
//! analysis widens toward that answer, never below the true depth.
//!
//! Runs on the final linear program, after `resolve_jump_targets`: jump
//! operands must already be program counters. An operand outside the program
//! makes the whole result unbounded rather than a panic.
//!
//! [`Program`]: crate::compiler::Program

use crate::compiler::Program;
use crate::consts::*;

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
    fn add(self, delta: i32) -> Self {
        match self {
            Self::Bounded(b) => {
                let next = (b as i64 + delta as i64).max(0);
                if next > WIDEN_LIMIT as i64 {
                    Self::Unbounded
                } else {
                    Self::Bounded(next as u32)
                }
            }
            Self::Unbounded => Self::Unbounded,
        }
    }
}

/// Bounds above this are as good as none: the consumer sizes small per-pool
/// buffers, and a four-figure depth already exceeds every size worth
/// declaring.
const WIDEN_LIMIT: u32 = 4096;

/// Abstract state at a program counter: which pools `OP_SEL` may have
/// selected on some path here (a bitmask over the pool index), and each
/// pool's depth upper bound.
#[derive(Clone, PartialEq)]
struct State {
    selected: u32,
    depths: [DepthBound; STORAGE_COUNT],
}

impl State {
    fn join(&self, other: &Self) -> Self {
        let mut depths = self.depths;
        for (d, o) in depths.iter_mut().zip(other.depths.iter()) {
            *d = d.join(*o);
        }
        State {
            selected: self.selected | other.selected,
            depths,
        }
    }
}

/// The sound direction for every ambiguity is "the pool got deeper":
///
/// * A push under an ambiguous `selected` adds one to every pool the mask
///   names — whichever pool really received it stays covered.
/// * A pop under an ambiguous `selected` subtracts from nothing: a pool that
///   was not the one popped keeps its old depth, so the old bound must stand.
/// * A pop under a unique `selected` does subtract — the pool's true depth
///   fell by exactly one on every path through this op.
fn transfer(state: &mut State, op: u8, operand: i32) {
    match op {
        OP_SEL => {
            let pool = operand as usize;
            state.selected = if pool < STORAGE_COUNT {
                1 << pool
            } else {
                (1 << STORAGE_COUNT) - 1
            };
        }
        OP_MOV => {
            if state.selected.count_ones() == 1 {
                let s = state.selected.trailing_zeros() as usize;
                state.depths[s] = state.depths[s].add(-1);
            }
            let target = operand as usize;
            if target < STORAGE_COUNT {
                state.depths[target] = state.depths[target].add(1);
            } else {
                for d in state.depths.iter_mut() {
                    *d = d.add(1);
                }
            }
        }
        _ => {
            let delta = OP_STACKADD[op as usize] - OP_STACKDEL[op as usize];
            if state.selected.count_ones() == 1 {
                let s = state.selected.trailing_zeros() as usize;
                state.depths[s] = state.depths[s].add(delta);
            } else if delta > 0 {
                for pool in 0..STORAGE_COUNT {
                    if state.selected & (1 << pool) != 0 {
                        state.depths[pool] = state.depths[pool].add(delta);
                    }
                }
            }
        }
    }
}

/// Per-pool depth upper bounds for `program`, or [`DepthBound::Unbounded`]
/// where no bound was proven.
pub fn max_pool_depths(program: &Program) -> [DepthBound; STORAGE_COUNT] {
    let mut result = [DepthBound::Bounded(0); STORAGE_COUNT];
    if program.size == 0 {
        return result;
    }

    let mut states: Vec<Option<State>> = vec![None; program.size];
    states[0] = Some(State {
        selected: 1,
        depths: [DepthBound::Bounded(0); STORAGE_COUNT],
    });
    let mut worklist: Vec<usize> = vec![0];

    let propagate =
        |states: &mut Vec<Option<State>>, worklist: &mut Vec<usize>, pc: usize, state: &State| {
            match &states[pc] {
                Some(old) => {
                    let joined = old.join(state);
                    if joined != *old {
                        states[pc] = Some(joined);
                        worklist.push(pc);
                    }
                }
                None => {
                    states[pc] = Some(state.clone());
                    worklist.push(pc);
                }
            }
        };

    while let Some(pc) = worklist.pop() {
        let Some(entry) = states[pc].clone() else {
            continue;
        };
        let op = program.get_op(pc);
        let operand = program.get_operand(pc);

        let mut state = entry;
        transfer(&mut state, op, operand);
        for (r, d) in result.iter_mut().zip(state.depths.iter()) {
            *r = r.join(*d);
        }

        match op {
            OP_HALT => {}
            OP_JMP => {
                let target = program.get_label(pc);
                if target < program.size {
                    propagate(&mut states, &mut worklist, target, &state);
                } else {
                    return [DepthBound::Unbounded; STORAGE_COUNT];
                }
            }
            OP_BRZ | OP_BRPOP1 | OP_BRPOP2 => {
                let target = program.get_label(pc);
                if target < program.size {
                    propagate(&mut states, &mut worklist, target, &state);
                } else {
                    return [DepthBound::Unbounded; STORAGE_COUNT];
                }
                if pc + 1 < program.size {
                    propagate(&mut states, &mut worklist, pc + 1, &state);
                }
            }
            _ => {
                if pc + 1 < program.size {
                    propagate(&mut states, &mut worklist, pc + 1, &state);
                }
            }
        }
    }

    result
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
        // 반반반: three pushes onto pool 0, then falls off the pane edge and
        // wraps; the wrap re-executes the row, but each revisit joins into the
        // same states, so the loop nets zero... it does not: re-running the
        // pushes grows the pool. The honest expectation is Unbounded for a
        // bare wrapping row of pushes.
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
    fn an_ambiguous_selected_keeps_every_candidates_old_bound_on_a_pop() {
        // Two SEL paths join, then a pop and a halt: neither pool 1 nor
        // pool 2 may take the credit, so both keep the pushed depth.
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
}
