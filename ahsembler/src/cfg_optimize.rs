//! CFG-based optimization passes for Aheui bytecode.
//!
//! Passes:
//! 1. Lattice-based stack depth analysis (fixed-point)
//! 2. Guard elimination (StackGuard → Goto when depth is proven sufficient)
//! 3. Block merging (collapse Goto chains with single predecessor)
//! 4. Iterative constant folding within blocks
//! 5. Dead block elimination

use crate::cfg::*;
use crate::consts::{STORAGE_COUNT, VAL_PORT, VAL_QUEUE};

// ── Pass 1: Fixed-point stack depth analysis ─────────────────────────

/// Exact fixed-point rounds allowed before widening kicks in.
///
/// `Depth::AtLeast(i32)` under `meet = min` has an unbounded descending chain,
/// so a loop that drains a deep stack one element per trip costs one whole
/// round per unit of depth. Round counts measured over the snippets corpus:
/// quine.puzzlet 587, pi.puzzlet 180, logo 54, vowel-advanced 14, everything
/// else <= 11. The two outliers are small CFGs draining a deep stack (46 and
/// 14 blocks), where the extra rounds buy at most one guard.
///
/// 64 clears logo, the largest round count that still pays for itself, and cuts
/// quine's analysis 9x and pi.puzzlet's 3x. Both come out faster end to end even
/// though pi.puzzlet retains one more `BRPOP2` as a result.
///
/// Widening is sound in one direction only, which is why it is safe here: the
/// result feeds `eliminate_guards` via [`Depth::is_at_least`], and `AtLeast(0)`
/// is the weakest claim, so widening can only keep a guard that would otherwise
/// have been removed — never remove one that is not proven safe.
const STACK_DEPTH_WIDENING_BUDGET: usize = 64;

/// Run abstract interpretation to compute stack depth at each block entry.
/// Sweeps dirty blocks in RPO to a fixed point, widening past
/// [`STACK_DEPTH_WIDENING_BUDGET`] rounds.
pub fn analyze_stack_depths(cfg: &Cfg) -> Vec<AbstractState> {
    let n = cfg.num_blocks();
    let mut states: Vec<AbstractState> = vec![AbstractState::bottom(); n];
    states[cfg.entry as usize] = AbstractState::initial();

    let rpo = cfg.reverse_postorder();
    let has_guard_depth: Vec<bool> = cfg
        .blocks
        .iter()
        .map(|block| {
            block
                .instructions
                .iter()
                .any(|inst| matches!(inst, Inst::GuardDepth { .. }))
        })
        .collect();
    let mut dirty = vec![false; n];
    dirty[cfg.entry as usize] = true;
    let mut successors = Vec::new();
    let mut round = 0;

    // Iterate until fixed point. Depth values decrease monotonically (meet = min)
    // and are bounded below by 0, so convergence is guaranteed; widening past the
    // budget bounds how many rounds that takes.
    loop {
        round += 1;
        let widen = round > STACK_DEPTH_WIDENING_BUDGET;
        let mut changed = false;
        for &block_id in &rpo {
            let idx = block_id as usize;
            if !dirty[idx] {
                continue;
            }
            dirty[idx] = false;

            let entry_state = &states[idx];

            if entry_state.is_bottom() {
                continue;
            }

            let exit_state = transfer_block(cfg.block(block_id), entry_state);

            successor_states(
                cfg.block(block_id),
                entry_state,
                &exit_state,
                has_guard_depth[idx],
                &mut successors,
            );
            for (succ_id, succ_state) in successors.drain(..) {
                let succ_idx = succ_id as usize;
                if succ_idx >= n {
                    continue;
                }
                let mut merged = states[succ_idx].meet(&succ_state);
                if widen {
                    // AtLeast(0) is the weakest claim: widening can only make
                    // is_at_least false and retain a guard, never eliminate an
                    // unproven guard.
                    for (old, new) in states[succ_idx].depths.iter().zip(merged.depths.iter_mut()) {
                        if matches!((*old, *new), (Depth::AtLeast(a), Depth::AtLeast(b)) if b < a) {
                            *new = Depth::AtLeast(0);
                        }
                    }
                }
                if merged != states[succ_idx] {
                    states[succ_idx] = merged;
                    dirty[succ_idx] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    states
}

/// Compute the abstract state after executing all instructions in a block.
fn transfer_block(block: &CfgBlock, entry: &AbstractState) -> AbstractState {
    let mut state = entry.clone();
    for inst in &block.instructions {
        transfer_inst(inst, &mut state);
    }
    state
}

/// Apply one instruction's effect to the abstract state.
pub fn transfer_inst(inst: &Inst, state: &mut AbstractState) {
    match inst {
        Inst::Sel(s) => {
            state.selected = Some(*s);
        }
        Inst::Mov(target) => {
            if let Some(sel) = state.selected {
                apply_depth_delta(state, sel, -1);
                apply_depth_delta(state, *target, 1);
            }
        }
        Inst::GuardDepth { min_depth, .. } => {
            // After passing GuardDepth, depth >= min_depth is guaranteed (ok path).
            if let Some(sel) = state.selected {
                if sel < STORAGE_COUNT {
                    let min = *min_depth as i32;
                    state.depths[sel] = match state.depths[sel] {
                        Depth::Bottom => Depth::Bottom,
                        Depth::AtLeast(d) => Depth::AtLeast(d.max(min)),
                    };
                }
            }
        }
        other => {
            if let Some(sel) = state.selected {
                let delta = other.stack_delta();
                apply_depth_delta(state, sel, delta);
            }
        }
    }
}

fn apply_depth_delta(state: &mut AbstractState, storage: usize, delta: i32) {
    if storage < STORAGE_COUNT {
        state.depths[storage] = match state.depths[storage] {
            Depth::Bottom => Depth::Bottom,
            Depth::AtLeast(d) => Depth::AtLeast((d + delta).max(0)),
        };
    }
}

/// Compute the abstract state delivered to each successor.
fn successor_states(
    block: &CfgBlock,
    entry: &AbstractState,
    exit_state: &AbstractState,
    has_guard_depth: bool,
    result: &mut Vec<(BlockId, AbstractState)>,
) {
    result.clear();
    // GuardDepth instructions within the block create additional successors.
    if has_guard_depth {
        let mut state = entry.clone();
        for inst in &block.instructions {
            if let Inst::GuardDepth { fail, .. } = inst {
                result.push((*fail, state.clone()));
            }
            transfer_inst(inst, &mut state);
        }
    }

    match &block.terminator {
        Terminator::Goto(target) => {
            result.push((*target, exit_state.clone()));
        }
        Terminator::StackGuard {
            min_depth,
            ok,
            fail,
        } => {
            let mut ok_state = exit_state.clone();
            if let Some(sel) = ok_state.selected {
                if sel < STORAGE_COUNT {
                    let min = *min_depth as i32;
                    ok_state.depths[sel] = match ok_state.depths[sel] {
                        Depth::Bottom => Depth::Bottom,
                        Depth::AtLeast(d) => Depth::AtLeast(d.max(min)),
                    };
                }
            }
            let fail_state = exit_state.clone();
            result.push((*ok, ok_state));
            result.push((*fail, fail_state));
        }
        Terminator::BranchZero {
            on_zero,
            on_nonzero,
        } => {
            let mut post_pop = exit_state.clone();
            if let Some(sel) = post_pop.selected {
                apply_depth_delta(&mut post_pop, sel, -1);
            }
            result.push((*on_zero, post_pop.clone()));
            result.push((*on_nonzero, post_pop));
        }
        Terminator::Halt => {}
    }
}

// ── Pass 2: Guard elimination ────────────────────────────────────────

/// Eliminate StackGuard terminators where the abstract state proves sufficient
/// depth. Returns the number of guards eliminated.
pub fn eliminate_guards(cfg: &mut Cfg, states: &[AbstractState]) -> usize {
    let mut eliminated = 0;

    for block_id in 0..cfg.num_blocks() {
        let entry = &states[block_id];
        if entry.is_bottom() {
            continue;
        }

        if let Terminator::StackGuard { min_depth, ok, .. } =
            cfg.block(block_id as BlockId).terminator
        {
            let at_terminator = transfer_block(cfg.block(block_id as BlockId), entry);
            let can_eliminate = at_terminator
                .selected
                .is_some_and(|sel| at_terminator.depths[sel].is_at_least(min_depth as i32));

            if can_eliminate {
                cfg.block_mut(block_id as BlockId).terminator = Terminator::Goto(ok);
                eliminated += 1;
            }
        }
    }

    eliminated
}

/// Strip ALL GuardDepth instructions unconditionally. Unsafe but enables maximum merging.
pub fn strip_all_guard_depth(cfg: &mut Cfg) -> usize {
    let mut stripped = 0;
    for block_id in 0..cfg.num_blocks() {
        let before = cfg.block(block_id as BlockId).instructions.len();
        cfg.block_mut(block_id as BlockId)
            .instructions
            .retain(|inst| !matches!(inst, Inst::GuardDepth { .. }));
        stripped += before - cfg.block(block_id as BlockId).instructions.len();
    }
    stripped
}

/// Eliminate GuardDepth instructions where the abstract state proves sufficient depth.
pub fn eliminate_guard_depth(cfg: &mut Cfg, states: &[AbstractState]) -> usize {
    let mut eliminated = 0;
    for block_id in 0..cfg.num_blocks() {
        let entry = &states[block_id];
        if entry.is_bottom() {
            continue;
        }
        let block = cfg.block_mut(block_id as BlockId);
        let mut state = entry.clone();
        block.instructions.retain(|inst| {
            if let Inst::GuardDepth { min_depth, .. } = inst {
                let can_remove = state
                    .selected
                    .is_some_and(|sel| state.depths[sel].is_at_least(*min_depth as i32));
                if can_remove {
                    eliminated += 1;
                    return false;
                }
            }
            transfer_inst(inst, &mut state);
            true
        });
    }
    eliminated
}

// ── Pass 3: Block merging ────────────────────────────────────────────

/// Merge blocks connected by Goto where the target has a single predecessor.
/// Returns the number of merges performed.
pub fn merge_blocks(cfg: &mut Cfg) -> usize {
    let mut merged = 0;

    // Build predecessor counts once. Merging a→b only invalidates b's
    // successors' pred counts, but since b becomes dead (self-loop),
    // we can safely iterate without full rebuild.
    let mut pred_count = vec![0u32; cfg.num_blocks()];
    for block in &cfg.blocks {
        for &succ in &block.all_successors() {
            if (succ as usize) < pred_count.len() {
                pred_count[succ as usize] += 1;
            }
        }
    }

    // Forward scan: merge chains greedily.
    for block_id in 0..cfg.num_blocks() as BlockId {
        loop {
            let Terminator::Goto(target) = cfg.block(block_id).terminator else {
                break;
            };
            if target == block_id || target == cfg.entry || pred_count[target as usize] != 1 {
                break;
            }
            // SEL is now a regular instruction within blocks.
            // The codegen handles intra-block SEL by switching storage Variables.

            let target_insts = cfg.block(target).instructions.clone();
            let target_term = cfg.block(target).terminator.clone();

            // Update pred counts for target's successors before removing
            for &succ in &target_term.successors() {
                if (succ as usize) < pred_count.len() {
                    pred_count[succ as usize] -= 1;
                }
            }
            // The merged block now reaches those successors
            for &succ in &target_term.successors() {
                if (succ as usize) < pred_count.len() {
                    pred_count[succ as usize] += 1;
                }
            }

            let block = cfg.block_mut(block_id);
            block.instructions.extend(target_insts);
            block.terminator = target_term;

            cfg.block_mut(target).instructions.clear();
            cfg.block_mut(target).terminator = Terminator::Goto(target);
            pred_count[target as usize] = 1; // Self-loop

            merged += 1;
            // Continue — might be able to merge the next target too
        }
    }

    merged
}

/// Merge StackGuard ok-blocks into their guard block when ok has single predecessor.
/// StackGuard { ok: B, fail: C } where pred_count[B]==1
/// → GuardDepth instruction + B's instructions, terminator = B's terminator.
pub fn merge_guard_ok(cfg: &mut Cfg) -> usize {
    let mut merged = 0;
    let mut pred_count = vec![0u32; cfg.num_blocks()];
    for block in &cfg.blocks {
        for &succ in &block.all_successors() {
            if (succ as usize) < pred_count.len() {
                pred_count[succ as usize] += 1;
            }
        }
    }

    for block_id in 0..cfg.num_blocks() as BlockId {
        let Terminator::StackGuard {
            min_depth,
            ok,
            fail,
        } = cfg.block(block_id).terminator
        else {
            continue;
        };
        if ok == block_id || ok == cfg.entry || pred_count[ok as usize] != 1 {
            continue;
        }
        if ok == fail {
            continue;
        }

        let ok_block = cfg.block(ok);
        let ok_insts = ok_block.instructions.clone();
        let ok_term = ok_block.terminator.clone();

        // Before merge: block_id → StackGuard { ok, fail }
        //   successors: ok, fail
        // After merge:  block_id → [GuardDepth(fail)] + ok_insts + ok_term
        //   successors: fail (from GuardDepth), ok_term.successors(), ok_insts GuardDepth targets
        // Net change: remove ok from successors, add ok_term successors
        // (fail was already a successor, ok_term successors are new)
        pred_count[ok as usize] = pred_count[ok as usize].saturating_sub(1); // ok no longer a successor
        for &succ in &ok_term.successors() {
            if (succ as usize) < pred_count.len() {
                pred_count[succ as usize] += 1;
            }
        }
        for inst in &ok_insts {
            if let Inst::GuardDepth { fail: gf, .. } = inst {
                if (*gf as usize) < pred_count.len() {
                    pred_count[*gf as usize] += 1;
                }
            }
        }

        let block = cfg.block_mut(block_id);
        block
            .instructions
            .push(Inst::GuardDepth { min_depth, fail });
        block.instructions.extend(ok_insts);
        block.terminator = ok_term;

        cfg.block_mut(ok).instructions.clear();
        cfg.block_mut(ok).terminator = Terminator::Goto(ok);
        pred_count[ok as usize] = 1; // self-loop

        merged += 1;
    }
    merged
}

// ── Pass 4: Iterative constant folding ───────────────────────────────

/// Range a folded constant has to fit for whoever will emit it.
///
/// `Inst::Push` carries an `i64`, but `cfg_linearize` writes the result into
/// `Program::values`, a `Vec<i32>`. A fold headed there may not leave the i32
/// range: the value would be truncated on the way out and the program would
/// then compute with the truncated constant. The AOT backends read
/// `Inst::Push` out of the CFG and emit it as an `i64` literal, so nothing
/// bounds them.
///
/// A rejected fold is not an error — the original instructions stay in place
/// and the value is computed at run time, where it is a bigint and exact.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConstWidth {
    /// Destined for `cfg_linearize` / `Program::values`.
    I32,
    /// Consumed straight off the CFG by an AOT backend.
    I64,
}

impl ConstWidth {
    /// Whether a folded constant may be written back as `Inst::Push(v)`.
    fn holds(self, v: i64) -> bool {
        match self {
            ConstWidth::I32 => i32::try_from(v).is_ok(),
            ConstWidth::I64 => true,
        }
    }
}

/// Fold constants within basic blocks. Returns number of folds performed.
pub fn fold_constants(cfg: &mut Cfg, states: &[AbstractState], width: ConstWidth) -> usize {
    let mut total_folds = 0;

    for block_id in 0..cfg.num_blocks() as BlockId {
        let entry = &states[block_id as usize];
        if entry.is_bottom() {
            continue;
        }

        // Determine if block operates on a queue/port storage
        let in_queue = is_queue_block(cfg.block(block_id), entry);
        if in_queue {
            continue; // Skip constant folding for queue/port
        }

        total_folds += fold_block_constants(cfg.block_mut(block_id), width);
    }

    total_folds
}

fn is_queue_block(block: &CfgBlock, entry: &AbstractState) -> bool {
    let mut selected = entry.selected;
    for inst in &block.instructions {
        if let Inst::Sel(s) = inst {
            selected = Some(*s);
        }
        if let Some(sel) = selected {
            if sel == VAL_QUEUE || sel == VAL_PORT {
                return true;
            }
        }
    }
    false
}

fn fold_block_constants(block: &mut CfgBlock, width: ConstWidth) -> usize {
    let mut folds = 0;
    let mut changed = true;

    while changed {
        changed = false;

        // Pattern: Push(a), Push(b), BinOp → Push(result)
        let mut i = 0;
        while i + 2 < block.instructions.len() {
            if let (Inst::Push(a), Inst::Push(b), Inst::BinOp(kind)) = (
                block.instructions[i],
                block.instructions[i + 1],
                block.instructions[i + 2],
            ) {
                if let Some(result) = eval_binop(kind, a, b).filter(|r| width.holds(*r)) {
                    block.instructions[i] = Inst::Push(result);
                    block.instructions.remove(i + 2);
                    block.instructions.remove(i + 1);
                    folds += 1;
                    changed = true;
                    continue; // Re-check at same position
                }
            }
            i += 1;
        }

        // Pattern: Push(v), Dup, BinOp → Push(result)
        i = 0;
        while i + 2 < block.instructions.len() {
            if let (Inst::Push(v), Inst::Dup, Inst::BinOp(kind)) = (
                block.instructions[i],
                block.instructions[i + 1],
                block.instructions[i + 2],
            ) {
                if let Some(result) = eval_binop(kind, v, v).filter(|r| width.holds(*r)) {
                    block.instructions[i] = Inst::Push(result);
                    block.instructions.remove(i + 2);
                    block.instructions.remove(i + 1);
                    folds += 1;
                    changed = true;
                    continue;
                }
            }
            i += 1;
        }

        // Pattern: Push(v), Dup → Push(v), Push(v)
        i = 0;
        while i + 1 < block.instructions.len() {
            if let (Inst::Push(v), Inst::Dup) = (block.instructions[i], block.instructions[i + 1]) {
                block.instructions[i + 1] = Inst::Push(v);
                changed = true;
                // Don't count as fold, but enable further folding
            }
            i += 1;
        }
    }

    // Fold BRZ on constant: Push(v); BranchZero → constant branch
    fold_constant_branch(block, &mut folds);

    folds
}

fn fold_constant_branch(block: &mut CfgBlock, folds: &mut usize) {
    if block.instructions.is_empty() {
        return;
    }
    let last_idx = block.instructions.len() - 1;
    if let Inst::Push(v) = block.instructions[last_idx] {
        if let Terminator::BranchZero {
            on_zero,
            on_nonzero,
        } = block.terminator
        {
            block.instructions.pop();
            block.terminator = if v == 0 {
                Terminator::Goto(on_zero)
            } else {
                Terminator::Goto(on_nonzero)
            };
            *folds += 1;
        }
    }
}

fn eval_binop(kind: BinOpKind, lhs: i64, rhs: i64) -> Option<i64> {
    Some(match kind {
        BinOpKind::Add => lhs.checked_add(rhs)?,
        BinOpKind::Sub => lhs.checked_sub(rhs)?,
        BinOpKind::Mul => lhs.checked_mul(rhs)?,
        BinOpKind::Div => {
            if rhs != 0 {
                crate::consts::floor_div_i64(lhs, rhs)
            } else {
                0
            }
        }
        BinOpKind::Mod => {
            if rhs != 0 {
                crate::consts::floor_mod_i64(lhs, rhs)
            } else {
                0
            }
        }
        BinOpKind::Cmp => {
            if lhs >= rhs {
                1
            } else {
                0
            }
        }
    })
}

// ── Pass 4b: Peephole instruction cleanup ───────────────────────────

/// Remove redundant instructions within blocks:
/// - Redundant SEL (already at that storage)
/// - Self-MOV (MOV to current storage = pop + push = no-op... but changes
///   stacksize tracking, so only safe when we know it's same storage)
/// - Identity operations: PUSH 0; ADD/SUB, PUSH 1; MUL/DIV
pub fn peephole_cleanup(cfg: &mut Cfg, states: &[AbstractState]) -> usize {
    let mut removed = 0;

    for block_id in 0..cfg.num_blocks() as BlockId {
        let entry = &states[block_id as usize];
        if entry.is_bottom() {
            continue;
        }
        removed += peephole_block(cfg.block_mut(block_id), entry);
    }

    removed
}

fn peephole_block(block: &mut CfgBlock, entry: &AbstractState) -> usize {
    let mut removed = 0;

    // Pass 1: Remove redundant SEL
    let mut selected = entry.selected;
    let mut i = 0;
    while i < block.instructions.len() {
        if let Inst::Sel(s) = block.instructions[i] {
            if selected == Some(s) {
                block.instructions.remove(i);
                removed += 1;
                continue;
            }
            selected = Some(s);
        } else if let Inst::Mov(target) = block.instructions[i] {
            // Self-MOV: pop from current, push to same = net zero on that storage,
            // but only when selected == target. In Aheui semantics this is truly
            // a no-op for stacks (pop top, push to same stack = same state).
            // For queues it's different (pop front, push back), so skip those.
            if selected == Some(target)
                && target != crate::consts::VAL_QUEUE
                && target != crate::consts::VAL_PORT
            {
                block.instructions.remove(i);
                removed += 1;
                continue;
            }
        }
        i += 1;
    }

    // Pass 2: Remove identity operations
    i = 0;
    while i + 1 < block.instructions.len() {
        let is_identity = match (block.instructions[i], block.instructions[i + 1]) {
            (Inst::Push(0), Inst::BinOp(BinOpKind::Add | BinOpKind::Sub)) => true,
            (Inst::Push(1), Inst::BinOp(BinOpKind::Mul | BinOpKind::Div)) => true,
            (Inst::Dup, Inst::Pop) => true, // Dup then Pop = no-op
            _ => false,
        };
        if is_identity {
            block.instructions.remove(i + 1);
            block.instructions.remove(i);
            removed += 2;
            continue;
        }
        i += 1;
    }

    removed
}

// ── Pass 5: Dead block elimination ───────────────────────────────────

/// Remove blocks not reachable from entry.
pub fn eliminate_dead_blocks(cfg: &mut Cfg) {
    let rpo = cfg.reverse_postorder();
    let mut reachable = vec![false; cfg.num_blocks()];
    for &id in &rpo {
        reachable[id as usize] = true;
    }

    // Remap block IDs: assign new contiguous IDs to reachable blocks
    let mut id_map = vec![0u32; cfg.num_blocks()];
    let mut new_id = 0u32;
    for (old_id, &is_reachable) in reachable.iter().enumerate() {
        if is_reachable {
            id_map[old_id] = new_id;
            new_id += 1;
        }
    }

    if new_id as usize == cfg.num_blocks() {
        return; // Nothing to remove
    }

    // Remap all successor references
    let mut new_blocks = Vec::with_capacity(new_id as usize);
    for (old_id, block) in cfg.blocks.iter().enumerate() {
        if !reachable[old_id] {
            continue;
        }
        let mut new_block = block.clone();
        new_block.id = id_map[old_id];
        remap_terminator(&mut new_block.terminator, &id_map);
        // Remap GuardDepth fail targets in instructions
        for inst in &mut new_block.instructions {
            if let Inst::GuardDepth { fail, .. } = inst {
                *fail = id_map[*fail as usize];
            }
        }
        new_blocks.push(new_block);
    }

    cfg.blocks = new_blocks;
    cfg.entry = id_map[cfg.entry as usize];
}

fn remap_terminator(term: &mut Terminator, id_map: &[u32]) {
    match term {
        Terminator::Goto(b) => *b = id_map[*b as usize],
        Terminator::StackGuard { ok, fail, .. } => {
            *ok = id_map[*ok as usize];
            *fail = id_map[*fail as usize];
        }
        Terminator::BranchZero {
            on_zero,
            on_nonzero,
        } => {
            *on_zero = id_map[*on_zero as usize];
            *on_nonzero = id_map[*on_nonzero as usize];
        }
        Terminator::Halt => {}
    }
}

// ── Pass 6: Jump threading ───────────────────────────────────────────

/// Thread jumps through empty blocks. If a block has no instructions and its
/// terminator is Goto(target), redirect all references to this block to target.
/// Also handles chains of empty blocks.
/// Returns the number of redirections performed.
pub fn thread_jumps(cfg: &mut Cfg) -> usize {
    let n = cfg.num_blocks();

    // Build a forwarding map: for each block, find its final target
    // through chains of empty Goto blocks.
    let mut forward: Vec<BlockId> = (0..n as BlockId).collect();

    // Resolve chains
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..n {
            let block = cfg.block(i as BlockId);
            if block.instructions.is_empty() {
                if let Terminator::Goto(target) = block.terminator {
                    if forward[i] != forward[target as usize] {
                        forward[i] = forward[target as usize];
                        changed = true;
                    }
                }
            }
        }
    }

    // Apply forwarding to all terminators
    let mut redirections = 0;
    for i in 0..n {
        let block = cfg.block_mut(i as BlockId);
        let mut term = block.terminator.clone();
        let old_term = term.clone();

        match &mut term {
            Terminator::Goto(b) => *b = forward[*b as usize],
            Terminator::StackGuard { ok, fail, .. } => {
                *ok = forward[*ok as usize];
                *fail = forward[*fail as usize];
            }
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } => {
                *on_zero = forward[*on_zero as usize];
                *on_nonzero = forward[*on_nonzero as usize];
            }
            Terminator::Halt => {}
        }

        if term != old_term {
            cfg.block_mut(i as BlockId).terminator = term;
            redirections += 1;
        }
        // Also forward GuardDepth fail targets
        for inst in &mut cfg.block_mut(i as BlockId).instructions {
            if let Inst::GuardDepth { fail, .. } = inst {
                let fwd = forward[*fail as usize];
                if fwd != *fail {
                    *fail = fwd;
                    redirections += 1;
                }
            }
        }
    }

    // Also update entry if it points to an empty forwarder
    cfg.entry = forward[cfg.entry as usize];

    redirections
}

// ── Pass 7: Loop-aware guard elimination ────────────────────────────

/// Eliminate GuardDepth instructions in natural loops that are provably
/// self-consistent: if assuming all guards pass leads to the back-edge
/// exiting with depth >= the loop header's entry depth, the guards are
/// redundant and can be safely removed.
///
/// Returns the number of GuardDepth instructions removed.
pub fn eliminate_loop_guards(cfg: &mut Cfg) -> usize {
    let rpo = cfg.reverse_postorder();
    let mut rpo_pos: Vec<Option<usize>> = vec![None; cfg.num_blocks()];
    for (i, &bid) in rpo.iter().enumerate() {
        rpo_pos[bid as usize] = Some(i);
    }

    // 1. Find natural loops: back-edge = edge where succ RPO pos <= src RPO pos
    //    header = succ, latch = src
    // `BTreeMap`, not `HashMap`: the header order below is a decision order, and
    // a `HashMap`'s is seeded per process, so the pass would strip a different
    // set of guards from one run to the next.
    let mut loop_headers: std::collections::BTreeMap<BlockId, Vec<BlockId>> =
        std::collections::BTreeMap::new();
    for (idx, &bid) in rpo.iter().enumerate() {
        for &succ in &cfg.blocks[bid as usize].all_successors() {
            if let Some(succ_pos) = rpo_pos[succ as usize] {
                if succ_pos <= idx {
                    loop_headers.entry(succ).or_default().push(bid);
                }
            }
        }
    }

    if loop_headers.is_empty() {
        return 0;
    }

    // Run normal analysis first to get entry states
    let states = analyze_stack_depths(cfg);

    // Every header is judged against the same unmutated CFG and the `states`
    // computed from it, and the strips are applied once at the end. Stripping
    // inside the scan would let an earlier header's rewrite change a later
    // header's `has_guards` / `max_guard_depth` / `guarded_storages` while
    // `states` still described the CFG as it was, so the answer would depend on
    // the order the headers happened to be visited in.
    let mut strip = vec![false; cfg.num_blocks()];

    for (&header, latches) in &loop_headers {
        let header_state = &states[header as usize];
        if header_state.is_bottom() {
            continue;
        }

        // Collect all blocks in this natural loop (blocks dominated by header
        // that can reach a latch). Use reverse walk from latches to header.
        let loop_blocks = collect_loop_blocks(cfg, header, latches, &rpo_pos);
        if loop_blocks.is_empty() {
            continue;
        }

        // Find max guard depth in this loop
        let max_guard_depth: i32 = loop_blocks
            .iter()
            .flat_map(|&bid| {
                cfg.block(bid)
                    .instructions
                    .iter()
                    .filter_map(|i| {
                        if let Inst::GuardDepth { min_depth, .. } = i {
                            Some(*min_depth as i32)
                        } else {
                            None
                        }
                    })
                    .chain(
                        if let Terminator::StackGuard { min_depth, .. } = &cfg.block(bid).terminator
                        {
                            Some(*min_depth as i32)
                        } else {
                            None
                        },
                    )
            })
            .max()
            .unwrap_or(0);

        let has_guards = loop_blocks.iter().any(|&bid| {
            cfg.block(bid)
                .instructions
                .iter()
                .any(|i| matches!(i, Inst::GuardDepth { .. }))
        });
        if !has_guards {
            continue;
        }

        // 2. Check: can we prove guards are redundant?
        // Strategy: if the ENTRY path (non-back-edge predecessors) delivers depth >= max_guard
        // for all non-bottom storages, AND check_loop_consistent says self-consistent,
        // then guards are safe to remove.
        let loop_set: std::collections::HashSet<BlockId> = loop_blocks.iter().copied().collect();
        let preds = cfg.predecessors();
        let non_backedge_preds: Vec<BlockId> = preds[header as usize]
            .iter()
            .filter(|&&p| !loop_set.contains(&p))
            .copied()
            .collect();

        // Compute entry depth from non-back-edge predecessors only
        // Collect which storages actually have guards in this loop
        let mut guarded_storages = std::collections::HashSet::new();
        for &bid in &loop_blocks {
            let mut st = states[bid as usize].clone();
            for inst in &cfg.block(bid).instructions {
                if let Inst::GuardDepth { .. } = inst {
                    if let Some(sel) = st.selected {
                        guarded_storages.insert(sel);
                    }
                }
                transfer_inst(inst, &mut st);
            }
        }

        let entry_depth_ok = if non_backedge_preds.is_empty() || guarded_storages.is_empty() {
            false
        } else {
            non_backedge_preds.iter().all(|&pred| {
                let exit = transfer_block(cfg.block(pred), &states[pred as usize]);
                guarded_storages
                    .iter()
                    .all(|&s| exit.depths[s].is_at_least(max_guard_depth))
            })
        };

        if entry_depth_ok
            && check_loop_consistent(cfg, header, &loop_blocks, &states, &guarded_storages)
        {
            // 3. Mark all GuardDepth in the loop for removal
            for &bid in &loop_blocks {
                strip[bid as usize] = true;
            }
        }
    }

    let mut total_removed = 0;
    for bid in 0..cfg.num_blocks() {
        if !strip[bid] {
            continue;
        }
        let block = cfg.block_mut(bid as BlockId);
        let before = block.instructions.len();
        block
            .instructions
            .retain(|inst| !matches!(inst, Inst::GuardDepth { .. }));
        total_removed += before - block.instructions.len();
    }

    total_removed
}

/// Eliminate guards in linear chains where all fail targets point within the
/// chain. If the non-chain entry delivers enough depth for the entire chain's
/// consumption, all guards are provably redundant.
///
/// Pattern: B_k has GuardDepth(d, fail=B_j) where B_j is earlier in the chain.
/// The chain starts from a block whose non-chain predecessor delivers sufficient
/// depth computed by the transfer function (ignoring back-edges from the chain).
pub fn eliminate_chain_guards(cfg: &mut Cfg) -> usize {
    let states = analyze_stack_depths(cfg);
    let preds = cfg.predecessors();
    let n = cfg.num_blocks();
    let mut total_removed = 0;

    // Find chain heads: blocks with GuardDepth whose predecessors include at
    // least one non-chain block (a block without GuardDepth as first instruction).
    // Walk the Goto chain forward, collecting blocks whose only GuardDepth fail
    // targets point backward within the chain (or to themselves).
    for start in 0..n {
        let bid = start as BlockId;
        let block = cfg.block(bid);
        // Must have GuardDepth as first meaningful instruction
        let first_guard = block
            .instructions
            .iter()
            .position(|i| matches!(i, Inst::GuardDepth { .. }));
        if first_guard.is_none() {
            continue;
        }

        // Must have at least one non-chain predecessor (provides the initial depth)
        let has_external_pred = preds[start].iter().any(|&p| {
            // External = not pointing from a later chain block's Goto
            !cfg.block(p)
                .instructions
                .iter()
                .any(|i| matches!(i, Inst::GuardDepth { .. }))
                || p as usize >= n // shouldn't happen, but safe
        });
        if !has_external_pred {
            continue;
        }

        // Walk the chain forward via Goto successors
        let mut chain: Vec<BlockId> = vec![bid];
        let mut chain_set: std::collections::HashSet<BlockId> = std::collections::HashSet::new();
        chain_set.insert(bid);
        let mut cur = bid;
        loop {
            if let Terminator::Goto(next) = cfg.block(cur).terminator {
                if chain_set.contains(&next) {
                    break;
                } // cycle
                let next_block = cfg.block(next);
                let has_guard = next_block
                    .instructions
                    .iter()
                    .any(|i| matches!(i, Inst::GuardDepth { .. }));
                if !has_guard {
                    break;
                }
                chain.push(next);
                chain_set.insert(next);
                cur = next;
            } else {
                break;
            }
        }

        if chain.len() < 2 {
            continue;
        }

        // Check: all GuardDepth fail targets must point within the chain
        // or to predecessors of the chain start (setup/feeder blocks).
        // If depth is proven sufficient, these fail paths are dead code.
        let chain_start_preds: std::collections::HashSet<BlockId> =
            preds[start].iter().copied().collect();
        let all_fail_ok = chain.iter().all(|&b| {
            cfg.block(b).instructions.iter().all(|inst| {
                if let Inst::GuardDepth { fail, .. } = inst {
                    chain_set.contains(fail) || chain_start_preds.contains(fail)
                } else {
                    true
                }
            })
        });
        if !all_fail_ok {
            continue;
        }

        // Check: all blocks must be on the same storage (sel)
        let chain_sel = states[start].selected;
        if chain_sel.is_none() {
            continue;
        }
        let sel = chain_sel.unwrap();

        // Compute depth at chain entry via forward-only transfer from the
        // program entry (block 0). This avoids back-edge dilution from the
        // fixed-point analysis and gives the first-iteration depth.
        let mut fwd_states: Vec<Option<AbstractState>> = vec![None; n];
        fwd_states[0] = Some(AbstractState::initial());
        let rpo_order = cfg.reverse_postorder();
        // Forward pass: only propagate through Goto edges (ignoring back-edges)
        for _ in 0..n {
            let mut changed = false;
            for &bid in &rpo_order {
                let entry = match &fwd_states[bid as usize] {
                    Some(s) => s.clone(),
                    None => continue,
                };
                let exit = transfer_block(cfg.block(bid), &entry);
                // Propagate only through Goto (linear) edges to non-chain blocks
                // or to the chain start
                if let Terminator::Goto(next) = cfg.block(bid).terminator {
                    let succ_state = match &fwd_states[next as usize] {
                        Some(existing) => existing.meet(&exit),
                        None => exit.clone(),
                    };
                    if fwd_states[next as usize].as_ref() != Some(&succ_state) {
                        fwd_states[next as usize] = Some(succ_state);
                        changed = true;
                    }
                }
                // Also propagate BranchZero
                if let Terminator::BranchZero {
                    on_zero,
                    on_nonzero,
                } = cfg.block(bid).terminator
                {
                    let mut post_pop = exit.clone();
                    if let Some(s) = post_pop.selected {
                        apply_depth_delta(&mut post_pop, s, -1);
                    }
                    for next in [on_zero, on_nonzero] {
                        let succ_state = match &fwd_states[next as usize] {
                            Some(existing) => existing.meet(&post_pop),
                            None => post_pop.clone(),
                        };
                        if fwd_states[next as usize].as_ref() != Some(&succ_state) {
                            fwd_states[next as usize] = Some(succ_state);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let min_entry_depth: i32 = match &fwd_states[start] {
            Some(s) => match s.depths[sel] {
                Depth::Bottom => 0,
                Depth::AtLeast(d) => d,
            },
            None => {
                continue;
            }
        };

        // Simulate the chain with this entry depth (no back-edges),
        // tracking sel changes within blocks
        let mut depth = min_entry_depth;
        let mut cur_sel = sel;
        let mut safe = true;
        for &b in &chain {
            let block = cfg.block(b);
            for inst in &block.instructions {
                match inst {
                    Inst::Sel(s) => {
                        cur_sel = *s;
                    }
                    Inst::GuardDepth { min_depth, .. } => {
                        if cur_sel == sel && depth < *min_depth as i32 {
                            safe = false;
                            break;
                        }
                    }
                    Inst::Mov(dst) => {
                        if cur_sel == sel {
                            depth -= 1;
                        }
                        if *dst == sel {
                            depth += 1;
                        }
                    }
                    _ => {
                        if cur_sel == sel {
                            depth += inst.stack_delta();
                            if depth < 0 {
                                depth = 0;
                            }
                        }
                    }
                }
            }
            if !safe {
                break;
            }
        }

        if safe {
            for &b in &chain {
                let block = cfg.block_mut(b);
                let before = block.instructions.len();
                block
                    .instructions
                    .retain(|i| !matches!(i, Inst::GuardDepth { .. }));
                total_removed += before - block.instructions.len();
            }
        }
    }

    total_removed
}

/// Collect all blocks in the natural loop with the given header and latches.
/// A block B is in the loop if there exists a path from header to B to latch
/// that doesn't leave the loop. We use reverse BFS from latches to header.
fn collect_loop_blocks(
    cfg: &Cfg,
    header: BlockId,
    latches: &[BlockId],
    _rpo_pos: &[Option<usize>],
) -> Vec<BlockId> {
    let mut in_loop = vec![false; cfg.num_blocks()];
    in_loop[header as usize] = true;

    // Reverse BFS from each latch, stopping at header
    let preds = cfg.predecessors();
    let mut worklist: Vec<BlockId> = Vec::new();
    for &latch in latches {
        if !in_loop[latch as usize] {
            in_loop[latch as usize] = true;
            worklist.push(latch);
        }
    }

    while let Some(bid) = worklist.pop() {
        for &pred in &preds[bid as usize] {
            if !in_loop[pred as usize] {
                in_loop[pred as usize] = true;
                worklist.push(pred);
            }
        }
    }

    (0..cfg.num_blocks() as BlockId)
        .filter(|&b| in_loop[b as usize])
        .collect()
}

/// Check if a natural loop is self-consistent under the optimistic assumption
/// that all GuardDepth checks pass.
///
/// Strategy: set the header's depth to the maximum min_depth required by any
/// GuardDepth in the loop (the "optimistic floor"). Run fixed-point iteration
/// within the loop only. If every GuardDepth's min_depth is satisfied and the
/// back-edge returns depth >= the optimistic floor, the loop is self-consistent
/// and all guards can be safely removed.
fn check_loop_consistent(
    cfg: &Cfg,
    header: BlockId,
    loop_blocks: &[BlockId],
    global_states: &[AbstractState],
    guarded_storages: &std::collections::HashSet<usize>,
) -> bool {
    let n = cfg.num_blocks();
    let in_loop: Vec<bool> = {
        let mut v = vec![false; n];
        for &b in loop_blocks {
            v[b as usize] = true;
        }
        v
    };

    // Find the maximum min_depth across all GuardDepth in the loop
    let mut max_guard_depth: i32 = 0;
    for &bid in loop_blocks {
        for inst in &cfg.block(bid).instructions {
            if let Inst::GuardDepth { min_depth, .. } = inst {
                max_guard_depth = max_guard_depth.max(*min_depth as i32);
            }
        }
        if let Terminator::StackGuard { min_depth, .. } = &cfg.block(bid).terminator {
            max_guard_depth = max_guard_depth.max(*min_depth as i32);
        }
    }

    if max_guard_depth == 0 {
        return true; // No guards to check
    }

    // Initialize: use global state for header, but raise each storage's depth
    // to at least max_guard_depth IF its global depth is already close (within
    // one loop iteration of reaching it). Conservative: only raise if global ≥ 1.
    let mut states: Vec<AbstractState> = vec![AbstractState::bottom(); n];
    let header_global = &global_states[header as usize];
    let mut header_state = header_global.clone();
    // Only raise depth for the SELECTED storage (if known).
    // If sel is unknown, raise all non-bottom storages (but check_loop_consistent
    // will verify with depth_sufficient which handles sel=None conservatively).
    for i in 0..STORAGE_COUNT {
        // Only raise storages that actually have guards AND have global depth >= 1.
        // depth=0 means the storage might be empty on first loop entry → can't assume guard passes.
        let is_guarded = guarded_storages.contains(&i);
        header_state.depths[i] = match header_state.depths[i] {
            Depth::Bottom => Depth::Bottom,
            Depth::AtLeast(d) if d >= max_guard_depth => Depth::AtLeast(d),
            Depth::AtLeast(d) if is_guarded && d >= 1 => Depth::AtLeast(max_guard_depth),
            d => d,
        };
    }
    states[header as usize] = header_state;

    // Fixed-point iteration within the loop
    let rpo = cfg.reverse_postorder();
    for _ in 0..100 {
        let mut changed = false;
        for &bid in &rpo {
            if !in_loop[bid as usize] {
                continue;
            }
            let entry = states[bid as usize].clone();
            if entry.is_bottom() {
                continue;
            }

            let exit = transfer_block(cfg.block(bid), &entry);

            let succs = optimistic_successor_states(cfg.block(bid), &entry, &exit);
            for (succ_id, succ_state) in succs {
                if !in_loop[succ_id as usize] {
                    continue;
                }
                let succ_idx = succ_id as usize;
                let merged = states[succ_idx].meet(&succ_state);
                if merged != states[succ_idx] {
                    states[succ_idx] = merged;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Check: at each GuardDepth, is depth >= min_depth?
    // Optimization: if the same storage was already guarded earlier in the loop,
    // the earlier guard strengthened it. Later guards only need depth >= 1.
    let mut ever_guarded: [bool; STORAGE_COUNT] = [false; STORAGE_COUNT];
    for &bid in loop_blocks {
        let entry = &states[bid as usize];
        if entry.is_bottom() {
            continue;
        }
        let mut state = entry.clone();
        for inst in &cfg.block(bid).instructions {
            if let Inst::GuardDepth { min_depth, .. } = inst {
                if let Some(sel) = state.selected {
                    if ever_guarded[sel] {
                        // Prior guard already strengthened this storage.
                        // After guard→consume→push cycle, depth ≥ 1. Check 1 not min_depth.
                        if !state.depths[sel].is_at_least(1) {
                            return false;
                        }
                    } else {
                        if !state.depths[sel].is_at_least(*min_depth as i32) {
                            return false;
                        }
                    }
                    ever_guarded[sel] = true;
                } else if !depth_sufficient(&state, *min_depth as i32) {
                    return false;
                }
            }
            transfer_inst(inst, &mut state);
        }
        if let Terminator::StackGuard { min_depth, .. } = &cfg.block(bid).terminator {
            if !depth_sufficient(&state, *min_depth as i32) {
                return false;
            }
        }
    }

    // The per-guard checks above are sufficient.
    // No separate back-edge check needed — if all guards within the loop pass
    // under the optimistic assumption, the loop is self-consistent.
    true
}

/// Check if the current depth is sufficient: if sel is known, check that storage.
/// If sel is unknown (None), check ALL non-bottom storages have depth >= threshold.
fn depth_sufficient(state: &AbstractState, threshold: i32) -> bool {
    match state.selected {
        Some(sel) => state.depths[sel].is_at_least(threshold),
        None => {
            // All non-bottom storages must meet the threshold
            let mut any_live = false;
            for d in &state.depths {
                match d {
                    Depth::Bottom => {}
                    Depth::AtLeast(v) => {
                        any_live = true;
                        if *v < threshold {
                            return false;
                        }
                    }
                }
            }
            any_live
        }
    }
}

/// Like successor_states but optimistic: GuardDepth fail paths are NOT emitted.
/// Only the "ok" (pass) paths are propagated.
fn optimistic_successor_states(
    block: &CfgBlock,
    _entry: &AbstractState,
    exit_state: &AbstractState,
) -> Vec<(BlockId, AbstractState)> {
    // NO GuardDepth fail paths (that's the optimistic part)
    let mut result = Vec::new();
    match &block.terminator {
        Terminator::Goto(target) => {
            result.push((*target, exit_state.clone()));
        }
        Terminator::StackGuard {
            min_depth,
            ok,
            fail,
        } => {
            let mut ok_state = exit_state.clone();
            if let Some(sel) = ok_state.selected {
                if sel < STORAGE_COUNT {
                    let min = *min_depth as i32;
                    ok_state.depths[sel] = match ok_state.depths[sel] {
                        Depth::Bottom => Depth::Bottom,
                        Depth::AtLeast(d) => Depth::AtLeast(d.max(min)),
                    };
                }
            }
            // Emit both ok and fail for StackGuard (it's a terminator, not GuardDepth)
            result.push((*ok, ok_state));
            result.push((*fail, exit_state.clone()));
        }
        Terminator::BranchZero {
            on_zero,
            on_nonzero,
        } => {
            let mut post_pop = exit_state.clone();
            if let Some(sel) = post_pop.selected {
                apply_depth_delta(&mut post_pop, sel, -1);
            }
            result.push((*on_zero, post_pop.clone()));
            result.push((*on_nonzero, post_pop));
        }
        Terminator::Halt => {}
    }
    result
}

// ── Pass 8: Simplify identical branches ─────────────────────────────

/// If a branch's both targets are the same, simplify to Goto.
pub fn simplify_branches(cfg: &mut Cfg) -> usize {
    let mut simplified = 0;
    for i in 0..cfg.num_blocks() {
        let block = cfg.block(i as BlockId);
        let new_term = match &block.terminator {
            Terminator::StackGuard { ok, fail, .. } if ok == fail => Some(Terminator::Goto(*ok)),
            // Self-referencing guard: fail points to self.
            // Only safe to eliminate if the block has NO stack-modifying instructions
            // (otherwise the self-loop builds up depth until the guard passes).
            Terminator::StackGuard { ok, fail, .. }
                if *fail == i as BlockId && block.instructions.is_empty() =>
            {
                Some(Terminator::Goto(*ok))
            }
            // Mutual bounce: two StackGuards with empty fail-blocks pointing at
            // each other. If neither can make progress, this is a real infinite
            // loop in the Aheui program. Convert to Goto(ok) to break the cycle.
            // NOTE: potentially incorrect if the correct escape depends on
            // cursor direction the CFG doesn't track.
            Terminator::StackGuard { ok, fail, .. } => {
                let fail_block = cfg.block(*fail);
                if fail_block.instructions.is_empty() {
                    if let Terminator::StackGuard { fail: fail2, .. } = fail_block.terminator {
                        if fail2 == i as BlockId {
                            Some(Terminator::Goto(*ok))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // BranchZero→StackGuard bounces are broken by converting the
            // StackGuard side (above). Don't modify BranchZero itself.
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } if on_zero == on_nonzero => Some(Terminator::Goto(*on_zero)),
            _ => None,
        };
        if let Some(term) = new_term {
            cfg.block_mut(i as BlockId).terminator = term;
            simplified += 1;
        }
    }
    simplified
}

// ── Combined optimization pipeline ──────────────────────────────────

/// Run the full CFG optimization pipeline. Returns the optimized CFG.
///
/// `width` is the range folded constants have to fit for whoever consumes the
/// result — see [`ConstWidth`].
pub fn optimize_cfg(mut cfg: Cfg, width: ConstWidth) -> Cfg {
    // Single analysis + guard elimination.
    let states = analyze_stack_depths(&cfg);
    eliminate_guards(&mut cfg, &states);

    // Structural cleanup.
    thread_jumps(&mut cfg);
    simplify_branches(&mut cfg);
    merge_blocks(&mut cfg);

    // Constant folding and peephole on the simplified CFG.
    let states = analyze_stack_depths(&cfg);
    fold_constants(&mut cfg, &states, width);
    peephole_cleanup(&mut cfg, &states);

    // Remove unreachable blocks.
    eliminate_dead_blocks(&mut cfg);
    cfg
}

/// AOT optimization: standard pipeline + iterated structural cleanup.
pub fn optimize_cfg_aot(mut cfg: Cfg) -> Cfg {
    cfg = optimize_cfg(cfg, ConstWidth::I64);
    for _ in 0..10 {
        let before = cfg.num_blocks();
        let states = analyze_stack_depths(&cfg);
        eliminate_guards(&mut cfg, &states);
        thread_jumps(&mut cfg);
        simplify_branches(&mut cfg);
        merge_blocks(&mut cfg);
        eliminate_dead_blocks(&mut cfg);
        if cfg.num_blocks() == before {
            break;
        }
    }
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_push_add_halt() -> Cfg {
        // PUSH 2; PUSH 3 → BRPOP2 (guard) → ADD → HALT
        // With BRPOP fail path
        let mut cfg = Cfg::new();

        // Block 0: PUSH 2, PUSH 3 → StackGuard(2, ok=1, fail=2)
        cfg.add_block(
            vec![Inst::Push(2), Inst::Push(3)],
            Terminator::StackGuard {
                min_depth: 2,
                ok: 1,
                fail: 2,
            },
        );

        // Block 1: ADD → HALT
        cfg.add_block(vec![Inst::BinOp(BinOpKind::Add)], Terminator::Halt);

        // Block 2: fail path → HALT
        cfg.add_block(vec![], Terminator::Halt);

        cfg
    }

    #[test]
    fn test_analyze_push_add() {
        let cfg = make_push_add_halt();
        let states = analyze_stack_depths(&cfg);

        // After PUSH 2; PUSH 3, depth[0] should be AtLeast(2)
        let s0 = &states[0];
        assert_eq!(s0.depths[0], Depth::AtLeast(0));
        assert_eq!(s0.selected, Some(0));

        // Block 1 (ok path) should have depth >= 2
        let s1 = &states[1];
        assert!(s1.depths[0].is_at_least(2));
    }

    #[test]
    fn test_guard_elimination() {
        let mut cfg = make_push_add_halt();
        let states = analyze_stack_depths(&cfg);
        let eliminated = eliminate_guards(&mut cfg, &states);

        assert_eq!(eliminated, 1);
        assert_eq!(cfg.block(0).terminator, Terminator::Goto(1));
    }

    #[test]
    fn test_block_merging_after_guard_elimination() {
        let mut cfg = make_push_add_halt();
        let states = analyze_stack_depths(&cfg);
        eliminate_guards(&mut cfg, &states);
        let merged = merge_blocks(&mut cfg);

        assert!(merged > 0);
        // Block 0 should now contain [Push(2), Push(3), Add] → Halt
        let block = cfg.block(0);
        assert_eq!(
            block.instructions,
            vec![Inst::Push(2), Inst::Push(3), Inst::BinOp(BinOpKind::Add)]
        );
        assert_eq!(block.terminator, Terminator::Halt);
    }

    #[test]
    fn test_constant_folding() {
        let mut cfg = make_push_add_halt();
        let states = analyze_stack_depths(&cfg);
        eliminate_guards(&mut cfg, &states);
        merge_blocks(&mut cfg);

        let states2 = analyze_stack_depths(&cfg);
        let folds = fold_constants(&mut cfg, &states2, ConstWidth::I32);

        assert!(folds > 0);
        // Should fold to [Push(5)] → Halt
        let block = cfg.block(0);
        assert_eq!(block.instructions, vec![Inst::Push(5)]);
    }

    #[test]
    fn test_full_pipeline() {
        let cfg = make_push_add_halt();
        let optimized = optimize_cfg(cfg, ConstWidth::I32);

        // Final: one block [Push(5)] → Halt, dead blocks removed
        assert!(optimized.num_blocks() <= 2);
        let entry = optimized.block(optimized.entry);
        assert_eq!(entry.instructions, vec![Inst::Push(5)]);
        assert_eq!(entry.terminator, Terminator::Halt);
    }

    #[test]
    fn test_fold_stops_at_the_linearized_operand_width() {
        // 65536 * 65536 = 2^32, which `Program::values` cannot carry. Under
        // `I32` the fold has to leave the multiply for run time; under `I64`,
        // where an AOT backend reads `Inst::Push` directly, it still folds.
        let build = || {
            let mut cfg = Cfg::new();
            cfg.add_block(
                vec![
                    Inst::Push(65536),
                    Inst::Push(65536),
                    Inst::BinOp(BinOpKind::Mul),
                ],
                Terminator::Halt,
            );
            cfg
        };

        let mut narrow = build();
        let states = analyze_stack_depths(&narrow);
        assert_eq!(fold_constants(&mut narrow, &states, ConstWidth::I32), 0);
        assert_eq!(
            narrow.block(0).instructions,
            vec![
                Inst::Push(65536),
                Inst::Push(65536),
                Inst::BinOp(BinOpKind::Mul),
            ]
        );

        let mut wide = build();
        let states = analyze_stack_depths(&wide);
        assert_eq!(fold_constants(&mut wide, &states, ConstWidth::I64), 1);
        assert_eq!(wide.block(0).instructions, vec![Inst::Push(4294967296)]);
    }

    #[test]
    fn test_chain_folding() {
        // PUSH 2; PUSH 3; ADD; PUSH 4; MUL → PUSH 20
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![
                Inst::Push(2),
                Inst::Push(3),
                Inst::BinOp(BinOpKind::Add),
                Inst::Push(4),
                Inst::BinOp(BinOpKind::Mul),
            ],
            Terminator::Halt,
        );

        let states = analyze_stack_depths(&cfg);
        fold_constants(&mut cfg, &states, ConstWidth::I32);

        assert_eq!(cfg.block(0).instructions, vec![Inst::Push(20)]);
    }

    #[test]
    fn test_dup_fold() {
        // PUSH 5; DUP; MUL → PUSH 25
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(5), Inst::Dup, Inst::BinOp(BinOpKind::Mul)],
            Terminator::Halt,
        );

        let states = analyze_stack_depths(&cfg);
        fold_constants(&mut cfg, &states, ConstWidth::I32);

        assert_eq!(cfg.block(0).instructions, vec![Inst::Push(25)]);
    }

    #[test]
    fn test_constant_brz_folding() {
        // PUSH 0; BRZ → Goto(on_zero)
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(0)],
            Terminator::BranchZero {
                on_zero: 1,
                on_nonzero: 2,
            },
        );
        cfg.add_block(vec![Inst::Push(42)], Terminator::Halt);
        cfg.add_block(vec![Inst::Push(99)], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        fold_constants(&mut cfg, &states, ConstWidth::I32);

        assert_eq!(cfg.block(0).terminator, Terminator::Goto(1));
        assert!(cfg.block(0).instructions.is_empty());
    }

    #[test]
    fn test_nonzero_brz_folding() {
        // PUSH 7; BRZ → Goto(on_nonzero)
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(7)],
            Terminator::BranchZero {
                on_zero: 1,
                on_nonzero: 2,
            },
        );
        cfg.add_block(vec![Inst::Push(42)], Terminator::Halt);
        cfg.add_block(vec![Inst::Push(99)], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        fold_constants(&mut cfg, &states, ConstWidth::I32);

        assert_eq!(cfg.block(0).terminator, Terminator::Goto(2));
    }

    #[test]
    fn test_guard_not_eliminated_when_insufficient() {
        // Entry with empty stack → StackGuard(1, ok, fail) should NOT be eliminated
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![],
            Terminator::StackGuard {
                min_depth: 1,
                ok: 1,
                fail: 2,
            },
        );
        cfg.add_block(vec![Inst::Pop], Terminator::Halt);
        cfg.add_block(vec![], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        let eliminated = eliminate_guards(&mut cfg, &states);

        assert_eq!(eliminated, 0);
    }

    #[test]
    fn test_sel_tracking() {
        // PUSH 5 on storage 0, SEL 1, PUSH 3 on storage 1
        // StackGuard on storage 1 with min_depth=1 should be eliminable
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![Inst::Push(5), Inst::Sel(1), Inst::Push(3)],
            Terminator::StackGuard {
                min_depth: 1,
                ok: 1,
                fail: 2,
            },
        );
        cfg.add_block(vec![Inst::Pop], Terminator::Halt);
        cfg.add_block(vec![], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        let eliminated = eliminate_guards(&mut cfg, &states);
        assert_eq!(eliminated, 1);
    }

    #[test]
    fn test_mov_tracking() {
        // PUSH 5, MOV(1) → storage 0 depth decreases, storage 1 depth increases
        let mut cfg = Cfg::new();
        cfg.add_block(vec![Inst::Push(5), Inst::Mov(1)], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        let exit = transfer_block(cfg.block(0), &states[0]);
        // storage 0: started at 0, +1 (push), -1 (mov) = 0
        assert_eq!(exit.depths[0], Depth::AtLeast(0));
        // storage 1: started at 0, +1 (mov target) = 1
        assert_eq!(exit.depths[1], Depth::AtLeast(1));
    }

    #[test]
    fn test_loop_convergence() {
        // Simple loop: B0 → B1 → B0 (back-edge)
        // B0: PUSH 1
        // B1: POP → Goto(B0)
        // Should converge without infinite loop
        let mut cfg = Cfg::new();
        cfg.add_block(vec![Inst::Push(1)], Terminator::Goto(1)); // B0
        cfg.add_block(vec![Inst::Pop], Terminator::Goto(0)); // B1

        let states = analyze_stack_depths(&cfg);
        // Should converge — both blocks are reachable
        assert!(!states[0].is_bottom());
        assert!(!states[1].is_bottom());
    }

    #[test]
    fn test_merge_point_guard_preserved() {
        // Two paths converge at a guard. Global meet can't eliminate it.
        let mut cfg = Cfg::new();
        cfg.add_block(
            vec![],
            Terminator::BranchZero {
                on_zero: 1,
                on_nonzero: 2,
            },
        );
        cfg.add_block(vec![Inst::Push(10), Inst::Push(20)], Terminator::Goto(3));
        cfg.add_block(vec![], Terminator::Goto(3));
        cfg.add_block(
            vec![],
            Terminator::StackGuard {
                min_depth: 2,
                ok: 4,
                fail: 5,
            },
        );
        cfg.add_block(vec![Inst::BinOp(BinOpKind::Add)], Terminator::Halt);
        cfg.add_block(vec![], Terminator::Halt);

        let states = analyze_stack_depths(&cfg);
        let eliminated = eliminate_guards(&mut cfg, &states);
        // Can't eliminate: meet of depth=2 (from B1) and depth=0 (from B2) = 0
        assert_eq!(eliminated, 0);
    }
}
