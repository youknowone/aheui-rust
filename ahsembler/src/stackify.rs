//! Stackifier: convert CFG to structured control flow scopes.
//!
//! Computes loop and forward-skip scopes from a CFG in RPO order.
//! Output is a Vec<Scope> that any backend can use to emit structured
//! code (loop/break/continue in Rust, labeled blocks in Wasm, etc.)

use crate::cfg::*;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Back-edge target: `loop { ... continue; ... break; }`
    Loop,
    /// Forward-skip target: `'label: { ... break 'label; }`
    Forward,
}

/// A scope [open, close) over RPO indices with an associated CFG label.
#[derive(Debug, Clone)]
pub struct Scope {
    pub kind: ScopeKind,
    pub label: BlockId,
    /// RPO index where scope opens (may be extended before header for nesting).
    pub open: usize,
    /// RPO index where scope closes (exclusive — scope covers [open, close)).
    pub close: usize,
}

/// Compute structured scopes from a CFG and its RPO ordering.
///
/// Returns scopes sorted by (open ascending, close descending) so that
/// wider scopes come first when they start at the same position.
/// All scopes are properly nested (no partial overlaps).
pub fn compute_scopes(cfg: &Cfg, rpo: &[BlockId]) -> Vec<Scope> {
    let n = rpo.len();
    let mut rpo_pos: HashMap<BlockId, usize> = HashMap::new();
    for (i, &bid) in rpo.iter().enumerate() {
        rpo_pos.insert(bid, i);
    }

    let mut scopes = Vec::new();

    // ── Loop scopes ─────────────────────────────────────────────────
    // A back edge (source → target where target_rpo <= source_rpo)
    // identifies `target` as a loop header. The loop extent is
    // [header_rpo, max_back_edge_source_rpo + 1).

    let mut loop_extent: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &bid) in rpo.iter().enumerate() {
        for &succ in &cfg.block(bid).all_successors() {
            if let Some(&succ_pos) = rpo_pos.get(&succ)
                && succ_pos <= idx
            {
                let e = loop_extent.entry(succ).or_insert(idx);
                *e = (*e).max(idx);
            }
        }
    }

    // Extend loop extents to ensure proper nesting.
    let mut changed = true;
    while changed {
        changed = false;
        let headers: Vec<BlockId> = loop_extent.keys().copied().collect();
        for &h1 in &headers {
            for &h2 in &headers {
                if h1 == h2 {
                    continue;
                }
                let h1_pos = rpo_pos[&h1];
                let h2_pos = rpo_pos[&h2];
                if h1_pos < h2_pos
                    && loop_extent[&h1] >= h2_pos
                    && loop_extent[&h1] < loop_extent[&h2]
                {
                    *loop_extent.get_mut(&h1).unwrap() = loop_extent[&h2];
                    changed = true;
                }
            }
        }
    }

    for (&header, &last_src) in &loop_extent {
        let hdr = rpo_pos[&header];
        scopes.push(Scope {
            kind: ScopeKind::Loop,
            label: header,
            open: hdr,
            close: last_src + 1,
        });
    }

    // ── Forward block scopes ────────────────────────────────────────
    // A forward skip is a non-fallthrough, non-back-edge jump to a
    // later RPO block. We create a labeled block scope [source, target)
    // so that `break 'label` can skip to the target.

    let mut fwd_sources: HashMap<BlockId, usize> = HashMap::new();
    for (idx, &bid) in rpo.iter().enumerate() {
        let next_rpo = if idx + 1 < n {
            Some(rpo[idx + 1])
        } else {
            None
        };
        for &succ in &cfg.block(bid).all_successors() {
            if let Some(&succ_pos) = rpo_pos.get(&succ)
                && succ_pos > idx
                && Some(succ) != next_rpo
                && succ_pos > idx + 1
            {
                let e = fwd_sources.entry(succ).or_insert(idx);
                *e = (*e).min(idx);
            }
        }
    }

    for (&target, &earliest_src) in &fwd_sources {
        let tp = rpo_pos[&target];
        scopes.push(Scope {
            kind: ScopeKind::Forward,
            label: target,
            open: earliest_src,
            close: tp,
        });
    }

    // ── Ensure proper nesting ───────────────────────────────────────
    // If two scopes partially overlap, extend one to contain the other.
    // NEVER extend a Forward scope's close (target is fixed).
    // Loop open MAY be extended; codegen uses _goto dispatch for
    // forward jumps that cross scope boundaries.
    let mut changed2 = true;
    while changed2 {
        changed2 = false;
        for i in 0..scopes.len() {
            for j in 0..scopes.len() {
                if i == j {
                    continue;
                }
                let (a_open, a_close) = (scopes[i].open, scopes[i].close);
                let (b_open, b_close) = (scopes[j].open, scopes[j].close);
                if a_open <= b_open && a_close > b_open && a_close < b_close {
                    if scopes[i].kind == ScopeKind::Forward {
                        if scopes[j].open > a_open {
                            scopes[j].open = a_open;
                            changed2 = true;
                        }
                    } else {
                        scopes[i].close = b_close;
                        changed2 = true;
                    }
                }
            }
        }
    }

    scopes.sort_by(|a, b| a.open.cmp(&b.open).then(b.close.cmp(&a.close)));

    scopes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_loop() {
        // B0 → B1 → B0 (back edge)
        let mut cfg = Cfg::new();
        cfg.add_block(vec![Inst::Push(1)], Terminator::Goto(1));
        cfg.add_block(vec![Inst::Pop], Terminator::Goto(0));

        let rpo = cfg.reverse_postorder();
        let scopes = compute_scopes(&cfg, &rpo);

        assert_eq!(scopes.len(), 1);
        assert_eq!(scopes[0].kind, ScopeKind::Loop);
        assert_eq!(scopes[0].open, 0);
    }
}
