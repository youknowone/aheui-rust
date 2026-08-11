//! The CFG optimization pipeline the AOT backends run.
//!
//! One copy, because the backends used to carry one each and they drifted: the C
//! and WebAssembly generators ran every pass unconditionally, so `-O` never
//! reached them and `-O0` emitted what `-O3` emitted.

use ahsembler::OptimizationLevel;
use ahsembler::cfg::Cfg;

/// Optimized CFG for an AOT backend at `level`.
pub fn optimize(source: &str, level: OptimizationLevel) -> Cfg {
    let mut cfg = ahsembler::compile_to_cfg_aot(source);
    optimize_cfg(&mut cfg, level);
    cfg
}

/// The pass sequence itself, for callers that already hold a `Cfg`.
pub fn optimize_cfg(cfg: &mut Cfg, level: OptimizationLevel) {
    if matches!(level, OptimizationLevel::O2 | OptimizationLevel::O3) {
        // Guard merging: fold StackGuard ok-paths into GuardDepth instructions
        for _ in 0..10 {
            let before = cfg.num_blocks();
            let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);
            ahsembler::cfg_optimize::eliminate_guards(cfg, &states);
            ahsembler::cfg_optimize::eliminate_guard_depth(cfg, &states);
            ahsembler::cfg_optimize::thread_jumps(cfg);
            ahsembler::cfg_optimize::merge_guard_ok(cfg);
            ahsembler::cfg_optimize::merge_blocks(cfg);
            ahsembler::cfg_optimize::eliminate_dead_blocks(cfg);
            if cfg.num_blocks() == before {
                break;
            }
        }
    }
    if matches!(level, OptimizationLevel::O3) {
        // Loop-aware guard elimination: remove guards in self-consistent loops
        let removed = ahsembler::cfg_optimize::eliminate_loop_guards(cfg);
        if removed > 0 {
            for _ in 0..10 {
                let before = cfg.num_blocks();
                let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);
                ahsembler::cfg_optimize::eliminate_guard_depth(cfg, &states);
                ahsembler::cfg_optimize::thread_jumps(cfg);
                ahsembler::cfg_optimize::merge_blocks(cfg);
                ahsembler::cfg_optimize::eliminate_dead_blocks(cfg);
                if cfg.num_blocks() == before {
                    break;
                }
            }
        }
        // Chain guard elimination: linear sequences with back-edge-only fail targets
        let chain_removed = ahsembler::cfg_optimize::eliminate_chain_guards(cfg);
        if chain_removed > 0 {
            for _ in 0..10 {
                let before = cfg.num_blocks();
                ahsembler::cfg_optimize::thread_jumps(cfg);
                ahsembler::cfg_optimize::merge_blocks(cfg);
                ahsembler::cfg_optimize::eliminate_dead_blocks(cfg);
                if cfg.num_blocks() == before {
                    break;
                }
            }
        }
        // Fold and peephole once, after both guard passes. Gating this on
        // `removed` skipped it whenever the loop-guard pass happened to find
        // nothing, even though the passes above rewrite blocks either way, and
        // nothing folded what `eliminate_chain_guards` reshaped.
        let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);
        ahsembler::cfg_optimize::fold_constants(
            cfg,
            &states,
            ahsembler::cfg_optimize::ConstWidth::I64,
        );
        ahsembler::cfg_optimize::peephole_cleanup(cfg, &states);
    }
}
