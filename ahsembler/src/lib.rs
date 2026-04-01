pub mod cfg;
pub mod cfg_build;
pub mod cfg_linearize;
pub mod cfg_optimize;
pub mod compiler;
pub mod consts;
pub mod stackify;

use compiler::{Compiler, Program};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptimizationLevel {
    /// No optimization. Raw bytecode from compiler serialization.
    O0,
    /// Bytecode-level: jump threading, dead code elimination, constant folding.
    /// Fast (~0.5ms for aheui.aheui). No CFG construction.
    O1,
    /// O1 + single-pass CFG guard elimination.
    /// Builds CFG, removes provably-safe BRPOP guards, re-linearizes.
    /// Moderate cost (+0.3-1ms). Best default for interpreter.
    O2,
    /// O2 + iterated CFG optimization (multi-pass guard elimination,
    /// constant folding, peephole, block merging) + post-linearize polish.
    /// Higher cost (+3-30ms) but removes more guards. Best for AOT/long-running.
    O3,
}

impl OptimizationLevel {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "0" => Ok(Self::O0),
            "1" => Ok(Self::O1),
            "2" => Ok(Self::O2),
            "3" => Ok(Self::O3),
            _ => Err(format!("invalid optimization level: {s}")),
        }
    }
}

pub fn compile(source: &str, optimization: OptimizationLevel) -> Program {
    let mut compiler = Compiler::new();
    compiler.compile(source);
    match optimization {
        OptimizationLevel::O0 => return compiler.into_program(),
        OptimizationLevel::O1 | OptimizationLevel::O2 | OptimizationLevel::O3 => {
            compiler.optimize1();
        }
    }
    match optimization {
        OptimizationLevel::O1 => compiler.into_program(),
        OptimizationLevel::O2 => {
            // Single-pass CFG: guard elimination + merge + linearize
            let mut cfg = cfg_build::build_cfg(&compiler);
            let states = cfg_optimize::analyze_stack_depths(&cfg);
            cfg_optimize::eliminate_guards(&mut cfg, &states);
            cfg_optimize::thread_jumps(&mut cfg);
            cfg_optimize::simplify_branches(&mut cfg);
            cfg_optimize::merge_blocks(&mut cfg);
            cfg_optimize::eliminate_dead_blocks(&mut cfg);
            cfg_linearize::linearize(&cfg)
        }
        OptimizationLevel::O3 => {
            // Iterated CFG: guard elimination + fold + peephole → re-linearize + polish
            let cfg = cfg_build::build_cfg(&compiler);
            let optimized = cfg_optimize::optimize_cfg(cfg);
            let program = cfg_linearize::linearize(&optimized);
            let mut post = Compiler::from_program(&program);
            post.remove_dead_labels();
            post.optimize1();
            post.into_program()
        }
        OptimizationLevel::O0 => unreachable!(),
    }
}

/// Compile using the hybrid optimizer pipeline:
/// 1. Full legacy O2 optimization (per-path guard elimination + constant folding)
/// 2. CFG-based optimizations (additional guard elimination, block merging,
///    jump threading, constant folding, dead block removal, layout)
pub fn compile_cfg(source: &str) -> Program {
    let mut compiler = Compiler::new();
    compiler.compile(source);

    // Phase 1: Full O2 legacy passes — fast per-path guard elimination.
    compiler.optimize2();

    // Phase 2: Single CFG pass — lattice analysis, path splitting, merging.
    let cfg = cfg_build::build_cfg(&compiler);
    let optimized = cfg_optimize::optimize_cfg(cfg);
    let program = cfg_linearize::linearize(&optimized);

    // Phase 3: Final O2 polish on the restructured layout.
    // Remove dead labels first — CFG linearization assigns fresh labels,
    // and some become dead after jump threading. Removing them lets O2's
    // constant folding see across former block boundaries.
    let mut post = Compiler::from_program(&program);
    post.remove_dead_labels();
    post.optimize2();
    post.into_program()
}

/// Compile source into an optimized CFG (for AOT compilation backends).
pub fn compile_to_cfg(source: &str) -> cfg::Cfg {
    let mut compiler = Compiler::new();
    compiler.compile(source);
    compiler.optimize2();
    let cfg = cfg_build::build_cfg(&compiler);
    cfg_optimize::optimize_cfg(cfg)
}

/// Aggressively optimized CFG for AOT backends.
pub fn compile_to_cfg_aot(source: &str) -> cfg::Cfg {
    let mut compiler = Compiler::new();
    compiler.compile(source);
    compiler.optimize2();
    let cfg = cfg_build::build_cfg(&compiler);
    cfg_optimize::optimize_cfg_aot(cfg)
}

pub fn assemble(
    source: &str,
    optimization: OptimizationLevel,
    writer: &mut impl std::io::Write,
) -> std::io::Result<()> {
    let program = compile(source, optimization);
    compiler::dump_asm(&program, writer)
}
