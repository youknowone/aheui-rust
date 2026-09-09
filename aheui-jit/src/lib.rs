//! Generated JIT artifacts and startup wiring for the canonical interpreter.
pub use aheuinterpreter::*;
pub mod jit;

include!(concat!(env!("OUT_DIR"), "/jit_trace_gen.rs"));

static ARTIFACTS: aheuinterpreter::jit::Artifacts = aheuinterpreter::jit::Artifacts {
    jitcode: jit::jitcode_runtime::pipeline_jitcode_by_name,
    prebuild_liveness: jit::jitcode_runtime::prebuild_pipeline_liveness,
    word_abi_fnaddrs: jit::jitcode_runtime::word_abi_fnaddrs,
};

pub fn init_gc_subsystem() {
    aheuinterpreter::jit::install(&ARTIFACTS);
    aheuinterpreter::init_gc_subsystem();
}

pub fn mainloop(program: &aheui::Program, threshold: u32) -> aheui_runtime::Val {
    init_gc_subsystem();
    aheuinterpreter::mainloop(program, Some(threshold))
}
