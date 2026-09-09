//! Build-time artifacts supplied by the JIT consumer, never by the interpreter.
//!
//! Like RPython's codewriter setup, these bindings are used when installing
//! JitCodes, not on the opcode dispatch hot path. The interpreter owns its
//! state and source; the consumer owns its generated tables.

use std::sync::{Arc, OnceLock};

pub struct Artifacts {
    pub jitcode: fn(&str) -> Option<Arc<majit_metainterp::JitCode>>,
    pub prebuild_liveness: fn(&mut majit_metainterp::Assembler),
    pub word_abi_fnaddrs: fn() -> Vec<i64>,
}

static ARTIFACTS: OnceLock<&'static Artifacts> = OnceLock::new();

pub fn install(artifacts: &'static Artifacts) {
    let installed = ARTIFACTS.get_or_init(|| artifacts);
    assert!(
        std::ptr::eq(*installed, artifacts),
        "JIT artifacts already installed by another consumer"
    );
}

pub(crate) fn artifacts() -> &'static Artifacts {
    ARTIFACTS
        .get()
        .copied()
        .expect("install Aheui JIT artifacts before enabling tracing")
}
