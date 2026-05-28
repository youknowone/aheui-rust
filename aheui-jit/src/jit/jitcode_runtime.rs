//! Runtime access to the build-time `pipeline.jitcodes` table.
//!
//! `MetaInterpStaticData.jitcodes` (warmspot.py:281-282) — the list of
//! `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py:89). The build script (`build.rs`) runs the
//! `majit_translate` pipeline over the Aheui interpreter sources and writes
//! `pipeline.jitcodes` to `$OUT_DIR/opcode_jitcodes.bin`. This module
//! deserializes that blob into `Vec<Arc<JitCode>>` shells — the `mainloop`
//! portal plus the per-storage-method sub-jitcodes that the macro-driven
//! dispatch will `BC_INLINE_CALL` into.
//!
//! Single-store model: the only persisted collection is `pipeline.jitcodes`,
//! in allocation order, matching `codewriter.py:80`'s invariant
//! `all_jitcodes[jitcode.index] is jitcode`.

use std::sync::Arc;

use majit_translate::jitcode::JitCode;

/// Deserialize the build-time `pipeline.jitcodes` blob.
///
/// `bincode::deserialize` produces fresh `Arc::new(...)` shells (refcount 1).
/// fnaddr / descr-pool rewiring is layered on by later callers; this entry
/// point only materializes the canonical bodies.
pub fn load_pipeline_jitcodes() -> Vec<Arc<JitCode>> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_jitcodes.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "aheui-jit: failed to deserialize opcode_jitcodes.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_pipeline_jitcodes_with_mainloop_portal() {
        let jitcodes = load_pipeline_jitcodes();
        assert!(
            !jitcodes.is_empty(),
            "pipeline must produce at least the mainloop portal jitcode",
        );
        assert!(
            jitcodes.iter().any(|jc| jc.name == "mainloop"),
            "pipeline jitcodes must include the `mainloop` portal; got {:?}",
            jitcodes.iter().map(|jc| jc.name.as_str()).collect::<Vec<_>>(),
        );
    }
}
