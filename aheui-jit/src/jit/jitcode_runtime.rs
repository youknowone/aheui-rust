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

use majit_metainterp::JitCode as RuntimeJitCode;
use majit_metainterp::jitcode::RuntimeBhDescr;
use majit_translate::jitcode::{BhDescr, JitCode};

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

/// Deserialize the build-time shared descr pool (`pipeline.descrs`).
///
/// `Assembler.descrs` (assembler.py:23) handed to
/// `BlackholeInterpBuilder.setup_descrs` (blackhole.py:59). Each 'd'/'j'
/// argcode operand in a `JitCode.code` byte stream is a 2-byte index into
/// this pool.
pub fn load_pipeline_descrs() -> Vec<BhDescr> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_descrs.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "aheui-jit: failed to deserialize opcode_descrs.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Build the runtime descr pool that a loaded portal/sub-jitcode resolves
/// its 'd'/'j' argcodes against.
///
/// `BhDescr::JitCode { jitcode_index, .. }` ('j' argcode, BC_INLINE_CALL)
/// becomes `RuntimeBhDescr::JitCode` pointing at the loaded callee shell so
/// `resolve_jitcode` (jitdriver.rs `parent.exec.descrs[idx].as_jitcode()`)
/// can chain into it. Every other variant (`Field`/`Array`/`Size`/`Call`)
/// is an ordinary 'd' descr — `CanonicalBhDescr` is the same type as the
/// build-time `BhDescr` (re-exported at `blackhole.rs`), so it carries
/// through unchanged. `Call` fnaddrs live in `constants_i` and are patched
/// separately, not in this pool.
pub fn build_runtime_descr_pool(jitcodes: &[Arc<JitCode>]) -> Vec<RuntimeBhDescr> {
    load_pipeline_descrs()
        .into_iter()
        .map(|bh| match bh {
            BhDescr::JitCode { jitcode_index, .. } => {
                let core = (*jitcodes[jitcode_index]).clone();
                RuntimeBhDescr::JitCode(Arc::new(RuntimeJitCode::from_canonical(core)))
            }
            other => RuntimeBhDescr::Descr(other),
        })
        .collect()
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

    #[test]
    fn descr_pool_resolves_jitcode_entries_to_loaded_callees() {
        let jitcodes = load_pipeline_jitcodes();
        let pool = build_runtime_descr_pool(&jitcodes);
        assert_eq!(
            pool.len(),
            load_pipeline_descrs().len(),
            "pool entry count must match pipeline.descrs",
        );
        let jitcode_slots = pool
            .iter()
            .filter_map(RuntimeBhDescr::as_jitcode)
            .count();
        assert!(
            jitcode_slots > 0,
            "the shared descr pool must carry the inter-jitcode BC_INLINE_CALL links",
        );
        // Every 'j' slot must resolve to a real callee body (the inline-call
        // chain mainloop->add->val_add->... is what the macro dispatch will
        // trace into).
        for slot in &pool {
            if let Some(callee) = slot.as_jitcode() {
                assert!(
                    !callee.name().is_empty(),
                    "resolved sub-jitcode must carry a name",
                );
            }
        }
    }
}
