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
/// This function materializes only the canonical bodies. Runtime registration
/// adds function addresses and the shared descriptor pool.
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
            other => RuntimeBhDescr::Descr(Box::new(other)),
        })
        .collect()
}

/// Materialize the pipeline jitcodes as runtime shells, each carrying the
/// shared descr pool as its `exec.descrs`.
///
/// The pool entries' 'd'/'j' operands are GLOBAL indices into
/// `pipeline.descrs`, and `resolve_jitcode` / the tracer read each
/// jitcode's own `exec.descrs[idx]`; so every loaded jitcode shares the
/// same pool and its internal `BC_INLINE_CALL` chain (mainloop→add→
/// val_add→alloc_node) resolves consistently. This is the runtime
/// registration the macro dispatch's `add_sub_jitcode` + `BC_INLINE_CALL`
/// emission targets via [`pipeline_jitcode_by_name`].
pub fn registered_pipeline_jitcodes() -> Vec<Arc<RuntimeJitCode>> {
    let canonical = load_pipeline_jitcodes();
    let pool = build_runtime_descr_pool(&canonical);
    canonical
        .iter()
        .map(|core| {
            let mut jc = RuntimeJitCode::from_canonical((**core).clone());
            jc.exec.descrs = pool.clone();
            Arc::new(jc)
        })
        .collect()
}

/// Look up a registered pipeline jitcode (shared pool installed) by name.
/// The macro dispatch resolves storage methods (`add`/`pop`/...) through
/// this to feed `JitCodeBuilder::add_sub_jitcode` + `inline_call_*`.
pub fn pipeline_jitcode_by_name(name: &str) -> Option<Arc<RuntimeJitCode>> {
    registered_pipeline_jitcodes()
        .into_iter()
        .find(|jc| jc.name() == name)
}

/// Map a mainloop storage call's last path segment (`stack_pop`,
/// `queue_add`, ...) to its monomorphic pipeline jitcode name and resolve
/// it. The `#[jit_interp]` `inline_pipeline_*` lowering passes the call's
/// last segment verbatim; the pipeline emitted a single jitcode per
/// storage operation (`make_jitcodes()`, codewriter.py:89) shared by the
/// stack and queue families, so both `stack_*` and `queue_*` collapse to
/// the same callee. `push`/`dup` have no standalone jitcode (not reached as
/// a separate `make_jitcodes()` entry) and intentionally return `None`.
pub fn pipeline_jitcode_for_call(call_name: &str) -> Option<Arc<RuntimeJitCode>> {
    let mapped = match call_name {
        "stack_add" | "queue_add" => "add",
        "stack_sub" | "queue_sub" => "sub",
        "stack_mul" | "queue_mul" => "mul",
        "stack_div" | "queue_div" => "div",
        "stack_mod" | "queue_mod" => "modulo",
        "stack_pop" | "queue_pop" => "pop",
        "stack_swap" | "queue_swap" => "swap",
        "stack_cmp" | "queue_cmp" => "cmp",
        other => other,
    };
    pipeline_jitcode_by_name(mapped)
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

    #[test]
    fn call_name_map_resolves_stack_and_queue_families_to_one_callee() {
        // Both families collapse to the same monomorphic storage jitcode.
        for (stack_call, queue_call, jitcode) in [
            ("stack_add", "queue_add", "add"),
            ("stack_sub", "queue_sub", "sub"),
            ("stack_mul", "queue_mul", "mul"),
            ("stack_div", "queue_div", "div"),
            ("stack_mod", "queue_mod", "modulo"),
            ("stack_pop", "queue_pop", "pop"),
            ("stack_swap", "queue_swap", "swap"),
            ("stack_cmp", "queue_cmp", "cmp"),
        ] {
            let from_stack = pipeline_jitcode_for_call(stack_call)
                .unwrap_or_else(|| panic!("{stack_call} must resolve a pipeline jitcode"));
            let from_queue = pipeline_jitcode_for_call(queue_call)
                .unwrap_or_else(|| panic!("{queue_call} must resolve a pipeline jitcode"));
            assert_eq!(from_stack.name(), jitcode);
            assert_eq!(from_queue.name(), jitcode);
        }
    }

    #[test]
    fn call_name_map_returns_none_for_ops_without_a_jitcode() {
        // `push`/`dup` were not emitted as standalone `make_jitcodes()`
        // entries, so the inline-pipeline path must not claim to find them.
        assert!(pipeline_jitcode_for_call("stack_push").is_none());
        assert!(pipeline_jitcode_for_call("stack_dup").is_none());
    }

    #[test]
    fn registered_jitcode_carries_shared_pool_for_inline_call_chain() {
        let pool_len = load_pipeline_descrs().len();
        let add = pipeline_jitcode_by_name("add")
            .expect("pipeline must produce a storage `add` jitcode the dispatch inlines");
        assert_eq!(
            add.exec.descrs.len(),
            pool_len,
            "each registered jitcode must carry the full shared descr pool so its \
             internal BC_INLINE_CALL operands resolve",
        );
        // The `add` body inline-calls deeper storage helpers; those 'j' slots
        // must resolve to named callees through this jitcode's own pool.
        assert!(
            add.exec
                .descrs
                .iter()
                .filter_map(RuntimeBhDescr::as_jitcode)
                .any(|callee| !callee.name().is_empty()),
            "the shared pool installed on `add` must carry resolvable sub-jitcode links",
        );
    }
}
