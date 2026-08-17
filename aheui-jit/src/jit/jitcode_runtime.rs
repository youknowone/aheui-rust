//! Runtime access to this crate's build-time jitcode table.
//!
//! `MetaInterpStaticData.jitcodes` (warmspot.py:281-282) — the list of
//! `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py:89). The build script (`build.rs`) runs the
//! `majit_translate` pipeline over the interpreter sources and writes
//! `pipeline.jitcodes`, the shared `pipeline.descrs`
//! (`Assembler.descrs`, assembler.py:23), symbolic function paths, and the
//! shared liveness table into `$OUT_DIR`.
//!
//! Only the embedding half lives here: reading those artifacts out of the
//! binary, binding process-local host addresses, and handing the result to
//! `majit_metainterp::EmbeddedJitCodeTable`, which owns the join and identity
//! rules. The artifacts are this crate's own `$OUT_DIR` output, which nothing
//! below it in the dependency graph can see, so the two halves cannot swap
//! places.

use std::sync::{Arc, OnceLock};

use majit_metainterp::EmbeddedJitCodeTable;
use majit_metainterp::JitCode as RuntimeJitCode;
use majit_translate::jitcode::{BhDescr, JitCode};

use aheui_runtime::storage::linkedlist::{ListBase, Node};

/// Deserialize the build-time `pipeline.jitcodes` blob.
fn load_pipeline_jitcodes() -> Vec<Arc<JitCode>> {
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
fn load_pipeline_descrs() -> Vec<BhDescr> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_descrs.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "aheui-jit: failed to deserialize opcode_descrs.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

fn load_symbolic_fnaddr_paths() -> Vec<(i64, String)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_symbolic_fnaddrs.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "aheui-jit: failed to deserialize opcode_symbolic_fnaddrs.bin ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

pub fn prebuild_pipeline_liveness(assembler: &mut majit_metainterp::Assembler) {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_liveness.bin"));
    assembler.prepend_embedded_liveness(BYTES);
}

// The graph pipeline lowers the shared LinkedList accessors as host calls, so
// these shims expose the base every storage embeds through the call-stub C ABI.
// The pointer they receive is a `pools` element, which already names that base.
extern "C" fn linked_list_head(storage: usize) -> i64 {
    unsafe { (*(storage as *const ListBase)).head as i64 }
}

extern "C" fn linked_list_set_head(storage: usize, head: i64) {
    unsafe { (*(storage as *mut ListBase)).head = head as *mut Node };
}

extern "C" fn linked_list_size(storage: usize) -> i64 {
    unsafe { (*(storage as *const ListBase)).size as i64 }
}

extern "C" fn linked_list_set_size(storage: usize, size: i64) {
    unsafe { (*(storage as *mut ListBase)).size = size as u32 };
}

extern "C" fn linked_list_free_node(node: usize) {
    aheui_runtime::storage::free_node(node as *mut Node);
}

fn runtime_fnaddr_bindings() -> [(&'static str, i64); 5] {
    [
        (
            "LinkedList::head",
            linked_list_head as *const () as usize as i64,
        ),
        (
            "LinkedList::set_head",
            linked_list_set_head as *const () as usize as i64,
        ),
        (
            "LinkedList::size",
            linked_list_size as *const () as usize as i64,
        ),
        (
            "LinkedList::set_size",
            linked_list_set_size as *const () as usize as i64,
        ),
        (
            "aheui_runtime::storage::free_node",
            linked_list_free_node as *const () as usize as i64,
        ),
    ]
}

/// The materialized table, built once.
///
/// Materializing per lookup would hand out a different `Arc` each call for the
/// same jitcode, which the identity the table exists to keep
/// (`codewriter.py:80 all_jitcodes[jitcode.index] is jitcode`) does not
/// survive: two callers naming one callee would inline-call into two objects.
/// It also re-decodes both blobs every time.
///
/// Installing the pool globally is part of the same one-time step — the shells
/// carry no pool of their own, so until it is installed their `d`/`j` operands
/// resolve to nothing.
fn pipeline_table() -> &'static EmbeddedJitCodeTable {
    static TABLE: OnceLock<&'static EmbeddedJitCodeTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let bindings = runtime_fnaddr_bindings();
        let table = EmbeddedJitCodeTable::materialize_with_symbolic_fnaddrs(
            &load_pipeline_jitcodes(),
            load_pipeline_descrs(),
            &load_symbolic_fnaddr_paths(),
            &bindings,
        );
        table.install_as_global_pool();
        table
    })
}

/// Look up a build-time jitcode by name.
///
/// This is what `__majit_pipeline_jitcode` answers: the `inline_pipeline_*`
/// call policies name their callee by the call's last path segment, which is
/// the only handle that side has. Everything internal to the table addresses
/// its callees by index instead.
pub fn pipeline_jitcode_by_name(name: &str) -> Option<Arc<RuntimeJitCode>> {
    pipeline_table().by_name(name).cloned()
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
            jitcodes
                .iter()
                .map(|jc| jc.name.as_str())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn descr_pool_resolves_jitcode_entries_to_loaded_callees() {
        let table = pipeline_table();
        assert_eq!(
            table.descrs().len(),
            load_pipeline_descrs().len(),
            "pool entry count must match pipeline.descrs",
        );
        let jitcode_slots = table
            .descrs()
            .iter()
            .filter_map(majit_metainterp::RuntimeBhDescr::as_jitcode)
            .count();
        assert!(
            jitcode_slots > 0,
            "the shared descr pool must carry the inter-jitcode BC_INLINE_CALL links",
        );
        for slot in table.descrs() {
            if let Some(callee) = slot.as_jitcode() {
                assert!(
                    !callee.name().is_empty(),
                    "resolved sub-jitcode must carry a name",
                );
            }
        }
    }

    /// A `j` slot and the name lookup must reach the same object, not two
    /// shells of one body — the tracer inline-calls through the first and the
    /// dispatch builder registers the second.
    #[test]
    fn a_named_lookup_and_a_pool_slot_reach_one_object() {
        let table = pipeline_table();
        let by_name = pipeline_jitcode_by_name("swap_base_known_two")
            .expect("the pipeline emits `swap_base_known_two`");
        let from_pool = table
            .descrs()
            .iter()
            .filter_map(majit_metainterp::RuntimeBhDescr::as_jitcode)
            .find(|callee| callee.name() == "swap_base_known_two")
            .expect("`swap_base_known_two` is inline-called, so a `j` slot names it");
        assert!(
            Arc::ptr_eq(&by_name, from_pool),
            "the dispatch builder registers what the name lookup returns and the \
             tracer follows what the pool holds; two shells make those different \
             jitcodes with the same body",
        );
    }

    /// The chain the dispatch traces into is deeper than one hop, so a callee
    /// reached through the pool must resolve its own operands.
    #[test]
    fn a_callee_reached_through_the_pool_resolves_its_own_inline_calls() {
        let table = pipeline_table();
        let swap_base = pipeline_jitcode_by_name("swap_base_known_two")
            .expect("the pipeline emits `swap_base_known_two`");
        let swap_nodes = pipeline_jitcode_by_name("swap_nodes_known_two")
            .expect("the pipeline emits `swap_nodes_known_two` under `swap_base_known_two`");
        for (index, _) in table.descrs().iter().enumerate() {
            // Both must answer the same pool, at every index — that is what
            // resolving through the global fallback rather than a per-shell
            // copy buys, and it is what a second hop needs.
            assert!(
                std::ptr::eq(
                    swap_base
                        .descr_at(index)
                        .expect("global pool covers every index"),
                    swap_nodes
                        .descr_at(index)
                        .expect("global pool covers every index"),
                ),
                "descr {index} must resolve to one entry for every jitcode; a \
                 per-shell pool makes the depth a lookup happened at decide \
                 whether it resolves at all",
            );
        }
    }
}
