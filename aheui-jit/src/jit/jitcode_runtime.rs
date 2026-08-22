//! Runtime access to this crate's build-time jitcode table.
//!
//! `MetaInterpStaticData.jitcodes` (warmspot.py) — the list of
//! `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py). The build script (`build.rs`) runs the
//! `majit_translate` pipeline over the interpreter sources and writes
//! `pipeline.jitcodes`, the shared `pipeline.descrs`
//! (`Assembler.descrs`, assembler.py), symbolic function paths, and the
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
/// `Assembler.descrs` (assembler.py) handed to
/// `BlackholeInterpBuilder.setup_descrs` (blackhole.py). Each 'd'/'j'
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

// Integer division and remainder are the only primitives the graph pipeline
// leaves as host calls — every other arithmetic operator it lowers to an IR op.
// A pipeline helper that divides therefore needs these bound, or its call site
// keeps the symbolic placeholder and jumps into it.
extern "C" fn wrapping_div_i64(numerator: i64, denominator: i64) -> i64 {
    numerator.wrapping_div(denominator)
}

extern "C" fn wrapping_rem_i64(numerator: i64, denominator: i64) -> i64 {
    numerator.wrapping_rem(denominator)
}

// The dual-mode flag is a static, and the pipeline spells a read of it as a call
// to the static's path. The macro-lowered dispatch reads it through its own
// registered helper; a pipeline helper needs this binding to reach the same bit.
extern "C" fn bigint_mode_flag() -> i64 {
    aheui_runtime::value::bigint_mode() as i64
}

// `aheui_runtime::band`'s six escapes. The band helpers reach `val_*` only
// through these, so binding them is what keeps the pipeline from having to lower
// the value layer's closures and generics — see the escape block in `band.rs`.
macro_rules! band_escape {
    ($shim:ident, $val_op:ident) => {
        extern "C" fn $shim(r2: i64, r1: i64) -> i64 {
            use aheui_runtime::value::{val_as_raw_i64, val_from_raw_i64, $val_op};
            val_as_raw_i64($val_op(val_from_raw_i64(r2), val_from_raw_i64(r1)))
        }
    };
}

band_escape!(band_promote_add, val_add);
band_escape!(band_promote_sub, val_sub);
band_escape!(band_promote_mul, val_mul);
band_escape!(band_promote_div, val_div);
band_escape!(band_promote_mod, val_mod);

extern "C" fn band_compare_ge(r2: i64, r1: i64) -> i64 {
    use aheui_runtime::value::{val_as_raw_i64, val_from_i32, val_from_raw_i64, val_ge};
    let ge = val_ge(&val_from_raw_i64(r2), &val_from_raw_i64(r1));
    val_as_raw_i64(val_from_i32(ge as i32))
}

fn runtime_fnaddr_bindings() -> [(&'static str, i64); 14] {
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
        (
            "core::num::<Impl>::wrapping_div",
            wrapping_div_i64 as *const () as usize as i64,
        ),
        (
            "core::num::<Impl>::wrapping_rem",
            wrapping_rem_i64 as *const () as usize as i64,
        ),
        (
            "aheui_runtime::value::bigint::BIGINT_MODE",
            bigint_mode_flag as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::promote_add",
            band_promote_add as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::promote_sub",
            band_promote_sub as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::promote_mul",
            band_promote_mul as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::promote_div",
            band_promote_div as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::promote_mod",
            band_promote_mod as *const () as usize as i64,
        ),
        (
            "aheui_runtime::band::compare_ge",
            band_compare_ge as *const () as usize as i64,
        ),
    ]
}

/// The materialized table, built once.
///
/// Materializing per lookup would hand out a different `Arc` each call for the
/// same jitcode, which the identity the table exists to keep
/// (`codewriter.py all_jitcodes[jitcode.index] is jitcode`) does not
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

    /// Every path `band.rs` leaves its fast paths through must be bound here.
    ///
    /// An unbound path keeps the `symbolic_fnaddr_for_path` placeholder, and the
    /// first trace that reaches the slow path calls it — the metainterp aborts
    /// naming the hash, but only once a program happens to overflow or divide.
    /// Checking the table directly names the path instead.
    #[test]
    fn band_escape_paths_are_bound() {
        let bound: Vec<&str> = runtime_fnaddr_bindings().iter().map(|(p, _)| *p).collect();
        let symbolic = load_symbolic_fnaddr_paths();
        for escape in [
            "aheui_runtime::band::promote_add",
            "aheui_runtime::band::promote_sub",
            "aheui_runtime::band::promote_mul",
            "aheui_runtime::band::promote_div",
            "aheui_runtime::band::promote_mod",
            "aheui_runtime::band::compare_ge",
            "aheui_runtime::value::bigint::BIGINT_MODE",
        ] {
            assert!(
                symbolic.iter().any(|(_, path)| path == escape),
                "`{escape}` is not a symbolic path the pipeline emitted; \
                 either it was renamed or the pipeline now lowers it inline",
            );
            assert!(bound.contains(&escape), "`{escape}` has no runtime binding");
        }
    }

    /// Every helper the mainloop names with `inline_pipeline_int` must end in a
    /// typed return opcode.
    ///
    /// `lower_value.rs` reads the callee's trailing return to learn which
    /// register carries the result, and a helper whose last block is a join
    /// instead of a return fails there — at aheui startup, as a panic, with no
    /// mention of which helper is at fault. Checking it here names the helper.
    #[test]
    fn band_helpers_end_in_a_typed_return() {
        use majit_metainterp::jitcode::JitCodeRuntimeExt as _;
        for name in [
            "pop_base_known_nonempty",
            "band_add",
            "band_sub",
            "band_mul",
            "band_div",
            "band_mod",
            "band_cmp",
            "band_add_raw",
            "band_sub_raw",
            "band_mul_raw",
            "band_div_raw",
            "band_mod_raw",
            "band_cmp_raw",
        ] {
            let jitcode = pipeline_jitcode_by_name(name)
                .unwrap_or_else(|| panic!("pipeline jitcode `{name}` is missing"));
            assert!(
                jitcode.trailing_return_info().is_some(),
                "`{name}` does not end in a typed return opcode; \
                 its last bytes are {:?}",
                &jitcode.code[jitcode.code.len().saturating_sub(8)..],
            );
        }
    }

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
