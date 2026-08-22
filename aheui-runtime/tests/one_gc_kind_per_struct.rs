//! Every macro block that names `Node` must classify its gc-kind the same way.
//!
//! A struct's gc-kind is a property of the type, not of the block that reaches
//! it, and the JIT folds that classification into the descriptor identity a
//! field access is minted under. Two blocks that disagree therefore hand the
//! optimizer two descriptors for one physical field: a store recorded against
//! one identity does not invalidate a load cached against the other, and the
//! load keeps serving the value from before the store.
//!
//! That is not a theoretical hazard here. `Node` is declared headerless — so
//! gc-managed — only in the blocks that also allocate one, while the read-only
//! helpers inherit the default. The portal and these helpers then split
//! `Node::value` and `Node::next` across two identities, and a `+= 2` written
//! through one is invisible to the very next read through the other.
//!
//! The declaration is per block and there is no cross-block channel to derive
//! it from, so completeness is what this checks: naming `Node` obliges a block
//! to state its gc-kind.

use std::path::Path;

const NODE: &str = "linkedlist::Node";
const CLASSIFICATION: &str = "headerless_structs";
const ATTRIBUTE: &str = "#[majit_macros::jit_inline(";
/// The attribute list's closing delimiter, at the column an outer attribute
/// puts it on.
const ATTRIBUTE_END: &str = "\n)]\n";

#[test]
fn every_block_naming_node_states_its_gc_kind() {
    let source = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("src/storage/linkedlist_jit.rs"),
    )
    .expect("the lowered linked-list helpers must be readable");

    let mut blocks = 0usize;
    let mut undeclared = Vec::new();
    let mut rest = source.as_str();
    while let Some(open) = rest.find(ATTRIBUTE) {
        let body = &rest[open..];
        let close = body
            .find(ATTRIBUTE_END)
            .expect("a jit_inline attribute list must be closed");
        let block = &body[..close];
        blocks += 1;
        if block.contains(NODE) && !block.contains(CLASSIFICATION) {
            // The item the attribute is attached to, for a locatable message.
            let item = body[close..]
                .lines()
                .find(|line| line.contains("fn "))
                .unwrap_or("<unknown>")
                .trim()
                .to_string();
            undeclared.push(item);
        }
        rest = &body[close + ATTRIBUTE_END.len()..];
    }

    assert!(blocks > 0, "no {ATTRIBUTE} block found — the scan missed");
    assert!(
        undeclared.is_empty(),
        "{} block(s) name {NODE} without declaring `{CLASSIFICATION}`, \
         so they mint its field descrs under a different identity than the \
         blocks that do: {undeclared:#?}",
        undeclared.len(),
    );
}
