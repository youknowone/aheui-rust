//! Verify `#[jit_struct]` descr registration for aheui's linked-list
//! storage types (Node, Stack, Queue, Port).

use majit_ir::descr::{Descr, GcCache, LLType};
use majit_ir::value::Type;
use majit_macros::jit_struct;

// ─────────────────────────────────────────────────────────────────────
// Aheui storage structures recoded as `#[jit_struct]`.
// These tests document the descr-registration end-state: once the
// tracer consumes descrs through GcCache lookup, the existing
// `linked_list_*` trait methods on `JitCodeSym` become redundant.
// ─────────────────────────────────────────────────────────────────────

/// Singly-linked list node.
#[jit_struct]
struct AheuiNode {
    value: i64,
    next: Option<Box<AheuiNode>>,
}

/// Stack backed by a singly-linked list.
#[jit_struct]
struct AheuiStack {
    head: Option<Box<AheuiNode>>,
    size: usize,
}

/// Queue backed by a singly-linked list.
#[jit_struct]
struct AheuiQueue {
    head: Option<Box<AheuiNode>>,
    tail: Option<Box<AheuiNode>>,
    size: usize,
}

/// Port (aheui-specific I/O storage).
#[jit_struct]
struct AheuiPort {
    head: Option<Box<AheuiNode>>,
    size: usize,
    last_push: i64,
}

#[test]
fn aheui_shapes_register_descrs() {
    let mut gc = GcCache::new();
    let node = AheuiNode::__majit_register_descrs(&mut gc);
    let stack = AheuiStack::__majit_register_descrs(&mut gc);
    let queue = AheuiQueue::__majit_register_descrs(&mut gc);
    let port = AheuiPort::__majit_register_descrs(&mut gc);

    // Each shape gets a distinct SizeDescr.
    for (a, b) in [
        (&node, &stack),
        (&node, &queue),
        (&node, &port),
        (&stack, &queue),
        (&stack, &port),
        (&queue, &port),
    ] {
        assert!(!std::sync::Arc::ptr_eq(a, b));
    }

    assert_eq!(AheuiNode::__MAJIT_FIELD_NAMES, &["value", "next"]);
    assert_eq!(AheuiStack::__MAJIT_FIELD_NAMES, &["head", "size"]);
    assert_eq!(AheuiQueue::__MAJIT_FIELD_NAMES, &["head", "tail", "size"]);
    assert_eq!(
        AheuiPort::__MAJIT_FIELD_NAMES,
        &["head", "size", "last_push"]
    );
}

#[test]
fn aheui_node_field_types_match() {
    let mut gc = GcCache::new();
    let _ = AheuiNode::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(AheuiNode::__majit_type_id());
    let fields = gc._cache_field.get(&key).unwrap();

    // `value: i64` → Int (bigint lowered to i64 for the integer trace).
    let value_fd = fields.get("value").unwrap().as_field_descr().unwrap();
    assert_eq!(value_fd.field_type(), Type::Int);

    // `next: Option<Box<Node>>` → Ref (the Node* in RPython).
    let next_fd = fields.get("next").unwrap().as_field_descr().unwrap();
    assert_eq!(next_fd.field_type(), Type::Ref);
}

#[test]
fn queue_tail_descr_distinct_from_head() {
    let mut gc = GcCache::new();
    let _ = AheuiQueue::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(AheuiQueue::__majit_type_id());
    let fields = gc._cache_field.get(&key).unwrap();
    let head = fields.get("head").unwrap();
    let tail = fields.get("tail").unwrap();
    assert!(!std::sync::Arc::ptr_eq(head, tail));
    let head_fd = head.as_field_descr().unwrap();
    let tail_fd = tail.as_field_descr().unwrap();
    assert_ne!(head_fd.offset(), tail_fd.offset());
    assert_eq!(head_fd.index_in_parent(), 0);
    assert_eq!(tail_fd.index_in_parent(), 1);
}
