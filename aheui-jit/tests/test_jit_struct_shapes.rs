//! Verify `#[jit_struct]` descr registration for aheui's linked-list
//! storage types (Node, ListBase, Stack, Queue, Port).

use majit_ir::descr::{Descr, GcCache, LLType};
use majit_ir::value::Type;
use majit_macros::jit_struct;

// Local `#[jit_struct]` mirrors isolate descriptor registration from the
// runtime storage implementation exercised by the tests. `#[jit_struct]` takes
// no attributes, so it cannot be told that a leading field is an inlined base;
// each mirror therefore spells out the FLATTENED field sequence the runtime
// type presents — the base's words first, in declaration order, then whatever
// the embedding type adds. That flattening is the whole subject here: the
// per-object field cache is keyed by slot, so the slot a storage's own field
// lands in is decided by how many words the base put ahead of it.

/// Singly-linked list node.
#[jit_struct]
struct AheuiNode {
    value: i64,
    next: Option<Box<AheuiNode>>,
}

/// The two words every storage carries. Declared once at the front of each
/// mirror below, mirroring the base the runtime types embed at offset 0.
#[jit_struct]
struct AheuiListBase {
    head: Option<Box<AheuiNode>>,
    size: usize,
}

/// Stack backed by a singly-linked list: the base and nothing else.
#[jit_struct]
struct AheuiStack {
    head: Option<Box<AheuiNode>>,
    size: usize,
}

/// Queue backed by a singly-linked list. `tail` is the queue's own word, so it
/// follows the base's two rather than sitting beside `head`.
#[jit_struct]
struct AheuiQueue {
    head: Option<Box<AheuiNode>>,
    size: usize,
    tail: Option<Box<AheuiNode>>,
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
    let base = AheuiListBase::__majit_register_descrs(&mut gc);
    let stack = AheuiStack::__majit_register_descrs(&mut gc);
    let queue = AheuiQueue::__majit_register_descrs(&mut gc);
    let port = AheuiPort::__majit_register_descrs(&mut gc);

    // Each shape gets a distinct SizeDescr. Sharing a field between a base and
    // the types that embed it does not merge their identities or their sizes —
    // only the descriptors for the words they have in common.
    let all = [&node, &base, &stack, &queue, &port];
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert!(!std::sync::Arc::ptr_eq(a, b));
        }
    }

    assert_eq!(AheuiNode::__MAJIT_FIELD_NAMES, &["value", "next"]);
    assert_eq!(AheuiListBase::__MAJIT_FIELD_NAMES, &["head", "size"]);
    // The base's two words lead every storage, in the base's own order.
    for (name, fields) in [
        ("Stack", AheuiStack::__MAJIT_FIELD_NAMES),
        ("Queue", AheuiQueue::__MAJIT_FIELD_NAMES),
        ("Port", AheuiPort::__MAJIT_FIELD_NAMES),
    ] {
        assert_eq!(
            &fields[..AheuiListBase::__MAJIT_FIELD_NAMES.len()],
            AheuiListBase::__MAJIT_FIELD_NAMES,
            "{name} must present the base's words first, or its own fields take \
             slots the base's already hold for the same object",
        );
    }
    assert_eq!(AheuiStack::__MAJIT_FIELD_NAMES, &["head", "size"]);
    assert_eq!(AheuiQueue::__MAJIT_FIELD_NAMES, &["head", "size", "tail"]);
    assert_eq!(
        AheuiPort::__MAJIT_FIELD_NAMES,
        &["head", "size", "last_push"]
    );
}

/// The runtime types this file mirrors present the same flattened word order.
///
/// `#[jit_struct]` numbers slots from declaration order, so a mirror that
/// drifts from the type it stands for keeps passing while describing a shape
/// nothing has. Tie the two together: the runtime base sits at offset 0 of each
/// storage, and each storage's own word sits after the base's whole extent.
#[test]
fn the_mirrors_stand_for_the_runtime_layout() {
    use aheui_runtime::storage::linkedlist::{ListBase, Port, Queue, Stack};

    assert_eq!(std::mem::offset_of!(Stack, base), 0);
    assert_eq!(std::mem::offset_of!(Queue, base), 0);
    assert_eq!(std::mem::offset_of!(Port, base), 0);
    assert_eq!(std::mem::offset_of!(ListBase, head), 0);

    // `tail` and `last_push` are declared after the base, so they begin at or
    // past its end — never inside the region `head` and `size` occupy.
    let base_extent = std::mem::size_of::<ListBase>();
    assert!(std::mem::offset_of!(Queue, tail) >= base_extent);
    assert!(std::mem::offset_of!(Port, last_push) >= base_extent);

    // Stack adds nothing, so it is the base's size exactly.
    assert_eq!(std::mem::size_of::<Stack>(), base_extent);
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

/// The queue's own word takes the slot after the base's, not one of theirs.
///
/// A field descriptor carries an `index_in_parent`, and the optimizer's
/// per-object field cache is keyed by it — two fields of one object that report
/// the same slot are served from the same entry, so a `tail` read can be
/// answered with what a `head` store left behind. `tail` is the queue's third
/// word because the base contributes two ahead of it; numbering it 0 is what a
/// layout built from the queue's own fields alone would produce.
#[test]
fn the_queues_own_word_follows_the_bases() {
    let mut gc = GcCache::new();
    let _ = AheuiQueue::__majit_register_descrs(&mut gc);
    let key = LLType::struct_key(AheuiQueue::__majit_type_id());
    let fields = gc._cache_field.get(&key).unwrap();
    let head = fields.get("head").unwrap();
    let size = fields.get("size").unwrap();
    let tail = fields.get("tail").unwrap();
    assert!(!std::sync::Arc::ptr_eq(head, tail));
    let head_fd = head.as_field_descr().unwrap();
    let size_fd = size.as_field_descr().unwrap();
    let tail_fd = tail.as_field_descr().unwrap();
    assert_ne!(head_fd.offset(), tail_fd.offset());
    assert_eq!(head_fd.index_in_parent(), 0);
    assert_eq!(size_fd.index_in_parent(), 1);
    assert_eq!(tail_fd.index_in_parent(), 2);
}
