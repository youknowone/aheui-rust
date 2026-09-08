#![cfg(feature = "bigint")]
#![allow(dead_code, unused_imports, unsafe_op_in_unsafe_fn)]

#[cfg(not(feature = "num-bigint"))]
use malachite_bigint::BigInt;
#[cfg(feature = "num-bigint")]
use num_bigint::BigInt;
include!("../src/runtime_templates/bigint_arena.rs.in");
include!("../src/runtime_templates/prelude_int_dual.rs.in");
include!("../src/runtime_templates/prelude_common.rs.in");

#[test]
fn reclaim_dead_bigints_preserving_all_storage_roots() {
    clear_bigints();
    let mut stack = [box_bigint(BigInt::from(i64::MAX))];
    let mut bases = [std::ptr::null_mut(); 28];
    let mut tops = bases;
    bases[0] = stack.as_mut_ptr();
    tops[0] = unsafe { bases[0].add(1) };
    let mut sp = SpecialStorage::new();
    let queued = box_bigint(BigInt::from(i64::MAX) + BigInt::from(1));
    let port = box_bigint(BigInt::from(i64::MAX) + BigInt::from(2));
    let last = box_bigint(BigInt::from(i64::MAX) + BigInt::from(3));
    sp.queue.push_back(queued);
    sp.queue.push_back(queued); // Duplicate pointers are roots, not ownership counts.
    sp.port.push(port);
    sp.port_last = last; // Sticky port value remains live after a pop.
    for _ in 0..10 {
        for _ in 0..100_000 {
            std::hint::black_box(box_bigint(BigInt::from(i64::MAX)));
        }
        unsafe {
            collect_bigints(&bases, &tops, &sp);
        }
        assert_eq!(BIGINT_ARENA.lock().unwrap().values.len(), 4);
        assert_eq!(to_big(stack[0].0), BigInt::from(i64::MAX));
        assert_eq!(
            to_big(sp.queue[0].0),
            BigInt::from(i64::MAX) + BigInt::from(1)
        );
        assert_eq!(
            to_big(sp.port[0].0),
            BigInt::from(i64::MAX) + BigInt::from(2)
        );
        assert_eq!(
            to_big(sp.port_last.0),
            BigInt::from(i64::MAX) + BigInt::from(3)
        );
    }
    tops[0] = bases[0];
    sp.queue.clear();
    sp.port.clear();
    sp.port_last = Int(1);
    unsafe {
        collect_bigints(&bases, &tops, &sp);
    }
    assert_eq!(BIGINT_ARENA.lock().unwrap().values.len(), 0);
    assert!(BIGINT_ARENA.lock().unwrap().values.capacity() <= 1024);
    // Limb storage, not just the number of boxes, must trigger collection.
    let huge = BigInt::from(1) << (BIGINT_MIN_COLLECTION_BYTES * 8);
    box_bigint(huge);
    assert_ne!(unsafe { cbig_collection_due }, 0);
    clear_bigints();
}
