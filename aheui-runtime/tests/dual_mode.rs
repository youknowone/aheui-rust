//! Dual mode end to end: start raw, overflow, and read everything back.
//!
//! In its own integration binary on purpose. The mode is a process-global
//! one-way switch, so a test that leaves mode 0 cannot share a process with
//! tests that assume mode 1 — and cargo runs `#[test]` fns in parallel
//! threads. For the same reason this file holds exactly one test.
#![cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]

use aheui_runtime::aheui::{STORAGE_COUNT, VAL_PORT, VAL_QUEUE};
use aheui_runtime::storage::{Storage, set_gc_roots};
use aheui_runtime::value::{
    bigint_mode, start_in_raw_mode, val_from_i32, val_from_raw_i64, val_to_i64,
};

#[test]
fn test_overflow_leaves_raw_mode_and_keeps_every_bystander() {
    start_in_raw_mode();
    assert!(!bigint_mode(), "mode 0 is where this starts");

    let mut storage = Storage::new();
    storage.refresh_pools();
    set_gc_roots(&mut storage as *mut Storage);

    let plain: Vec<usize> = (0..STORAGE_COUNT)
        .filter(|&idx| idx != VAL_QUEUE && idx != VAL_PORT)
        .collect();
    let (arith, bystander) = (plain[0], plain[1]);

    // Bystanders: untouched by the arithmetic, and therefore only correct
    // afterwards if the flip walked them. An even one is the interesting case
    // — as a tagged word it would read as a bigint pointer.
    storage.dispatch_mut(bystander).push(val_from_i32(12_345));
    storage.dispatch_mut(bystander).push(val_from_i32(-98_766));
    storage.dispatch_mut(VAL_QUEUE).push(val_from_i32(7));
    storage.dispatch_mut(VAL_PORT).push(val_from_i32(-8));
    storage.port.last_push = val_from_i32(1_000_000);

    // i64::MAX + 1 does not fit the machine word, which is the only way out
    // of mode 0.
    storage.dispatch_mut(arith).push(val_from_raw_i64(i64::MAX));
    storage.dispatch_mut(arith).push(val_from_i32(1));
    storage.dispatch_mut(arith).add();

    assert!(bigint_mode(), "the overflow did not leave mode 0");

    let sum = storage.dispatch_mut(arith).pop();
    assert_eq!(
        format!("{sum}"),
        "9223372036854775808",
        "the promoting op's own operands were not fixed up"
    );

    assert_eq!(val_to_i64(&storage.port.last_push), 1_000_000);
    assert_eq!(val_to_i64(&storage.dispatch_mut(VAL_PORT).pop()), -8);
    assert_eq!(val_to_i64(&storage.dispatch_mut(VAL_QUEUE).pop()), 7);
    assert_eq!(val_to_i64(&storage.dispatch_mut(bystander).pop()), -98_766);
    assert_eq!(val_to_i64(&storage.dispatch_mut(bystander).pop()), 12_345);
}
