/// Storage system for Aheui: 28 storage spaces (Stacks, Queue, Port).
///
/// Ported from rpaheui/aheui/storage/linkedlist.py.
use std::collections::VecDeque;

use crate::aheui::{STORAGE_COUNT, VAL_PORT, VAL_QUEUE};
use crate::value::*;

/// Common storage operations.
pub enum StorageKind {
    Stack(AheuiStack),
    Queue(AheuiQueue),
    Port(AheuiPort),
}

impl StorageKind {
    pub fn push(&mut self, value: impl Into<Val>) {
        match self {
            StorageKind::Stack(s) => s.push(value),
            StorageKind::Queue(q) => q.push(value),
            StorageKind::Port(p) => p.push(value),
        }
    }

    pub fn pop(&mut self) -> Val {
        match self {
            StorageKind::Stack(s) => s.pop(),
            StorageKind::Queue(q) => q.pop(),
            StorageKind::Port(p) => p.pop(),
        }
    }

    pub fn dup(&mut self) {
        match self {
            StorageKind::Stack(s) => s.dup(),
            StorageKind::Queue(q) => q.dup(),
            StorageKind::Port(p) => p.dup(),
        }
    }

    pub fn swap(&mut self) {
        match self {
            StorageKind::Stack(s) => s.swap(),
            StorageKind::Queue(q) => q.swap(),
            StorageKind::Port(p) => p.swap(),
        }
    }

    pub fn add(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_add),
            StorageKind::Queue(q) => q.binop(val_add),
            StorageKind::Port(p) => p.binop(val_add),
        }
    }

    pub fn sub(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_sub),
            StorageKind::Queue(q) => q.binop(val_sub),
            StorageKind::Port(p) => p.binop(val_sub),
        }
    }

    pub fn mul(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_mul),
            StorageKind::Queue(q) => q.binop(val_mul),
            StorageKind::Port(p) => p.binop(val_mul),
        }
    }

    pub fn div(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_div),
            StorageKind::Queue(q) => q.binop(val_div),
            StorageKind::Port(p) => p.binop(val_div),
        }
    }

    pub fn modulo(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(val_mod),
            StorageKind::Queue(q) => q.binop(val_mod),
            StorageKind::Port(p) => p.binop(val_mod),
        }
    }

    pub fn cmp(&mut self) {
        match self {
            StorageKind::Stack(s) => s.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
            StorageKind::Queue(q) => q.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
            StorageKind::Port(p) => p.binop(|a, b| {
                if val_ge(&a, &b) {
                    val_from_i32(1)
                } else {
                    val_from_i32(0)
                }
            }),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            StorageKind::Stack(s) => s.data.len(),
            StorageKind::Queue(q) => q.data.len(),
            StorageKind::Port(p) => p.data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Peek at element at index i as i64 (for JIT extract/runtime).
    /// For bigint mode, extracts i64 from Small variant.
    pub fn peek_at(&self, i: usize) -> i64 {
        match self {
            StorageKind::Stack(s) => val_to_i64(&s.data[i]),
            StorageKind::Queue(q) => val_to_i64(&q.data[i]),
            StorageKind::Port(p) => val_to_i64(&p.data[i]),
        }
    }

    /// Check whether all values fit in i32 range (for JIT i32 acceleration).
    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    pub fn all_values_i32(&self) -> bool {
        fn check_i32(v: &Val) -> bool {
            if v.is_small() {
                let raw = v.to_i64();
                raw >= i32::MIN as i64 && raw <= i32::MAX as i64
            } else {
                false
            }
        }
        match self {
            StorageKind::Stack(s) => s.data.iter().all(check_i32),
            StorageKind::Queue(q) => q.data.iter().all(check_i32),
            StorageKind::Port(p) => p.data.iter().all(check_i32),
        }
    }

    /// Check whether all values are small (tagged pointer i63 range).
    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    pub fn all_values_small(&self) -> bool {
        fn check(v: &Val) -> bool {
            v.is_small()
        }
        match self {
            StorageKind::Stack(s) => s.data.iter().all(check),
            StorageKind::Queue(q) => q.data.iter().all(check),
            StorageKind::Port(p) => p.data.iter().all(check),
        }
    }

    /// Clear all elements.
    pub fn clear(&mut self) {
        match self {
            StorageKind::Stack(s) => s.data.clear(),
            StorageKind::Queue(q) => q.data.clear(),
            StorageKind::Port(p) => p.data.clear(),
        }
    }

    /// Check whether all values are representable as i64 (for JIT compatibility).
    #[cfg(not(feature = "bigint"))]
    #[inline(always)]
    pub fn all_jit_compatible(&self) -> bool {
        true
    }

    /// Check whether all values are Small (i64) for JIT compatibility.
    #[cfg(feature = "bigint")]
    pub fn all_jit_compatible(&self) -> bool {
        match self {
            StorageKind::Stack(s) => s.data.iter().all(|v| v.is_small()),
            StorageKind::Queue(q) => q.data.iter().all(|v| v.is_small()),
            StorageKind::Port(p) => p.data.iter().all(|v| v.is_small()),
        }
    }
}

/// Stack storage (LIFO). Used for 26 of the 28 storage slots.
///
/// Binary ops: pop top (r1), peek next (r2), result = r2 op r1, replace top.
pub struct AheuiStack {
    pub data: Vec<Val>,
}

impl AheuiStack {
    pub fn new() -> Self {
        AheuiStack {
            data: Vec::with_capacity(16),
        }
    }

    pub fn push(&mut self, value: impl Into<Val>) {
        self.data.push(value.into());
    }

    pub fn pop(&mut self) -> Val {
        self.data.pop().expect("stack underflow")
    }

    pub fn dup(&mut self) {
        let top = *self.data.last().expect("stack underflow on dup");
        self.data.push(top);
    }

    pub fn swap(&mut self) {
        let len = self.data.len();
        assert!(len >= 2, "stack underflow on swap");
        self.data.swap(len - 1, len - 2);
    }

    /// Pop r1 (top), peek r2 (new top), compute f(r2, r1), replace top with result.
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.data.pop().expect("stack underflow on binop");
        let r2 = *self.data.last().expect("stack underflow on binop");
        let result = f(r2, r1);
        *self.data.last_mut().unwrap() = result;
    }
}

/// Queue storage (FIFO). Used for storage slot 21.
///
/// Binary ops: pop front (r1), pop front (r2), result = r2 op r1, push to back.
pub struct AheuiQueue {
    pub data: VecDeque<Val>,
}

impl AheuiQueue {
    pub fn new() -> Self {
        AheuiQueue {
            data: VecDeque::with_capacity(16),
        }
    }

    pub fn push(&mut self, value: impl Into<Val>) {
        self.data.push_back(value.into());
    }

    pub fn pop(&mut self) -> Val {
        self.data.pop_front().expect("queue underflow")
    }

    pub fn dup(&mut self) {
        let front = *self.data.front().expect("queue underflow on dup");
        self.data.push_front(front);
    }

    pub fn swap(&mut self) {
        assert!(self.data.len() >= 2, "queue underflow on swap");
        self.data.swap(0, 1);
    }

    /// Pop r1 (front), pop r2 (front), compute f(r2, r1), push result to back.
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.data.pop_front().expect("queue underflow on binop");
        let r2 = self.data.pop_front().expect("queue underflow on binop");
        let result = f(r2, r1);
        self.data.push_back(result);
    }
}

/// Port storage (slot 27). Like Stack but dup pushes last_push value.
pub struct AheuiPort {
    pub data: Vec<Val>,
    pub last_push: Val,
}

impl AheuiPort {
    pub fn new() -> Self {
        AheuiPort {
            data: Vec::with_capacity(16),
            last_push: val_from_i32(0),
        }
    }

    pub fn push(&mut self, value: impl Into<Val>) {
        let v = value.into();
        self.last_push = v;
        self.data.push(v);
    }

    pub fn pop(&mut self) -> Val {
        self.data.pop().expect("port underflow")
    }

    pub fn dup(&mut self) {
        let v = self.last_push;
        self.push(v);
    }

    pub fn swap(&mut self) {
        let len = self.data.len();
        assert!(len >= 2, "port underflow on swap");
        self.data.swap(len - 1, len - 2);
    }

    /// Same as Stack: pop r1, peek r2, compute f(r2, r1), replace top.
    pub fn binop(&mut self, f: impl FnOnce(Val, Val) -> Val) {
        let r1 = self.data.pop().expect("port underflow on binop");
        let r2 = *self.data.last().expect("port underflow on binop");
        let result = f(r2, r1);
        *self.data.last_mut().unwrap() = result;
    }
}

/// Pool of 28 storage spaces.
pub struct StoragePool {
    pub pools: Vec<StorageKind>,
}

impl StoragePool {
    pub fn new() -> Self {
        let mut pools = Vec::with_capacity(STORAGE_COUNT);
        for i in 0..STORAGE_COUNT {
            if i == VAL_QUEUE {
                pools.push(StorageKind::Queue(AheuiQueue::new()));
            } else if i == VAL_PORT {
                pools.push(StorageKind::Port(AheuiPort::new()));
            } else {
                pools.push(StorageKind::Stack(AheuiStack::new()));
            }
        }
        StoragePool { pools }
    }

    pub fn get(&self, idx: usize) -> &StorageKind {
        &self.pools[idx]
    }

    pub fn get_mut(&mut self, idx: usize) -> &mut StorageKind {
        &mut self.pools[idx]
    }

    /// Check whether all storage values fit in i32 range.
    ///
    /// When all values are i32, the JIT can use plain IntAdd/IntSub/IntMul
    /// without overflow checks, since i32*i32 fits in i64.
    #[cfg(not(any(feature = "num-bigint", feature = "malachite-bigint")))]
    #[inline(always)]
    pub fn all_jit_compatible(&self) -> bool {
        true
    }

    #[cfg(any(feature = "num-bigint", feature = "malachite-bigint"))]
    #[inline(always)]
    pub fn all_jit_compatible(&self) -> bool {
        self.pools.iter().all(|s| s.all_values_i32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_basic() {
        let mut s = AheuiStack::new();
        s.push(10_i64);
        s.push(20_i64);
        assert_eq!(val_to_i64(&s.pop()), 20);
        assert_eq!(val_to_i64(&s.pop()), 10);
    }

    #[test]
    fn test_stack_binop() {
        let mut s = AheuiStack::new();
        s.push(10_i64);
        s.push(3_i64);
        // add: r1=3, r2=10, result=13, replaces top
        s.binop(val_add);
        assert_eq!(s.data.len(), 1);
        assert_eq!(val_to_i64(&s.pop()), 13);
    }

    #[test]
    fn test_stack_dup_swap() {
        let mut s = AheuiStack::new();
        s.push(5_i64);
        s.dup();
        assert_eq!(s.data.len(), 2);
        s.push(7_i64);
        s.swap();
        assert_eq!(val_to_i64(&s.pop()), 5); // was second-to-top
        assert_eq!(val_to_i64(&s.pop()), 7); // was top, now swapped
    }

    #[test]
    fn test_queue_basic() {
        let mut q = AheuiQueue::new();
        q.push(1_i64);
        q.push(2_i64);
        q.push(3_i64);
        assert_eq!(val_to_i64(&q.pop()), 1); // FIFO
        assert_eq!(val_to_i64(&q.pop()), 2);
    }

    #[test]
    fn test_queue_binop() {
        let mut q = AheuiQueue::new();
        q.push(10_i64);
        q.push(3_i64);
        // add: r1=10(front), r2=3(front), result=13, pushed to back
        q.binop(val_add);
        assert_eq!(q.data.len(), 1);
        assert_eq!(val_to_i64(&q.pop()), 13);
    }

    #[test]
    fn test_port_dup() {
        let mut p = AheuiPort::new();
        p.push(5_i64);
        p.push(10_i64);
        p.dup(); // duplicates last_push=10
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 5);
    }

    #[test]
    fn test_storage_pool() {
        let pool = StoragePool::new();
        assert_eq!(pool.pools.len(), 28);
        assert!(matches!(pool.get(0), StorageKind::Stack(_)));
        assert!(matches!(pool.get(VAL_QUEUE), StorageKind::Queue(_)));
        assert!(matches!(pool.get(VAL_PORT), StorageKind::Port(_)));
    }
}
