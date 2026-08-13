//! Array-backed storage: the three Aheui pools over a flat `Val` buffer.
//!
//! `rpaheui/aheui/storage/array.py` is CPython-only (`assert not PYR`) because
//! `class Stack(list)` / `class Queue(deque)` do not translate; it is not a
//! source we can port line-by-line. What does not translate is the *inheritance
//! from a builtin container*, not the array representation itself, so the
//! layouts here are written against `linkedlist.py`'s observable semantics
//! instead, which is the behaviour both backends must agree on.
//!
//! Two places where `array.py` diverges from `linkedlist.py` are not copied,
//! because `linkedlist.py` is the semantics the corpus pins:
//!
//! * `array.py:51-52` `Queue.dup` is `appendleft(self[0])`, and in its
//!   orientation `self[0]` is the BACK, so it duplicates the most recently
//!   pushed element. `linkedlist.py:112-116` duplicates the FRONT.
//! * `array.py:82` makes `Port = Stack`, dropping the `last_push` shadow that
//!   `linkedlist.py:134-145` keeps and that `Port::dup` reads.
//!
//! Layout: every pool is `#[repr(C)]` with the buffer base pointer at offset 0
//! and the element count at offset 8, so `Storage::pools` can keep punning all
//! three through one pointer type exactly as it does for the linked list. The
//! buffer is a hand-rolled allocation rather than a `Vec` field: `Vec`'s field
//! order is unspecified, and the JIT bakes in `offset_of!` for the base.
//!
//! Growth may reallocate. Any operation that can push may move the buffer, so a
//! base pointer read before a push must not be reused after it. The linked
//! list had no such hazard — `head`/`next` stores never invalidated an
//! unrelated pointer.
use crate::value::*;

/// Element count each pool allocates on first push, then doubles.
const INITIAL_CAPACITY: u32 = 16;

/// Allocate a `Val` buffer of `cap` elements.
///
/// # Safety
/// `cap` must be non-zero.
unsafe fn alloc_buffer(cap: u32) -> *mut Val {
    debug_assert!(cap > 0);
    let layout = std::alloc::Layout::array::<Val>(cap as usize).expect("storage buffer layout");
    let ptr = unsafe { std::alloc::alloc(layout) } as *mut Val;
    assert!(!ptr.is_null(), "storage buffer allocation failed");
    ptr
}

/// Grow `data` from `old_cap` to `new_cap` elements, preserving the first
/// `live` elements. Returns the (possibly moved) base pointer.
///
/// # Safety
/// `data` must be null with `old_cap == 0`, or a buffer of `old_cap` elements
/// with at least `live` initialized.
unsafe fn grow_buffer(data: *mut Val, old_cap: u32, new_cap: u32, live: u32) -> *mut Val {
    debug_assert!(new_cap > old_cap);
    let fresh = unsafe { alloc_buffer(new_cap) };
    if !data.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(data, fresh, live as usize);
            let layout =
                std::alloc::Layout::array::<Val>(old_cap as usize).expect("storage buffer layout");
            std::alloc::dealloc(data as *mut u8, layout);
        }
    }
    fresh
}

/// Release a pool's buffer.
///
/// # Safety
/// `data` must be null with `cap == 0`, or the live allocation of `cap`
/// elements.
unsafe fn free_buffer(data: *mut Val, cap: u32) {
    if data.is_null() {
        return;
    }
    let layout = std::alloc::Layout::array::<Val>(cap as usize).expect("storage buffer layout");
    unsafe { std::alloc::dealloc(data as *mut u8, layout) };
}

/// Shared operations over an array-backed pool.
///
/// Mirrors the `LinkedList` trait's split: the subclass supplies `push` /
/// `dup` / `_get_2_values` / `_put_value` and this trait derives the
/// arithmetic from them, so the two backends compute each opcode through the
/// same shape (`linkedlist.py:35-64`).
pub trait ArrayStorage {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push(&mut self, value: Val);
    fn pop(&mut self) -> Val;
    fn dup(&mut self);
    fn swap(&mut self);
    fn _get_2_values(&mut self) -> (Val, Val);
    fn _put_value(&mut self, value: Val);

    // linkedlist.py:35-38
    fn add(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_add(r2, r1));
    }

    // linkedlist.py:40-43
    fn sub(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_sub(r2, r1));
    }

    // linkedlist.py:45-48
    fn mul(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mul(r2, r1));
    }

    // linkedlist.py:50-53
    fn div(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_div(r2, r1));
    }

    // linkedlist.py:55-58 — Python spells this `mod`.
    fn modulo(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mod(r2, r1));
    }

    // linkedlist.py:60-64
    fn cmp(&mut self) {
        let (r1, r2) = self._get_2_values();
        let r = if val_ge(&r2, &r1) {
            val_from_i32(1)
        } else {
            val_from_i32(0)
        };
        self._put_value(r);
    }
}

// Stack layout.
// Top of stack is `data[size - 1]`, matching `array.py:8-42`'s right end and
// `linkedlist.py:76-91`'s head.

/// `linkedlist.py:67-91 class Stack` over a flat buffer.
#[repr(C)]
pub struct Stack {
    pub data: *mut Val,
    /// The element count. `u32` rather than `usize` so the JIT's field descr
    /// is sub-word and `intbounds` can bound a load of it; without an upper
    /// bound a depth `+ 1` may overflow, the sum goes rangeless, and every
    /// re-check of the depth has to be guarded again.
    pub size: u32,
    pub cap: u32,
}

impl Stack {
    pub fn new() -> Self {
        Stack {
            data: std::ptr::null_mut(),
            size: 0,
            cap: 0,
        }
    }

    /// Ensure room for one more element, reallocating if full.
    ///
    /// Callers must re-read `self.data` afterwards: this may move the buffer.
    fn reserve_one(&mut self) {
        if self.size < self.cap {
            return;
        }
        let new_cap = if self.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.cap * 2
        };
        self.data = unsafe { grow_buffer(self.data, self.cap, new_cap, self.size) };
        self.cap = new_cap;
    }

    /// The top element without popping. Panics when empty.
    pub fn top(&self) -> Val {
        assert!(self.size > 0, "top of empty stack");
        unsafe { *self.data.add(self.size as usize - 1) }
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

// The buffer is a plain `std::alloc` allocation, not a GC object and not
// nursery-managed, so nothing else would ever reclaim it. `Storage` owns every
// pool inline, so this runs when the storage does.
impl Drop for Stack {
    fn drop(&mut self) {
        unsafe { free_buffer(self.data, self.cap) };
    }
}

impl ArrayStorage for Stack {
    fn len(&self) -> usize {
        self.size as usize
    }

    // linkedlist.py:76-80
    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            unsafe {
                *self.data.add(self.size as usize) = value;
            }
            self.size += 1;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:22-28
    fn pop(&mut self) -> Val {
        assert!(self.size > 0, "pop from empty stack");
        self.size -= 1;
        unsafe { *self.data.add(self.size as usize) }
    }

    // linkedlist.py:82-83 — `self.push(self.head.value)`.
    fn dup(&mut self) {
        let top = self.top();
        self.push(top);
    }

    // linkedlist.py:30-33 — swap the two topmost values in place.
    fn swap(&mut self) {
        assert!(self.size >= 2, "swap on <2 elements");
        unsafe {
            let a = self.data.add(self.size as usize - 1);
            let b = self.data.add(self.size as usize - 2);
            std::ptr::swap(a, b);
        }
    }

    // linkedlist.py:87-88 — `return self.pop(), self.head.value`.
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.top();
        (r1, r2)
    }

    // linkedlist.py:90-91 — `self.head.value = value`, i.e. overwrite the top
    // rather than push.
    fn _put_value(&mut self, value: Val) {
        assert!(self.size > 0, "_put_value on empty stack");
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            unsafe {
                *self.data.add(self.size as usize - 1) = value;
            }
            maybe_collect_bigints();
        });
    }
}

// Queue layout.
// FIFO: push appends at the back, pop takes from the front. A ring over
// `[front, front + size)` keeps both ends O(1) without the unbounded drift a
// moving front index alone would cause in a long-running program.
//
// The linked list carries one node MORE than `size` (a tail sentinel,
// `linkedlist.py:98-101`); that is an artifact of the chain representation and
// has no counterpart here. Any invariant checker that budgets `size + 1` for
// the queue is checking the chain, not the semantics.

/// `linkedlist.py:94-122 class Queue` over a ring buffer.
#[repr(C)]
pub struct Queue {
    pub data: *mut Val,
    /// Element count, narrowed for the same reason as [`Stack::size`].
    pub size: u32,
    pub cap: u32,
    /// Index of the front element; the ring spans `[front, front + size)`.
    pub front: u32,
}

impl Queue {
    pub fn new() -> Self {
        Queue {
            data: std::ptr::null_mut(),
            size: 0,
            cap: 0,
            front: 0,
        }
    }

    fn slot(&self, offset: u32) -> *mut Val {
        debug_assert!(self.cap > 0);
        let idx = (self.front as usize + offset as usize) % self.cap as usize;
        unsafe { self.data.add(idx) }
    }

    /// Ensure room for one more element. On growth the ring is unrolled so
    /// `front` returns to 0, which keeps the copy a pair of contiguous runs.
    fn reserve_one(&mut self) {
        if self.size < self.cap {
            return;
        }
        let new_cap = if self.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.cap * 2
        };
        let fresh = unsafe { alloc_buffer(new_cap) };
        for i in 0..self.size {
            unsafe {
                *fresh.add(i as usize) = *self.slot(i);
            }
        }
        if !self.data.is_null() {
            let layout =
                std::alloc::Layout::array::<Val>(self.cap as usize).expect("storage buffer layout");
            unsafe { std::alloc::dealloc(self.data as *mut u8, layout) };
        }
        self.data = fresh;
        self.cap = new_cap;
        self.front = 0;
    }

    /// The front element without popping. Panics when empty.
    pub fn front_value(&self) -> Val {
        assert!(self.size > 0, "front of empty queue");
        unsafe { *self.slot(0) }
    }

    /// Insert at the FRONT — the shape `dup` needs (`linkedlist.py:112-116`).
    fn push_front(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            self.front = if self.front == 0 {
                self.cap - 1
            } else {
                self.front - 1
            };
            unsafe {
                *self.slot(0) = value;
            }
            self.size += 1;
            maybe_collect_bigints();
        });
    }
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

// The buffer is a plain `std::alloc` allocation, not a GC object and not
// nursery-managed, so nothing else would ever reclaim it. `Storage` owns every
// pool inline, so this runs when the storage does.
impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { free_buffer(self.data, self.cap) };
    }
}

impl ArrayStorage for Queue {
    fn len(&self) -> usize {
        self.size as usize
    }

    // linkedlist.py:103-110 — append at the back.
    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            let size = self.size;
            unsafe {
                *self.slot(size) = value;
            }
            self.size += 1;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:22-28 — take from the front.
    fn pop(&mut self) -> Val {
        assert!(self.size > 0, "pop from empty queue");
        let value = unsafe { *self.slot(0) };
        self.front = (self.front + 1) % self.cap;
        self.size -= 1;
        value
    }

    // linkedlist.py:112-116 — duplicate the FRONT and insert at the front.
    // NOT `array.py:51-52`, which duplicates the back.
    fn dup(&mut self) {
        let front = self.front_value();
        self.push_front(front);
    }

    // linkedlist.py:30-33 — swap the two frontmost values in place.
    fn swap(&mut self) {
        assert!(self.size >= 2, "swap on <2 elements");
        unsafe {
            std::ptr::swap(self.slot(0), self.slot(1));
        }
    }

    // linkedlist.py:118-119 — `return self.pop(), self.pop()`, both off the
    // front.
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.pop();
        (r1, r2)
    }

    // linkedlist.py:121-122 — `self.push(value)`, i.e. the arithmetic result
    // lands at the BACK, not where the operands were.
    fn _put_value(&mut self, value: Val) {
        self.push(value);
    }
}

// Port layout.

/// `linkedlist.py:125-148 class Port` — a stack plus the `last_push` shadow.
#[repr(C)]
pub struct Port {
    pub data: *mut Val,
    /// Element count, narrowed for the same reason as [`Stack::size`].
    pub size: u32,
    pub cap: u32,
    pub last_push: Val,
}

impl Port {
    pub fn new() -> Self {
        Port {
            data: std::ptr::null_mut(),
            size: 0,
            cap: 0,
            last_push: val_from_i32(0),
        }
    }

    fn reserve_one(&mut self) {
        if self.size < self.cap {
            return;
        }
        let new_cap = if self.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.cap * 2
        };
        self.data = unsafe { grow_buffer(self.data, self.cap, new_cap, self.size) };
        self.cap = new_cap;
    }

    pub fn top(&self) -> Val {
        assert!(self.size > 0, "top of empty port");
        unsafe { *self.data.add(self.size as usize - 1) }
    }
}

impl Default for Port {
    fn default() -> Self {
        Self::new()
    }
}

// The buffer is a plain `std::alloc` allocation, not a GC object and not
// nursery-managed, so nothing else would ever reclaim it. `Storage` owns every
// pool inline, so this runs when the storage does.
impl Drop for Port {
    fn drop(&mut self) {
        unsafe { free_buffer(self.data, self.cap) };
    }
}

impl ArrayStorage for Port {
    fn len(&self) -> usize {
        self.size as usize
    }

    // linkedlist.py:134-139 — push records `last_push`.
    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            unsafe {
                *self.data.add(self.size as usize) = value;
            }
            self.size += 1;
            self.last_push = value;
            maybe_collect_bigints();
        });
    }

    // linkedlist.py:22-28
    fn pop(&mut self) -> Val {
        assert!(self.size > 0, "pop from empty port");
        self.size -= 1;
        unsafe { *self.data.add(self.size as usize) }
    }

    // linkedlist.py:141-142 — `self.push(self.last_push)`: the SHADOW, not the
    // top. After `push 5; push 10; add` the top is 15 but `last_push` is 10.
    fn dup(&mut self) {
        self.push(self.last_push);
    }

    // linkedlist.py:30-33
    fn swap(&mut self) {
        assert!(self.size >= 2, "swap on <2 elements");
        unsafe {
            let a = self.data.add(self.size as usize - 1);
            let b = self.data.add(self.size as usize - 2);
            std::ptr::swap(a, b);
        }
    }

    // linkedlist.py:144-145
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.top();
        (r1, r2)
    }

    // linkedlist.py:147-148 — overwrites the top and deliberately leaves
    // `last_push` alone, which is what makes `dup` observably differ from
    // duplicating the top.
    fn _put_value(&mut self, value: Val) {
        assert!(self.size > 0, "_put_value on empty port");
        unsafe {
            *self.data.add(self.size as usize - 1) = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The six cases below mirror `linkedlist.rs`'s tests one-for-one, so the
    // two backends are pinned to the same observable behaviour.

    #[test]
    fn test_stack_basic() {
        let mut s = Stack::new();
        s.push(val_from_i32(10));
        s.push(val_from_i32(20));
        assert_eq!(val_to_i64(&s.pop()), 20);
        assert_eq!(val_to_i64(&s.pop()), 10);
    }

    #[test]
    fn test_stack_add() {
        let mut s = Stack::new();
        s.push(val_from_i32(10));
        s.push(val_from_i32(3));
        // r1=3, r2=10, result=13 replaces the top.
        s.add();
        assert_eq!(s.size, 1);
        assert_eq!(val_to_i64(&s.pop()), 13);
    }

    #[test]
    fn test_stack_dup_swap() {
        let mut s = Stack::new();
        s.push(val_from_i32(5));
        s.dup();
        assert_eq!(s.size, 2);
        s.push(val_from_i32(7));
        s.swap();
        assert_eq!(val_to_i64(&s.pop()), 5);
        assert_eq!(val_to_i64(&s.pop()), 7);
    }

    #[test]
    fn test_queue_basic() {
        let mut q = Queue::new();
        q.push(val_from_i32(1));
        q.push(val_from_i32(2));
        q.push(val_from_i32(3));
        assert_eq!(val_to_i64(&q.pop()), 1);
        assert_eq!(val_to_i64(&q.pop()), 2);
    }

    #[test]
    fn test_queue_add() {
        let mut q = Queue::new();
        q.push(val_from_i32(10));
        q.push(val_from_i32(3));
        // r1=10 and r2=3 both come off the FRONT; 13 goes to the BACK.
        q.add();
        assert_eq!(q.size, 1);
        assert_eq!(val_to_i64(&q.pop()), 13);
    }

    #[test]
    fn test_port_dup() {
        let mut p = Port::new();
        p.push(val_from_i32(5));
        p.push(val_from_i32(10));
        p.dup(); // duplicates last_push=10
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 5);
    }

    // Cases the linked list could not get wrong, but the array can.

    #[test]
    fn stack_survives_growth() {
        // Push past INITIAL_CAPACITY so the buffer reallocates at least twice,
        // then check every element survived the moves in order.
        let mut s = Stack::new();
        let n = (INITIAL_CAPACITY * 4) as i32;
        for i in 0..n {
            s.push(val_from_i32(i));
        }
        assert_eq!(s.size as i32, n);
        for i in (0..n).rev() {
            assert_eq!(val_to_i64(&s.pop()), i as i64);
        }
        assert_eq!(s.size, 0);
    }

    #[test]
    fn queue_wraps_and_survives_growth() {
        // Drive the ring past a wrap before forcing a grow, so the unroll in
        // `reserve_one` has to copy two runs rather than one.
        let mut q = Queue::new();
        for i in 0..INITIAL_CAPACITY as i32 {
            q.push(val_from_i32(i));
        }
        for i in 0..(INITIAL_CAPACITY as i32 / 2) {
            assert_eq!(val_to_i64(&q.pop()), i as i64);
        }
        assert!(q.front > 0, "expected the ring to have advanced");
        let n = INITIAL_CAPACITY as i32 * 3;
        for i in INITIAL_CAPACITY as i32..n {
            q.push(val_from_i32(i));
        }
        for i in (INITIAL_CAPACITY as i32 / 2)..n {
            assert_eq!(val_to_i64(&q.pop()), i as i64);
        }
        assert_eq!(q.size, 0);
    }

    #[test]
    fn port_put_value_leaves_last_push_alone() {
        // `_put_value` must not touch the shadow: after `5, 10, add` the top
        // is 15 while `dup` still duplicates 10.
        let mut p = Port::new();
        p.push(val_from_i32(5));
        p.push(val_from_i32(10));
        p.add();
        assert_eq!(p.size, 1);
        assert_eq!(val_to_i64(&p.top()), 15);
        p.dup();
        assert_eq!(val_to_i64(&p.pop()), 10);
        assert_eq!(val_to_i64(&p.pop()), 15);
    }

    #[test]
    fn queue_dup_duplicates_the_front() {
        // `linkedlist.py:112-116`, NOT `array.py:51-52` which takes the back.
        let mut q = Queue::new();
        q.push(val_from_i32(1));
        q.push(val_from_i32(2));
        q.dup();
        assert_eq!(q.size, 3);
        assert_eq!(val_to_i64(&q.pop()), 1);
        assert_eq!(val_to_i64(&q.pop()), 1);
        assert_eq!(val_to_i64(&q.pop()), 2);
    }
}

/// Differential tests: the array backend must agree with the linked list on
/// every operation, since the corpus pins the linked list's behaviour and the
/// conversion is supposed to be observationally invisible.
#[cfg(test)]
mod differential {
    use super::ArrayStorage;
    use crate::storage::linkedlist::{self, LinkedList};
    use crate::value::*;

    /// Bigint-safe equality: `Val` has no `PartialEq`, and `val_to_i64`
    /// panics once a value promotes past i64 — which repeated `Mul` reaches
    /// quickly, and which is exactly the dual-mode path worth covering. Two
    /// values are equal when neither is strictly greater.
    fn vals_eq(a: &Val, b: &Val) -> bool {
        val_ge(a, b) && val_ge(b, a)
    }

    fn opt_vals_eq(a: &Option<Val>, b: &Option<Val>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(x), Some(y)) => vals_eq(x, y),
            _ => false,
        }
    }

    fn seq_eq(a: &[Val], b: &[Val]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| vals_eq(x, y))
    }

    /// For failure messages only — a promoted value renders as `<big>`, so
    /// never compare on this.
    fn render(v: &Val) -> String {
        match v.try_to_i64() {
            Some(i) => i.to_string(),
            None => "<big>".to_string(),
        }
    }

    fn render_seq(vs: &[Val]) -> Vec<String> {
        vs.iter().map(render).collect()
    }

    /// Deterministic LCG so a failure reproduces from its seed alone.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    /// The operations both backends expose, minus `div`/`modulo` (a zero
    /// divisor traps in both, which would test the panic path rather than
    /// agreement).
    #[derive(Clone, Copy, Debug)]
    enum Op {
        Push(i32),
        Pop,
        Dup,
        Swap,
        Add,
        Sub,
        Mul,
        Cmp,
    }

    fn pick(rng: &mut Rng) -> Op {
        match rng.next() % 8 {
            0 | 1 => Op::Push((rng.next() % 1000) as i32),
            2 => Op::Pop,
            3 => Op::Dup,
            4 => Op::Swap,
            5 => Op::Add,
            6 => Op::Sub,
            7 => Op::Mul,
            _ => Op::Cmp,
        }
    }

    /// Minimum depth an op needs, so the same sequence is legal for both
    /// backends and neither hits its underflow assert. This mirrors the
    /// `req_size` guard the interpreter applies ahead of every opcode.
    fn required_depth(op: Op) -> usize {
        match op {
            Op::Push(_) => 0,
            Op::Pop | Op::Dup => 1,
            Op::Swap | Op::Add | Op::Sub | Op::Mul | Op::Cmp => 2,
        }
    }

    fn run_array<S: ArrayStorage>(pool: &mut S, op: Op) -> Option<Val> {
        match op {
            Op::Push(v) => {
                pool.push(val_from_i32(v));
                None
            }
            Op::Pop => Some(pool.pop()),
            Op::Dup => {
                pool.dup();
                None
            }
            Op::Swap => {
                pool.swap();
                None
            }
            Op::Add => {
                pool.add();
                None
            }
            Op::Sub => {
                pool.sub();
                None
            }
            Op::Mul => {
                pool.mul();
                None
            }
            Op::Cmp => {
                pool.cmp();
                None
            }
        }
    }

    fn run_list<S: LinkedList>(pool: &mut S, op: Op) -> Option<Val> {
        match op {
            Op::Push(v) => {
                pool.push(val_from_i32(v));
                None
            }
            Op::Pop => Some(pool.pop()),
            Op::Dup => {
                pool.dup();
                None
            }
            Op::Swap => {
                pool.swap();
                None
            }
            Op::Add => {
                pool.add();
                None
            }
            Op::Sub => {
                pool.sub();
                None
            }
            Op::Mul => {
                pool.mul();
                None
            }
            Op::Cmp => {
                pool.cmp();
                None
            }
        }
    }

    /// Drain both pools and compare the full remaining contents, so a
    /// divergence buried below the top is caught rather than only the top.
    fn drain_array<S: ArrayStorage>(pool: &mut S) -> Vec<Val> {
        let mut out = Vec::new();
        while !pool.is_empty() {
            out.push(pool.pop());
        }
        out
    }

    fn drain_list<S: LinkedList>(pool: &mut S) -> Vec<Val> {
        let mut out = Vec::new();
        while pool.__len__() > 0 {
            out.push(pool.pop());
        }
        out
    }

    #[test]
    fn stack_matches_linked_list() {
        let _lock = crate::storage::nursery_test_lock();
        for seed in 0..64u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1);
            let mut arr = super::Stack::new();
            let mut list = linkedlist::Stack::new();
            for step in 0..200 {
                let op = pick(&mut rng);
                if arr.len() < required_depth(op) {
                    continue;
                }
                let a = run_array(&mut arr, op);
                let l = run_list(&mut list, op);
                assert!(
                    opt_vals_eq(&a, &l),
                    "seed {seed} step {step} op {op:?} returned differently: array {:?} vs list {:?}",
                    a.as_ref().map(render),
                    l.as_ref().map(render)
                );
                assert_eq!(
                    arr.len(),
                    list.__len__(),
                    "seed {seed} step {step} op {op:?} left different depths"
                );
            }
            let arr_rest = drain_array(&mut arr);
            let list_rest = drain_list(&mut list);
            assert!(
                seq_eq(&arr_rest, &list_rest),
                "seed {seed}: final contents differ: array {:?} vs list {:?}",
                render_seq(&arr_rest),
                render_seq(&list_rest)
            );
        }
    }

    #[test]
    fn queue_matches_linked_list() {
        let _lock = crate::storage::nursery_test_lock();
        for seed in 0..64u64 {
            let mut rng = Rng(seed.wrapping_mul(0xD1B54A32D192ED03) | 1);
            let mut arr = super::Queue::new();
            let mut list = linkedlist::Queue::new();
            for step in 0..200 {
                let op = pick(&mut rng);
                if arr.len() < required_depth(op) {
                    continue;
                }
                let a = run_array(&mut arr, op);
                let l = run_list(&mut list, op);
                assert!(
                    opt_vals_eq(&a, &l),
                    "seed {seed} step {step} op {op:?} returned differently: array {:?} vs list {:?}",
                    a.as_ref().map(render),
                    l.as_ref().map(render)
                );
                assert_eq!(
                    arr.len(),
                    list.__len__(),
                    "seed {seed} step {step} op {op:?} left different depths"
                );
            }
            let arr_rest = drain_array(&mut arr);
            let list_rest = drain_list(&mut list);
            assert!(
                seq_eq(&arr_rest, &list_rest),
                "seed {seed}: final contents differ: array {:?} vs list {:?}",
                render_seq(&arr_rest),
                render_seq(&list_rest)
            );
        }
    }

    #[test]
    fn port_matches_linked_list() {
        let _lock = crate::storage::nursery_test_lock();
        for seed in 0..64u64 {
            let mut rng = Rng(seed.wrapping_mul(0xBF58476D1CE4E5B9) | 1);
            let mut arr = super::Port::new();
            let mut list = linkedlist::Port::new();
            for step in 0..200 {
                let op = pick(&mut rng);
                if arr.len() < required_depth(op) {
                    continue;
                }
                let a = run_array(&mut arr, op);
                let l = run_list(&mut list, op);
                assert!(
                    opt_vals_eq(&a, &l),
                    "seed {seed} step {step} op {op:?} returned differently: array {:?} vs list {:?}",
                    a.as_ref().map(render),
                    l.as_ref().map(render)
                );
                assert_eq!(
                    arr.len(),
                    list.__len__(),
                    "seed {seed} step {step} op {op:?} left different depths"
                );
            }
            let arr_rest = drain_array(&mut arr);
            let list_rest = drain_list(&mut list);
            assert!(
                seq_eq(&arr_rest, &list_rest),
                "seed {seed}: final contents differ: array {:?} vs list {:?}",
                render_seq(&arr_rest),
                render_seq(&list_rest)
            );
        }
    }
}

#[cfg(test)]
mod drop_tests {
    use super::*;

    /// Churn many pools through growth and drop. The buffer is a plain
    /// `std::alloc` allocation that nothing else reclaims, so a missing or
    /// wrong-layout `dealloc` shows up here as a leak or an allocator abort
    /// rather than staying silent.
    #[test]
    fn dropping_pools_releases_their_buffers() {
        for _ in 0..256 {
            let mut s = Stack::new();
            for i in 0..(INITIAL_CAPACITY as i32 * 3) {
                s.push(val_from_i32(i));
            }
            let mut q = Queue::new();
            for i in 0..(INITIAL_CAPACITY as i32 * 3) {
                q.push(val_from_i32(i));
            }
            // Advance the ring so the drop path sees a non-zero `front`.
            for _ in 0..(INITIAL_CAPACITY / 2) {
                q.pop();
            }
            let mut p = Port::new();
            for i in 0..(INITIAL_CAPACITY as i32 * 3) {
                p.push(val_from_i32(i));
            }
        }
    }

    /// A pool that never pushed holds a null buffer; dropping it must not
    /// call `dealloc` on null.
    #[test]
    fn dropping_an_untouched_pool_is_a_no_op() {
        for _ in 0..256 {
            let _ = Stack::new();
            let _ = Queue::new();
            let _ = Port::new();
        }
    }
}
