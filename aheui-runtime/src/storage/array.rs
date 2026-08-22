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
//! * `array.py`'s `Queue.dup` is `appendleft(self[0])`, and in its
//!   orientation `self[0]` is the BACK, so it duplicates the most recently
//!   pushed element. `linkedlist.py` duplicates the FRONT.
//! * `array.py` makes `Port = Stack`, dropping the `last_push` shadow that
//!   `linkedlist.py` keeps and that `Port::dup` reads.
//!
//! Layout: every pool is `#[repr(C)]` and embeds [`ArrayBase`] at offset 0,
//! so the words they share are declared once and an access resolves against
//! the struct that declares it — the shape the linked-list backend gets from
//! `ListBase`. The buffer is a hand-rolled allocation rather than a `Vec`
//! field: `Vec`'s field order is unspecified, and the JIT bakes in
//! `offset_of!` for the base.
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

/// Whether to stamp a buffer with the `0x5EEDDEA` sentinel before releasing
/// it, mirroring the nursery's `AHEUI_GC_POISON` quarantine stamp in `mod.rs`.
///
/// This exists because a stale array-element read is otherwise SILENT and
/// usually *correct*.  `grow_buffer` copies the live prefix into a fresh
/// allocation and frees the old one, so a base pointer cached across the grow
/// points into recycled malloc memory — which, for a same-size-class block,
/// still holds the very bytes that were copied out of it.  The wrong read then
/// returns the right value, stdout stays byte-identical, and no stdout/exit
/// oracle can see the defect until an unrelated allocation happens to recycle
/// the block.  Stamping the buffer turns that into a deterministic sentinel
/// that reaches the consumer on the first read.
///
/// The env lookup is cached: unlike the nursery's per-collect stamp this runs
/// on every pool growth.
fn poison_on_free() -> bool {
    static POISON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *POISON.get_or_init(|| std::env::var_os("AHEUI_GC_POISON").is_some())
}

/// Stamp every element of a buffer that is about to be released.
///
/// # Safety
/// `data` must be the live allocation of `cap` elements.
unsafe fn poison_buffer(data: *mut Val, cap: u32) {
    for i in 0..cap as usize {
        unsafe { *data.add(i) = crate::value::val_from_i32(0x5EEDDEA) };
    }
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
            // After the copy, before the free: the old contents are already
            // safe in `fresh`, so anything still reading through the old base
            // is reading a pointer that a realloc invalidated.
            if poison_on_free() {
                poison_buffer(data, old_cap);
            }
            let layout =
                std::alloc::Layout::array::<Val>(old_cap as usize).expect("storage buffer layout");
            std::alloc::dealloc(data as *mut u8, layout);
        }
    }
    fresh
}

/// Release a pool's buffer.
///
/// Each pool calls this from its `Drop`: the buffer is a plain `std::alloc`
/// allocation, not a GC object and not nursery-managed, so nothing else would
/// ever reclaim it. `Storage` owns every pool inline, so the drops run when
/// the storage does.
///
/// # Safety
/// `data` must be null with `cap == 0`, or the live allocation of `cap`
/// elements.
unsafe fn free_buffer(data: *mut Val, cap: u32) {
    if data.is_null() {
        return;
    }
    if poison_on_free() {
        unsafe { poison_buffer(data, cap) };
    }
    let layout = std::alloc::Layout::array::<Val>(cap as usize).expect("storage buffer layout");
    unsafe { std::alloc::dealloc(data as *mut u8, layout) };
}

/// Shared operations over an array-backed pool.
///
/// Mirrors the `LinkedList` trait's split: the subclass supplies `push` /
/// `dup` / `_get_2_values` / `_put_value` and this trait derives the
/// arithmetic from them, so the two backends compute each opcode through the
/// same shape.
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

    fn add(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_add(r2, r1));
    }

    fn sub(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_sub(r2, r1));
    }

    fn mul(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mul(r2, r1));
    }

    fn div(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_div(r2, r1));
    }

    // Python spells this `mod`.
    fn modulo(&mut self) {
        let (r1, r2) = self._get_2_values();
        self._put_value(val_mod(r2, r1));
    }

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

/// The fields every pool declares, as a struct the pools embed at offset 0.
///
/// Declared once rather than repeated per pool: reaching `data`/`size`/`cap`
/// by reinterpreting one pool's address as another's would give one physical
/// field a descriptor per nominal pool it can be spelled through, and a
/// per-object cache keyed on the descriptor then serves one pool's value out
/// of another's slot.
#[repr(C)]
pub struct ArrayBase {
    /// Base pointer of the element buffer. Null until the first push.
    pub data: *mut Val,
    /// The element count. `u32` rather than `usize` so the JIT's field descr
    /// is sub-word and `intbounds` can bound a load of it; without an upper
    /// bound a depth `+ 1` may overflow, the sum goes rangeless, and every
    /// re-check of the depth has to be guarded again.
    pub size: u32,
    /// Elements the buffer has room for. A push past it must reallocate, so
    /// the JIT guards `size < cap` and leaves growth to the interpreter.
    pub cap: u32,
}

impl ArrayBase {
    /// An empty pool: no buffer, so the first push allocates.
    pub const fn new() -> Self {
        ArrayBase {
            data: std::ptr::null_mut(),
            size: 0,
            cap: 0,
        }
    }
}

impl Default for ArrayBase {
    fn default() -> Self {
        Self::new()
    }
}

/// `linkedlist.py`'s `Stack` over a flat buffer.
///
/// The top is `data[size - 1]` — `array.py`'s right end, and the element
/// `linkedlist.py` keeps at `head`.
#[repr(C)]
pub struct Stack {
    pub base: ArrayBase,
}

impl Stack {
    pub fn new() -> Self {
        Stack {
            base: ArrayBase::new(),
        }
    }

    /// Ensure room for one more element, reallocating if full.
    ///
    /// Callers must re-read `self.base.data` afterwards: this may move the buffer.
    fn reserve_one(&mut self) {
        if self.base.size < self.base.cap {
            return;
        }
        let new_cap = if self.base.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.base.cap * 2
        };
        self.base.data = unsafe { grow_buffer(self.base.data, self.base.cap, new_cap, self.base.size) };
        self.base.cap = new_cap;
    }

    /// The top element without popping. Panics when empty.
    pub fn top(&self) -> Val {
        assert!(self.base.size > 0, "top of empty stack");
        unsafe { *self.base.data.add(self.base.size as usize - 1) }
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe { free_buffer(self.base.data, self.base.cap) };
    }
}

impl ArrayStorage for Stack {
    fn len(&self) -> usize {
        self.base.size as usize
    }

    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            unsafe {
                *self.base.data.add(self.base.size as usize) = value;
            }
            self.base.size += 1;
            maybe_collect_bigints();
        });
    }

    fn pop(&mut self) -> Val {
        assert!(self.base.size > 0, "pop from empty stack");
        self.base.size -= 1;
        unsafe { *self.base.data.add(self.base.size as usize) }
    }

    // `self.push(self.head.value)`.
    fn dup(&mut self) {
        let top = self.top();
        self.push(top);
    }

    // swap the two topmost values in place.
    fn swap(&mut self) {
        assert!(self.base.size >= 2, "swap on <2 elements");
        unsafe {
            let a = self.base.data.add(self.base.size as usize - 1);
            let b = self.base.data.add(self.base.size as usize - 2);
            std::ptr::swap(a, b);
        }
    }

    // `return self.pop(), self.head.value`.
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.top();
        (r1, r2)
    }

    // `self.head.value = value`, i.e. overwrite the top
    // rather than push.
    fn _put_value(&mut self, value: Val) {
        assert!(self.base.size > 0, "_put_value on empty stack");
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            unsafe {
                *self.base.data.add(self.base.size as usize - 1) = value;
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
// `linkedlist.py`); that is an artifact of the chain representation and
// has no counterpart here. Any invariant checker that budgets `size + 1` for
// the queue is checking the chain, not the semantics.

/// `linkedlist.py`'s `Queue` over a ring buffer.
#[repr(C)]
pub struct Queue {
    pub base: ArrayBase,
    /// Index of the front element; the ring spans `[front, front + size)`.
    pub front: u32,
}

impl Queue {
    pub fn new() -> Self {
        Queue {
            base: ArrayBase::new(),
            front: 0,
        }
    }

    fn slot(&self, offset: u32) -> *mut Val {
        debug_assert!(self.base.cap > 0);
        let idx = (self.front as usize + offset as usize) % self.base.cap as usize;
        unsafe { self.base.data.add(idx) }
    }

    /// Ensure room for one more element. On growth the ring is unrolled so
    /// `front` returns to 0, which keeps the copy a pair of contiguous runs.
    fn reserve_one(&mut self) {
        if self.base.size < self.base.cap {
            return;
        }
        let new_cap = if self.base.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.base.cap * 2
        };
        let fresh = unsafe { alloc_buffer(new_cap) };
        for i in 0..self.base.size {
            unsafe {
                *fresh.add(i as usize) = *self.slot(i);
            }
        }
        if !self.base.data.is_null() {
            let layout =
                std::alloc::Layout::array::<Val>(self.base.cap as usize).expect("storage buffer layout");
            unsafe { std::alloc::dealloc(self.base.data as *mut u8, layout) };
        }
        self.base.data = fresh;
        self.base.cap = new_cap;
        self.front = 0;
    }

    /// The front element without popping. Panics when empty.
    pub fn front_value(&self) -> Val {
        assert!(self.base.size > 0, "front of empty queue");
        unsafe { *self.slot(0) }
    }

    /// Insert at the FRONT — the shape `dup` needs.
    fn push_front(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            self.front = if self.front == 0 {
                self.base.cap - 1
            } else {
                self.front - 1
            };
            unsafe {
                *self.slot(0) = value;
            }
            self.base.size += 1;
            maybe_collect_bigints();
        });
    }
}

impl Default for Queue {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        unsafe { free_buffer(self.base.data, self.base.cap) };
    }
}

impl ArrayStorage for Queue {
    fn len(&self) -> usize {
        self.base.size as usize
    }

    // append at the back.
    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            let size = self.base.size;
            unsafe {
                *self.slot(size) = value;
            }
            self.base.size += 1;
            maybe_collect_bigints();
        });
    }

    // take from the front.
    fn pop(&mut self) -> Val {
        assert!(self.base.size > 0, "pop from empty queue");
        let value = unsafe { *self.slot(0) };
        self.front = (self.front + 1) % self.base.cap;
        self.base.size -= 1;
        value
    }

    // duplicate the FRONT and insert at the front.
    // NOT `array.py`, which duplicates the back.
    fn dup(&mut self) {
        let front = self.front_value();
        self.push_front(front);
    }

    // swap the two frontmost values in place.
    fn swap(&mut self) {
        assert!(self.base.size >= 2, "swap on <2 elements");
        unsafe {
            std::ptr::swap(self.slot(0), self.slot(1));
        }
    }

    // `return self.pop(), self.pop()`, both off the
    // front.
    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.pop();
        (r1, r2)
    }

    // `self.push(value)`, i.e. the arithmetic result
    // lands at the BACK, not where the operands were.
    fn _put_value(&mut self, value: Val) {
        self.push(value);
    }
}

// Port layout.

/// `linkedlist.py`'s `Port` — a stack plus the `last_push` shadow.
#[repr(C)]
pub struct Port {
    pub base: ArrayBase,
    pub last_push: Val,
}

impl Port {
    pub fn new() -> Self {
        Port {
            base: ArrayBase::new(),
            last_push: val_from_i32(0),
        }
    }

    fn reserve_one(&mut self) {
        if self.base.size < self.base.cap {
            return;
        }
        let new_cap = if self.base.cap == 0 {
            INITIAL_CAPACITY
        } else {
            self.base.cap * 2
        };
        self.base.data = unsafe { grow_buffer(self.base.data, self.base.cap, new_cap, self.base.size) };
        self.base.cap = new_cap;
    }

    pub fn top(&self) -> Val {
        assert!(self.base.size > 0, "top of empty port");
        unsafe { *self.base.data.add(self.base.size as usize - 1) }
    }
}

impl Default for Port {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Port {
    fn drop(&mut self) {
        unsafe { free_buffer(self.base.data, self.base.cap) };
    }
}

impl ArrayStorage for Port {
    fn len(&self) -> usize {
        self.base.size as usize
    }

    // push records `last_push`.
    fn push(&mut self, value: Val) {
        let mut root = value;
        with_bigint_transient_root(&mut root, || {
            self.reserve_one();
            unsafe {
                *self.base.data.add(self.base.size as usize) = value;
            }
            self.base.size += 1;
            self.last_push = value;
            maybe_collect_bigints();
        });
    }

    fn pop(&mut self) -> Val {
        assert!(self.base.size > 0, "pop from empty port");
        self.base.size -= 1;
        unsafe { *self.base.data.add(self.base.size as usize) }
    }

    // `self.push(self.last_push)`: the SHADOW, not the
    // top. After `push 5; push 10; add` the top is 15 but `last_push` is 10.
    fn dup(&mut self) {
        self.push(self.last_push);
    }

    fn swap(&mut self) {
        assert!(self.base.size >= 2, "swap on <2 elements");
        unsafe {
            let a = self.base.data.add(self.base.size as usize - 1);
            let b = self.base.data.add(self.base.size as usize - 2);
            std::ptr::swap(a, b);
        }
    }

    fn _get_2_values(&mut self) -> (Val, Val) {
        let r1 = self.pop();
        let r2 = self.top();
        (r1, r2)
    }

    // overwrites the top and deliberately leaves
    // `last_push` alone, which is what makes `dup` observably differ from
    // duplicating the top.
    fn _put_value(&mut self, value: Val) {
        assert!(self.base.size > 0, "_put_value on empty port");
        unsafe {
            *self.base.data.add(self.base.size as usize - 1) = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Behaviour shared with the linked list is pinned by
    // `backend_equivalence`, which drives both backends through the same
    // operation streams. What is left here is what only the array can get
    // wrong: the buffer, the ring, and the release path.

    /// The freed-buffer sentinel must land on every element.
    ///
    /// It is the only way a stale array-element read becomes visible: a base
    /// pointer cached across a realloc reads recycled memory that still holds
    /// the copied-out bytes, so the wrong read returns the right value.
    #[test]
    fn poison_buffer_stamps_every_element() {
        let cap = 4u32;
        unsafe {
            let data = alloc_buffer(cap);
            for i in 0..cap as usize {
                *data.add(i) = val_from_i32(7);
            }
            poison_buffer(data, cap);
            for i in 0..cap as usize {
                assert_eq!(
                    val_to_i64(&*data.add(i)),
                    0x5EEDDEA,
                    "element {i} of a released buffer was left unstamped"
                );
            }
            let layout =
                std::alloc::Layout::array::<Val>(cap as usize).expect("storage buffer layout");
            std::alloc::dealloc(data as *mut u8, layout);
        }
    }

    #[test]
    fn stack_survives_growth() {
        // Push past INITIAL_CAPACITY so the buffer reallocates at least twice,
        // then check every element survived the moves in order.
        let mut s = Stack::new();
        let n = (INITIAL_CAPACITY * 4) as i32;
        for i in 0..n {
            s.push(val_from_i32(i));
        }
        assert_eq!(s.base.size as i32, n);
        for i in (0..n).rev() {
            assert_eq!(val_to_i64(&s.pop()), i as i64);
        }
        assert_eq!(s.base.size, 0);
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
        assert_eq!(q.base.size, 0);
    }

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
