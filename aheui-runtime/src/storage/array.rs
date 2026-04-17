//! Array-backed storage. Parity stub for `rpaheui/aheui/storage/array.py`.
//!
//! In rpaheui this is the CPython-only fast-path (`Stack(list)` /
//! `Queue(deque)`); RPython always uses `linkedlist.py` because the
//! JIT drives allocation through `Node` objects that `OptVirtualize`
//! can eliminate.
//!
//! Our port follows the RPython path — the JIT tracer expects
//! linked-list `Node` allocations it can virtualize. The module
//! exists so the directory layout mirrors rpaheui; no types are
//! exported yet.
