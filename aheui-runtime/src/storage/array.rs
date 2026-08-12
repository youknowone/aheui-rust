//! Array-backed storage. Parity stub for `rpaheui/aheui/storage/array.py`.
//!
//! In rpaheui, `array.py` is the CPython-only fast-path (`Stack(list)` /
//! `Queue(deque)`), while `linkedlist.py` is the RPython path using
//! linked-list `Node` objects.
//!
//! In our port those nodes escape into `Stack.head`, so they are not
//! virtualized; whether to move to array-backed storage remains an open
//! question. This module exists so the directory layout mirrors rpaheui;
//! no types are exported yet.
