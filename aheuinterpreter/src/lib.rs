//! Pure Aheui interpreter (no JIT).
//!
//! Thin crate around [`aheui_runtime`]: re-exports runtime modules so
//! `interp.rs` can reference them via `crate::*`, matching the flat
//! layout of rpaheui (`aheui/aheui.py`).
pub mod interp;

pub use aheui_runtime as runtime;
pub use aheui_runtime::Val;
pub use aheui_runtime::aheui;
pub use aheui_runtime::ahsembler;
pub use aheui_runtime::io;
pub use aheui_runtime::storage;
pub use aheui_runtime::value;
