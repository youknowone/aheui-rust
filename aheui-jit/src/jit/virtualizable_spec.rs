//! Canonical virtualizable field/array specification for `Storage`.
//!
//! Data-only module shared by runtime code and `build.rs`.

pub const AHEUI_VABLE_OWNER_ROOT: &str = "Storage";

/// No scalar virtualizable fields.
/// `Storage` is accessed via dynamic indexing (`dispatch_mut(selected)`),
/// which the graph pipeline classifies via call_effects instead.
pub const AHEUI_VABLE_FIELDS: &[(&str, usize)] = &[];

/// No virtualizable arrays at graph pipeline level.
/// Storage stacks are managed through the `#[jit_interp]` macro,
/// not the graph pipeline's vable rewrite.
pub const AHEUI_VABLE_ARRAYS: &[(&str, usize)] = &[];
