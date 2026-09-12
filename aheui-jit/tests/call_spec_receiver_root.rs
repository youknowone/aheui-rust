//! `AHEUI_CALL_OWNER_ROOT` must name a trait that exists.
//!
//! `build.rs` turns every `CallTargetSpec::Method` into
//! `majit_translate::CallTarget::method(name, Some(receiver_root))`, where
//! `receiver_root` is the plain string `AHEUI_CALL_OWNER_ROOT`. Nothing links
//! that string to the trait it names: rename the trait, or point `Storage`'s
//! dispatch at a different one, and every effect override goes on being
//! constructed, matches nothing, and takes no effect. There is no error, and
//! the only symptom is that calls the codewriter should have classified
//! `Elidable`/`Residual` fall back to whatever the analyzer infers.
//!
//! Checked against the source text rather than the type system because
//! `call_spec` is a data-only module shared with `build.rs` and cannot depend
//! on `aheui-runtime`.

use std::path::{Path, PathBuf};

/// `<repo>/aheui/aheui-jit` -> `<repo>/aheui`
fn aheui_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent")
        .to_path_buf()
}

/// A missing file must FAIL, never skip. A test whose job is to compare two
/// files and which passes when it cannot find them is the failure mode it
/// exists to prevent, one level up.
fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {} — {e}", path.display()))
}

/// Every `pub trait NAME` declared under `aheui-runtime/src/storage/`.
fn storage_traits() -> Vec<String> {
    let dir = aheui_root().join("aheui-runtime/src/storage");
    let mut found = Vec::new();
    for entry in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot list {} — {e}", dir.display()))
        .flatten()
    {
        let path = entry.path();
        if path.extension().is_none_or(|ext| ext != "rs") {
            continue;
        }
        for line in read(&path).lines() {
            let Some(rest) = line.trim().strip_prefix("pub trait ") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                found.push(name);
            }
        }
    }
    found
}

#[test]
fn the_receiver_root_names_a_storage_trait_that_exists() {
    let root = aheui_jit::jit::call_spec::AHEUI_CALL_OWNER_ROOT;
    let traits = storage_traits();
    assert!(
        !traits.is_empty(),
        "test setup: found no `pub trait` under aheui-runtime/src/storage, so \
         this test would pass for the wrong reason"
    );
    assert!(
        traits.iter().any(|t| t == root),
        "AHEUI_CALL_OWNER_ROOT is {root:?}, which is not a trait declared under \
         aheui-runtime/src/storage (found {traits:?}). `build.rs` hands this \
         string to `CallTarget::method` as the receiver root; a root that \
         matches no trait makes all {} effect overrides no-ops, with no error \
         and no symptom other than ten calls being classified by inference \
         instead of by this table",
        aheui_jit::jit::call_spec::AHEUI_CALL_EFFECTS.len(),
    );
}

/// …and the root has to be the trait `Storage` actually dispatches through,
/// not merely one that exists. Both storage backends declare a trait, so the
/// check above stays green while the overrides are keyed to whichever backend
/// is not the one in use.
#[test]
fn the_receiver_root_is_the_trait_storage_dispatches_through() {
    let root = aheui_jit::jit::call_spec::AHEUI_CALL_OWNER_ROOT;
    let src = read(&aheui_root().join("aheui-runtime/src/storage/mod.rs"));
    let dispatch = src
        .lines()
        .find(|line| line.contains("pub fn dispatch_mut"))
        .expect("storage/mod.rs declares `pub fn dispatch_mut`");
    assert!(
        dispatch.contains(root),
        "AHEUI_CALL_OWNER_ROOT is {root:?} but `Storage::dispatch_mut` returns \
         `{}`. The overrides are keyed to a trait the interpreter no longer \
         dispatches through, so they match nothing",
        dispatch.trim(),
    );
}
