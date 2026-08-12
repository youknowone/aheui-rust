//! Fixture-path resolution shared by the integration tests.
//!
//! The corpus is `github.com/aheui/snippets`. Resolution matches `check.sh`:
//! `$AHEUI_SNIPPETS` first, then this repo's tracked `tests -> snippets` link,
//! then a sibling `rpaheui/snippets` checkout. The latter is whatever that
//! machine last pulled, so it is not interchangeable with a pinned checkout.
//!
//! The `aheui.aheui` self-interpreter is a separate project and is in none of
//! them, so its caller gets an `Option` and skips loudly rather than failing.

/// Computes 2^64 three times, then exercises division, remainder, and compare
/// after the first multiplication has promoted the execution to bigint mode.
pub const DUAL_MODE_POST_PROMOTION_OPS: &str = "반빠따빠따빠따빠따빠따빠따받나망반빠따빠따빠따빠따빠따빠따받라망반빠따빠따빠따빠따빠따빠따받자망희";
pub const DUAL_MODE_POST_PROMOTION_OUTPUT: &str = "614891469123651720511";

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct ScratchDir(PathBuf);

impl ScratchDir {
    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Creates a process-unique scratch directory and removes it after the test.
pub fn scratch_dir(label: &str) -> ScratchDir {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "compaheuiler_{label}_{}_{}",
            std::process::id(),
            serial
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => return ScratchDir(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => panic!("cannot create scratch directory {}: {e}", path.display()),
        }
    }
}

/// Compile generated dual-mode C together with its Rust BigInt bridge.
pub fn compile_c_bigint(scratch: &Path, name: &str, c_code: &str) -> PathBuf {
    let dir = scratch.join(name);
    let package = format!("compaheuiler-c-test-{name}");
    let target_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target/c-bigint-tests");
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let c_path = dir.join("program.c");
    let object_path = dir.join("program.o");
    std::fs::write(&c_path, c_code).unwrap();
    std::fs::write(dir.join("src/main.rs"), compaheuiler::c_bigint_bridge_rs()).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = {package:?}
version = "0.0.0"
edition = "2024"
rust-version = "1.85"

[dependencies]
{}

[profile.release]
opt-level = 1
"#,
            compaheuiler::c_bigint_dependency_toml()
        ),
    )
    .unwrap();
    let output = Command::new("cc")
        .args(["-O1", "-std=c99", "-DCOMPAHEUILER_RUST_BIGINT", "-c"])
        .arg(&c_path)
        .args(["-o"])
        .arg(&object_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "generated C failed to compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::write(
        dir.join("build.rs"),
        format!(
            "fn main() {{\n  let object = {object:?};\n  println!(\"cargo:rerun-if-changed={{object}}\");\n  println!(\"cargo:rustc-link-arg-bin={package}={{object}}\");\n}}\n",
            object = object_path.to_string_lossy(),
        ),
    )
    .unwrap();
    let output = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", &target_dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "Rust BigInt bridge failed to link: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    target_dir.join("release").join(package)
}

/// Root of the snippet corpus.
///
/// Panics when it cannot be found: an unresolvable root is a misconfigured
/// checkout, not a missing optional fixture.
pub fn snippets_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        if let Some(p) = std::env::var_os("AHEUI_SNIPPETS") {
            return PathBuf::from(p);
        }
        // The tracked `tests -> snippets` link, found by walking up so this
        // works from any crate. `is_dir` also rejects an uninitialized target.
        let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
        while let Some(d) = dir {
            let candidate = d.join("tests");
            if candidate.is_dir() {
                return candidate;
            }
            dir = d.parent();
        }
        let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
        while let Some(d) = dir {
            let candidate = d.join("rpaheui").join("snippets");
            if candidate.is_dir() {
                return candidate;
            }
            dir = d.parent();
        }
        panic!(
            "snippet corpus not found in or above {}; \
             run 'git submodule update --init' or set AHEUI_SNIPPETS",
            env!("CARGO_MANIFEST_DIR"),
        )
    })
}

/// `<snippets>/<rel>`, without checking that it exists.
pub fn snippet_path(rel: &str) -> PathBuf {
    snippets_dir().join(rel)
}

/// Read a snippet, or `None` when this corpus does not carry it.
pub fn read_snippet(rel: &str) -> Option<String> {
    std::fs::read_to_string(snippet_path(rel)).ok()
}

/// Read a snippet that every corpus is expected to have.
pub fn require_snippet(rel: &str) -> String {
    let path = snippet_path(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read snippet {}: {e}", path.display()))
}

/// The `aheui.aheui` self-interpreter source, which is a separate project and
/// is in no snippet corpus. `AHEUI_SELF_INTERP` names it; there is no default.
pub fn read_self_interp() -> Option<String> {
    std::fs::read_to_string(std::env::var_os("AHEUI_SELF_INTERP")?).ok()
}

/// Report a fixture this checkout does not have. Prints rather than fails —
/// but prints, so the gap stays visible in the test log instead of reading as
/// coverage.
pub fn skip(test: &str, fixture: &str) {
    eprintln!("SKIP {test}: fixture {fixture} not available in this checkout");
}
