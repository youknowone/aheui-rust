mod common;

/// Dumps the generated Rust for a few large programs, for eyeballing codegen
/// changes. Sources that this checkout does not carry are skipped.
///
/// `#[ignore]` because this asserts nothing: it is a tool, and a run of it can
/// only ever pass. Ask for it by name —
/// `cargo test -p compaheuiler --test codegen_debug -- --ignored` — and read
/// what it leaves in `target/codegen/`.
#[test]
#[ignore = "writes files to look at; gates nothing"]
fn gen_all() {
    let out_dir = common::codegen_dir();

    for (name, rel) in [
        ("99b", "99bottles/99bottles.aheui"),
        ("logo", "logo/logo.aheui"),
    ] {
        let Some(src) = common::read_snippet(rel) else {
            common::skip("gen_all", rel);
            continue;
        };
        std::fs::write(
            out_dir.join(format!("aheui_{name}.rs")),
            compaheuiler::compile_to_rs(&src),
        )
        .unwrap();
    }
    match common::read_self_interp() {
        Some(src) => std::fs::write(
            out_dir.join("aheui_self.rs"),
            compaheuiler::compile_to_rs(&src),
        )
        .unwrap(),
        None => common::skip("gen_all", "aheui.aheui (set AHEUI_SELF_INTERP)"),
    }
    eprintln!("generated Rust written to {}", out_dir.display());
}
