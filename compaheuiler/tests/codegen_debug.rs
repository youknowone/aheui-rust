mod common;

/// Dumps the generated Rust for a few large programs to `/tmp`, for eyeballing
/// codegen changes. Sources that this checkout does not carry are skipped.
#[test]
fn gen_all() {
    for (n, rel) in [
        ("99b", "99bottles/99bottles.aheui"),
        ("logo", "logo/logo.aheui"),
    ] {
        let Some(src) = common::read_snippet(rel) else {
            common::skip("gen_all", rel);
            continue;
        };
        std::fs::write(
            format!("/tmp/aheui_{n}.rs"),
            compaheuiler::compile_to_rs(&src),
        )
        .unwrap();
    }
    match common::read_self_interp() {
        Some(src) => {
            std::fs::write("/tmp/aheui_self.rs", compaheuiler::compile_to_rs(&src)).unwrap()
        }
        None => common::skip("gen_all", "aheui.aheui (set AHEUI_SELF_INTERP)"),
    }
}
