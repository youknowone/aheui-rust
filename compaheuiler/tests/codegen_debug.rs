#[test]
fn gen_all() {
    let base = "/Users/al03219714/Projects/pypy-compaheuiler/aheui/tests";
    for (n, p) in [
        ("99b", "99bottles/99bottles.aheui"),
        ("self", "aheui.aheui"),
        ("logo", "logo/logo.aheui"),
    ] {
        let src = std::fs::read_to_string(format!("{base}/{p}")).unwrap();
        std::fs::write(
            format!("/tmp/aheui_{n}.rs"),
            compaheuiler::compile_to_rs(&src),
        )
        .unwrap();
    }
}
