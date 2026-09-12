/// `파` (swap) on a storage the program never pushed to.
///
/// The generated Rust is checked for existing, not written out: codegen for a
/// swap with nothing under it used to be the interesting case, and what makes
/// it interesting is whether `compile_to_rs` returns at all.
#[test]
fn empty_swap_generates_a_program() {
    let code = compaheuiler::compile_to_rs("뱐희파반망희");
    assert!(
        code.contains("fn main("),
        "codegen produced no program:\n{code}"
    );
}
