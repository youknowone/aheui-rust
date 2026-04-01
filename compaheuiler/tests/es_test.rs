#[test]
fn gen_es() {
    let code = compaheuiler::compile_to_rs("뱐희파반망희");
    std::fs::write("/tmp/aheui_emptyswap.rs", &code).unwrap();
}
