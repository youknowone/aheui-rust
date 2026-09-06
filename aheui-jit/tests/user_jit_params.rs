#[test]
fn cli_uses_rpython_parameters_and_validates_aheui_stack_cap() {
    for text in [
        "threshold=10,stack_cap=8,vec=1",
        "enable_opts=",
        "off",
        "default",
    ] {
        assert!(aheui_jit::set_user_jit_params(text).is_ok(), "{text:?}");
    }
    for text in [
        "vectorize=1",
        "max_inline_depth=7",
        "stack_cap=3",
        "stack_cap=1",
        "threshold=1,",
        "threshold =1",
        "enable_opts=a=b",
        "stack_cap=8,off",
    ] {
        assert!(aheui_jit::set_user_jit_params(text).is_err(), "{text:?}");
    }
}
