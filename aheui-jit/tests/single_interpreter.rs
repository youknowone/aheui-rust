//! Disabled and enabled JIT runs share the interpreter's state and dispatch.
mod common;

#[test]
fn ordinary_execution_then_tracing_uses_the_same_portal() {
    let program = common::drain_loop(
        common::ITERATIONS,
        &[(ahsembler::consts::OP_POP, common::NONE)],
    );
    let ordinary = aheuinterpreter::interp::mainloop(&program);
    let stats = aheuinterpreter::last_jit_stats().unwrap();
    assert_eq!(stats.loops_compiled, 0);

    let traced = aheui_jit::mainloop(&program, common::THRESHOLD);
    assert_eq!(ordinary, traced);
    common::assert_compiled("canonical interpreter after an ordinary run");
}
