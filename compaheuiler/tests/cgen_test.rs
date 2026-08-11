mod common;

use std::process::Command;
use std::time::Instant;

fn compile_and_run_c(name: &str, source: &str) -> (String, f64, f64) {
    let c_code = compaheuiler::compile_to_c(source);
    compile_and_run_c_code(name, &c_code)
}

fn compile_and_run_c_code(name: &str, c_code: &str) -> (String, f64, f64) {
    let c_path = format!("/tmp/aheui_{name}.c");
    let bin_path = format!("/tmp/aheui_{name}_c");
    std::fs::write(&c_path, c_code).unwrap();

    let t = Instant::now();
    let status = Command::new("cc")
        .args(["-O2", "-std=c99", "-o", &bin_path, &c_path])
        .status()
        .unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(status.success(), "cc failed for {name}");

    let t = Instant::now();
    let output = Command::new(&bin_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let run_ms = t.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (stdout, compile_ms, run_ms)
}

fn compile_and_run_c_bigint_code(name: &str, c_code: &str) -> (String, f64, f64) {
    let scratch = common::scratch_dir(name);
    let t = Instant::now();
    let bin_path = common::compile_c_bigint(scratch.path(), name, c_code);
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    let output = Command::new(bin_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let run_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(output.status.success(), "linked C/Rust program failed");
    (
        String::from_utf8_lossy(&output.stdout).to_string(),
        compile_ms,
        run_ms,
    )
}

#[cfg(feature = "bigint")]
#[test]
fn test_2e65_c_bigint() {
    let source = common::require_snippet("integer/2e65-print.aheui");
    let expected = common::require_snippet("integer/2e65-print.out");
    let c_code = compaheuiler::compile_to_c_bigint(&source);
    let (out, compile_ms, run_ms) = compile_and_run_c_bigint_code("2e65_bigint", &c_code);
    eprintln!("2e65(c bigint): cc={compile_ms:.0}ms run={run_ms:.0}ms out={out:?}");
    assert_eq!(out.trim_end(), expected.trim_end());
}

#[cfg(feature = "bigint")]
#[test]
fn test_post_promotion_ops_c_bigint() {
    let c_code = compaheuiler::compile_to_c_bigint(common::DUAL_MODE_POST_PROMOTION_OPS);
    let (out, _, _) = compile_and_run_c_bigint_code("post_promotion_bigint", &c_code);
    assert_eq!(out, common::DUAL_MODE_POST_PROMOTION_OUTPUT);
}

#[test]
fn test_add_c() {
    let (out, compile_ms, _) = compile_and_run_c("add", "반받다망희");
    eprintln!("add(c): cc={compile_ms:.0}ms out={out:?}");
    assert_eq!(out, "5");
}

#[test]
fn test_hello_c() {
    let source = common::require_snippet("hello-world/hello-world.puzzlet.aheui");
    let (out, compile_ms, run_ms) = compile_and_run_c("hello", &source);
    eprintln!("hello(c): cc={compile_ms:.0}ms  run={run_ms:.0}ms  out={out:?}");
    let expected = common::require_snippet("hello-world/hello-world.puzzlet.out");
    assert_eq!(out, expected);
}

#[test]
fn test_logo_c() {
    let source = common::require_snippet("logo/logo.aheui");
    let (out, compile_ms, run_ms) = compile_and_run_c("logo", &source);
    eprintln!(
        "logo(c): cc={compile_ms:.0}ms  run={run_ms:.0}ms  output={}bytes",
        out.len()
    );
    assert!(
        out.starts_with("P1 615 810"),
        "output starts with: {:?}",
        &out[..20.min(out.len())]
    );
}
