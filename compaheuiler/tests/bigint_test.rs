#![cfg(feature = "bigint")]

mod common;

use std::process::Command;
use std::time::Instant;

/// Every case builds through the same scratch cargo project, so only one may
/// hold it at a time — the test harness runs cases in parallel, and two of
/// them writing `src/main.rs` interleaved leaves each running the other's
/// binary. Same reason `aheui-runtime` serialises its nursery tests.
static PROJ: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn compile_and_run_bigint(source: &str, stdin_data: &str) -> (String, i32, f64, f64) {
    compile_and_run_bigint_with_opt(source, stdin_data, ahsembler::OptimizationLevel::O3)
}

fn compile_and_run_bigint_with_opt(
    source: &str,
    stdin_data: &str,
    opt: ahsembler::OptimizationLevel,
) -> (String, i32, f64, f64) {
    let _guard = PROJ.lock().unwrap_or_else(|e| e.into_inner());
    let rs = compaheuiler::compile_to_rs_bigint_opt(source, opt);

    // Create temp cargo project
    let dir = "/tmp/aheui_bigint_proj";
    std::fs::create_dir_all(format!("{dir}/src")).ok();
    std::fs::write(format!("{dir}/src/main.rs"), &rs).unwrap();
    #[cfg(feature = "num-bigint")]
    let bigint_dep = r#"num-bigint = "0.4""#;
    #[cfg(not(feature = "num-bigint"))]
    let bigint_dep = r#"malachite-bigint = "0.9""#;
    std::fs::write(
        format!("{dir}/Cargo.toml"),
        format!(
            r#"
[package]
name = "aheui-bigint-test"
version = "0.0.1"
edition = "2021"

[dependencies]
{bigint_dep}
num-traits = "0.2"

[profile.release]
opt-level = 2
"#
        ),
    )
    .unwrap();

    let t = Instant::now();
    let status = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(dir)
        .status()
        .expect("cargo build failed");
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(status.success(), "bigint compilation failed");

    let bin = format!("{dir}/target/release/aheui-bigint-test");
    let t = Instant::now();
    let output = if stdin_data.is_empty() {
        Command::new(&bin).output().expect("execution failed")
    } else {
        use std::io::Write;
        let mut child = Command::new(&bin)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(stdin_data.as_bytes())
            .unwrap();
        child.wait_with_output().unwrap()
    };
    let run_ms = t.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (
        stdout,
        output.status.code().unwrap_or(-1),
        compile_ms,
        run_ms,
    )
}

#[test]
fn test_special_storage_promotion_updates_live_registers() {
    // Keep 7 and 8 in storage 0's generated locals, overflow 2^32 * 2^32 on
    // the queue, then consume those locals after the representation changes.
    let source = "밝밣상박빠따빠따빠따빠따빠따빠따사다망하";
    let (out, exit, _, _) =
        compile_and_run_bigint_with_opt(source, "", ahsembler::OptimizationLevel::O0);
    assert_eq!(out, "15");
    assert_eq!(exit, 0);
}

#[test]
fn test_halt_reads_special_storage() {
    let (out, exit, _, _) =
        compile_and_run_bigint_with_opt("상밝하", "", ahsembler::OptimizationLevel::O0);
    assert_eq!(out, "");
    assert_eq!(exit, 7);
}

#[test]
fn test_read_num_boxes_large_i64_after_promotion() {
    // Promote with 2^64, discard that value, then read an i64 which is outside
    // the tagged immediate range. It must be boxed rather than shifted.
    let source = "반빠따빠따빠따빠따빠따빠따마방망하";
    let (out, exit, _, _) = compile_and_run_bigint_with_opt(
        source,
        "5000000000000000000\n",
        ahsembler::OptimizationLevel::O0,
    );
    assert_eq!(out, "5000000000000000000");
    assert_eq!(exit, 0);
}

#[test]
fn test_add_bigint() {
    let (out, _exit, compile_ms, run_ms) = compile_and_run_bigint("반받다망희", "");
    eprintln!("bigint add: compile={compile_ms:.0}ms run={run_ms:.0}ms out={out:?}");
    assert_eq!(out, "5");
}

#[test]
fn test_post_promotion_ops_bigint() {
    let (out, exit, _, _) = compile_and_run_bigint(common::DUAL_MODE_POST_PROMOTION_OPS, "");
    assert_eq!(exit, 0);
    assert_eq!(out, common::DUAL_MODE_POST_PROMOTION_OUTPUT);
}

#[test]
fn test_loop_carried_storage_survives_bigint_promotion() {
    let source = common::require_snippet("factorial/factorial.aheui");
    let (out, exit, _, _) = compile_and_run_bigint(&source, "21\n");
    assert_eq!(exit, 0);
    assert_eq!(out, "51090942171709440000");
}

#[test]
fn test_logo_bigint() {
    let src = common::require_snippet("logo/logo.aheui");
    let (out, _exit, compile_ms, run_ms) = compile_and_run_bigint(&src, "");
    eprintln!(
        "bigint logo: compile={compile_ms:.0}ms run={run_ms:.0}ms output={}bytes",
        out.len()
    );
    assert_eq!(out.len(), 996310);
}

#[test]
fn test_2e65_bigint() {
    let (Some(src), Some(expected)) = (
        common::read_snippet("integer/2e65-print.aheui"),
        common::read_snippet("integer/2e65-print.out"),
    ) else {
        common::skip("test_2e65_bigint", "integer/2e65-print.*");
        return;
    };
    let (out, _exit, compile_ms, run_ms) = compile_and_run_bigint(&src, "");
    eprintln!(
        "bigint 2e65: compile={compile_ms:.0}ms run={run_ms:.0}ms out={:?}",
        &out[..out.len().min(30)]
    );
    // Allow trailing newline difference
    assert!(
        out == expected || format!("{out}\n") == expected,
        "2e65 output mismatch: got {:?}",
        &out[..out.len().min(30)]
    );
}

#[test]
fn test_2e65_noprint_bigint() {
    let Some(src) = common::read_snippet("integer/2e65.aheui") else {
        common::skip("test_2e65_noprint_bigint", "integer/2e65.aheui");
        return;
    };
    let rs = compaheuiler::compile_to_rs_bigint(&src);
    std::fs::write("/tmp/aheui_2e65_gen.rs", &rs).unwrap();
    eprintln!("Generated {} bytes to /tmp/aheui_2e65_gen.rs", rs.len());

    let (out, exit, compile_ms, run_ms) = compile_and_run_bigint(&src, "");
    eprintln!(
        "bigint 2e65 (noprint): compile={compile_ms:.0}ms run={run_ms:.0}ms exit={exit} out={out:?}"
    );
}
