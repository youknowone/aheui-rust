#![cfg(feature = "bigint")]

use std::process::Command;
use std::time::Instant;

fn compile_and_run_bigint(source: &str, stdin_data: &str) -> (String, i32, f64, f64) {
    let rs = compaheuiler::compile_to_rs_bigint(source);

    // Create temp cargo project
    let dir = "/tmp/aheui_bigint_proj";
    std::fs::create_dir_all(&format!("{dir}/src")).ok();
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
fn test_add_bigint() {
    let (out, _exit, compile_ms, run_ms) = compile_and_run_bigint("반받다망희", "");
    eprintln!("bigint add: compile={compile_ms:.0}ms run={run_ms:.0}ms out={out:?}");
    assert_eq!(out, "5");
}

#[test]
fn test_logo_bigint() {
    let src =
        std::fs::read_to_string("/Users/al03219714/Projects/pypy/rpaheui/snippets/logo/logo.aheui")
            .unwrap();
    let (out, _exit, compile_ms, run_ms) = compile_and_run_bigint(&src, "");
    eprintln!(
        "bigint logo: compile={compile_ms:.0}ms run={run_ms:.0}ms output={}bytes",
        out.len()
    );
    assert_eq!(out.len(), 996310);
}

#[test]
fn test_2e65_bigint() {
    let src = std::fs::read_to_string(
        "/Users/al03219714/Projects/pypy/rpaheui/snippets/integer/2e65-print.aheui",
    )
    .unwrap();
    let expected = std::fs::read_to_string(
        "/Users/al03219714/Projects/pypy/rpaheui/snippets/integer/2e65-print.out",
    )
    .unwrap();
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
    let src = std::fs::read_to_string(
        "/Users/al03219714/Projects/pypy/rpaheui/snippets/integer/2e65.aheui",
    )
    .unwrap();
    let rs = compaheuiler::compile_to_rs_bigint(&src);
    std::fs::write("/tmp/aheui_2e65_gen.rs", &rs).unwrap();
    eprintln!("Generated {} bytes to /tmp/aheui_2e65_gen.rs", rs.len());

    let (out, exit, compile_ms, run_ms) = compile_and_run_bigint(&src, "");
    eprintln!(
        "bigint 2e65 (noprint): compile={compile_ms:.0}ms run={run_ms:.0}ms exit={exit} out={out:?}"
    );
}
