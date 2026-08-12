mod common;

use std::process::Command;
use std::time::Instant;

#[test]
fn test_quine40() {
    let source = common::require_snippet("quine/quine.puzzlet.40col.aheui");

    // Generate Rust code
    let rs_code = compaheuiler::compile_to_rs(&source);
    let rs_path = "/tmp/aheui_quine40.rs";
    let bin_path = "/tmp/aheui_quine40";
    std::fs::write(rs_path, &rs_code).unwrap();
    eprintln!("Generated {} lines", rs_code.lines().count());

    // Compile
    let t = Instant::now();
    let status = Command::new("rustc")
        .args([
            "-C",
            "opt-level=3",
            "-C",
            "target-cpu=native",
            "-o",
            bin_path,
            rs_path,
        ])
        .status()
        .unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(status.success(), "rustc failed");
    eprintln!("rustc: {compile_ms:.0}ms");

    // Run
    let t = Instant::now();
    let output = Command::new(bin_path)
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    let run_ms = t.elapsed().as_secs_f64() * 1000.0;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    eprintln!("run: {run_ms:.0}ms  output={}bytes", stdout.len());

    // Quine check: output should equal source. The file on disk ends with a
    // newline the program itself never emits, so allow that one byte — the
    // same tolerance `all_snippets_test` applies to the `.out` references.
    assert!(
        stdout == source || format!("{stdout}\n") == source,
        "Not a quine! Output differs from source ({} vs {} bytes)",
        stdout.len(),
        source.len(),
    );
    eprintln!("✓ Quine verified!");
}
