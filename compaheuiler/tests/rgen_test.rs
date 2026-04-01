use std::process::Command;
use std::time::Instant;

fn compile_and_run_rs(name: &str, source: &str) -> (String, f64, f64) {
    let rs_code = compaheuiler::compile_to_rs(source);
    let rs_path = format!("/tmp/aheui_{name}.rs");
    let bin_path = format!("/tmp/aheui_{name}_rs");
    std::fs::write(&rs_path, &rs_code).unwrap();

    let t = Instant::now();
    let status = Command::new("rustc")
        .args([
            "-C",
            "opt-level=3",
            "-C",
            "target-cpu=native",
            "-o",
            &bin_path,
            &rs_path,
        ])
        .status()
        .unwrap();
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;
    assert!(status.success(), "rustc failed for {name}");

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

#[test]
fn test_add_rs() {
    let (out, compile_ms, _) = compile_and_run_rs("add", "반받다망희");
    eprintln!("add: rustc={compile_ms:.0}ms out={out:?}");
    assert_eq!(out, "5");
}

#[test]
fn test_logo_rs() {
    let source =
        std::fs::read_to_string("snippets/logo/logo.aheui")
            .unwrap();
    let (out, compile_ms, run_ms) = compile_and_run_rs("logo", &source);
    eprintln!(
        "logo(rust): rustc={compile_ms:.0}ms  run={run_ms:.0}ms  output={}bytes",
        out.len()
    );
    assert!(
        out.starts_with("P1 615 810"),
        "output starts with: {:?}",
        &out[..20.min(out.len())]
    );
}

fn compile_and_run_rs_stdin(name: &str, source: &str, stdin: &[u8]) -> (String, i32, f64) {
    let rs_code = compaheuiler::compile_to_rs(source);
    let rs_path = format!("/tmp/aheui_{name}.rs");
    let bin_path = format!("/tmp/aheui_{name}_rs");
    std::fs::write(&rs_path, &rs_code).unwrap();
    let status = Command::new("rustc")
        .args(["-C", "opt-level=2", "-o", &bin_path, &rs_path])
        .stderr(std::process::Stdio::piped())
        .status()
        .unwrap();
    assert!(status.success(), "rustc failed for {name}");
    let t = Instant::now();
    let mut child = Command::new(&bin_path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.take().unwrap().write_all(stdin).ok();
    let out = child.wait_with_output().unwrap();
    let run_ms = t.elapsed().as_secs_f64() * 1000.0;
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        out.status.code().unwrap_or(-1),
        run_ms,
    )
}

#[test]
fn test_99bottles_rs() {
    let base = "tests";
    let source = std::fs::read_to_string(format!("{base}/99bottles/99bottles.aheui")).unwrap();
    let expected = std::fs::read_to_string(format!("{base}/99bottles/99bottles.out")).unwrap();
    let (out, _, run_ms) = compile_and_run_rs_stdin("99bottles", &source, b"");
    eprintln!("99bottles: {run_ms:.0}ms  {} bytes", out.len());
    assert_eq!(out, expected, "99bottles output mismatch");
}

#[test]
fn test_aheui_self_interp() {
    let aheui_src = std::fs::read_to_string(
        "tests/aheui.aheui",
    )
    .unwrap();
    let test_prog = "밝희";
    let (out, exit, run_ms) =
        compile_and_run_rs_stdin("aheui_self", &aheui_src, test_prog.as_bytes());
    eprintln!("aheui.aheui(밝희): {run_ms:.0}ms  out={out:?}  exit={exit}");
    // Reference interpreter also gives exit=7 for aheui.aheui(밝희)
    assert_eq!(exit, 7, "aheui.aheui(밝희) exit mismatch");

    // Also test with 반희 (push 2, halt → exit=2 via self-interp)
    let (out2, exit2, run_ms2) =
        compile_and_run_rs_stdin("aheui_self2", &aheui_src, "반희".as_bytes());
    eprintln!("aheui.aheui(반희): {run_ms2:.0}ms  out={out2:?}  exit={exit2}");
    assert_eq!(exit2, 2, "aheui.aheui(반희) exit mismatch");
}
