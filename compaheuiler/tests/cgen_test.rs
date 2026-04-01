use std::process::Command;
use std::time::Instant;

fn compile_and_run_c(name: &str, source: &str) -> (String, f64, f64) {
    let c_code = compaheuiler::compile_to_c(source);
    let c_path = format!("/tmp/aheui_{name}.c");
    let bin_path = format!("/tmp/aheui_{name}_c");
    std::fs::write(&c_path, &c_code).unwrap();

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

#[test]
fn test_add_c() {
    let (out, compile_ms, _) = compile_and_run_c("add", "반받다망희");
    eprintln!("add(c): cc={compile_ms:.0}ms out={out:?}");
    assert_eq!(out, "5");
}

#[test]
fn test_hello_c() {
    let source = std::fs::read_to_string(
        "/Users/al03219714/Projects/pypy-compaheuiler/aheui/tests/hello-world/hello-world.puzzlet.aheui"
    ).unwrap();
    let (out, compile_ms, run_ms) = compile_and_run_c("hello", &source);
    eprintln!("hello(c): cc={compile_ms:.0}ms  run={run_ms:.0}ms  out={out:?}");
    let expected = std::fs::read_to_string(
        "/Users/al03219714/Projects/pypy-compaheuiler/aheui/tests/hello-world/hello-world.puzzlet.out"
    ).unwrap();
    assert_eq!(out, expected);
}

#[test]
fn test_logo_c() {
    let source =
        std::fs::read_to_string("/Users/al03219714/Projects/pypy/rpaheui/snippets/logo/logo.aheui")
            .unwrap();
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
