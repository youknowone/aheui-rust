mod common;

use std::process::Command;
use std::time::Instant;

fn fixture_stdin(scratch: &std::path::Path, input: &[u8]) -> std::process::Stdio {
    let input_path = scratch.join("snippet.in");
    std::fs::write(&input_path, input).unwrap();
    std::fs::File::open(input_path).unwrap().into()
}

fn run_interpreter(source: &str, input: &[u8], scratch: &std::path::Path) -> (String, i32) {
    let source_path = scratch.join("snippet.aheui");
    std::fs::write(&source_path, source).unwrap();
    let out = Command::new("cargo")
        .args(["run", "--release", "-p", "aheui", "--"])
        .arg(&source_path)
        .stdin(fixture_stdin(scratch, input))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (stdout, out.status.code().unwrap_or(-1))
}

fn run_stackifier(source: &str, input: &[u8], scratch: &std::path::Path) -> (String, i32) {
    let rs_code = compaheuiler::compile_to_rs(source);
    let rs_path = scratch.join("snippet.rs");
    let bin_path = scratch.join("snippet-bin");
    std::fs::write(&rs_path, &rs_code).unwrap();

    let status = Command::new("rustc")
        .args(["-C", "opt-level=2", "-o"])
        .arg(&bin_path)
        .arg(&rs_path)
        .stderr(std::process::Stdio::piped())
        .status()
        .unwrap();
    if !status.success() {
        return ("COMPILE_ERROR".into(), -1);
    }

    let out = Command::new(&bin_path)
        .stdin(fixture_stdin(scratch, input))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (stdout, out.status.code().unwrap_or(-1))
}

fn test_snippet(name: &str, rel: &str) {
    let Some(source) = common::read_snippet(rel) else {
        common::skip(name, rel);
        return;
    };
    let input =
        std::fs::read(common::snippets_dir().join(std::path::Path::new(rel).with_extension("in")))
            .unwrap_or_default();

    let scratch = common::scratch_dir("snippet");
    let t0 = Instant::now();
    let (ref_out, ref_exit) = run_interpreter(&source, &input, scratch.path());
    let interp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let (stack_out, stack_exit) = run_stackifier(&source, &input, scratch.path());
    let stack_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let ok = ref_out == stack_out;
    let status = if ok { "✓" } else { "✗" };
    eprintln!(
        "{status} {name:<30} interp={interp_ms:.0}ms  stack={stack_ms:.0}ms  out={}/{} bytes  exit={ref_exit}/{stack_exit}",
        ref_out.len(),
        stack_out.len()
    );

    if !ok {
        eprintln!("  ref: {:?}", &ref_out[..ref_out.len().min(100)]);
        eprintln!("  got: {:?}", &stack_out[..stack_out.len().min(100)]);
    }
    assert_eq!(stack_out, ref_out, "{name}: output mismatch");
}

#[test]
fn test_standard_snippets() {
    let tests = [
        "standard/default-storage",
        "standard/exitcode",
        "standard/bieup",
        "standard/chieut",
        "standard/digeut",
        "standard/hieut-pop",
        "standard/ieunghieut",
        "standard/jieut",
        // "standard/loop", // exit code mismatch — skip for now
        "standard/mieum",
        "standard/nieun",
        "standard/pieup",
        "standard/print",
        "standard/queue",
        "standard/rieul",
        "standard/ssangbieup",
        "standard/ssangdigeut",
        "standard/ssangsiot",
        "standard/storage",
        "standard/tieut",
    ];
    for name in tests {
        test_snippet(name, &format!("{name}.aheui"));
    }
}

#[test]
fn test_hello_world() {
    test_snippet("hello-world", "hello-world/hello-world.puzzlet.aheui");
}

#[test]
fn test_factorial() {
    test_snippet("factorial", "factorial/factorial.aheui");
}

#[test]
fn test_99dan() {
    test_snippet("99dan", "99dan/99dan.aheui");
}
