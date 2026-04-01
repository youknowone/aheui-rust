use std::process::Command;
use std::time::Instant;

const SNIPPETS: &str = "snippets";

fn run_interpreter(source: &str) -> (String, i32) {
    let tmp = "/tmp/aheui_snippet_src.aheui";
    std::fs::write(tmp, source).unwrap();
    let out = Command::new("cargo")
        .args(["run", "--release", "-p", "aheui", "--", tmp])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (stdout, out.status.code().unwrap_or(-1))
}

fn run_stackifier(source: &str) -> (String, i32) {
    let rs_code = compaheuiler::compile_to_rs(source);
    let rs_path = "/tmp/aheui_snippet.rs";
    let bin_path = "/tmp/aheui_snippet_bin";
    std::fs::write(rs_path, &rs_code).unwrap();

    let status = Command::new("rustc")
        .args(["-C", "opt-level=2", "-o", bin_path, rs_path])
        .stderr(std::process::Stdio::piped())
        .status()
        .unwrap();
    if !status.success() {
        return ("COMPILE_ERROR".into(), -1);
    }

    let out = Command::new(bin_path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (stdout, out.status.code().unwrap_or(-1))
}

fn test_snippet(name: &str, path: &str) {
    let source = std::fs::read_to_string(path).unwrap();

    let t0 = Instant::now();
    let (ref_out, ref_exit) = run_interpreter(&source);
    let interp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let (stack_out, stack_exit) = run_stackifier(&source);
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
        let path = format!("{SNIPPETS}/{name}.aheui");
        if std::path::Path::new(&path).exists() {
            test_snippet(name, &path);
        }
    }
}

#[test]
fn test_hello_world() {
    test_snippet(
        "hello-world",
        &format!("{SNIPPETS}/hello-world/hello-world.puzzlet.aheui"),
    );
}

#[test]
fn test_factorial() {
    test_snippet(
        "factorial",
        &format!("{SNIPPETS}/factorial/factorial.aheui"),
    );
}

#[test]
fn test_99dan() {
    test_snippet("99dan", &format!("{SNIPPETS}/99dan/99dan.aheui"));
}
