mod common;

use std::process::Command;
use std::time::Instant;

fn fixture_stdin(scratch: &std::path::Path, input: &[u8]) -> std::process::Stdio {
    let input_path = scratch.join("snippet.in");
    std::fs::write(&input_path, input).unwrap();
    std::fs::File::open(input_path).unwrap().into()
}

/// The reference interpreter, built once for this whole test binary.
///
/// Building it per snippet through `cargo run` hides its own failures: a
/// build that fails exits with a status this harness cannot tell apart from
/// a program that halted with that value, so the reference contributes no
/// output and every snippet reports an output mismatch against a
/// compilation that was in fact correct. Build it once, and let a failure
/// to build say so.
///
/// The crate under test may be built against a patched dependency set
/// (`cargo --config …`) and cargo passes no such flag down to a test
/// process. A reference built against a different dependency set is not a
/// reference, so it travels in `AHEUI_CARGO_CONFIG` — the same variable
/// `scripts/extract-llbc.sh` reads for the same reason.
fn reference_interpreter() -> &'static std::path::Path {
    static BIN: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    BIN.get_or_init(|| {
        let mut cargo = Command::new("cargo");
        cargo.args(["build", "--release", "-p", "aheui"]);
        if let Ok(config) = std::env::var("AHEUI_CARGO_CONFIG") {
            cargo.args(["--config", &config]);
        }
        let out = cargo.output().expect("cargo build");
        assert!(
            out.status.success(),
            "the reference interpreter did not build, so every comparison below \
             would be against no output at all. When the workspace is built \
             against a patched dependency set, pass the same one here in \
             AHEUI_CARGO_CONFIG.\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let target = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .parent()
                    .expect("this crate is a workspace member")
                    .join("target")
            });
        let bin = target.join("release").join("aheui");
        assert!(
            bin.is_file(),
            "the reference interpreter built but is not at {}",
            bin.display()
        );
        bin
    })
    .as_path()
}

fn run_interpreter(source: &str, input: &[u8], scratch: &std::path::Path) -> (String, i32) {
    let source_path = scratch.join("snippet.aheui");
    std::fs::write(&source_path, source).unwrap();
    let out = Command::new(reference_interpreter())
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
