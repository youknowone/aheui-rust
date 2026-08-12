//! Every AOT emitter has to produce the same source for the same input.
//!
//! A `std::collections::HashMap` seeds its hasher once per process, so a pass
//! that consumes one's iteration order emits something different on each run
//! while looking perfectly stable within a single one. Catching that needs
//! separate processes, which is why this spawns the binary instead of calling
//! the library.

use std::path::PathBuf;
use std::process::Command;

/// `logo` is the program whose loop-guard elimination used to vary, which moved
/// whole blocks in and out of the emitted source; `99bottles` keeps enough live
/// registers at an overflow-capable binop to expose the emitters' own ordering.
const PROGRAMS: [&str; 2] = [
    "snippets/logo/logo.aheui",
    "snippets/99bottles/99bottles.aheui",
];
const CODEGENS: [&str; 4] = ["rust", "c", "wasm32-wasi", "wasm32-web"];
const RUNS: usize = 8;

fn emit(codegen: &str, program: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(program);
    assert!(
        path.is_file(),
        "{} is missing; the snippets submodule is not checked out",
        path.display()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_aheui"))
        .args(["build", "-O3", "--codegen", codegen, "--emit", "asm"])
        .arg(&path)
        .output()
        .expect("could not run the aheui binary");
    assert!(
        out.status.success(),
        "{codegen} failed on {program}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Where two emissions first disagree, as a `line, column` pair for the failure
/// message — the sources are megabytes, so printing them is no help.
fn first_difference(a: &[u8], b: &[u8]) -> String {
    let at = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(a.len());
    let line = 1 + a[..at].iter().filter(|&&c| c == b'\n').count();
    let col = 1 + at
        - a[..at]
            .iter()
            .rposition(|&c| c == b'\n')
            .map_or(0, |i| i + 1);
    format!(
        "line {line}, column {col} (byte {at}; lengths {} and {})",
        a.len(),
        b.len()
    )
}

#[test]
fn aot_emitters_are_reproducible_across_processes() {
    for program in PROGRAMS {
        for codegen in CODEGENS {
            let first = emit(codegen, program);
            for run in 2..=RUNS {
                let again = emit(codegen, program);
                assert!(
                    first == again,
                    "{codegen} emitted a different source for {program} on run {run}; \
                     they diverge at {}",
                    first_difference(&first, &again)
                );
            }
        }
    }
}
