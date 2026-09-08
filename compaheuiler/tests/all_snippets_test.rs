mod common;

use std::process::Command;

/// Names that require bigint to produce correct output (i64 overflows).
/// `integer/64bit` prints 2^65 to probe the implementation's integer range;
/// the corpus has carried it under both names, so both are listed.
const BIGINT_SNIPPETS: &[&str] = &["integer/2e65-print", "integer/64bit"];

fn needs_bigint(name: &str) -> bool {
    BIGINT_SNIPPETS.contains(&name)
}

fn compile_and_run(source: &str, stdin_data: &[u8]) -> (String, i32) {
    let rs_code = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compaheuiler::compile_to_rs(source)
    })) {
        Ok(code) => code,
        Err(_) => return ("CODEGEN_PANIC".into(), -1),
    };
    let scratch = common::scratch_dir("allsnip");
    let rs_path = scratch.path().join("aheui_allsnip.rs");
    let bin_path = scratch.path().join("aheui_allsnip");
    // Preserve the generated source for failure diagnosis. Outlives the run,
    // so `target/` rather than the scratch directory removed with it.
    std::fs::write(
        common::codegen_dir().join("aheui_allsnip_debug.rs"),
        &rs_code,
    )
    .ok();
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

    let mut child = Command::new(bin_path)
        .stdin(if stdin_data.is_empty() {
            std::process::Stdio::null()
        } else {
            std::process::Stdio::piped()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    if !stdin_data.is_empty() {
        use std::io::Write;
        child.stdin.as_mut().unwrap().write_all(stdin_data).unwrap();
        drop(child.stdin.take());
    }
    let (stdout_bytes, status) = common::bounded_output(child, std::time::Duration::from_secs(5));
    let stdout = String::from_utf8_lossy(&stdout_bytes).to_string();
    (stdout, status.code().unwrap_or(-1))
}

fn compile_and_run_bigint(source: &str, stdin_data: &[u8]) -> (String, i32) {
    let rs_code = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compaheuiler::compile_to_rs_bigint(source)
    })) {
        Ok(code) => code,
        Err(_) => return ("CODEGEN_PANIC".into(), -1),
    };
    // Own scratch project: `bigint_test` builds against a different
    // malachite-bigint version, and sharing one directory would make the two
    // suites rebuild the dependency for each other on every alternation.
    let package = format!("aheui-allsnip-bigint-test-{}", std::process::id());
    let dir = common::build_dir(&package);
    let target = common::build_dir("allsnip-bigint-proj/target");
    std::fs::create_dir_all(dir.join("src")).ok();
    std::fs::write(dir.join("src/main.rs"), &rs_code).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            r#"
# Its own workspace root: the project sits under `target/`, inside the aheui
# workspace directory, and cargo would otherwise refuse to build a package it
# finds there but no member list names.
[workspace]

[package]
name = "{package}"
version = "0.0.1"
edition = "2021"

[dependencies]
malachite-bigint = "0.2"
num-traits = "0.2"

[profile.release]
opt-level = 2
"#
        ),
    )
    .unwrap();
    let status = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .unwrap();
    if !status.success() {
        return ("COMPILE_ERROR".into(), -1);
    }
    let bin = target.join("release").join(&package);

    let mut child = Command::new(&bin)
        .stdin(if stdin_data.is_empty() {
            std::process::Stdio::null()
        } else {
            std::process::Stdio::piped()
        })
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    if !stdin_data.is_empty() {
        use std::io::Write;
        child.stdin.as_mut().unwrap().write_all(stdin_data).unwrap();
        drop(child.stdin.take());
    }
    let (stdout_bytes, status) = common::bounded_output(child, std::time::Duration::from_secs(5));
    (
        String::from_utf8_lossy(&stdout_bytes).to_string(),
        status.code().unwrap_or(-1),
    )
}

fn test_snippet(
    name: &str,
    aheui_path: &str,
    out_path: &str,
    in_path: Option<&str>,
    exitcode_path: Option<&str>,
) -> bool {
    let source = std::fs::read_to_string(aheui_path).unwrap();
    let expected = std::fs::read_to_string(out_path).unwrap();
    let stdin_data = in_path
        .map(|p| std::fs::read(p).unwrap())
        .unwrap_or_default();
    let expected_exit: Option<i32> =
        exitcode_path.and_then(|p| std::fs::read_to_string(p).ok()?.trim().parse().ok());

    // Preserve each generated source under a stable snippet-specific name.
    if let Ok(rs) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compaheuiler::compile_to_rs(&source)
    })) {
        let safe_name = name.replace('/', "_");
        std::fs::write(
            common::codegen_dir().join(format!("snip_{safe_name}.rs")),
            &rs,
        )
        .ok();
    }
    let (got, exit) = if needs_bigint(name) {
        compile_and_run_bigint(&source, &stdin_data)
    } else {
        compile_and_run(&source, &stdin_data)
    };

    // Allow trailing newline difference
    let out_ok = got == expected
        || format!("{got}\n") == expected
        || got == format!("{}\n", expected.trim_end());
    let exit_ok = expected_exit.is_none_or(|e| e == exit);

    if out_ok && exit_ok {
        eprintln!("  ✓ {name}");
        true
    } else {
        eprintln!("  ✗ {name}");
        if !out_ok {
            eprintln!(
                "    output: got {} bytes, expected {} bytes",
                got.len(),
                expected.len()
            );
            if got.len() < 200 && expected.len() < 200 {
                eprintln!("    got:  {:?}", got);
                eprintln!("    want: {:?}", expected);
            }
        }
        if !exit_ok {
            eprintln!("    exit: got {exit}, expected {:?}", expected_exit);
        }
        false
    }
}

#[test]
fn test_all_snippets() {
    type SnippetEntry = (
        String,
        std::path::PathBuf,
        std::path::PathBuf,
        Option<std::path::PathBuf>,
        Option<std::path::PathBuf>,
    );

    let snippets_dir = common::snippets_dir();
    let mut tested = 0;
    let mut passed = 0;
    let mut failed = 0;
    let mut entries: Vec<_> = Vec::new();

    // Collect all .out files recursively
    fn walk(dir: &std::path::Path, entries: &mut Vec<SnippetEntry>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, entries);
                    continue;
                }
                if path.extension().is_none_or(|e| e != "out") {
                    continue;
                }
                let base = path.file_stem().unwrap().to_str().unwrap().to_string();
                let aheui = path.with_extension("aheui");
                if !aheui.exists() {
                    continue;
                }
                let in_file = path.with_extension("in");
                let exitcode = path.with_extension("exitcode");
                let name = format!(
                    "{}/{}",
                    path.parent()
                        .unwrap()
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap(),
                    base
                );
                entries.push((
                    name,
                    aheui,
                    path.clone(),
                    if in_file.exists() {
                        Some(in_file)
                    } else {
                        None
                    },
                    if exitcode.exists() {
                        Some(exitcode)
                    } else {
                        None
                    },
                ));
            }
        }
    }

    walk(snippets_dir, &mut entries);
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    for (name, aheui, out, in_file, exitcode) in &entries {
        let ok = test_snippet(
            name,
            aheui.to_str().unwrap(),
            out.to_str().unwrap(),
            in_file.as_ref().map(|p| p.to_str().unwrap()),
            exitcode.as_ref().map(|p| p.to_str().unwrap()),
        );
        if ok {
            passed += 1;
        } else {
            failed += 1;
        }
        tested += 1;
    }

    eprintln!("\n  {passed}/{tested} passed, {failed} failed");
    assert_eq!(failed, 0, "{failed} snippets failed");
}
