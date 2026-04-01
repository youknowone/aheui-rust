use std::process::Command;

const SNIPPETS: &str = "/Users/al03219714/Projects/pypy/rpaheui/snippets";

/// Names that require bigint to produce correct output (i64 overflows).
const BIGINT_SNIPPETS: &[&str] = &["integer/2e65-print"];

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
    let rs_path = "/tmp/aheui_allsnip.rs";
    let bin_path = "/tmp/aheui_allsnip";
    // Save for debugging
    std::fs::write("/tmp/aheui_allsnip_debug.rs", &rs_code).ok();
    std::fs::write(rs_path, &rs_code).unwrap();
    let status = Command::new("rustc")
        .args(["-C", "opt-level=2", "-o", bin_path, rs_path])
        .stderr(std::process::Stdio::piped())
        .status()
        .unwrap();
    if !status.success() {
        return ("COMPILE_ERROR".into(), -1);
    }

    use std::io::Read;
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
    // Read stdout with limit to prevent OOM
    let mut stdout_bytes = Vec::new();
    let max_output = 10 * 1024 * 1024; // 10MB limit
    if let Some(ref mut out) = child.stdout {
        let mut buf = [0u8; 8192];
        loop {
            match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    stdout_bytes.extend_from_slice(&buf[..n]);
                    if stdout_bytes.len() > max_output {
                        let _ = child.kill();
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
    // Wait with timeout (5 seconds)
    let status = match child.try_wait() {
        Ok(Some(s)) => s,
        _ => {
            std::thread::sleep(std::time::Duration::from_secs(5));
            match child.try_wait() {
                Ok(Some(s)) => s,
                _ => {
                    let _ = child.kill();
                    child.wait().unwrap()
                }
            }
        }
    };
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
    let dir = "/tmp/aheui_bigint_proj";
    std::fs::create_dir_all(&format!("{dir}/src")).ok();
    std::fs::write(format!("{dir}/src/main.rs"), &rs_code).unwrap();
    std::fs::write(
        format!("{dir}/Cargo.toml"),
        r#"
[package]
name = "aheui-bigint-test"
version = "0.0.1"
edition = "2021"

[dependencies]
malachite-bigint = "0.2"
num-traits = "0.2"

[profile.release]
opt-level = 2
"#,
    )
    .unwrap();
    let status = Command::new("cargo")
        .args(["build", "--release", "--quiet"])
        .current_dir(dir)
        .status()
        .unwrap();
    if !status.success() {
        return ("COMPILE_ERROR".into(), -1);
    }
    let bin = format!("{dir}/target/release/aheui-bigint-test");

    use std::io::Read;
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
    let mut stdout_bytes = Vec::new();
    if let Some(ref mut out) = child.stdout {
        let mut buf = [0u8; 8192];
        loop {
            match out.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    stdout_bytes.extend_from_slice(&buf[..n]);
                    if stdout_bytes.len() > 10 * 1024 * 1024 {
                        let _ = child.kill();
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    }
    let status = match child.try_wait() {
        Ok(Some(s)) => s,
        _ => {
            std::thread::sleep(std::time::Duration::from_secs(5));
            match child.try_wait() {
                Ok(Some(s)) => s,
                _ => {
                    let _ = child.kill();
                    child.wait().unwrap()
                }
            }
        }
    };
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

    // Save generated code for debugging
    if let Ok(rs) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        compaheuiler::compile_to_rs(&source)
    })) {
        let safe_name = name.replace('/', "_");
        std::fs::write(format!("/tmp/snip_{safe_name}.rs"), &rs).ok();
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
    let exit_ok = expected_exit.map_or(true, |e| e == exit);

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
                eprintln!("    got:  {:?}", &got);
                eprintln!("    want: {:?}", &expected);
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
    let snippets_dir = std::path::Path::new(SNIPPETS);
    let mut tested = 0;
    let mut passed = 0;
    let mut failed = 0;
    let mut entries: Vec<_> = Vec::new();

    // Collect all .out files recursively
    fn walk(
        dir: &std::path::Path,
        entries: &mut Vec<(
            String,
            std::path::PathBuf,
            std::path::PathBuf,
            Option<std::path::PathBuf>,
            Option<std::path::PathBuf>,
        )>,
    ) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for entry in rd {
                let entry = entry.unwrap();
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, entries);
                    continue;
                }
                if path.extension().map_or(true, |e| e != "out") {
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
