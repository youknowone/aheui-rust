#![cfg(feature = "jit")]

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn early_popnum_guard_resumes_with_the_live_state_frame() {
    let program =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("snippets/99dan/99dan.aheui");
    let mut child = Command::new(env!("CARGO_BIN_EXE_aheui"))
        .arg(program)
        .env("MAJIT_THRESHOLD", "50")
        .env("MAJIT_STATS", "1")
        .env_remove("MAJIT_LOG")
        .env_remove("AHEUI_BANDS")
        .env_remove("AHEUI_CAP")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // A tracing panic used to be caught and followed by an infinite compiled
    // loop. Bound the child lifetime and both output buffers on regressions.
    let read = |stream: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            stream.take(65536).read_to_end(&mut bytes).unwrap();
            bytes
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout.join().unwrap();
    let stderr = String::from_utf8_lossy(&stderr.join().unwrap()).into_owned();
    assert!(status.is_some_and(|s| s.success()), "{status:?}: {stderr}");
    let expected: String = (2..=9)
        .flat_map(|a| (1..=9).map(move |b| format!("{a}*{b}={}\n", a * b)))
        .collect();
    assert_eq!(stdout, expected.as_bytes());
    assert!(!stderr.contains("panicked"), "{stderr}");
    let loops = stderr
        .split_whitespace()
        .find_map(|word| word.strip_prefix("loops_compiled="))
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap();
    assert!(
        loops > 0,
        "the regression must exercise compiled code: {stderr}"
    );
    assert!(stderr.contains("loops_aborted=0"), "{stderr}");
}
