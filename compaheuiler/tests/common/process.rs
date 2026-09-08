/// Drain stdout concurrently: a silent loop must not block before its deadline.
pub fn bounded_output(
    mut child: std::process::Child,
    timeout: std::time::Duration,
) -> (Vec<u8>, std::process::ExitStatus) {
    use std::io::Read;
    const LIMIT: usize = 10 * 1024 * 1024;
    let stdout = child.stdout.take().expect("piped stdout");
    let (send, receive) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = stdout.take((LIMIT + 1) as u64).read_to_end(&mut bytes);
        let _ = send.send((bytes, result));
    });
    let deadline = std::time::Instant::now() + timeout;
    let mut output = None;
    let status = loop {
        if output.is_none() {
            output = receive.try_recv().ok();
        }
        if std::time::Instant::now() >= deadline
            || output
                .as_ref()
                .is_some_and(|(bytes, _)| bytes.len() > LIMIT)
        {
            let _ = child.kill();
            break child.wait().expect("reap timed-out child");
        }
        if let Some(status) = child.try_wait().expect("poll child") {
            break status;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    let (bytes, result) = output.unwrap_or_else(|| receive.recv().expect("stdout reader"));
    reader.join().expect("stdout thread");
    result.expect("read stdout");
    (bytes, status)
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[test]
    fn unlimited_output_is_killed_at_the_capture_limit() {
        let child = std::process::Command::new("yes")
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let (output, status) = super::bounded_output(child, std::time::Duration::from_secs(3));
        assert_eq!(output.len(), 10 * 1024 * 1024 + 1);
        assert!(!status.success());
    }

    #[cfg(unix)]
    #[test]
    fn silent_child_is_killed_and_reaped_before_reading_eof() {
        let child = std::process::Command::new("sh")
            .args(["-c", "while :; do :; done"])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let start = std::time::Instant::now();
        let (output, status) = super::bounded_output(child, std::time::Duration::from_millis(100));
        assert!(output.is_empty());
        assert!(!status.success());
        assert!(start.elapsed() < std::time::Duration::from_secs(3));
    }

    #[cfg(unix)]
    #[test]
    fn complete_output_and_exit_status_are_preserved() {
        let child = std::process::Command::new("sh")
            .args(["-c", "printf complete; exit 7"])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let (output, status) = super::bounded_output(child, std::time::Duration::from_secs(3));
        assert_eq!(output, b"complete");
        assert_eq!(status.code(), Some(7));
    }
}
