"""Shared text format for Aheui benchmark baselines."""

from pathlib import Path
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time


def bounded_run(command, *, input=b"", timeout=60, cwd=None, env=None, memory_mib=1024, file_mib=8):
    """Capture to bounded files, not an unbounded communicate() byte buffer.

    The outer run-limited.py watchdog also bounds aggregate RSS. These local
    limits protect ordinary direct invocations of corpus scripts.
    """
    limit = 8 * 1024 * 1024
    if file_mib <= 0:
        raise ValueError("file_mib must be positive")

    def limits():
        def lower(kind, amount):
            hard = resource.getrlimit(kind)[1]
            if hard != resource.RLIM_INFINITY:
                amount = min(amount, hard)
            resource.setrlimit(kind, (amount, amount))
        lower(resource.RLIMIT_CORE, 0)
        # Cargo also writes Git packs, object files and binaries. Their budget
        # is independent of the captured output budget checked below.
        lower(resource.RLIMIT_FSIZE, file_mib * 1024 * 1024)
        if sys.platform.startswith("linux"):
            lower(resource.RLIMIT_DATA, memory_mib * 1024 * 1024)

    environment = dict(os.environ if env is None else env)
    environment.update(CARGO_BUILD_JOBS="1", RUST_TEST_THREADS="1")
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        stdin.write(input or b"")
        stdin.seek(0)
        proc = subprocess.Popen(command, cwd=cwd, env=environment, stdin=stdin,
                                stdout=stdout, stderr=stderr, start_new_session=True, preexec_fn=limits)
        expired = False
        overflowed = False
        deadline = time.monotonic() + timeout
        try:
            while True:
                if max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size) >= limit:
                    overflowed = True
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    expired = True
                    break
                try:
                    proc.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
        finally:
            # Include children which inherited a pipe or outlived their parent.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(limit), stderr.read(limit)
        if expired:
            raise subprocess.TimeoutExpired(command, timeout, output=out, stderr=err)
        if overflowed or len(out) == limit or len(err) == limit:
            return subprocess.CompletedProcess(command, 125, out, err + b"\noutput limit exceeded\n")
        return subprocess.CompletedProcess(command, proc.returncode, out, err)


def parse_fields(text: str) -> dict[str, str]:
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


def format_fields(fields: dict) -> str:
    return "".join(f"{key}={value}\n" for key, value in sorted(fields.items()))


def read_int_fields(path: Path) -> dict[str, int]:
    return {key: int(value) for key, value in parse_fields(path.read_text()).items()}


def write_fields(path: Path, fields: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_fields(fields))
