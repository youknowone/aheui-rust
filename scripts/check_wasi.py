"""Exercise a real WASI JIT trace, not just guest compilation/layout extraction."""

import os
from pathlib import Path
import subprocess

from bench_support import parse_fields, bounded_run

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runner = ROOT / "target/release/aheui-wasm-runner"
    guest = ROOT / "target/wasm32-wasip1/release/aheui.wasm"
    # The runner preopens host paths; the WASI guest does not inherit host cwd.
    source = str(ROOT / "snippets/standard/loop.aheui")
    command = [str(runner), str(guest), source]
    env = dict(os.environ, MAJIT_STATS="1", MAJIT_THRESHOLD="50")
    # Wasmtime's compiled-module cache exceeds the default 8 MiB file limit.
    # Captured stdout/stderr remain capped independently by bounded_run.
    jit = bounded_run(command, cwd=ROOT, env=env, timeout=180, file_mib=64)
    control = bounded_run(
        [str(runner), str(guest), "--no-jit", source],
        cwd=ROOT, env=env, timeout=180, file_mib=64,
    )
    assert jit.returncode == control.returncode == 0, (
        jit.returncode, control.returncode, jit.stderr, control.stderr,
    )
    expected = (ROOT / "snippets/standard/loop.out").read_bytes()
    assert jit.stdout == control.stdout == expected, "WASI output mismatch"
    lines = jit.stderr.decode(errors="replace").splitlines()
    stats = {}
    for line in lines:
        if "[jit-stats]" in line:
            stats.update(parse_fields(line.split("[jit-stats]", 1)[1].strip().replace(" ", "\n")))
    assert int(stats.get("loops_compiled", 0)) > 0, jit.stderr
    for name in ("loops_aborted", "internal_compile_panics"):
        assert int(stats.get(name, 0)) == 0, jit.stderr
    print("WASI output and actual JIT compilation passed")


if __name__ == "__main__":
    main()
