#!/usr/bin/env python3
"""Build and time every pinned snippet across the six supported backends."""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = ROOT / "snippets"
WORK = ROOT / "target" / "snippet-matrix"
BIN_DIR = WORK / "bin"
PROGRAM_DIR = WORK / "programs"
GENERATED_TARGET = WORK / "generated-target"
TIMING_RE = re.compile(rb"\[snippet-matrix\] median_ns=(\d+) repeats=(\d+) exit=(-?\d+)")

COMPA_BACKENDS = ("compaheuiler/rust", "compaheuiler/c", "compaheuiler/cranelift")
INTERP_FEATURES = {
    "aheuinterpreter/malachite": "malachite-bigint",
    "aheuinterpreter/num-bigint": "num-bigint",
    "aheuinterpreter/rbigint": "runtime-rbigint",
}
BACKENDS = COMPA_BACKENDS + tuple(INTERP_FEATURES)


@dataclasses.dataclass(frozen=True)
class Snippet:
    name: str
    source: Path
    expected: bytes
    stdin: bytes
    expected_exit: int | None


@dataclasses.dataclass
class BackendResult:
    times_ns: dict[str, int] = dataclasses.field(default_factory=dict)
    errors: dict[str, str] = dataclasses.field(default_factory=dict)


def run_command(
    command: list[str | os.PathLike[str]],
    *,
    input_bytes: bytes | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    capture: bool = True,
    log: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    display = " ".join(str(part) for part in command)
    if log:
        print(f"+ {display}", flush=True)
    return subprocess.run(
        [str(part) for part in command],
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
        timeout=timeout,
        check=False,
    )


def require_success(proc: subprocess.CompletedProcess[bytes], what: str) -> None:
    if proc.returncode == 0:
        return
    stderr = proc.stderr.decode("utf-8", "replace")[-8_000:] if proc.stderr else ""
    raise RuntimeError(f"{what} failed with exit {proc.returncode}\n{stderr}")


def discover_snippets(selected: list[str]) -> list[Snippet]:
    if not SNIPPETS.is_dir():
        raise RuntimeError("snippets submodule is missing; checkout with submodules enabled")
    snippets = []
    for output_path in sorted(SNIPPETS.rglob("*.out")):
        source = output_path.with_suffix(".aheui")
        if not source.exists():
            continue
        relative = source.relative_to(SNIPPETS).with_suffix("")
        stdin_path = source.with_suffix(".in")
        exit_path = source.with_suffix(".exitcode")
        snippets.append(
            Snippet(
                name=relative.as_posix(),
                source=source,
                expected=output_path.read_bytes(),
                stdin=stdin_path.read_bytes() if stdin_path.exists() else b"",
                expected_exit=int(exit_path.read_text().strip()) if exit_path.exists() else None,
            )
        )
    if selected:
        by_name = {snippet.name: snippet for snippet in snippets}
        missing = sorted(set(selected) - by_name.keys())
        if missing:
            raise RuntimeError(f"unknown snippets: {', '.join(missing)}")
        snippets = [by_name[name] for name in selected]
    if not snippets:
        raise RuntimeError("no snippets with reference output were found")
    return snippets


def copy_executable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(destination.stat().st_mode | 0o111)


def build_backends() -> dict[str, Path]:
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    PROGRAM_DIR.mkdir(parents=True, exist_ok=True)

    # No `--locked`: `Cargo.lock` is not tracked (`.gitignore`), so a fresh
    # checkout has none to honour and cargo refuses to create one under the
    # flag. The majit dependencies are pinned by exact git rev in
    # `Cargo.toml`, which is where this matrix's reproducibility comes from.
    compa_build = run_command(
        [
            "cargo",
            "build",
            "--release",
            "-p",
            "compaheuiler",
            "--features",
            "cranelift,malachite-bigint",
            "--bin",
            "compaheuiler",
            "--example",
            "cranelift_snippet_runner",
        ],
        timeout=1_800,
    )
    require_success(compa_build, "compaheuiler backend build")
    compa = BIN_DIR / "compaheuiler"
    cranelift_runner = BIN_DIR / "cranelift-snippet-runner"
    copy_executable(ROOT / "target/release/compaheuiler", compa)
    copy_executable(
        ROOT / "target/release/examples/cranelift_snippet_runner",
        cranelift_runner,
    )

    binaries = {
        "compaheuiler/compiler": compa,
        "compaheuiler/cranelift": cranelift_runner,
    }
    for backend, feature in INTERP_FEATURES.items():
        build = run_command(
            [
                "cargo",
                "build",
                "--release",
                "-p",
                "aheuinterpreter",
                "--no-default-features",
                "--features",
                feature,
                "--example",
                "snippet_runner",
            ],
            timeout=1_800,
        )
        require_success(build, f"{backend} build")
        destination = BIN_DIR / backend.replace("/", "-")
        copy_executable(ROOT / "target/release/examples/snippet_runner", destination)
        binaries[backend] = destination
    return binaries


def compile_compaheuiler_programs(
    snippets: list[Snippet], compiler: Path
) -> dict[str, dict[str, Path]]:
    binaries: dict[str, dict[str, Path]] = {backend: {} for backend in COMPA_BACKENDS[:2]}
    env = dict(os.environ)
    env["COMPAHEUILER_GENERATED_TARGET_DIR"] = str(GENERATED_TARGET.resolve())
    for backend, codegen in zip(COMPA_BACKENDS[:2], ("rust", "c")):
        backend_dir = PROGRAM_DIR / codegen
        backend_dir.mkdir(parents=True, exist_ok=True)
        for index, snippet in enumerate(snippets, 1):
            output = backend_dir / snippet.name.replace("/", "-")
            print(f"[{backend}] compile {index}/{len(snippets)} {snippet.name}", flush=True)
            proc = run_command(
                [
                    compiler,
                    snippet.source,
                    "-O",
                    "3",
                    "--codegen",
                    codegen,
                    "-o",
                    output,
                ],
                env=env,
                timeout=600,
                log=False,
            )
            require_success(proc, f"{backend} compile {snippet.name}")
            binaries[backend][snippet.name] = output
    return binaries


def output_matches(actual: bytes, expected: bytes) -> bool:
    return (
        actual == expected
        or actual + b"\n" == expected
        or actual == expected.rstrip() + b"\n"
    )


def validate_run(snippet: Snippet, proc: subprocess.CompletedProcess[bytes]) -> str | None:
    if proc.returncode < 0:
        return f"terminated by signal {-proc.returncode}"
    if not output_matches(proc.stdout, snippet.expected):
        return f"stdout {len(proc.stdout)} bytes, expected {len(snippet.expected)}"
    if snippet.expected_exit is not None and proc.returncode != snippet.expected_exit:
        return f"exit {proc.returncode}, expected {snippet.expected_exit}"
    return None


def time_external(
    command: list[str | os.PathLike[str]], snippet: Snippet, repeats: int
) -> tuple[int | None, str | None]:
    elapsed = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        try:
            proc = run_command(command, input_bytes=snippet.stdin, timeout=180, log=False)
        except subprocess.TimeoutExpired:
            return None, "timed out after 180 seconds"
        elapsed.append(time.perf_counter_ns() - started)
        error = validate_run(snippet, proc)
        if error:
            return None, error
    return int(statistics.median(elapsed)), None


def time_cranelift(
    runner: Path, snippet: Snippet, repeats: int
) -> tuple[int | None, str | None]:
    try:
        proc = run_command(
            [runner, snippet.source, str(repeats)],
            input_bytes=snippet.stdin,
            timeout=300,
            log=False,
        )
    except subprocess.TimeoutExpired:
        return None, "timed out after 300 seconds"
    error = validate_run(snippet, proc)
    if error:
        return None, error
    match = TIMING_RE.search(proc.stderr)
    if not match:
        return None, "Cranelift runner did not report its execution time"
    return int(match.group(1)), None


def run_matrix(
    snippets: list[Snippet],
    binaries: dict[str, Path],
    compiled: dict[str, dict[str, Path]],
    repeats: int,
) -> dict[str, BackendResult]:
    results = {backend: BackendResult() for backend in BACKENDS}
    for index, snippet in enumerate(snippets, 1):
        print(f"run {index}/{len(snippets)} {snippet.name}", flush=True)
        for backend in COMPA_BACKENDS[:2]:
            elapsed, error = time_external([compiled[backend][snippet.name]], snippet, repeats)
            if error:
                results[backend].errors[snippet.name] = error
            else:
                results[backend].times_ns[snippet.name] = elapsed or 0

        elapsed, error = time_cranelift(binaries["compaheuiler/cranelift"], snippet, repeats)
        if error:
            results["compaheuiler/cranelift"].errors[snippet.name] = error
        else:
            results["compaheuiler/cranelift"].times_ns[snippet.name] = elapsed or 0

        for backend in INTERP_FEATURES:
            elapsed, error = time_external(
                [binaries[backend], snippet.source], snippet, repeats
            )
            if error:
                results[backend].errors[snippet.name] = error
            else:
                results[backend].times_ns[snippet.name] = elapsed or 0
    return results


def format_ms(nanoseconds: int) -> str:
    milliseconds = nanoseconds / 1_000_000
    if milliseconds < 1:
        return f"{milliseconds:.3f}"
    if milliseconds < 100:
        return f"{milliseconds:.2f}"
    return f"{milliseconds:.1f}"


def make_report(
    snippets: list[Snippet], results: dict[str, BackendResult], repeats: int
) -> str:
    labels = {
        "compaheuiler/rust": "compaheuiler Rust",
        "compaheuiler/c": "compaheuiler C",
        "compaheuiler/cranelift": "compaheuiler Cranelift",
        "aheuinterpreter/malachite": "aheuinterpreter malachite",
        "aheuinterpreter/num-bigint": "aheuinterpreter num-bigint",
        "aheuinterpreter/rbigint": "aheuinterpreter rbigint",
    }
    lines = [
        "<!-- aheui-snippet-matrix -->",
        "## 전체 snippet backend 검사",
        "",
        f"고정 corpus의 {len(snippets)}개 프로그램을 여섯 backend에서 각각 한 번 준비하고, "
        f"같은 실행을 정확성 검사와 속도 측정에 함께 사용했습니다. 시간은 {repeats}회 중앙값입니다.",
        "",
        "| backend | 정확성 | snippet 중앙값의 합계 |",
        "|---|---:|---:|",
    ]
    for backend in BACKENDS:
        result = results[backend]
        passed = len(snippets) - len(result.errors)
        total = sum(result.times_ns.values())
        status = "✅" if not result.errors else "❌"
        lines.append(
            f"| {labels[backend]} | {status} {passed}/{len(snippets)} | {format_ms(total)} ms |"
        )

    lines.extend(
        [
            "",
            "<details>",
            "<summary>snippet별 실행 시간</summary>",
            "",
            "| snippet | Rust | C | Cranelift | malachite | num-bigint | rbigint |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for snippet in snippets:
        cells = []
        for backend in BACKENDS:
            result = results[backend]
            if snippet.name in result.errors:
                cells.append("❌")
            else:
                cells.append(f"{format_ms(result.times_ns[snippet.name])} ms")
        lines.append(f"| `{snippet.name}` | " + " | ".join(cells) + " |")
    lines.extend(["", "</details>", ""])

    failures = [
        (backend, name, error)
        for backend, result in results.items()
        for name, error in result.errors.items()
    ]
    if failures:
        lines.extend(["### 실패", ""])
        for backend, name, error in failures:
            lines.append(f"- `{backend}` / `{name}`: {error}")
        lines.append("")

    lines.extend(
        [
            "측정 범위:",
            "",
            "- Rust/C는 생성된 실행 파일의 process wall time입니다.",
            "- Cranelift는 source와 CFG를 한 번 컴파일한 뒤 `JitFunction::execute_buffered`만 잰 시간입니다.",
            "- aheuinterpreter는 source parse와 process 시작을 포함한 CLI wall time입니다.",
            "- timing은 정보 제공용이며 regression threshold로 사용하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def make_setup_failure_report(error: BaseException) -> str:
    return "\n".join(
        [
            "<!-- aheui-snippet-matrix -->",
            "## 전체 snippet backend 검사 실패",
            "",
            "backend 준비 또는 생성물 컴파일 중 실패해 실행 시간 표를 만들지 못했습니다.",
            "",
            "```text",
            str(error)[-8_000:],
            "```",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=WORK / "report.md")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--snippet",
        action="append",
        default=[],
        help="run only this exact corpus-relative snippet name; may be repeated",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.unlink(missing_ok=True)
    try:
        snippets = discover_snippets(args.snippet)
        binaries = build_backends()
        compiled = compile_compaheuiler_programs(snippets, binaries["compaheuiler/compiler"])
        results = run_matrix(snippets, binaries, compiled, args.repeats)
        report = make_report(snippets, results, args.repeats)
        status = 1 if any(result.errors for result in results.values()) else 0
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        report = make_setup_failure_report(error)
        status = 1
    args.report.write_text(report)
    print(report)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
