#!/usr/bin/env python3
"""Record/check emitted backend operation counts; show one program's census.

Usage: opcensus.py {record,check,show} [corpus/program ...]. Runs with
MAJIT_LOG at threshold 50. Counts may fall; rises require investigation.
Output size and exit status must match exactly. Trace count is informational.
Baselines live beside the corresponding jitstress .jitstats files.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from bench_support import read_int_fields as read_baseline, write_fields as write_baseline

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "target" / "release" / "aheui"
SNIPPETS = REPO / "snippets"
BENCH = REPO / "bench"

# `pyre/check.py`'s `*_jitstress` threshold. See the module docstring for why
# the census does not run at the production one.
CENSUS_THRESHOLD = "50"

# `[dynasm] emit[N]: Opcode ...`     an op whose result is kept.
# `[dynasm] discard[N]: Opcode ...`  an op emitted for its side effect (stores).
# `[dynasm] guard[N]: Opcode ...`    a guard.
EMIT_RE = re.compile(r"^\[dynasm\] (?:emit|discard|guard)\[\d+\]: ([A-Za-z_0-9]+)")
# Opens each compiled trace, so it is how the traces are counted apart.
ASSEMBLE_RE = re.compile(r"^\[dynasm\] _assemble: \d+ ops → \d+ ra_ops")

# Exact-match fields: a move in either direction fails. Everything else that is
# gated may fall freely and fails only on a rise.
EXACT_FIELDS = ("out_bytes", "exit")
# Reported, gated by nothing.
UNGATED_FIELDS = ("traces",)


def baseline_path(name: str) -> Path:
    """Beside the `.jitstats` baseline for the same run.

    A directory under `bench/` names the CONFIGURATION a baseline was taken
    under, and the extension names the instrument that read it. This census
    runs at `CENSUS_THRESHOLD`, which is `jitstats.py`'s jitstress threshold —
    the same configuration — so it belongs in that directory rather than one
    named after the instrument. Splitting by instrument put two descriptions of
    a single run in two trees, where nothing lines them up: `traces` here and
    `loops_compiled` + `bridges_compiled` there count the same event, and
    `exit` is recorded twice.
    """
    return BENCH / "jitstress" / f"{name}.opcensus"


def census(program: Path) -> dict[str, int]:
    """Run one program under `MAJIT_LOG` and count what the backend emitted."""
    env = dict(os.environ, MAJIT_LOG="1", MAJIT_THRESHOLD=CENSUS_THRESHOLD)
    # The sibling `.in` is the input the committed `.out` was produced from.
    # Without it those programs read EOF and take a path the reference never
    # described — and never reach the workload whose trace this counts.
    stdin_file = program.with_suffix(".in")
    stdin_bytes = stdin_file.read_bytes() if stdin_file.exists() else b""
    proc = subprocess.run(
        [str(BINARY), str(program)],
        input=stdin_bytes,
        capture_output=True,
        env=env,
    )

    fields: dict[str, int] = {"traces": 0, "total_ops": 0}
    for line in proc.stderr.decode("utf-8", "replace").splitlines():
        if ASSEMBLE_RE.match(line):
            fields["traces"] += 1
            continue
        matched = EMIT_RE.match(line)
        if matched:
            key = f"op.{matched.group(1)}"
            fields[key] = fields.get(key, 0) + 1
            fields["total_ops"] += 1
    fields["out_bytes"] = len(proc.stdout)
    fields["exit"] = proc.returncode
    return fields


def compare(new: dict[str, int], old: dict[str, int]) -> list[str]:
    """Every gated field that regressed, as one message each."""
    failures = []
    for field in EXACT_FIELDS:
        if new.get(field) != old.get(field):
            failures.append(f"{field}: {old.get(field)} -> {new.get(field)} (must not move)")
    for key in sorted(set(new) | set(old)):
        if key in EXACT_FIELDS or key in UNGATED_FIELDS:
            continue
        before, after = old.get(key, 0), new.get(key, 0)
        if after > before:
            failures.append(f"{key}: {before} -> {after} (+{after - before})")
    return failures


def moved(new: dict[str, int], old: dict[str, int]) -> list[str]:
    """Every field that moved at all, gated or not — the readable half."""
    lines = []
    for key in sorted(set(new) | set(old)):
        before, after = old.get(key, 0), new.get(key, 0)
        if before != after:
            lines.append(f"    {key}: {before} -> {after} ({after - before:+d})")
    return lines


def corpus() -> dict[str, Path]:
    """Every pinned corpus program, keyed the way the baselines are."""
    found: dict[str, Path] = {}
    if not SNIPPETS.is_dir():
        return found
    for source in sorted(SNIPPETS.glob("*/*.aheui")):
        found[f"{source.parent.name}/{source.stem}"] = source
    return found


def selected(args: list[str]) -> list[tuple[str, Path]]:
    available = corpus()
    if args:
        picked = []
        for name in args:
            if name not in available:
                print(f"  FAIL: {name}: no such corpus program", file=sys.stderr)
                return []
            picked.append((name, available[name]))
        return picked
    # With no arguments: every program that already has a baseline, plus — for
    # `record` — nothing else. A program that compiles nothing has an all-zero
    # census that gates only that it still compiles nothing, which is worth
    # having but is not what this file is for.
    return [(name, path) for name, path in available.items() if baseline_path(name).exists()]


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("record", "check", "show"):
        print(__doc__)
        return 2
    mode, args = argv[1], argv[2:]

    if not BINARY.exists():
        print(f"no binary at {BINARY}; cargo build -p aheui --release", file=sys.stderr)
        return 2

    if mode == "show":
        if not args:
            print("show needs one corpus/program", file=sys.stderr)
            return 2
        available = corpus()
        if args[0] not in available:
            print(f"no such corpus program: {args[0]}", file=sys.stderr)
            return 2
        fields = census(available[args[0]])
        for key, value in sorted(fields.items()):
            print(f"{key}={value}")
        return 0

    if mode == "record":
        # `record` with no arguments only rewrites what is already committed.
        # Naming a program is how a new one joins the gate.
        items = selected(args)
        if not items:
            print("nothing to record", file=sys.stderr)
            return 1
        for name, path in items:
            fields = census(path)
            write_baseline(baseline_path(name), fields)
            print(f"  {name}: {fields['total_ops']} ops in {fields['traces']} traces")
        return 0

    failed = 0
    items = selected(args)
    if not items:
        print("  SKIP: no op-census baselines and no corpus")
        return 0
    for name, path in items:
        path_to_baseline = baseline_path(name)
        if not path_to_baseline.exists():
            print(f"  FAIL: {name}: no baseline — record it with scripts/opcensus.py record {name}")
            failed += 1
            continue
        new, old = census(path), read_baseline(path_to_baseline)
        failures = compare(new, old)
        if failures:
            failed += 1
            print(f"  FAIL: {name}")
            for message in failures:
                print(f"    {message}")
        else:
            delta = new["total_ops"] - old["total_ops"]
            note = f" ({delta:+d})" if delta else ""
            print(f"  ok: {name}: {new['total_ops']} ops{note}")
            for line in moved(new, old):
                print(line)
    if failed:
        print(f"\n{failed} program(s) emit more ops than their baseline.")
        print("A rise is the regression. If it is deliberate, re-record and say what bought it.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
