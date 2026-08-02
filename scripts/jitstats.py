#!/usr/bin/env python3
"""Record and gate the aheui JIT's `[jit-stats]` counters.

Same file format and the same regression floor as `pyre/check.py`, so one
reading applies to both: a sorted `key=value` text file per fixture, compared
field-by-field with a per-field direction.

Each fixture is also run twice — once with the JIT and once with the threshold
raised out of reach — and the two runs must agree byte-for-byte on stdout and
on the exit code. That A/B is the correctness half: without it a baseline can
go green on a run that miscompiled.

    scripts/jitstats.py record [fixture ...]
    scripts/jitstats.py check  [fixture ...]

With no fixture arguments both modes cover the whole committed set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "target" / "release" / "aheui"
SAMPLES = REPO / "aheui-wasm" / "web" / "samples"
BENCH = REPO / "bench"

# The fixtures live in-tree (`aheui-wasm/web/samples/`) rather than in the
# rpaheui snippet corpus: the corpus is an unpinned sibling checkout, and a
# baseline keyed to a fixture nobody can reproduce gates nothing. `logo` is the
# only sample that reaches the 1039 back-edge threshold and actually compiles;
# the rest are carried anyway because their badness counters start at 0, and a
# rise off 0 is exactly what the floor exists to catch.
FIXTURES = [
    "logo",
    "99bottles",
    "hello",
    "hello-world",
]

# `pyre/check.py` JITSTATS_BADNESS_FIELDS / RISE_BOUNDED / FALL.
BADNESS_FIELDS = ("loops_aborted", "internal_compile_panics")
RISE_BOUNDED_FIELDS = ("guard_failures",)
FALL_FIELDS = ("loops_compiled",)
# `bridges_compiled` is recorded but gated by nothing: it moves in both
# directions under ordinary tuning.
SNAPSHOT_FIELDS = (
    *BADNESS_FIELDS,
    *RISE_BOUNDED_FIELDS,
    *FALL_FIELDS,
    "bridges_compiled",
)

# High enough that no fixture reaches it, so `mainloop` runs with the tracer
# never firing. This is the right control: `--no-jit` would run a DIFFERENT
# interpreter (aheuinterpreter), and it only exists on a `naive` build.
NO_COMPILE_THRESHOLD = "1000000000"


def fixture_path(name: str) -> Path:
    return SAMPLES / f"{name}.aheui"


def baseline_path(name: str) -> Path:
    return BENCH / f"{name}.jitstats"


def run(name: str, *, compile_: bool) -> tuple[bytes, int, dict[str, str]]:
    """Run one fixture; return its stdout, exit code and `[jit-stats]` fields."""
    env = dict(os.environ, MAJIT_STATS="1")
    if not compile_:
        env["MAJIT_THRESHOLD"] = NO_COMPILE_THRESHOLD
    proc = subprocess.run(
        [str(BINARY), str(fixture_path(name))],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=env,
    )
    fields: dict[str, str] = {}
    for line in proc.stderr.decode("utf-8", "replace").splitlines():
        if not line.startswith("[jit-stats]"):
            continue
        # Merge every line rather than keeping the last: a diag line emitted
        # after the counters would otherwise displace them and silently disarm
        # the floor (the lesson `pyre/check.py` records at `_jit_stats_snapshot`).
        for token in line[len("[jit-stats]") :].split():
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
    return proc.stdout, proc.returncode, fields


def snapshot(fields: dict[str, str]) -> str:
    return "".join(
        f"{k}={fields[k]}\n" for k in sorted(SNAPSHOT_FIELDS) if k in fields
    )


def parse(text: str) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line
    )


def floor_regression(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """`pyre/check.py` `_jit_stats_regression_floor`, verbatim policy."""
    failures = []
    for field in (*BADNESS_FIELDS, *RISE_BOUNDED_FIELDS, *FALL_FIELDS):
        if field not in BADNESS_FIELDS and field not in old:
            # Count-valued with no baseline entry: ungated.
            continue
        base = int(old.get(field, 0))
        cur = int(new.get(field, 0))
        if field in RISE_BOUNDED_FIELDS:
            regressed = cur > base + max(base // 4, 2)
        elif field in FALL_FIELDS:
            regressed = cur < base
        else:
            regressed = cur > base
        if regressed:
            failures.append(f"{field} {base} -> {cur}")
    return failures


def check_one(name: str, *, record: bool) -> bool:
    """Run the A/B, then either record or gate. True on success."""
    if not fixture_path(name).exists():
        print(f"  FAIL: {name}: no such fixture ({fixture_path(name)})")
        return False

    jit_out, jit_code, fields = run(name, compile_=True)
    control_out, control_code, _ = run(name, compile_=False)

    if jit_out != control_out:
        print(
            f"  FAIL: {name}: JIT stdout differs from the un-compiled run "
            f"({len(jit_out)} vs {len(control_out)} bytes)"
        )
        return False
    if jit_code != control_code:
        print(
            f"  FAIL: {name}: JIT exit {jit_code} != un-compiled exit {control_code}"
        )
        return False
    if not fields:
        print(f"  FAIL: {name}: no [jit-stats] line (is MAJIT_STATS honored?)")
        return False

    current = snapshot(fields)
    path = baseline_path(name)
    if record:
        BENCH.mkdir(exist_ok=True)
        path.write_text(current)
        print(f"  recorded {path.relative_to(REPO)}: {current.split()}")
        return True

    if not path.exists():
        print(
            f"  FAIL: {name}: no committed baseline ({path.relative_to(REPO)})"
            f" — record it with scripts/jitstats.py record {name}"
        )
        return False
    failures = floor_regression(parse(path.read_text()), parse(current))
    if failures:
        print(f"  FAIL: {name}: jit-stats regression: {', '.join(failures)}")
        return False
    print(f"  PASS: {name} ({len(jit_out)} bytes, exit {jit_code})")
    return True


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("record", "check"):
        print(__doc__)
        return 2
    if not BINARY.exists():
        print(f"aheui binary not built: {BINARY}", file=sys.stderr)
        return 1
    record = argv[1] == "record"
    names = argv[2:] or FIXTURES
    failed = sum(0 if check_one(n, record=record) else 1 for n in names)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
