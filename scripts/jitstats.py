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
    scripts/jitstats.py survey

With no fixture arguments both modes cover the whole committed set. `survey`
prints the counters for every rpaheui corpus program that has a reference
output, gating nothing — it answers which programs the JIT engages with at all.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
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


class Run:
    """One execution of a fixture."""

    def __init__(self, stdout: bytes, code: int, fields: dict[str, str], secs: float):
        self.stdout = stdout
        self.code = code
        self.fields = fields
        self.secs = secs


def run(name: str, *, compile_: bool) -> Run:
    """Run one fixture; capture its stdout, exit code and `[jit-stats]` fields."""
    env = dict(os.environ, MAJIT_STATS="1")
    if not compile_:
        env["MAJIT_THRESHOLD"] = NO_COMPILE_THRESHOLD
    start = time.monotonic()
    proc = subprocess.run(
        [str(BINARY), str(fixture_path(name))],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        env=env,
    )
    elapsed = time.monotonic() - start
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
    return Run(proc.stdout, proc.returncode, fields, elapsed)


def snapshot(fields: dict[str, str]) -> str:
    return "".join(
        f"{k}={fields[k]}\n" for k in sorted(SNAPSHOT_FIELDS) if k in fields
    )


def parse(text: str) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in text.splitlines() if "=" in line
    )


def field_verdict(field: str, base: int, cur: int) -> tuple[bool, str]:
    """Whether `field` regressed, and the headroom left before it would.

    The headroom is the point of printing this: a counter sitting one step
    below its gate is worth seeing before the run that trips it.
    """
    if field in RISE_BOUNDED_FIELDS:
        limit = base + max(base // 4, 2)
        return cur > limit, f"gate >{limit}"
    if field in FALL_FIELDS:
        return cur < base, f"gate <{base}"
    return cur > base, f"gate >{base}"


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


HEADER = (
    f"  {'fixture':<14}{'loops':>6}{'bridges':>8}{'aborted':>8}"
    f"{'guards':>8}{'panics':>7}{'out':>9}{'exit':>5}{'jit':>8}{'nojit':>8}{'x':>6}"
)


def counter_row(name: str, fields: dict[str, str], jit: Run, control: Run) -> str:
    def n(key: str) -> str:
        return fields.get(key, "-")

    speedup = control.secs / jit.secs if jit.secs > 0 else 0.0
    return (
        f"  {name:<14}{n('loops_compiled'):>6}{n('bridges_compiled'):>8}"
        f"{n('loops_aborted'):>8}{n('guard_failures'):>8}"
        f"{n('internal_compile_panics'):>7}{len(jit.stdout):>9}{jit.code:>5}"
        f"{jit.secs:>7.2f}s{control.secs:>7.2f}s{speedup:>5.1f}x"
    )


ABORT_REASONS = (
    "abort_too_long",
    "abort_bridge",
    "abort_bad_loop",
    "abort_escape",
    "abort_force_quasiimmut",
    "abort_segmented_trace",
)


def abort_breakdown(fields: dict[str, str]) -> list[str]:
    """`Counters.ABORT_*` for a run that aborted, so the total has a cause.

    Not part of the recorded surface: these are diagnosis, and pinning them
    would gate the same event twice through `loops_aborted`.
    """
    if fields.get("loops_aborted", "0") == "0":
        return []
    reasons = [
        f"{r.removeprefix('abort_')}={fields[r]}"
        for r in ABORT_REASONS
        if fields.get(r, "0") != "0"
    ]
    return [f"aborts: {' '.join(reasons)}"] if reasons else []


def movement(old: dict[str, str], new: dict[str, str]) -> list[str]:
    """Every gated counter that moved, with the headroom left before its gate."""
    moved = []
    for field in (*BADNESS_FIELDS, *RISE_BOUNDED_FIELDS, *FALL_FIELDS):
        if field not in BADNESS_FIELDS and field not in old:
            continue
        base = int(old.get(field, 0))
        cur = int(new.get(field, 0))
        if cur == base:
            continue
        _, gate = field_verdict(field, base, cur)
        moved.append(f"{field} {base} -> {cur} ({gate})")
    return moved


def check_one(name: str, *, record: bool) -> bool:
    """Run the A/B, then either record or gate. True on success."""
    if not fixture_path(name).exists():
        print(f"  FAIL: {name}: no such fixture ({fixture_path(name)})")
        return False

    jit = run(name, compile_=True)
    control = run(name, compile_=False)

    if jit.stdout != control.stdout:
        print(
            f"  FAIL: {name}: JIT stdout differs from the un-compiled run "
            f"({len(jit.stdout)} vs {len(control.stdout)} bytes)"
        )
        return False
    if jit.code != control.code:
        print(
            f"  FAIL: {name}: JIT exit {jit.code} != un-compiled exit {control.code}"
        )
        return False
    if not jit.fields:
        print(f"  FAIL: {name}: no [jit-stats] line (is MAJIT_STATS honored?)")
        return False

    # The counters themselves, always — a run that only says PASS tells you
    # nothing about what the JIT actually did, and the numbers are the point.
    print(counter_row(name, jit.fields, jit, control))
    for line in abort_breakdown(jit.fields):
        print(f"      {line}")

    current = snapshot(jit.fields)
    path = baseline_path(name)
    if record:
        BENCH.mkdir(exist_ok=True)
        path.write_text(current)
        print(f"      recorded {path.relative_to(REPO)}")
        return True

    if not path.exists():
        print(
            f"      FAIL: no committed baseline ({path.relative_to(REPO)})"
            f" — record it with scripts/jitstats.py record {name}"
        )
        return False
    baseline = parse(path.read_text())
    parsed = parse(current)
    for line in movement(baseline, parsed):
        print(f"      moved: {line}")
    failures = floor_regression(baseline, parsed)
    if failures:
        print(f"      FAIL: jit-stats regression: {', '.join(failures)}")
        return False
    return True


def find_snippets() -> Path | None:
    """`check.sh find_snippets`: `$AHEUI_SNIPPETS`, else walk up for the corpus."""
    env = os.environ.get("AHEUI_SNIPPETS")
    if env and Path(env).is_dir():
        return Path(env)
    for parent in [REPO, *REPO.parents]:
        candidate = parent / "rpaheui" / "snippets"
        if candidate.is_dir():
            return candidate
    return None


def survey(paths: list[Path]) -> None:
    """Print counters for programs that are NOT gated.

    The committed fixture set is in-tree and reproducible; the rpaheui corpus is
    an unpinned sibling checkout, so nothing here can hold a baseline. Printing
    it anyway answers the question the gate cannot: which programs does the JIT
    engage with at all, and which are candidates worth vendoring.
    """
    for path in paths:
        env = dict(os.environ, MAJIT_STATS="1")
        start = time.monotonic()
        proc = subprocess.run(
            [str(BINARY), str(path)],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=env,
        )
        elapsed = time.monotonic() - start
        fields: dict[str, str] = {}
        for line in proc.stderr.decode("utf-8", "replace").splitlines():
            if line.startswith("[jit-stats]"):
                for token in line[len("[jit-stats]") :].split():
                    key, sep, value = token.partition("=")
                    if sep:
                        fields[key] = value
        compiled = fields.get("loops_compiled", "0")
        aborted = fields.get("loops_aborted", "0")
        guards = fields.get("guard_failures", "0")
        mark = " *" if compiled != "0" or aborted != "0" else ""
        print(
            f"  {path.stem[:20]:<22}{compiled:>6}{aborted:>8}{guards:>8}"
            f"{len(proc.stdout):>10}{proc.returncode:>5}{elapsed:>7.2f}s{mark}"
        )
        for line in abort_breakdown(fields):
            print(f"      {line}")


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in ("record", "check", "survey"):
        print(__doc__)
        return 2
    if not BINARY.exists():
        print(f"aheui binary not built: {BINARY}", file=sys.stderr)
        return 1

    if argv[1] == "survey":
        corpus = find_snippets()
        if corpus is None:
            print("rpaheui snippet corpus not found; set AHEUI_SNIPPETS")
            return 1
        # Only programs with a committed reference output — the rest are
        # unverifiable, and an unverified number is worse than no number.
        paths = sorted(
            p for p in corpus.rglob("*.aheui") if p.with_suffix(".out").exists()
        )
        print(f"  corpus survey ({len(paths)} programs, informational — nothing gated)")
        print(
            f"  {'program':<22}{'loops':>6}{'aborted':>8}{'guards':>8}"
            f"{'out':>10}{'exit':>5}{'time':>8}"
        )
        survey(paths)
        print("  * = the JIT engaged")
        return 0

    record = argv[1] == "record"
    names = argv[2:] or FIXTURES
    print(HEADER)
    failed = sum(0 if check_one(n, record=record) else 1 for n in names)
    verb = "recorded" if record else "checked"
    print(f"  {len(names)} {verb}, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
