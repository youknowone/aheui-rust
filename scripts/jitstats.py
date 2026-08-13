#!/usr/bin/env python3
"""Record and gate the aheui JIT's `[jit-stats]` counters.

Same file format and the same regression floor as `pyre/check.py`, so one
reading applies to both: a sorted `key=value` text file per program, compared
field-by-field with a per-field direction.

Each gated program is also run twice — once with the JIT and once with the
threshold raised out of reach — and the two runs must agree byte-for-byte on
stdout and on the exit code. That A/B is the correctness half: without it a
baseline can go green on a run that miscompiled.

    scripts/jitstats.py record [--jitstress] [fixture | corpus/program ...]
    scripts/jitstats.py check  [--jitstress] [fixture | corpus/program ...]
    scripts/jitstats.py survey
    scripts/jitstats.py dump  [-m NOTE]
    scripts/jitstats.py trend [program ...] [--all]

With no arguments both modes cover the whole committed set: in-tree fixtures,
plus corpus programs when `snippets/` is present as this repo's pinned
submodule, plus a JIT-stressed copy of that pinned corpus at the same threshold
`pyre/check.py` uses for its `*_jitstress` axis. `survey` is the ungated view
for every rpaheui corpus program with a reference output; it is also what
`check` falls back to when the corpus came from `$AHEUI_SNIPPETS` or a sibling
checkout instead of the submodule.

`check` and `survey` each append one row per program run to a local ledger
(`bench/history.jsonl`; `AHEUI_JITSTATS_HISTORY` moves it), and `dump` is the
two of them in one command. `trend` reads the ledger back and prints, per
program, only the runs where a counter actually moved.

That is the whole methodology: a baseline states what the JIT does *today* and
says nothing about the run before it, so without a ledger every "did that
change help?" costs a rebuild of the old tree. Here the record accumulates as
a side effect of running the gate, and `--no-log` is the way to opt out.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BINARY = REPO / "target" / "release" / "aheui"
SNIPPETS = REPO / "snippets"
BENCH = REPO / "bench"

# These paths preserve the six programs that originally established the
# committed baselines. The files live in the pinned snippets submodule, so the
# fixture names remain stable without keeping duplicate samples in the tree.
FIXTURES = {
    "logo": "logo/logo.aheui",
    "99bottles": "99bottles/99bottles.aheui",
    "factorial": "factorial/factorial.aheui",
    "fibonacci": "fibonacci/fibonacci.codroc.aheui",
    "hello": "hello-world/hello.puzzlet.aheui",
    "hello-world": "hello-world/hello-world.puzzlet.aheui",
}

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

# `pyre/check.py`'s `*_jitstress` axis: the same programs under a threshold
# low enough that the merge-point machinery is actually reached. At the
# production threshold only 3 of the 62 pinned programs compile anything and
# none of them exercises `compile_trace`'s JUMP into an existing procedure
# token, so that path is uncovered by a green production gate. Measured over
# the corpus: 1039 -> 3 programs engaged, 200 -> 7, 50 -> 10, and 50 is also
# the best sampled value for that JUMP path (10 is worse). Zero aborts and
# zero A/B mismatches at 50, so it is a floor to hold, not a known-bad state.
JITSTRESS_THRESHOLD = "50"

# Local by default, and not committed: the timings in it are machine-specific,
# and four worktrees of this repo appending to one tracked file would conflict
# on every run. Point `AHEUI_JITSTATS_HISTORY` at a shared path to pool them.
HISTORY = Path(os.environ.get("AHEUI_JITSTATS_HISTORY", BENCH / "history.jsonl"))

# `aheui-jit` path-depends on `../../majit/*`, i.e. on the pyre working tree.
# A row that names only the aheui commit cannot be attributed afterwards —
# most of what moves these counters is a majit change, not an aheui one.
MAJIT_REPO = REPO.parent


def fixture_path(name: str) -> Path:
    return SNIPPETS / FIXTURES[name]


def baseline_path(name: str) -> Path:
    return BENCH / f"{name}.jitstats"


def corpus_baseline_path(name: str) -> Path:
    return BENCH / "corpus" / f"{name}.jitstats"


def jitstress_baseline_path(name: str) -> Path:
    return BENCH / "jitstress" / f"{name}.jitstats"


class Run:
    """One execution of a fixture."""

    def __init__(
        self,
        stdout: bytes,
        code: int,
        fields: dict[str, str],
        secs: float,
        env: dict[str, str],
    ):
        self.stdout = stdout
        self.code = code
        self.fields = fields
        self.secs = secs
        self.env = env


def run_env(*, compile_: bool, threshold: str | None = None) -> dict[str, str]:
    env = dict(os.environ, MAJIT_STATS="1")
    if threshold is not None:
        env["MAJIT_THRESHOLD"] = threshold
    elif not compile_:
        env["MAJIT_THRESHOLD"] = NO_COMPILE_THRESHOLD
    return env


def comparable_env(env: dict[str, str]) -> dict[str, str]:
    """The tuning knobs that made this JIT row, including harness-set knobs."""
    return {
        k: v
        for k, v in env.items()
        if (k.startswith("MAJIT_") or k.startswith("AHEUI_")) and k != "MAJIT_STATS"
    }


def run_path(path: Path, *, compile_: bool, threshold: str | None = None) -> Run:
    """Run one program; capture its stdout, exit code and `[jit-stats]` fields."""
    env = run_env(compile_=compile_, threshold=threshold)
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
        if not line.startswith("[jit-stats]"):
            continue
        # Merge every line rather than keeping the last: a diag line emitted
        # after the counters would otherwise displace them and silently disarm
        # the floor (the lesson `pyre/check.py` records at `_jit_stats_snapshot`).
        for token in line[len("[jit-stats]") :].split():
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
    return Run(proc.stdout, proc.returncode, fields, elapsed, comparable_env(env))


def run(name: str, *, compile_: bool) -> Run:
    return run_path(fixture_path(name), compile_=compile_)


def snapshot(fields: dict[str, str]) -> str:
    return "".join(
        f"{k}={fields[k]}\n" for k in sorted(SNAPSHOT_FIELDS) if k in fields
    )


def corpus_snapshot(jit: "Run") -> str:
    fields = {**jit.fields, "exit": str(jit.code), "out_sha": stdout_sha(jit.stdout)}
    keys = sorted((*SNAPSHOT_FIELDS, "exit", "out_sha"))
    return "".join(f"{k}={fields[k]}\n" for k in keys if k in fields)


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


def git(repo: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", "replace").strip()


def git_head(repo: Path) -> str:
    """`repo`'s short HEAD, suffixed `-dirty` when tracked files are modified."""
    head = git(repo, "rev-parse", "--short", "HEAD")
    if head is None:
        return "?"
    return f"{head}-dirty" if git(repo, "status", "--porcelain", "-uno") else head


def head_committed_at(repo: Path) -> float | None:
    stamp = git(repo, "log", "-1", "--format=%ct")
    return float(stamp) if stamp else None


def build_is_stale() -> bool:
    """Whether the binary predates the commits the row is about to name.

    A row is stamped with the trees' CURRENT commits, but the binary was built
    from whatever they held at build time. A peer rebasing the majit tree in
    between silently misattributes the numbers to a commit that never produced
    them — the exact failure this ledger exists to prevent — so detect it and
    say so rather than record a confident lie.
    """
    if not BINARY.exists():
        return False
    heads = [t for t in (head_committed_at(REPO), head_committed_at(MAJIT_REPO)) if t]
    return bool(heads) and BINARY.stat().st_mtime < max(heads)


def tuning_env() -> dict[str, str]:
    """The `MAJIT_*` / `AHEUI_*` knobs inherited by the harness.

    A row produced under `MAJIT_TRACE_LIMIT=5000` is not comparable with one
    produced at the production limit, and nothing else in the row would say so.
    """
    return {
        k: v
        for k, v in os.environ.items()
        if (k.startswith("MAJIT_") or k.startswith("AHEUI_")) and k != "MAJIT_STATS"
    }


def out_match(stdout: bytes, reference: bytes) -> str:
    """`ok` / `+nl` / `≠` — how stdout compares with a reference.

    Three states rather than a bool with a tolerance. Most of the rpaheui
    corpus's `.out` files carry a trailing newline this interpreter does not
    emit, and folding that into `ok` would also hide a real newline change;
    calling it `+nl` keeps it visible AND keeps it distinct from a program
    whose output genuinely differs. What the ledger watches for is a state
    that *changes*, so a program sitting at `+nl` or `≠` from its first run is
    not noise.
    """
    if stdout == reference:
        return "ok"
    if reference == stdout + b"\n":
        return "+nl"
    return "≠"


def stdout_sha(stdout: bytes) -> str:
    return hashlib.sha256(stdout).hexdigest()[:12]


def ledger_row(
    kind: str, program: str, jit: Run, control: Run | None, out_ok: str
) -> dict:
    return {
        "kind": kind,
        "program": program,
        "exit": jit.code,
        "out_bytes": len(jit.stdout),
        "out_sha": stdout_sha(jit.stdout),
        "out_ok": out_ok,
        "secs": round(jit.secs, 3),
        "nojit_secs": round(control.secs, 3) if control is not None else None,
        "env": jit.env,
        # Every `[jit-stats]` field verbatim, `mc_diag` included: the decline
        # census is what says WHY a program compiled nothing, and a ledger that
        # kept only the totals could not answer that after the fact.
        "stats": jit.fields,
    }


def append_rows(mode: str, note: str, rows: list[dict]) -> None:
    """Stamp `rows` with their provenance and append them to the ledger."""
    if not rows:
        return
    stamp = {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "note": note,
        "aheui": git_head(REPO),
        "majit": git_head(MAJIT_REPO),
        "env": tuning_env(),
        "stale_build": build_is_stale(),
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({**stamp, **row}, sort_keys=True) + "\n")
    print(f"  logged {len(rows)} row(s) to {HISTORY}  [{stamp['aheui']}/{stamp['majit']}]")
    if stamp["stale_build"]:
        print(
            "  ⚠ the binary is older than one of those commits — these rows are\n"
            "    attributed to a tree that did not build them. Rebuild and re-run."
        )


def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    rows = []
    for line in HISTORY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def check_one(name: str, *, record: bool) -> tuple[bool, dict | None]:
    """Run the A/B, then either record or gate. True on success, plus its row."""
    if not fixture_path(name).exists():
        print(f"  FAIL: {name}: no such fixture ({fixture_path(name)})")
        return False, None

    jit = run(name, compile_=True)
    control = run(name, compile_=False)
    # Exact, deliberately: this A/B is one interpreter against itself, so even
    # a trailing-newline difference is a miscompile, not a file convention.
    out_ok = "ok" if jit.stdout == control.stdout and jit.code == control.code else "≠"
    # Built before the verdict, and returned on every path: a run that
    # miscompiled is precisely the row the ledger must not be missing.
    row = ledger_row("fixture", name, jit, control, out_ok)

    if jit.stdout != control.stdout:
        print(
            f"  FAIL: {name}: JIT stdout differs from the un-compiled run "
            f"({len(jit.stdout)} vs {len(control.stdout)} bytes)"
        )
        return False, row
    if jit.code != control.code:
        print(
            f"  FAIL: {name}: JIT exit {jit.code} != un-compiled exit {control.code}"
        )
        return False, row
    if not jit.fields:
        print(f"  FAIL: {name}: no [jit-stats] line (is MAJIT_STATS honored?)")
        return False, row

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
        return True, row

    if not path.exists():
        print(
            f"      FAIL: no committed baseline ({path.relative_to(REPO)})"
            f" — record it with scripts/jitstats.py record {name}"
        )
        return False, row
    baseline = parse(path.read_text())
    parsed = parse(current)
    for line in movement(baseline, parsed):
        print(f"      moved: {line}")
    failures = floor_regression(baseline, parsed)
    if failures:
        print(f"      FAIL: jit-stats regression: {', '.join(failures)}")
        return False, row
    return True, row


def check_corpus_one(
    name: str,
    source_path: Path,
    *,
    record: bool,
    gated: bool,
    kind: str = "corpus",
    threshold: str | None = None,
) -> tuple[bool, dict | None]:
    """Run the corpus A/B, then either record, gate, or survey it."""
    jit = run_path(source_path, compile_=True, threshold=threshold)
    control = run_path(source_path, compile_=False)
    reference = source_path.with_suffix(".out").read_bytes()
    out_ok = out_match(jit.stdout, reference)
    row_name = jitstress_name(name) if kind == "jitstress" else name
    row = ledger_row(kind, row_name, jit, control, out_ok)

    print(counter_row(row_name, jit.fields, jit, control))
    for line in abort_breakdown(jit.fields):
        print(f"      {line}")

    if not gated:
        return True, row

    ok = True
    if jit.stdout != control.stdout:
        print(
            f"      FAIL: JIT stdout differs from the un-compiled run "
            f"({len(jit.stdout)} vs {len(control.stdout)} bytes)"
        )
        ok = False
    if jit.code != control.code:
        print(f"      FAIL: JIT exit {jit.code} != un-compiled exit {control.code}")
        ok = False
    if not jit.fields:
        print(f"      FAIL: no [jit-stats] line (is MAJIT_STATS honored?)")
        ok = False
    if not ok:
        return False, row

    current = corpus_snapshot(jit)
    baseline_file = (
        jitstress_baseline_path(name)
        if kind == "jitstress"
        else corpus_baseline_path(name)
    )
    if record:
        baseline_file.parent.mkdir(parents=True, exist_ok=True)
        baseline_file.write_text(current)
        print(f"      recorded {baseline_file.relative_to(REPO)}")
        return ok, row

    if not baseline_file.exists():
        jitstress_flag = " --jitstress" if kind == "jitstress" else ""
        print(
            f"      FAIL: no committed baseline ({baseline_file.relative_to(REPO)})"
            f" — record it with scripts/jitstats.py record{jitstress_flag} {name}"
        )
        return False, row

    baseline = parse(baseline_file.read_text())
    parsed = parse(current)
    for line in movement(baseline, parsed):
        print(f"      moved: {line}")
    failures = floor_regression(baseline, parsed)
    for field in ("out_sha", "exit"):
        if baseline.get(field) != parsed.get(field):
            failures.append(f"{field} {baseline.get(field)} -> {parsed.get(field)}")
    if failures:
        print(f"      FAIL: jit-stats regression: {', '.join(failures)}")
        return False, row
    return ok, row


def find_snippets() -> tuple[Path, str] | None:
    """`check.sh find_snippets`: the `snippets/` submodule, `$AHEUI_SNIPPETS`,
    else walk up for a sibling checkout.

    The submodule comes first because it is the only one of the three that
    names a commit: `github.com/aheui/snippets` pinned by this repo's gitlink.
    A sibling checkout is whatever that machine last pulled, and the two are
    not interchangeable — `pi/pi.jinseo` alone differs between the canonical
    repo and one fork by a digit count that moves its output from 1006 bytes to
    15001 and its uncompiled run from 0.5s to two minutes. The fallbacks stay
    for a tree checked out without `--recurse-submodules`, and `survey` says
    which one it used.
    """
    submodule = REPO / "snippets"
    if (submodule / ".git").exists():
        return submodule, "submodule"
    env = os.environ.get("AHEUI_SNIPPETS")
    if env and Path(env).is_dir():
        return Path(env), "env"
    for parent in [REPO, *REPO.parents]:
        candidate = parent / "rpaheui" / "snippets"
        if candidate.is_dir():
            return candidate, "sibling"
    return None


SURVEY_HEADER = (
    f"  {'program':<22}{'loops':>6}{'aborted':>8}{'guards':>8}"
    f"{'out':>10}{'exit':>5}{'ref':>6}{'time':>8}"
)


def survey(paths: list[Path]) -> list[dict]:
    """Print, and return ledger rows for, programs that are NOT gated.

    The pinned `snippets/` submodule can hold baselines. `$AHEUI_SNIPPETS` and
    the sibling-checkout fallback cannot: they name whatever that machine last
    pulled, and `pi/pi.jinseo` is known to differ enough between repos to turn
    one baseline into nonsense for the other. This view stays useful for those
    unpinned sources and for exploring which programs the JIT engages with.

    Correctness here is the committed `.out` beside each program, which costs
    nothing and catches what the counters cannot: a row whose numbers improved
    while stdout changed is visibly disqualified. Two things make a corpus
    program sit at a non-`ok` state permanently — most `.out` files carry a
    trailing newline this interpreter does not emit (`+nl`), and a program
    that reads stdin gets EOF here because the run has none (`≠`, e.g.
    `bahmanghui` prints -1 for the integer it cannot read). Neither is a
    defect, which is why what the ledger watches is a state that *changes*.
    """
    rows = []
    for path in paths:
        name = corpus_name(path)
        jit = run_path(path, compile_=True)
        out_ok = out_match(jit.stdout, path.with_suffix(".out").read_bytes())
        compiled = jit.fields.get("loops_compiled", "0")
        aborted = jit.fields.get("loops_aborted", "0")
        guards = jit.fields.get("guard_failures", "0")
        engaged = compiled != "0" or aborted != "0"
        # The un-compiled control run only where a ratio would mean something.
        # On a program the tracer never fires for it is a duplicate run.
        control = run_path(path, compile_=False) if engaged else None
        print(
            f"  {name[:20]:<22}{compiled:>6}{aborted:>8}{guards:>8}"
            f"{len(jit.stdout):>10}{jit.code:>5}{out_ok:>6}"
            f"{jit.secs:>7.2f}s{' *' if engaged else ''}"
        )
        for line in abort_breakdown(jit.fields):
            print(f"      {line}")
        rows.append(ledger_row("corpus", name, jit, control, out_ok))
    return rows


# ── the history view ────────────────────────────────────────────────────────

# What counts as a change. Timings are deliberately excluded: they are noise on
# a shared machine, and a history that treats every run as a change point is a
# log, not a history. The `ABORT_*` breakdown IS included even though nothing
# gates it — `abort_too_long 784 -> 7` is the shape of a fix, while the
# `loops_aborted` total it rolls up into can move for unrelated reasons.
TREND_FIELDS = (*SNAPSHOT_FIELDS, *ABORT_REASONS)


def trend_key(row: dict) -> tuple:
    stats = row.get("stats", {})
    return (
        *(stats.get(field, "0") for field in TREND_FIELDS),
        row.get("out_sha"),
        row.get("exit"),
        row.get("out_ok"),
    )


def trend_deltas(old: dict, new: dict) -> list[str]:
    moved = [
        f"{field} {old.get('stats', {}).get(field, '0')}"
        f"->{new.get('stats', {}).get(field, '0')}"
        for field in TREND_FIELDS
        if old.get("stats", {}).get(field, "0") != new.get("stats", {}).get(field, "0")
    ]
    for key, label in (("out_sha", "stdout"), ("exit", "exit"), ("out_ok", "out_ok")):
        if old.get(key) != new.get(key):
            moved.append(f"{label} {old.get(key)}->{new.get(key)}")
    return moved


TREND_HEADER = (
    f"    {'when':<18}{'aheui/majit':<28}{'loops':>6}{'brdg':>6}"
    f"{'abrtd':>7}{'guards':>8}{'out':>5}{'secs':>8}"
)


def trend_line(row: dict) -> str:
    stats = row.get("stats", {})
    who = f"{row.get('aheui', '?')}/{row.get('majit', '?')}"
    out = row.get("out_ok") or "-"
    return (
        # UTC, and said so: a bare `09:03` on a KST machine reads as local.
        f"    {row.get('at', '?')[:16] + 'Z':<18}{who[:27]:<28}"
        f"{stats.get('loops_compiled', '-'):>6}{stats.get('bridges_compiled', '-'):>6}"
        f"{stats.get('loops_aborted', '-'):>7}{stats.get('guard_failures', '-'):>8}"
        f"{out:>5}{row.get('secs') or 0.0:>8.2f}"
    )


def trend(programs: list[str], *, show_all: bool) -> int:
    """Per program, the runs where something moved — and what moved."""
    rows = load_history()
    if not rows:
        print(f"  ledger is empty ({HISTORY}) — run scripts/jitstats.py dump")
        return 1

    by_program: dict[str, list[dict]] = {}
    for row in rows:
        if programs and not any(trend_program_matches(row["program"], p) for p in programs):
            continue
        by_program.setdefault(row["program"], []).append(row)
    if not by_program:
        print(f"  no ledger rows for {', '.join(programs)}")
        return 1

    for program, history in by_program.items():
        print(f"\n  {program}  ({len(history)} run(s))")
        print(TREND_HEADER)
        previous = None
        unchanged = 0
        for row in history:
            if not show_all and previous is not None and trend_key(row) == trend_key(previous):
                unchanged += 1
                continue
            if unchanged:
                print(f"      … {unchanged} run(s) with nothing moved")
                unchanged = 0
            print(trend_line(row))
            if previous is not None:
                deltas = trend_deltas(previous, row)
                if deltas:
                    print(f"      Δ {', '.join(deltas)}")
            if row.get("note"):
                print(f"      note: {row['note']}")
            if row.get("env"):
                knobs = " ".join(f"{k}={v}" for k, v in sorted(row["env"].items()))
                print(f"      ⚠ tuned run, not comparable with the rest: {knobs}")
            if row.get("stale_build"):
                print("      ⚠ binary older than these commits — misattributed")
            previous = row
        if unchanged:
            print(f"      … {unchanged} run(s) with nothing moved")
    return 0


def jitstress_name(name: str) -> str:
    return f"{name} [jitstress]"


def trend_program_matches(row_program: str, requested: str) -> bool:
    return row_program == requested or row_program == jitstress_name(requested)


def corpus_name(path: Path) -> str:
    # `<dir>/<stem>`, the way bench/README.md's survey table names them —
    # a bare stem collides across the corpus's subdirectories.
    return f"{path.parent.name}/{path.stem}"


def corpus_paths() -> tuple[list[Path], str, Path] | None:
    found = find_snippets()
    if found is None:
        return None
    corpus, source = found
    # Only programs with a committed reference output — the rest are
    # unverifiable, and an unverified number is worse than no number.
    paths = sorted(p for p in corpus.rglob("*.aheui") if p.with_suffix(".out").exists())
    return paths, source, corpus


def corpus_map(paths: list[Path]) -> dict[str, Path]:
    return {corpus_name(path): path for path in paths}


def run_corpus_checks(
    items: list[tuple[str, Path]],
    *,
    source: str,
    record: bool,
    kind: str = "corpus",
) -> tuple[int, list[dict]]:
    gated = source == "submodule"
    unpinned = f"  corpus is unpinned — surveyed, not gated ({source} source)"
    if not gated:
        print(unpinned)
    print(HEADER)
    rows: list[dict] = []
    failed = 0
    for name, path in items:
        ok, row = check_corpus_one(
            name,
            path,
            record=record,
            gated=gated,
            kind=kind,
            threshold=JITSTRESS_THRESHOLD if kind == "jitstress" else None,
        )
        failed += 0 if ok else 1
        if row is not None:
            rows.append(row)
    if not gated:
        print(unpinned)
    return failed if gated else 0, rows


def fixture_or_corpus(name: str, corpus: dict[str, Path]) -> tuple[str, Path | None]:
    if name in FIXTURES:
        return "fixture", None
    if name in corpus:
        return "corpus", corpus[name]
    return "missing", None


def jitstress_requested(name: str) -> str:
    return name.removesuffix(" [jitstress]")


def dump(note: str, *, log: bool) -> int:
    """Gate the fixtures, survey the corpus, append every row.

    The exploratory command to run after a change: it leaves behind the corpus
    rows that make the NEXT change measurable, without turning that sweep into
    a verdict.
    """
    print(HEADER)
    rows: list[dict] = []
    failed = 0
    for name in FIXTURES:
        ok, row = check_one(name, record=False)
        failed += 0 if ok else 1
        if row is not None:
            rows.append(row)
    print(f"  {len(FIXTURES)} checked, {failed} failed")

    found = corpus_paths()
    if found is None:
        print("\n  rpaheui corpus not found (set AHEUI_SNIPPETS) — fixtures only")
    else:
        paths, _, _ = found
        print(f"\n  corpus survey ({len(paths)} programs, informational — nothing gated)")
        print(SURVEY_HEADER)
        rows.extend(survey(paths))
        print("  * = the JIT engaged")

    if log:
        print()
        append_rows("dump", note, rows)
    return 1 if failed else 0


MODES = ("record", "check", "survey", "dump", "trend")


def take_option(args: list[str], *names: str) -> str | None:
    """Pull `--opt VALUE` out of `args` in place; None when it is absent."""
    for name in names:
        if name in args:
            index = args.index(name)
            del args[index]
            return args.pop(index) if index < len(args) else ""
    return None


def take_flag(args: list[str], *names: str) -> bool:
    found = False
    for name in names:
        while name in args:
            args.remove(name)
            found = True
    return found


def main(argv: list[str]) -> int:
    args = list(argv[1:])
    if not args or args[0] not in MODES:
        print(__doc__)
        return 2
    mode = args.pop(0)
    note = take_option(args, "-m", "--note") or ""
    show_all = take_flag(args, "--all")
    jitstress_only = take_flag(args, "--jitstress")
    log = not take_flag(args, "--no-log")

    if mode == "trend":
        return trend(args, show_all=show_all)

    if not BINARY.exists():
        print(f"aheui binary not built: {BINARY}", file=sys.stderr)
        return 1

    if mode == "dump":
        return dump(note, log=log)

    if mode == "survey":
        found = corpus_paths()
        if found is None:
            print("rpaheui snippet corpus not found; set AHEUI_SNIPPETS")
            return 1
        paths, source, _ = found
        print(f"  corpus survey ({len(paths)} programs, informational — nothing gated)")
        print(f"  source: {source}")
        print(SURVEY_HEADER)
        rows = survey(paths)
        print("  * = the JIT engaged")
        if log:
            append_rows(mode, note, rows)
        return 0

    record = mode == "record"
    found = corpus_paths()
    paths: list[Path] = []
    source = ""
    corpus: dict[str, Path] = {}
    if found is not None:
        paths, source, _ = found
        corpus = corpus_map(paths)

    print(HEADER)
    rows = []
    failed = 0
    fixture_count = 0
    corpus_items: list[tuple[str, Path]] = []
    jitstress_items: list[tuple[str, Path]] = []

    if args:
        for name in args:
            if jitstress_only:
                jitstress_target = jitstress_requested(name)
                path = corpus.get(jitstress_target)
                if path is None:
                    corpus_hint = "corpus program" if found is not None else "fixture"
                    print(f"  FAIL: {name}: no such jitstress {corpus_hint}")
                    failed += 1
                else:
                    jitstress_items.append((jitstress_target, path))
                continue
            kind, path = fixture_or_corpus(name, corpus)
            if kind == "fixture":
                ok, row = check_one(name, record=record)
                fixture_count += 1
                failed += 0 if ok else 1
                if row is not None:
                    rows.append(row)
            elif kind == "corpus" and path is not None:
                corpus_items.append((name, path))
            else:
                fixture = fixture_path(name)
                corpus_hint = " or corpus program" if found is not None else ""
                print(f"  FAIL: {name}: no such fixture{corpus_hint} ({fixture})")
                failed += 1
    else:
        if paths:
            all_corpus_items = [(corpus_name(path), path) for path in paths]
            if source == "submodule":
                jitstress_items = all_corpus_items
            if not jitstress_only:
                corpus_items = all_corpus_items
        if not jitstress_only:
            for name in FIXTURES:
                ok, row = check_one(name, record=record)
                fixture_count += 1
                failed += 0 if ok else 1
                if row is not None:
                    rows.append(row)

    corpus_failed = 0
    if corpus_items:
        print()
        corpus_failed, corpus_rows = run_corpus_checks(
            corpus_items, source=source, record=record
        )
        failed += corpus_failed
        rows.extend(corpus_rows)
    jitstress_failed = 0
    if jitstress_items:
        if source != "submodule":
            print(f"\n  jitstress corpus is unpinned — not gated ({source} source)")
        else:
            print()
            jitstress_failed, jitstress_rows = run_corpus_checks(
                jitstress_items, source=source, record=record, kind="jitstress"
            )
            failed += jitstress_failed
            rows.extend(jitstress_rows)
    verb = "recorded" if record else "checked"
    corpus_verb = verb if source == "submodule" else "surveyed"
    jitstress_verb = verb if source == "submodule" else "skipped"
    if corpus_items and jitstress_items and source == "submodule":
        print(
            f"  {fixture_count} fixtures + {len(corpus_items)} corpus"
            f" + {len(jitstress_items)} jitstress {verb}, {failed} failed"
        )
    elif corpus_items:
        print(f"  {fixture_count} fixtures + {len(corpus_items)} corpus {corpus_verb}, {failed} failed")
    elif jitstress_items:
        print(
            f"  {len(jitstress_items)} jitstress {jitstress_verb}, {failed} failed"
        )
    else:
        print(f"  {fixture_count} fixtures {verb}, {failed} failed")
    if log:
        append_rows(mode, note, rows)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
