# aheui JIT stats baselines

One `<fixture>.jitstats` per gated program, in the same format
`pyre/bench/synth/*.jitstats` uses: sorted `key=value` lines holding the
counters `majit_metainterp::JitStats` exposes.

```
bridges_compiled=0
guard_failures=12
internal_compile_panics=0
loops_aborted=0
loops_compiled=3
```

Record and gate with:

```sh
cargo build -p aheui --release
python3 scripts/jitstats.py record        # rewrite every baseline
python3 scripts/jitstats.py check         # what check.sh section 5 runs
python3 scripts/jitstats.py record logo   # one fixture
python3 scripts/jitstats.py survey        # the whole rpaheui corpus, ungated
python3 scripts/jitstats.py dump          # check + survey, appended to the ledger
python3 scripts/jitstats.py trend         # what moved between runs, per program
```

`check` prints the counters for every fixture, not just pass/fail — a run that
only says PASS says nothing about what the JIT did. Any gated counter that moved
is listed with the headroom left before its gate, so a number sitting one step
below the threshold is visible before the run that trips it.

A fixture that aborted also gets its `Counters.ABORT_*` breakdown. Those are
diagnosis, not part of the recorded surface: pinning them would gate the same
event twice through `loops_aborted`. They exist because `JitStats` carries only
the total and the profiler's own `print_stats` is behind `MAJIT_LOG`, which is
far too slow to enable on a workload big enough to abort interestingly.

## survey

`survey` runs every rpaheui corpus program that has a reference output and
prints its counters, gating nothing — the corpus is an unpinned sibling
checkout, so nothing there can hold a baseline. It answers the question the gate
cannot: which programs does the JIT engage with at all.

At the production 1039 back-edge threshold, **3 of 55** do:

| program | loops | aborted | note |
|---|---|---|---|
| `logo` | 1 | 0 | the gated fixture |
| `standard/loop` | 1 | 0 | 92 bytes, sub-10ms |
| `pi/pi.jinseo` | 11 | 39 | `too_long=7 bad_loop=31 segmented=1`, ~3.4s |

Everything else compiles nothing — they never reach the threshold, so their
baselines are all-zero and gate only the badness fields.

Fixtures are the in-tree programs under `aheui-wasm/web/samples/`, not the
rpaheui snippet corpus: the corpus is an unpinned sibling checkout, so a
baseline keyed to it is not reproducible on another machine.

## The ledger — `dump` and `trend`

A baseline states what the JIT does *today*. It says nothing about the run
before it, so without a record every "did that change help?" costs a rebuild of
the old tree. The ledger is that record, and it fills itself: `check` and
`survey` each append one row per program they run.

```sh
python3 scripts/jitstats.py dump -m "merge-point segmenting trigger"
python3 scripts/jitstats.py trend                 # every program, change points only
python3 scripts/jitstats.py trend pi/pi.jinseo    # one program
python3 scripts/jitstats.py trend logo --all      # every row, not just the changes
python3 scripts/jitstats.py check --no-log        # opt out for one run
```

`dump` is the command to run after a change: it reaches the same verdict
`check` does — the gated fixtures, with the un-compiled A/B — and then sweeps
the whole corpus, so the programs that actually stress the JIT (`pi/pi.jinseo`)
are in the record even though nothing can gate them.

Corpus rows are checked against the `.out` committed beside each program, in
three states rather than a pass/fail with a tolerance: `ok`, `+nl` when the
only difference is the trailing newline most of those files carry and this
interpreter does not emit, and `≠` otherwise — which for a handful of programs
just means they read stdin and the survey gives them none (`bahmanghui` prints
`-1` for the integer it cannot read). Folding `+nl` into `ok` would also hide a
real newline change, so it stays its own state; what disqualifies a row is a
state that *changes*, not one that has always been `≠`. The fixtures' own A/B
is exact by contrast — it compares one interpreter against itself, so even a
newline difference there is a miscompile.

Each row carries the counters, the full `ABORT_*` breakdown, the `mc_diag`
decline census, stdout's length/sha/exit, the wall times, and the `-m` note —
plus **both** commits: the aheui HEAD and the majit one. `aheui-jit`
path-depends on `../../majit/*`, so most of what moves these numbers is a majit
change; a row naming only one of the two cannot be attributed afterwards. A
`-dirty` suffix means that tree had uncommitted changes, and any `MAJIT_*` /
`AHEUI_*` knob in the environment is recorded too — a row taken under
`MAJIT_TRACE_LIMIT=5000` is not comparable with one at the production limit,
and nothing else in the row would say so.

`trend` prints only the runs where something moved, with a `Δ` naming what.
Timings are deliberately not part of that test — they are noise on a shared
machine, and a history where every run is a change point is a log, not a
history. The `ABORT_*` fields *are*, even though nothing gates them:
`abort_too_long 784 -> 7` is the shape of a fix, while the `loops_aborted`
total it rolls up into moves for unrelated reasons too.

The file is `bench/history.jsonl`, gitignored — the timings are
machine-specific and four worktrees of this repo would conflict on it every
run. `AHEUI_JITSTATS_HISTORY` points it somewhere shared. When a run is worth
keeping for good, promote it by hand into the survey table above; the ledger is
the raw high-frequency record, the table is the curated one.

## What each counter gates

Copied from `pyre/check.py`'s `_jit_stats_regression_floor` so the two suites
read the same way:

| counter | fails on |
|---|---|
| `loops_aborted` | any rise above the baseline |
| `internal_compile_panics` | any rise (healthy value is 0) |
| `guard_failures` | a rise past `base + max(base // 4, 2)` |
| `loops_compiled` | any fall below the baseline |
| `bridges_compiled` | never — it moves both ways under ordinary tuning |

A baseline is a record of what the JIT does today, not a target: it pins known
declines so a *new* one is visible. Re-record only when the change that moved a
number is understood and intended.

## The correctness half

Before comparing counters, every fixture runs twice — once normally and once
with `MAJIT_THRESHOLD` raised out of reach so the tracer never fires — and the
two runs must agree byte-for-byte on stdout and on the exit code. Without that,
a baseline can go green on a run that miscompiled.

`MAJIT_THRESHOLD` is the right control rather than `--no-jit`: the flag selects
a *different* interpreter (`aheuinterpreter`), and it only exists on a `naive`
build, while a huge threshold keeps `aheui_jit::mainloop` on the same path with
the tracer idle.
