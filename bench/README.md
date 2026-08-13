# aheui JIT stats baselines

One `.jitstats` per gated program, in the same format
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
python3 scripts/jitstats.py record logo/logo  # one corpus program
python3 scripts/jitstats.py record pi/pi.jinseo
python3 scripts/jitstats.py record --jitstress pi/pi.jinseo  # its jitstress row
python3 scripts/jitstats.py survey        # the rpaheui corpus, ungated
python3 scripts/jitstats.py trend         # what moved between runs, per program
```

`check` prints the counters for every gated program, not just pass/fail — a run that
only says PASS says nothing about what the JIT did. Any gated counter that moved
is listed with the headroom left before its gate, so a number sitting one step
below the threshold is visible before the run that trips it.

A program that aborted also gets its `Counters.ABORT_*` breakdown. Those are
diagnosis, not part of the recorded surface: pinning them would gate the same
event twice through `loops_aborted`. They exist because `JitStats` carries only
the total and the profiler's own `print_stats` is behind `MAJIT_LOG`, which is
far too slow to enable on a workload big enough to abort interestingly.

## survey

`survey` runs every rpaheui corpus program that has a reference output and
prints its counters, gating nothing. That is the right mode for `$AHEUI_SNIPPETS`
or a sibling `rpaheui/snippets` checkout: those sources name whatever that
machine last pulled, so a baseline keyed to them is not reproducible elsewhere.
When the corpus comes from this repo's pinned `snippets/` submodule, `check`
and `record` gate it under `bench/corpus/<dir>/<stem>.jitstats`.

At the production 1039 back-edge threshold, **3 of the 62 pinned programs**
reach it (submodule `4961b05`, majit `2674bdcb06b`, aheui `13e4eb9`):

| program | loops | bridges | aborted | guards | jit / uncompiled | out |
|---|---|---|---|---|---|---|
| `logo` | 1 | 1 | 0 | 201 | 0.46s / 4.56s | 996310, `ok` |
| `pi/pi.jinseo` | 3 | 1 | 0 | 674 | 0.04s / 0.63s | 1005, `+nl` |
| `standard/loop` | 1 | 0 | 0 | 1 | 0.006s / 0.14s | 1, `ok` |

Everything else compiles nothing at the production threshold — they never reach
it, so their production baselines are all-zero and gate only the badness fields.

`pi/pi.jinseo` is the program that exercises the merge-point machinery, and it
is also the one place where which corpus you point at decides what you measure.
The pinned `aheui/snippets` version prints 1006 bytes; a fork carries the same
algorithm with a larger digit count that prints 15001 and takes 123s
uncompiled. The larger fork is not gated because it is not pinned, but it is
why the canonical `pi/pi.jinseo` row matters.

The pinned corpus therefore has a second gated axis under
`bench/jitstress/<dir>/<stem>.jitstats`, run with `MAJIT_THRESHOLD=50`. This
mirrors `pyre/check.py`'s `*_jitstress` rows: the same program under a lower
threshold, with its own baseline and the same byte-exact JIT-vs-uncompiled A/B.
Measured over the pinned corpus, production `1039` engaged 3 programs and hit
`compile_trace`'s JUMP into an existing procedure token 0 times for
`pi/pi.jinseo`; `200` engaged 7 and hit it 5 times; `50` engaged 10 and hit it
8 times. `10` was worse for that JUMP path at 7. The `50` sweep had zero aborts
and zero A/B mismatches, so the jitstress axis closes the coverage gap without
turning the unpinned fork into a gate.

The corpus and jitstress axes use every pinned snippet with a reference output;
fallback snippet sources remain survey-only because they do not identify a
pinned commit.

## The ledger and `trend`

A baseline states what the JIT does *today*. It says nothing about the run
before it, so without a record every "did that change help?" costs a rebuild of
the old tree. The ledger is that record, and it fills itself: `check` and
`survey` each append one row per program they run.

```sh
python3 scripts/jitstats.py check -m "merge-point segmenting trigger"
python3 scripts/jitstats.py trend                 # every program, change points only
python3 scripts/jitstats.py trend pi/pi.jinseo    # one program
python3 scripts/jitstats.py trend logo/logo --all # every row, not just the changes
python3 scripts/jitstats.py check --no-log        # opt out for one run
```

`check` is the exploratory command to run after a change. It gates both corpus
axes when the pinned submodule is present; with a fallback source it records an
ungated survey instead.

Corpus survey rows are checked against the `.out` committed beside each
program, in three states rather than a pass/fail with a tolerance: `ok`, `+nl` when the
only difference is the trailing newline most of those files carry and this
interpreter does not emit, and `≠` otherwise — which for a handful of programs
just means they read stdin and the survey gives them none (`bahmanghui` prints
`-1` for the integer it cannot read). Folding `+nl` into `ok` would also hide a
real newline change, so it stays its own state. Pinned corpus and jitstress
gating are stricter: the baseline records stdout's SHA prefix and the exit
code, and any change in either fails. The pinned corpus and jitstress A/B is
exact — it compares one interpreter against itself, so even a newline
difference there is a miscompile.

Each row carries the counters, the full `ABORT_*` breakdown, the `mc_diag`
decline census, stdout's length/sha/exit, the wall times, and the `-m` note —
plus **both** commits: the aheui HEAD and the majit one. `aheui-jit`
path-depends on `../../majit/*`, so most of what moves these numbers is a majit
change; a row naming only one of the two cannot be attributed afterwards. A
`-dirty` suffix means that tree had uncommitted changes, and any `MAJIT_*` /
`AHEUI_*` knob in the environment is recorded too — a row taken under
`MAJIT_TRACE_LIMIT=5000` is not comparable with one at the production limit,
and nothing else in the row would say so.

A row names the trees' commits *as of the run*, but the binary was built from
whatever they held earlier. When the majit tree moves in between — a peer
rebasing it is enough — the numbers get attributed to a commit that never
produced them, which is the one failure this ledger exists to prevent. So the
binary's mtime is compared against both HEAD commit times and the row carries
`stale_build`; `check` says so loudly and `trend` marks the row. Rebuild and
re-run rather than reading a flagged row.

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
read the same way. The production corpus and jitstress corpus use the same field
directions; jitstress only changes the JIT threshold and keeps a separate
baseline identity.

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

Before comparing counters, every pinned corpus program runs twice
— once normally and once with `MAJIT_THRESHOLD` raised out of reach so the
tracer never fires — and the two runs must agree byte-for-byte on stdout and on
the exit code. Without that, a baseline can go green on a run that miscompiled.

`MAJIT_THRESHOLD` is the right control rather than `--no-jit`: the flag selects
a *different* interpreter (`aheuinterpreter`), and it only exists on a `naive`
build, while a huge threshold keeps `aheui_jit::mainloop` on the same path with
the tracer idle.
