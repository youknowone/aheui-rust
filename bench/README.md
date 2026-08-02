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
```

Fixtures are the in-tree programs under `aheui-wasm/web/samples/`, not the
rpaheui snippet corpus: the corpus is an unpinned sibling checkout, so a
baseline keyed to it is not reproducible on another machine.

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
