# Aheui Rust

This workspace contains the Aheui interpreter, generated majit JIT, and
compaheuiler's C/Rust/Cranelift/WAT ahead-of-time compilers. Raw integer execution
promotes to tagged BigInt only when needed; this dual mode is intentional.

`aheuinterpreter` owns the fast interpreter state and opcode dispatch. Ordinary
execution and tracing use that same source; `aheui-jit` supplies generated
artifacts, their loader, and startup bindings. The interpreter does not depend
on the JIT consumer. Builds without the `jit` feature retain concrete execution
but omit tracing metadata. The portal still uses majit's source macros alongside
LLBC-generated helpers; this is not yet an LLBC-only generation pipeline.
The artifact build translates an explicit set of runtime helpers, not a second
unused copy of the portal or the JIT engine's own setup routines.

## Standalone non-JIT build

```sh
git submodule update --init
cargo build --release -p compaheuiler
cargo build --release -p aheui --no-default-features --features naive,malachite-bigint
```

See [the compiler guide](compaheuiler/README.md) and
[browser library guide](aheui-wasm/README.md).

## Generated JIT bootstrap

Runtime helpers are translated through Charon LLBC and majit; interpreter macros
generate the portal. A Git dependency fetch alone does not generate the helper
inputs. Use this supported checkout layout:

```text
pyre/                  # youknowone/pyre, exact SHA from Cargo.toml
  scripts/
  majit/
  aheui/               # this checkout, including snippets submodule
```

From `pyre/`, check out the majit revision recorded in this repository's
`[workspace.dependencies]`, then run:

```sh
python3 scripts/install-charon.py
cd aheui
git submodule update --init
mkdir -p .cargo
cp .github/majit-ci-config.toml .cargo/config.toml
cargo generate-lockfile
python3 scripts/check_config.py
python3 scripts/extract-llbc.py
cargo build --release -p aheui --no-default-features --features jit,dynasm,malachite-bigint
target/release/aheui snippets/hello-world/hello.puzzlet.aheui
```

The local Cargo patch must point at that same majit checkout. Re-extract after
runtime source changes; extraction checks its inputs for freshness. Interpreter
portal changes take effect through its Rust build. Full interpreter LLBC
extraction remains available separately for translator census work.
To select Cranelift JIT, replace `dynasm` with `cranelift`; `aot-cranelift` is a
separate compiler feature. Do not combine native backend features.

## Verification

Do not launch full builds or corpus tests concurrently. On macOS, run potentially
runaway tests inside a Linux VM/container with an explicit RAM cap: macOS rejects
the POSIX data/address-space limits used by our Linux supervisor. A sampled RSS
watchdog alone is not a hard aggregate-memory guarantee.

The parent Pyre checkout provides `scripts/run-limited.py`: use it inside the
memory-capped environment to enforce a process-tree RSS budget, per-process
allocation limit, timeout, output cap, and single protected command at a time.
It forces `CARGO_BUILD_JOBS=1`, `RUST_TEST_THREADS=1` and serial LLBC layouts.
It refuses native macOS execution. For example, in a Linux VM capped at 4 GiB:

```sh
python3 ../scripts/run-limited.py --memory-mib 3072 --process-mib 2048 --file-mib 512 --seconds 1800 -- cargo test -p compaheuiler --test bigint_test process::tests
```

Corpus scripts spool stdout/stderr into size-limited temporary files instead of
accumulating unlimited output in RAM. This does not replace the VM memory cap:
the program under test can still have an allocation or non-termination bug.

Generated C/Rust programs reclaim unreachable boxed integers at block entries,
including inner self-loops. Stack slots, queues, ports and the sticky port value
are roots; register temporaries are flushed before control leaves a block.
Raw/tagged representation is unchanged. Collection is deferred during raw mode
and within arithmetic operations, and its threshold grows with live bigint data.
The bigint integration tests force collection at every tagged block entry;
`bigint_collection` also checks reclamation and root preservation directly.

```sh
cargo test --workspace --features dynasm --no-fail-fast
python3 scripts/jitstats.py check
python3 scripts/jitstats.py sweep
python3 scripts/opcensus.py check
```

Run these against the freshly built release executable. Benchmark history records
resolved majit source at measurement time and the executable's SHA-256; a source
revision alone is not proof of what an older executable contains. Preserve
baseline targets unless a change is intentional and justified. See
[benchmark details](bench/README.md). `check.sh` also runs controlled-machine
timing checks; set `AHEUI_SELF_INTERP` to the separate `aheui.aheui` source to
enable self-interpreter checks. Temporary diagnostic artifacts are reported by
the script and retained for inspection.

For WASI JIT, install `wasm32-wasip1`, build the guest with
`--no-default-features --features jit,naive,malachite-bigint,wasm-host`, build
`aheui-wasm-runner` natively, then run `python3 scripts/check_wasi.py`.
The runner uses Wasmtime's user cache, never an adjacent `.cwasm` from the input
directory. `AHEUI_WASM_NO_MODULE_CACHE=1` disables it for cold-load measurements.
