//! Native wasmtime host for the `wasm32-wasip1` build of aheui.
//!
//! The guest is an ordinary WASI command — it opens its program from a
//! preopened directory and writes to stdout — but its JIT needs more than WASI
//! provides. A wasm module cannot compile or instantiate another wasm module,
//! and it cannot call a function whose signature it only knows at runtime, so
//! the wasm backend leaves both to its embedder and reaches them through the
//! `majit_host.*` imports this runner satisfies:
//!
//!   * `jit_compile_wasm` / `jit_replace_wasm` / `jit_free_wasm` — compile an
//!     emitted trace module against the guest's own memory and function table,
//!     and publish it under a table index the guest keeps as its handle.
//!   * `jit_execute_wasm` — call a published trace by that index.
//!   * `jit_call_host` — perform a residual call on the guest's behalf,
//!     reflecting the callee's real wasm signature.
//!
//! The guest must therefore export `memory` and `__indirect_function_table`,
//! and its table must be growable:
//!
//! ```text
//! RUSTFLAGS="-C link-arg=--export-table -C link-arg=--growable-table" \
//!   cargo build --release --target wasm32-wasip1 -p aheui \
//!     --no-default-features --features jit,naive,malachite-bigint,wasm-host
//! ```
//!
//! Unlike a `wasm32-unknown-unknown` guest, a WASI command has a real
//! environment, so `MAJIT_THRESHOLD`, `MAJIT_STATS` and the rest are inherited
//! from this process and take effect inside the guest.

use majit_backend_wasm_host::CALL_RESULT_OFS;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use wasmtime::{Caller, Config, Engine, Error, Func, Linker, Module, Result, Store};
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

/// Per-store host state shared by every import callback.
struct Host {
    jit: majit_backend_wasm_host::TraceState,
    wasi: WasiP1Ctx,
    /// `MAJIT_STATS` diagnostics: trace modules compiled, time spent compiling
    /// them, traces entered through the import, and residual calls reflected
    /// back into the guest.
    compile_count: u64,
    compile_time_ns: u128,
    /// Only counts entries that cross the `jit_execute_wasm` import. A guest
    /// built against plain wasm imports calls its traces straight through the
    /// shared table instead — the handle IS the table slot — so this stays 0
    /// there and the guest's own tally is the one that counts trace entries.
    execute_count: u64,
    call_count: u64,
    /// Nanoseconds spent turning the guest module into something runnable,
    /// reported so a measurement can tell that cost apart from the program's.
    guest_load_time_ns: u128,
    /// Whether that time went into compiling the guest rather than reading back
    /// an artefact compiled by an earlier run.
    guest_compiled: bool,
}

impl majit_backend_wasm_host::HostState for Host {
    fn traces(&self) -> &majit_backend_wasm_host::TraceState {
        &self.jit
    }
    fn traces_mut(&mut self) -> &mut majit_backend_wasm_host::TraceState {
        &mut self.jit
    }
}

impl Host {
    fn new(wasi: WasiP1Ctx) -> Self {
        Self {
            wasi,
            jit: Default::default(),
            compile_count: 0,
            compile_time_ns: 0,
            execute_count: 0,
            call_count: 0,
            guest_load_time_ns: 0,
            guest_compiled: false,
        }
    }
}

fn fatal(msg: &str) -> ! {
    eprintln!("aheui-wasm-runner: {msg}");
    std::process::exit(2)
}

/// A `--dir` mapping. `HOST` alone preopens `HOST` under its own name.
struct DirMapping {
    host: PathBuf,
    guest: String,
}

fn parse_dir(spec: &str) -> DirMapping {
    match spec.split_once("::") {
        Some((host, guest)) => DirMapping {
            host: PathBuf::from(host),
            guest: guest.to_string(),
        },
        None => DirMapping {
            host: PathBuf::from(spec),
            guest: spec.to_string(),
        },
    }
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let mut dirs: Vec<DirMapping> = Vec::new();
    let mut module_path: Option<PathBuf> = None;
    let mut guest_args: Vec<String> = Vec::new();

    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--dir" => match argv.next() {
                Some(spec) => dirs.push(parse_dir(&spec)),
                None => fatal("--dir needs HOST[::GUEST]"),
            },
            "-h" | "--help" => {
                eprintln!(
                    "usage: aheui-wasm-runner [--dir HOST[::GUEST]]... <module.wasm> [args...]"
                );
                std::process::exit(0);
            }
            _ => {
                module_path = Some(PathBuf::from(&arg));
                guest_args.extend(argv);
                break;
            }
        }
    }
    let Some(module_path) = module_path else {
        fatal("no module given; see --help");
    };
    // With no explicit mapping the guest sees the host filesystem under its own
    // paths, so a program named on the command line resolves the same way it
    // would natively. Anything narrower is spelled with `--dir`.
    if dirs.is_empty() {
        dirs.push(DirMapping {
            host: PathBuf::from("/"),
            guest: "/".to_string(),
        });
    }

    match run(&module_path, &dirs, &guest_args) {
        Ok(code) => std::process::exit(code),
        Err(e) => fatal(&format!("{e:?}")),
    }
}

/// Whether to count the wasm instructions this run executes.
///
/// Fuel charges one unit per executed instruction, so the count is a property
/// of the program rather than of the machine it runs on: it stays put while
/// wall clock moves with whatever else the host happens to be doing. Reading it
/// costs the run its speed, because every basic block gains the accounting, so
/// it answers "how much work" and never "how fast".
fn fuel_count_enabled() -> bool {
    std::env::var_os("AHEUI_WASM_FUEL_COUNT").is_some()
}

/// Wasmtime owns content/config keys, atomic cache publication and validation.
/// Never deserialize an untrusted sidecar next to a guest supplied by the user.
fn load_guest(
    engine: &Engine,
    module_path: &Path,
    cache: Option<&wasmtime::Cache>,
) -> Result<(Module, bool)> {
    let hits = cache.map_or(0, wasmtime::Cache::cache_hits);
    let module = Module::from_file(engine, module_path)?;
    let compiled = cache.is_none_or(|cache| cache.cache_hits() == hits);
    Ok((module, compiled))
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn engine(directory: &Path) -> (Engine, wasmtime::Cache) {
        let mut cache_config = wasmtime::CacheConfig::new();
        cache_config.with_directory(directory);
        let cache = wasmtime::Cache::new(cache_config).unwrap();
        let mut config = Config::new();
        config.cache(Some(cache.clone()));
        (Engine::new(&config).unwrap(), cache)
    }

    fn value(module: &Module, engine: &Engine) -> i32 {
        let mut store = Store::new(engine, ());
        let instance = wasmtime::Instance::new(&mut store, module, &[]).unwrap();
        instance
            .get_typed_func::<(), i32>(&mut store, "value")
            .unwrap()
            .call(&mut store, ())
            .unwrap()
    }

    #[test]
    fn cache_ignores_adjacent_artifacts_and_keys_by_content() {
        let temp = tempfile::tempdir().unwrap();
        let (engine, cache) = engine(&temp.path().join("cache"));
        let guest = temp.path().join("guest.wasm");
        std::fs::write(
            guest.with_extension("wasm.cwasm"),
            b"untrusted adjacent file",
        )
        .unwrap();
        std::fs::write(
            &guest,
            "(module (func (export \"value\") (result i32) i32.const 7))",
        )
        .unwrap();
        let (module, compiled) = load_guest(&engine, &guest, Some(&cache)).unwrap();
        assert!(compiled);
        assert_eq!(value(&module, &engine), 7);
        assert!(!load_guest(&engine, &guest, Some(&cache)).unwrap().1);
        std::fs::write(
            &guest,
            "(module (func (export \"value\") (result i32) i32.const 9))",
        )
        .unwrap();
        let (replacement, compiled) = load_guest(&engine, &guest, Some(&cache)).unwrap();
        assert!(compiled);
        assert_eq!(value(&replacement, &engine), 9);
        assert_eq!(value(&module, &engine), 7);
    }

    #[test]
    fn concurrent_cache_publication_keeps_live_modules_valid() {
        let temp = tempfile::tempdir().unwrap();
        let (engine, cache) = engine(&temp.path().join("cache"));
        let guest = temp.path().join("guest.wasm");
        std::fs::write(
            &guest,
            "(module (func (export \"value\") (result i32) i32.const 7))",
        )
        .unwrap();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    let (module, _) = load_guest(&engine, &guest, Some(&cache)).unwrap();
                    assert_eq!(value(&module, &engine), 7);
                });
            }
        });
    }
}

fn run(module_path: &Path, dirs: &[DirMapping], guest_args: &[String]) -> Result<i32> {
    let mut config = Config::new();
    config.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable);
    let counting_fuel = fuel_count_enabled();
    config.consume_fuel(counting_fuel);
    let cache = if std::env::var_os("AHEUI_WASM_NO_MODULE_CACHE").is_some() {
        None
    } else {
        match wasmtime::Cache::new(wasmtime::CacheConfig::new()) {
            Ok(cache) => Some(cache),
            Err(error) => {
                eprintln!("[module cache disabled] {error}");
                None
            }
        }
    };
    config.cache(cache.clone());
    let engine = Engine::new(&config)?;
    // Recorded before the first compile, so a dumped trace can name its
    // `call_indirect` targets by the guest's own symbols.
    *GUEST_MODULE_PATH.lock().unwrap() = Some(module_path.to_path_buf());
    let load_started = std::time::Instant::now();
    let (module, guest_compiled) = load_guest(&engine, module_path, cache.as_ref())?;
    let guest_load_time_ns = load_started.elapsed().as_nanos();

    let mut builder = WasiCtxBuilder::new();
    builder.inherit_stdio().inherit_env();
    // argv[0] is the program name the guest's own argument parser skips.
    builder.arg(module_path.to_string_lossy().into_owned());
    for a in guest_args {
        builder.arg(a);
    }
    for d in dirs {
        builder
            .preopened_dir(&d.host, &d.guest, DirPerms::all(), FilePerms::all())
            .map_err(|e| Error::msg(format!("preopen {} as {}: {e}", d.host.display(), d.guest)))?;
    }
    let mut store = Store::new(&engine, Host::new(builder.build_p1()));
    // Half the range, so the guest's own accounting has room to subtract
    // without the store ever running dry mid-program.
    let fuel_budget = u64::MAX / 2;
    if counting_fuel {
        store.set_fuel(fuel_budget)?;
    }

    let mut linker: Linker<Host> = Linker::new(&engine);
    p1::add_to_linker_sync(&mut linker, |h: &mut Host| &mut h.wasi)?;
    add_majit_host(&mut linker)?;

    let instance = linker.instantiate(&mut store, &module)?;
    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| Error::msg("guest is missing its `memory` export"))?;
    let table = instance
        .get_table(&mut store, "__indirect_function_table")
        .ok_or_else(|| {
            Error::msg(
                "guest is missing its `__indirect_function_table` export \
                 (build with -C link-arg=--export-table -C link-arg=--growable-table)",
            )
        })?;
    // The table's current size is the first slot a later `table.grow` returns,
    // i.e. the first trace handle; everything below it belongs to the guest.
    let trace_base = table.size(&store);
    let host = store.data_mut();
    host.guest_load_time_ns = guest_load_time_ns;
    host.guest_compiled = guest_compiled;
    host.jit.memory = Some(memory);
    host.jit.table = Some(table);
    host.jit.trace_base = trace_base;

    let start = instance.get_typed_func::<(), ()>(&mut store, "_start")?;
    let outcome = start.call(&mut store, ());
    let code = match outcome {
        Ok(()) => 0,
        // A WASI command leaves through `proc_exit`, which unwinds as a trap
        // carrying the status. That is a normal exit, not a failure.
        Err(e) => match e.downcast_ref::<wasmtime_wasi::I32Exit>() {
            Some(exit) => exit.0,
            None => {
                report_stats(store.data());
                return Err(e);
            }
        },
    };
    if counting_fuel {
        let left = store.get_fuel().unwrap_or(fuel_budget);
        eprintln!(
            "[jit-stats] wasm_instructions_executed={}",
            fuel_budget.saturating_sub(left)
        );
    }
    report_stats(store.data());
    call_hist_dump();
    Ok(code)
}

/// Report what the host did on the JIT's behalf, alongside the `[jit-stats]`
/// lines the guest prints for itself under the same gate.
fn report_stats(host: &Host) {
    if std::env::var_os("MAJIT_STATS").is_none() {
        return;
    }
    eprintln!(
        "[jit-stats] host_compiles={} host_compile_ms={} host_execute_imports={} \
         host_calls={}",
        host.compile_count,
        host.compile_time_ns / 1_000_000,
        host.execute_count,
        host.call_count,
    );
    eprintln!(
        "[jit-stats] guest_load_ms={} guest_compiled={}",
        host.guest_load_time_ns / 1_000_000,
        host.guest_compiled,
    );
}

fn add_majit_host(linker: &mut Linker<Host>) -> Result<()> {
    linker.func_wrap(
        "majit_host",
        "jit_compile_wasm",
        |mut caller: Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_compile(&mut caller, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_compile_wasm] {e:?}");
                    0
                }
            }
        },
    )?;
    linker.func_wrap(
        "majit_host",
        "jit_replace_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, bytes_ptr: u32, bytes_len: u32| -> u32 {
            match jit_replace(&mut caller, func_id, bytes_ptr, bytes_len) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("[jit_replace_wasm] {e:?}");
                    0
                }
            }
        },
    )?;
    linker.func_wrap(
        "majit_host",
        "jit_execute_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32, frame_ptr: u32| -> u32 {
            match jit_execute(&mut caller, func_id, frame_ptr) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[jit_execute_wasm] {e:?}");
                    0
                }
            }
        },
    )?;
    linker.func_wrap(
        "majit_host",
        "jit_free_wasm",
        |mut caller: Caller<'_, Host>, func_id: u32| {
            let _ = majit_backend_wasm_host::free(&mut caller, func_id);
        },
    )?;
    // Residual-call trampoline for the recording and blackhole paths. A
    // compiled trace reaches its residual targets through `env.jit_call` on the
    // child module; the guest's own metainterpreter cannot reflect a function's
    // wasm type to build a matching `call_indirect`, so it routes residual calls
    // here instead. Same call area and same reflection as `env.jit_call`, so a
    // target whose signature is not the uniform `(i64…) -> i64` is coerced
    // rather than trapping on an indirect-call type mismatch.
    linker.func_wrap(
        "majit_host",
        "jit_call_host",
        |mut caller: Caller<'_, Host>, frame_ptr: u32| {
            if let Err(e) = jit_call_trampoline(&mut caller, frame_ptr, CALL_RESULT_OFS as u32) {
                eprintln!("[jit_call_host] {e:?}");
            }
        },
    )?;
    Ok(())
}

/// Residual crossings per callee, under `AHEUI_WASM_CALL_HIST`.
///
/// Every entry here is a call the compiled trace could not make in-guest, so
/// the histogram is the work list for narrowing the trampoline: a callee near
/// the top is one whose declared wasm signature the trace could match directly.
static CALL_HIST: std::sync::Mutex<Option<std::collections::BTreeMap<u32, u64>>> =
    std::sync::Mutex::new(None);

fn call_hist_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("AHEUI_WASM_CALL_HIST").is_some())
}

fn call_hist_record(slot: u32) {
    if !call_hist_enabled() {
        return;
    }
    let mut g = CALL_HIST.lock().unwrap();
    *g.get_or_insert_with(Default::default)
        .entry(slot)
        .or_insert(0) += 1;
}

fn call_hist_dump() {
    if !call_hist_enabled() {
        return;
    }
    let g = CALL_HIST.lock().unwrap();
    let Some(m) = g.as_ref() else { return };
    let total: u64 = m.values().sum();
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    let names = guest_slot_names();
    eprintln!("[call-hist] total={total} distinct={}", v.len());
    for (slot, n) in v.into_iter().take(20) {
        let pct = *n as f64 * 100.0 / total.max(1) as f64;
        let name = names.get(slot).map(String::as_str).unwrap_or("?");
        eprintln!("[call-hist] slot={slot} n={n} pct={pct:.1} {name}");
    }
}

/// Path of the guest module, so a dumped trace's `call_indirect` targets can
/// be resolved against the table the guest itself populated.
static GUEST_MODULE_PATH: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

/// Directory named by `AHEUI_WASM_DUMP_TRACES`, or `None` when the knob is
/// unset. Every compiled trace is written there as both `.wasm` and WAT.
fn trace_dump_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("AHEUI_WASM_DUMP_TRACES")?);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        eprintln!("[dump-traces] {}: {e}", dir.display());
        return None;
    }
    Some(dir)
}

/// Table slot to symbol, read out of the guest module's element and name
/// sections. A trace calls its residual targets by table index, and a trap
/// report names a wasm offset -- neither reads as anything without this map.
fn guest_slot_names() -> std::collections::BTreeMap<u32, String> {
    use wasmparser::{ElementItems, ElementKind, Name, Operator, Payload};

    let mut out = std::collections::BTreeMap::new();
    let path = GUEST_MODULE_PATH.lock().unwrap().clone();
    let Some(path) = path else { return out };
    let Ok(bytes) = std::fs::read(&path) else {
        return out;
    };

    let mut slot_to_func: std::collections::BTreeMap<u32, u32> = Default::default();
    let mut func_to_name: std::collections::BTreeMap<u32, String> = Default::default();
    for payload in wasmparser::Parser::new(0).parse_all(&bytes) {
        match payload {
            Ok(Payload::ElementSection(reader)) => {
                for element in reader {
                    let Ok(element) = element else { continue };
                    // Only an active segment puts entries in the table; a
                    // passive or declared one is never indexed by a call.
                    let ElementKind::Active { offset_expr, .. } = element.kind else {
                        continue;
                    };
                    let mut ops = offset_expr.get_operators_reader().into_iter();
                    let base = match ops.next() {
                        Some(Ok(Operator::I32Const { value })) => value as u32,
                        _ => continue,
                    };
                    let ElementItems::Functions(funcs) = element.items else {
                        continue;
                    };
                    for (i, func) in funcs.into_iter().enumerate() {
                        let Ok(func) = func else { continue };
                        slot_to_func.insert(base + i as u32, func);
                    }
                }
            }
            Ok(Payload::CustomSection(c)) if c.name() == "name" => {
                let reader = wasmparser::NameSectionReader::new(wasmparser::BinaryReader::new(
                    c.data(),
                    c.data_offset(),
                ));
                for subsection in reader {
                    let Ok(Name::Function(map)) = subsection else {
                        continue;
                    };
                    for naming in map {
                        let Ok(naming) = naming else { continue };
                        func_to_name.insert(
                            naming.index,
                            rustc_demangle::demangle(naming.name).to_string(),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    for (slot, func) in slot_to_func {
        if let Some(name) = func_to_name.get(&func) {
            out.insert(slot, name.clone());
        }
    }
    out
}

/// Write one trace module beside its disassembly. The WAT carries binary
/// offsets, so a trap reported at an offset inside the trace names the
/// instruction that took it.
fn dump_trace(bytes: &[u8], seq: u64) {
    let Some(dir) = trace_dump_dir() else { return };
    let stem = dir.join(format!("trace-{seq:04}"));
    let _ = std::fs::write(stem.with_extension("wasm"), bytes);
    let mut wat = String::new();
    let mut cfg = wasmprinter::Config::new();
    cfg.print_offsets(true);
    match cfg.print(bytes, &mut wasmprinter::PrintFmtWrite(&mut wat)) {
        Ok(()) => {
            let _ = std::fs::write(stem.with_extension("wat"), &wat);
        }
        Err(e) => eprintln!("[dump-traces] wat print failed for {seq}: {e}"),
    }
    // The slot map is the same for every trace; write it once beside them.
    let slots = dir.join("guest-slots.txt");
    if !slots.exists() {
        let names = guest_slot_names();
        let text: String = names
            .iter()
            .map(|(slot, name)| format!("{slot}\t{name}\n"))
            .collect();
        let _ = std::fs::write(&slots, text);
    }
    eprintln!(
        "[dump-traces] wrote {} ({} bytes)",
        stem.display(),
        bytes.len()
    );
}

fn jit_compile_trace(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<(Func, Option<Func>)> {
    caller.data_mut().compile_count += 1;
    let memory = majit_backend_wasm_host::memory(caller)?;

    let mut bytes = vec![0u8; bytes_len as usize];
    memory.read(&*caller, bytes_ptr as usize, &mut bytes)?;
    dump_trace(&bytes, caller.data().compile_count);

    let engine = caller.engine().clone();
    let compile_start = std::time::Instant::now();
    let module_result = Module::new(&engine, &bytes);
    caller.data_mut().compile_time_ns += compile_start.elapsed().as_nanos();
    let module = match module_result {
        Ok(m) => m,
        Err(e) => {
            if std::env::var_os("AHEUI_WASM_DUMP_BAD_TRACE").is_some() {
                let path = "/tmp/aheui_bad_trace.wasm";
                let _ = std::fs::write(path, &bytes);
                eprintln!("[jit_compile_wasm] dumped {} bytes to {path}", bytes.len());
                match wasmprinter::print_bytes(&bytes) {
                    Ok(wat) => eprintln!("--- WAT ---\n{wat}\n--- /WAT ---"),
                    Err(pe) => eprintln!("[jit_compile_wasm] wat print failed: {pe}"),
                }
            }
            return Err(e);
        }
    };

    let (trace, trace_wide, _) =
        majit_backend_wasm_host::instantiate(caller, &module, jit_call_trampoline)?;
    Ok((trace, trace_wide))
}
/// Compile and instantiate a trace, then append its export to the table.
fn jit_compile(caller: &mut Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32) -> Result<u32> {
    let (trace, trace_wide) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    let slot = majit_backend_wasm_host::publish(caller, trace, trace_wide)?;
    Ok(slot)
}
/// Compile and instantiate a trace, then replace an existing trace slot.
fn jit_replace(
    caller: &mut Caller<'_, Host>,
    func_id: u32,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32> {
    if (func_id as u64) < caller.data().jit.trace_base {
        return Err(Error::msg("cannot replace a guest function"));
    }
    let (trace, trace_wide) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    let slot = majit_backend_wasm_host::replace(caller, func_id, trace, trace_wide)?;
    Ok(slot)
}
/// Run a previously compiled trace, returning its guard-exit index.
fn jit_execute(caller: &mut Caller<'_, Host>, func_id: u32, frame_ptr: u32) -> Result<u32> {
    caller.data_mut().execute_count += 1;
    let ret = majit_backend_wasm_host::execute(caller, func_id, frame_ptr)?;
    Ok(ret)
}
/// Perform one residual call on the guest's behalf.
///
/// The guest has written the callee's table index, its argument count and the
/// arguments into the call area; this reads the callee's declared wasm type,
/// coerces each argument to it, calls it, and writes the result back.
fn jit_call_trampoline(
    caller: &mut Caller<'_, Host>,
    frame_ptr: u32,
    call_area_ofs: u32,
) -> Result<()> {
    caller.data_mut().call_count += 1;
    majit_backend_wasm_host::residual_call(caller, frame_ptr, call_area_ofs, |slot, live| {
        call_hist_record(slot);
        if !live {
            report_dead_call_slot(slot);
        }
    })
}
/// Name a call-area FUNC field that indexes no live function, once per value.
fn report_dead_call_slot(func_ptr: u32) {
    use std::sync::Mutex;
    static SEEN: Mutex<Option<HashSet<u32>>> = Mutex::new(None);
    let mut g = SEEN.lock().unwrap();
    if g.get_or_insert_with(HashSet::new).insert(func_ptr) {
        eprintln!("[jit_call] call-area func slot {func_ptr} holds no live function");
    }
}
