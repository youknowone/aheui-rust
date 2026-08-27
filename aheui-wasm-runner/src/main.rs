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

use majit_backend_wasm::codegen::{
    CALL_ARGS_OFS, CALL_FUNC_OFS, CALL_NARGS_OFS, CALL_RESULT_OFS, MAX_CALL_ARGS,
};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use wasmtime::{
    Caller, Config, Engine, Error, Extern, Func, Instance, Linker, Memory, Module, Ref, Result,
    Store, Table, Val, ValType,
};
use wasmtime_wasi::p1::{self, WasiP1Ctx};
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtxBuilder};

/// Per-store host state shared by every import callback.
struct Host {
    wasi: WasiP1Ctx,
    /// The guest's exported linear memory, shared with every trace module.
    memory: Option<Memory>,
    /// The guest's `__indirect_function_table`, which is both the trace
    /// registry and the trampoline's callee lookup.
    table: Option<Table>,
    /// First table slot that is a JIT trace. The guest's own functions occupy
    /// `[0, trace_base)`; `jit_compile` only ever appends, so a slot at or
    /// above this is a trace and nothing below it may be executed or freed as
    /// one. The table is the sole record of trace liveness — the slot IS the
    /// handle — and it also roots each trace's instance for the store's life.
    trace_base: u64,
    /// Trace slots whose compile exported a `trace_wide`, and whose `slot + 1`
    /// is therefore a published call target. The reserved spare slot holds the
    /// narrow function when a compile had no wide entry, so the table alone
    /// cannot tell the two apart; a replacement that would drop the wide entry
    /// is rejected against this set rather than leaving `slot + 1` pointing at
    /// the replaced compile.
    wide_slots: HashSet<u32>,
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
}

impl Host {
    fn new(wasi: WasiP1Ctx) -> Self {
        Self {
            wasi,
            memory: None,
            table: None,
            trace_base: 0,
            wide_slots: HashSet::new(),
            compile_count: 0,
            compile_time_ns: 0,
            execute_count: 0,
            call_count: 0,
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

fn run(module_path: &Path, dirs: &[DirMapping], guest_args: &[String]) -> Result<i32> {
    let mut config = Config::new();
    config.wasm_backtrace_details(wasmtime::WasmBacktraceDetails::Enable);
    let engine = Engine::new(&config)?;
    let module = Module::from_file(&engine, module_path)?;

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
    host.memory = Some(memory);
    host.table = Some(table);
    host.trace_base = trace_base;

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
    report_stats(store.data());
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
            // Only a trace slot may be cleared; nulling a guest slot would
            // corrupt the shared dispatch table.
            if (func_id as u64) < caller.data().trace_base {
                return;
            }
            // `jit_compile` appends the entry as a pair, so `func_id + 1` holds
            // this same trace — the wide entry where one was published, a spare
            // copy of the narrow function otherwise. Clearing only the first
            // half would leave the freed trace reachable through
            // `call_indirect func_id + 1`, so both halves go together.
            caller.data_mut().wide_slots.remove(&func_id);
            if let Some(table) = caller.data().table {
                let _ = table.set(&mut caller, func_id as u64, Ref::Func(None));
                let _ = table.set(&mut caller, func_id as u64 + 1, Ref::Func(None));
            }
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

fn jit_compile_trace(
    caller: &mut Caller<'_, Host>,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<(Table, Func, Option<Func>)> {
    caller.data_mut().compile_count += 1;
    let memory = host_memory(caller)?;
    let table = host_table(caller)?;

    let mut bytes = vec![0u8; bytes_len as usize];
    memory.read(&*caller, bytes_ptr as usize, &mut bytes)?;

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

    // A fresh trampoline per trace; it reads all state from `caller.data()`.
    let jit_call = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32| {
            if let Err(e) =
                jit_call_trampoline(&mut inner, frame_ptr as u32, CALL_RESULT_OFS as u32)
            {
                eprintln!("[jit_call] {e:?}");
            }
        },
    );
    let jit_call_compact = Func::wrap(
        &mut *caller,
        |mut inner: Caller<'_, Host>, frame_ptr: i32, call_area_ofs: i32| {
            if let Err(e) = jit_call_trampoline(&mut inner, frame_ptr as u32, call_area_ofs as u32)
            {
                eprintln!("[jit_call_compact] {e:?}");
            }
        },
    );

    // Supply imports in the module's declared order.
    let mut externs: Vec<Extern> = Vec::new();
    for import in module.imports() {
        match (import.module(), import.name()) {
            ("env", "memory") => externs.push(Extern::Memory(memory)),
            ("env", "jit_call") => externs.push(Extern::Func(jit_call)),
            ("env", "jit_call_compact") => externs.push(Extern::Func(jit_call_compact)),
            ("env", "__indirect_function_table") => externs.push(Extern::Table(table)),
            (m, n) => {
                return Err(Error::msg(format!(
                    "trace module has unexpected import {m}.{n}"
                )));
            }
        }
    }

    let instance = Instance::new(&mut *caller, &module, &externs)?;
    let trace = instance
        .get_func(&mut *caller, "trace")
        .ok_or_else(|| Error::msg("trace module is missing its `trace` export"))?;
    let trace_wide = instance.get_func(&mut *caller, "trace_wide");
    Ok((table, trace, trace_wide))
}

/// Compile and instantiate a trace, then append its export to the table.
fn jit_compile(caller: &mut Caller<'_, Host>, bytes_ptr: u32, bytes_len: u32) -> Result<u32> {
    let (table, trace, trace_wide) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    // The pair is appended even for a narrow module. An emitted module names
    // its wide entry `handle + 1`, and `jit_replace` may install a wide entry
    // where this compile had none; without the reservation that write would
    // land on the next trace's own entry. The spare slot holds the narrow
    // function, which nothing calls.
    let slot = table.grow(&mut *caller, 2, Ref::Func(Some(trace)))? as u32;
    if let Some(wide) = trace_wide {
        table.set(&mut *caller, slot as u64 + 1, Ref::Func(Some(wide)))?;
        caller.data_mut().wide_slots.insert(slot);
    }
    Ok(slot)
}

/// Compile and instantiate a trace, then replace an existing trace slot.
fn jit_replace(
    caller: &mut Caller<'_, Host>,
    func_id: u32,
    bytes_ptr: u32,
    bytes_len: u32,
) -> Result<u32> {
    if (func_id as u64) < caller.data().trace_base {
        return Err(Error::msg(format!(
            "jit_replace_wasm: id {func_id} is not a trace slot"
        )));
    }
    let (table, trace, trace_wide) = jit_compile_trace(caller, bytes_ptr, bytes_len)?;
    if !matches!(
        table.get(&mut *caller, func_id as u64),
        Some(Ref::Func(Some(_)))
    ) {
        return Err(Error::msg(format!(
            "jit_replace_wasm: id {func_id} is not a live trace"
        )));
    }
    // Modules emitted while this slot was wide carry `call_indirect func_id +
    // 1` baked in. A narrow replacement cannot retract those, so accepting one
    // would leave the pair straddling two compiles.
    if trace_wide.is_none() && caller.data().wide_slots.contains(&func_id) {
        return Err(Error::msg(format!(
            "jit_replace_wasm: id {func_id} has a published wide entry the replacement does not"
        )));
    }
    table.set(&mut *caller, func_id as u64, Ref::Func(Some(trace)))?;
    if let Some(wide) = trace_wide {
        table.set(&mut *caller, func_id as u64 + 1, Ref::Func(Some(wide)))?;
        caller.data_mut().wide_slots.insert(func_id);
    }
    Ok(func_id)
}

/// Run a previously compiled trace, returning its guard-exit index.
fn jit_execute(caller: &mut Caller<'_, Host>, func_id: u32, frame_ptr: u32) -> Result<u32> {
    caller.data_mut().execute_count += 1;
    if (func_id as u64) < caller.data().trace_base {
        return Err(Error::msg(format!(
            "jit_execute_wasm: id {func_id} is not a trace slot"
        )));
    }
    let table = host_table(caller)?;
    // The handle IS the table slot; dispatch by index, the same lookup an
    // in-module `call_indirect` would perform. A freed trace misses here.
    let trace = match table.get(&mut *caller, func_id as u64) {
        Some(Ref::Func(Some(f))) => f,
        _ => {
            return Err(Error::msg(format!(
                "jit_execute_wasm: id {func_id} is not a live trace (unknown or freed)"
            )));
        }
    };
    let mut results = [Val::I32(0)];
    trace.call(&mut *caller, &[Val::I32(frame_ptr as i32)], &mut results)?;
    Ok(match results[0] {
        Val::I32(x) => x as u32,
        _ => 0,
    })
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
    let memory = host_memory(caller)?;
    let table = host_table(caller)?;
    let call_area = frame_ptr as usize + call_area_ofs as usize;
    let arg_ofs = call_area + (CALL_ARGS_OFS - CALL_RESULT_OFS) as usize;
    let func_ptr = read_u32(
        &memory,
        &*caller,
        call_area + (CALL_FUNC_OFS - CALL_RESULT_OFS) as usize,
    );

    let func = match table.get(&mut *caller, func_ptr as u64) {
        Some(Ref::Func(Some(f))) => f,
        // Nothing to call, and the 0 written in its place is indistinguishable
        // from a null the callee could have returned — so say the slot out loud
        // here rather than letting a wrong result surface much later.
        _ => {
            report_dead_call_slot(func_ptr);
            write_i64(&memory, &mut *caller, call_area, 0)?;
            return Ok(());
        }
    };

    let ty = func.ty(&*caller);
    let params: Vec<ValType> = ty.params().collect();
    if params.len() > MAX_CALL_ARGS {
        return Err(Error::msg(format!(
            "residual callee declares {} params, past the {MAX_CALL_ARGS} the call area holds",
            params.len()
        )));
    }
    let mut args: Vec<Val> = Vec::with_capacity(params.len());
    for (i, pty) in params.iter().enumerate() {
        let raw = read_i64(&memory, &*caller, arg_ofs + i * 8);
        args.push(match pty {
            ValType::I32 => Val::I32(raw as i32),
            ValType::I64 => Val::I64(raw),
            // Floats cross the call area as their raw bit pattern in an i64 slot.
            ValType::F32 => Val::F32(raw as u32),
            ValType::F64 => Val::F64(raw as u64),
            other => {
                return Err(Error::msg(format!(
                    "unsupported residual-call param type {other:?}"
                )));
            }
        });
    }

    let mut results: Vec<Val> = ty
        .results()
        .map(|t| match t {
            ValType::I64 => Val::I64(0),
            ValType::F32 => Val::F32(0),
            ValType::F64 => Val::F64(0),
            _ => Val::I32(0),
        })
        .collect();

    // A trapping residual target is reported as a zero result rather than
    // aborting the whole run, matching the browser glue's try/catch.
    if let Err(e) = func.call(&mut *caller, &args, &mut results) {
        eprintln!("[jit_call] residual target trapped: {e:?}");
        write_i64(&memory, &mut *caller, call_area, 0)?;
        return Ok(());
    }

    let result = match results.first() {
        Some(Val::I32(x)) => (*x as u32) as i64, // zero-extend; high word stays 0
        Some(Val::I64(x)) => *x,
        Some(Val::F64(x)) => *x as i64,
        Some(Val::F32(x)) => (*x as u64) as i64,
        _ => 0,
    };
    write_i64(&memory, &mut *caller, call_area, result)?;
    Ok(())
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

fn host_memory(caller: &Caller<'_, Host>) -> Result<Memory> {
    caller
        .data()
        .memory
        .ok_or_else(|| Error::msg("guest memory not initialized"))
}

fn host_table(caller: &Caller<'_, Host>) -> Result<Table> {
    caller
        .data()
        .table
        .ok_or_else(|| Error::msg("guest function table not initialized"))
}

fn read_u32(mem: &Memory, store: impl wasmtime::AsContext, off: usize) -> u32 {
    let mut buf = [0u8; 4];
    let _ = mem.read(store, off, &mut buf);
    u32::from_le_bytes(buf)
}

fn read_i64(mem: &Memory, store: impl wasmtime::AsContext, off: usize) -> i64 {
    let mut buf = [0u8; 8];
    let _ = mem.read(store, off, &mut buf);
    i64::from_le_bytes(buf)
}

fn write_i64(mem: &Memory, store: impl wasmtime::AsContextMut, off: usize, v: i64) -> Result<()> {
    mem.write(store, off, &v.to_le_bytes())?;
    Ok(())
}

// `CALL_NARGS_OFS` is part of the call-area ABI the guest writes but this host
// does not read: the callee's own declared arity is authoritative, and a
// disagreement between the two is a guest-side bug that the reflected call
// would answer with garbage rather than a trap. Naming it keeps the import
// list complete and the unused-import lint honest about which fields are read.
const _: u64 = CALL_NARGS_OFS;
