//! Rtyper census for the Aheui interpreter and value helpers.
//!
//! Runs the production analyze pipeline (`analyze_multiple_pipeline_with_modules`)
//! over the Charon-extracted aheui LLBC set (`build/llbc/*.ullbc`, produced by
//! `aheui/scripts/extract-llbc.sh`) and surfaces the two-phase prepass census
//! dispositions for whichever portal each probe binds.
//!
//! This is a measurement, not an acceptance gate: it prints which functions
//! the pipeline accepts and where translation stops. Run as
//!
//! ```sh
//! PYRE_RTYPER_VERBOSE=1 cargo test --release -p aheui-jit \
//!     --test test_aheui_census -- --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` is required: each probe re-exports
//! `PYRE_MIR_FRONTEND_LLBC` for its own artefact set, and the pipeline re-seeds
//! process-global registries (`STRUCT_ORIGIN_REGISTRY`, …) on every invocation.

use majit_translate::{AnalyzeConfig, CallPath, HostStaticAddrs, JitDriverSpec, PipelineConfig};

/// Resolve the named LLBC artefacts and export `PYRE_MIR_FRONTEND_LLBC`.
///
/// Returns `false` — skip cleanly, saying so — when any is absent.
fn llbc_ready(required: &[&str]) -> bool {
    let llbc_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("build")
        .join("llbc");
    let mut paths = Vec::with_capacity(required.len());
    for name in required {
        let p = llbc_dir.join(name);
        if !p.exists() {
            eprintln!(
                "skipping: build/llbc/{name} missing — run `aheui/scripts/extract-llbc.sh`"
            );
            return false;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    let joined = std::env::join_paths(&paths).expect("join llbc paths");
    // SAFETY: serialized test binary; set before any worker spawns.
    unsafe { std::env::set_var("PYRE_MIR_FRONTEND_LLBC", joined) };
    true
}

/// A portal and its green/red layout, with every other knob left at the
/// pipeline's default.
fn driver(portal: &[&str], greens: &[&str], reds: &[&str]) -> JitDriverSpec {
    JitDriverSpec {
        portal: CallPath::from_segments(portal.iter().copied()),
        greens: greens.iter().map(|s| s.to_string()).collect(),
        reds: reds.iter().map(|s| s.to_string()).collect(),
        green_kinds: Vec::new(),
        red_kinds: Vec::new(),
        autoreds: false,
        virtualizables: Vec::new(),
        red_types: Vec::new(),
    }
}

/// Extract the panic message from a `catch_unwind` error payload.
fn panic_message(err: &Box<dyn std::any::Any + Send>) -> &str {
    err.downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| err.downcast_ref::<&str>().copied())
        .unwrap_or("<non-string panic>")
}

/// Run the pipeline under `driver` and print what it accepted.
///
/// Unsupported Aheui shapes may make the jitcode-emission tail panic after the
/// census histograms have been printed, so the panic is caught and reported
/// rather than allowed to discard the census output.
fn run_census(label: &str, driver: JitDriverSpec) {
    let config = AnalyzeConfig {
        pipeline: PipelineConfig {
            transform: Default::default(),
            jit_drivers: vec![driver],
            register_trait_families: Vec::new(),
        },
    };
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        majit_translate::analyze_multiple_pipeline_with_modules(
            &[],
            &config,
            None,
            &|_, _| None,
            &[],
            HostStaticAddrs::default(),
        )
    }));
    match outcome {
        Ok(result) => {
            eprintln!("=== {label}: pipeline completed ===");
            eprintln!("jitcodes emitted: {}", result.jitcodes.len());
            let mut names: Vec<String> = result
                .jitcodes_by_path
                .keys()
                .map(|k| k.canonical_key())
                .collect();
            names.sort_unstable();
            eprintln!("jitcode paths: {names:#?}");
            let mut insns: Vec<&String> = result.insns.keys().collect();
            insns.sort_unstable();
            eprintln!("insn vocabulary: {insns:?}");
        }
        Err(err) => {
            eprintln!("=== {label}: pipeline panicked after census ===");
            eprintln!("panic: {}", panic_message(&err));
        }
    }
}

/// Census the `val_*` helper subtree directly. The `mainloop` traversal stops
/// at the `dispatch_mut`/`LinkedList` call-target boundary before reaching
/// `val_add`, so this probe starts at `val_add` to measure the arithmetic
/// helpers' own phase-A and phase-B dispositions.
///
/// The portal path is module-qualified because portal resolution is exact: the
/// two builds carry their own `val_add` (`value::bigint` / `value::smallint`),
/// so the leaf alone does not name one.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_bigint_val_helpers() {
    if llbc_ready(&["aheui-runtime.ullbc", "aheuinterpreter.ullbc"]) {
        run_census(
            "aheui census (bigint val helpers)",
            driver(&["value", "bigint", "val_add"], &[], &[]),
        );
    }
}

/// The same probe against the smallint build, where `Val = i64` and
/// `val_add = wrapping_add`.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_smallint_val_helpers() {
    if llbc_ready(&["aheui-runtime-smallint.ullbc"]) {
        run_census(
            "aheui census (smallint val helpers)",
            driver(&["value", "smallint", "val_add"], &[], &[]),
        );
    }
}

/// The interpreter's own portal, under the green/red layout `interp::mainloop`
/// declares.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_mainloop() {
    if llbc_ready(&["aheui-runtime.ullbc", "aheuinterpreter.ullbc"]) {
        run_census(
            "aheui mainloop census",
            driver(
                &["interp", "mainloop"],
                &["pc", "stackok", "is_queue", "program"],
                &["stacksize", "storage", "selected"],
            ),
        );
    }
}
