//! M0 instrumentation for the aheui rtyper-penetration epic.
//!
//! Runs the production analyze pipeline (`analyze_multiple_pipeline_with_modules`)
//! over the Charon-extracted aheui LLBC set (`build/llbc/aheui-runtime.ullbc`
//! + `build/llbc/aheuinterpreter.ullbc`, produced by
//! `aheui/scripts/extract-llbc.sh`) with the portal
//! bound to `aheuinterpreter::interp::mainloop`, and surfaces the two-phase
//! prepass census dispositions for the mainloop closure (`val_*`,
//! `Storage`, …).
//!
//! This is a measurement, not an acceptance gate: it prints what the
//! pipeline can and cannot digest today. Run as
//!
//! ```sh
//! PYRE_RTYPER_VERBOSE=1 cargo test --release -p aheui-jit \
//!     --test test_aheui_census -- --nocapture
//! ```

use majit_translate::{AnalyzeConfig, HostStaticAddrs, PipelineConfig, PortalSpec};

/// Resolve the named aheui LLBC artefacts and export `PYRE_MIR_FRONTEND_LLBC`.
/// Returns `false` (skip cleanly) when any is absent. Tests sharing this
/// process each re-export the var for their own artefact set; run with
/// `--test-threads=1`.
fn ensure_aheui_llbc_env(required: &[&str]) -> bool {
    let workspace_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("build")
        .join("llbc");
    let llbc_dir = workspace_root;
    let mut paths = Vec::with_capacity(required.len());
    for name in required {
        let p = llbc_dir.join(name);
        if !p.exists() {
            return false;
        }
        paths.push(p.to_string_lossy().into_owned());
    }
    let joined = std::env::join_paths(&paths).expect("join llbc paths");
    // SAFETY: serialized test binary; set before any worker spawns.
    unsafe { std::env::set_var("PYRE_MIR_FRONTEND_LLBC", joined) };
    true
}

/// Census the `val_*` helper subtree directly. The `mainloop` BFS closure
/// (`aheui_census_m0`) is currently truncated before reaching `val_add`
/// (call-target resolution fails at `dispatch_mut`/`LinkedList` level), so
/// this second probe seeds the BFS from `val_add` itself to measure the
/// arithmetic helpers' own phaseA/phaseB disposition.
///
/// Process-global registries (`STRUCT_ORIGIN_REGISTRY`, …) are re-seeded by
/// each pipeline invocation; run serialized (`--test-threads=1`) to keep the
/// two probes from racing on them.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_m0_val_helpers() {
    if !ensure_aheui_llbc_env(&["aheui-runtime.ullbc", "aheuinterpreter.ullbc"]) {
        eprintln!(
            "skipping: build/llbc/{{aheui-runtime,aheuinterpreter}}.ullbc missing — \
             run `aheui/scripts/extract-llbc.sh`"
        );
        return;
    }
    run_census_with_portal(val_add_portal());
}

/// Same probe against the smallint build (`Val = i64`,
/// `val_add = wrapping_add`) — the rpaheui-smallint-parity mode M1 targets.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_m0_val_helpers_smallint() {
    if !ensure_aheui_llbc_env(&["aheui-runtime-smallint.ullbc"]) {
        eprintln!(
            "skipping: build/llbc/aheui-runtime-smallint.ullbc missing — \
             run `aheui/scripts/extract-llbc.sh`"
        );
        return;
    }
    run_census_with_portal(val_add_portal());
}

fn val_add_portal() -> PortalSpec {
    PortalSpec {
        name: "val_add".to_string(),
        greens: Vec::new(),
        reds: Vec::new(),
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

fn run_census_with_portal(portal: PortalSpec) {
    let config = AnalyzeConfig {
        pipeline: PipelineConfig {
            portal: Some(portal),
            ..Default::default()
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
            eprintln!("=== aheui census: pipeline completed ===");
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
            eprintln!("=== aheui census: pipeline panicked after census ===");
            eprintln!("panic: {}", panic_message(&err));
        }
    }
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "release-only: runs full LLBC translation; use `cargo test --release --test test_aheui_census`"
)]
fn aheui_census_m0() {
    if !ensure_aheui_llbc_env(&["aheui-runtime.ullbc", "aheuinterpreter.ullbc"]) {
        eprintln!(
            "skipping: build/llbc/{{aheui-runtime,aheuinterpreter}}.ullbc missing — \
             run `aheui/scripts/extract-llbc.sh`"
        );
        return;
    }

    let config = AnalyzeConfig {
        pipeline: PipelineConfig {
            // Portal + green/red layout documented in
            // `aheui/aheuinterpreter/src/interp.rs` (mirrors
            // rpaheui/aheui/aheui.py greens/reds).
            portal: Some(PortalSpec {
                name: "mainloop".to_string(),
                greens: ["pc", "stackok", "is_queue", "program"]
                    .map(String::from)
                    .to_vec(),
                reds: ["stacksize", "storage", "selected"]
                    .map(String::from)
                    .to_vec(),
                virtualizables: Vec::new(),
                red_types: Vec::new(),
            }),
            ..Default::default()
        },
    };

    // The drain / jitcode-emission tail may well panic on aheui shapes the
    // pipeline has never seen; the census histograms are printed before
    // that point, so capture the panic and report it instead of dying.
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
            eprintln!("=== aheui census M0: pipeline completed ===");
            eprintln!("jitcodes emitted: {}", result.jitcodes.len());
            let mut names: Vec<&str> = result
                .jitcodes_by_path
                .keys()
                .map(|k| k.segments.last().map(|s| s.as_str()).unwrap_or(""))
                .collect();
            names.sort_unstable();
            eprintln!("jitcode leaves: {names:?}");
        }
        Err(err) => {
            eprintln!("=== aheui census M0: pipeline panicked after census ===");
            eprintln!("panic: {}", panic_message(&err));
        }
    }
}
