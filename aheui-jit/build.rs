//! Build script for aheui-jit: analyzes the Aheui interpreter through the
//! majit-translate graph pipeline — the same path pyre-jit takes — and emits
//! the generated trace code.

#[path = "src/jit/call_spec.rs"]
mod call_spec;
#[path = "src/jit/virtualizable_spec.rs"]
mod virtualizable_spec;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let base = format!("{manifest_dir}/..");

    // The majit-translate graph pipeline lowers Charon-extracted MIR
    // (`.ullbc`). The shared front-end auto-discovers only the parent
    // repo's pyre artefact pair,
    // so point it at aheui's own crate LLBC (extracted by
    // `scripts/extract-llbc.py` into `<aheui>/build/llbc/`) via
    // `MAJIT_MIR_FRONTEND_LLBC`, which the front-end honours ahead of
    // auto-discovery. An explicit env override still wins.
    if std::env::var_os("MAJIT_MIR_FRONTEND_LLBC").is_none() {
        let llbc_dir = std::path::Path::new(&base).join("build").join("llbc");
        let rt = llbc_dir.join("aheui-runtime.ullbc");
        let interp = llbc_dir.join("aheuinterpreter.ullbc");
        if rt.exists() && interp.exists() {
            let mut paths: Vec<std::path::PathBuf> = vec![rt, interp];
            // Cross-target layout sidecars go LAST. `build/llbc` is one set
            // shared by every build, and struct layout is not: a pointer is 4
            // bytes on wasm32, so `ListBase.size` sits at offset 4 there and at
            // 8 on a 64-bit host, and a descr carrying the host offset names a
            // word past the end of the struct that the JIT then reads and
            // writes. The front-end merges exact layouts last-writer-wins and
            // everything else first-writer-wins, so appending the sidecars puts
            // their target offsets on top while their body-stripped tables lose
            // to the host artefacts. A missing sidecar is a hard error rather
            // than a silent fallback to layouts that do not describe this
            // target.
            let target = std::env::var("TARGET").unwrap_or_default();
            let host = std::env::var("HOST").unwrap_or_default();
            if majit_translate::layout::is_cross_target(&target, &host) {
                for stem in ["aheui-runtime", "aheuinterpreter"] {
                    let name = majit_translate::layout::layout_sidecar_filename(stem, &target);
                    let sidecar = llbc_dir.join(&name);
                    assert!(
                        sidecar.exists(),
                        "aheui layout sidecar {} is missing under {}.\n\
                         Re-run `aheui/scripts/extract-llbc.py`, whose \
                         `LAYOUT_TARGETS` names the cross targets that get one.",
                        name,
                        llbc_dir.display()
                    );
                    paths.push(sidecar);
                }
            }
            let joined =
                std::env::join_paths(paths).expect("aheui LLBC paths contain no path separator");
            // SAFETY: build scripts are single-threaded; no other thread
            // observes the environment during this set.
            unsafe { std::env::set_var("MAJIT_MIR_FRONTEND_LLBC", joined) };
        } else {
            panic!(
                "aheui LLBC missing under {}.\n\
                 Run `aheui/scripts/extract-llbc.py` to produce \
                 `aheui-runtime.ullbc` + `aheuinterpreter.ullbc` \
                 (install with the parent repo's \
                 `python3 scripts/install-charon.py`), or set \
                 `MAJIT_MIR_FRONTEND_LLBC` explicitly.",
                llbc_dir.display()
            );
        }
        println!("cargo::rerun-if-changed={}/build/llbc", base);
    }

    let source_dirs = [
        format!("{base}/aheuinterpreter/src"),
        format!("{base}/aheui-runtime/src"),
    ];

    let mut source_paths = Vec::new();
    for dir in &source_dirs {
        source_paths.extend(majit_translate::module_path::collect_rs_files(dir));
    }

    eprintln!(
        "[aheui-jit build.rs] reading {} source files from {} dirs",
        source_paths.len(),
        source_dirs.len(),
    );

    // The graph surface comes from the Charon-extracted LLBC set
    // (`MAJIT_MIR_FRONTEND_LLBC` above). `module_paths[i]` is the
    // crate-stripped module path of the i-th source file, derived by the
    // translator that consumes it — the spelling is its invariant, not a
    // label this build script gets to choose.
    // This pipeline needs no host vinfo or static-singleton configuration.
    // Concrete function bindings are installed by jit::jitcode_runtime.
    let module_paths: Vec<String> = source_paths
        .iter()
        .map(|p| majit_translate::module_path::module_path_from_source_file(p))
        .collect();
    let module_path_refs: Vec<&str> = module_paths.iter().map(|s| s.as_str()).collect();
    let vinfo_factory: &majit_translate::VirtualizableInfoFactory<'_> = &|_, _| None;
    let pipeline = majit_translate::analyze_multiple_pipeline_with_modules(
        &module_path_refs,
        &majit_translate::AnalyzeConfig {
            pipeline: majit_translate::PipelineConfig {
                transform: majit_translate::GraphTransformConfig {
                    vable_fields: virtualizable_spec::AHEUI_VABLE_FIELDS
                        .iter()
                        .map(|(name, idx)| {
                            majit_translate::VirtualizableFieldDescriptor::new(
                                *name,
                                Some(virtualizable_spec::AHEUI_VABLE_OWNER_ROOT.to_string()),
                                *idx,
                            )
                        })
                        .collect(),
                    vable_arrays: virtualizable_spec::AHEUI_VABLE_ARRAYS
                        .iter()
                        .map(|(name, idx)| {
                            majit_translate::VirtualizableFieldDescriptor::new(
                                *name,
                                Some(virtualizable_spec::AHEUI_VABLE_OWNER_ROOT.to_string()),
                                *idx,
                            )
                        })
                        .collect(),
                    call_effects: build_call_effect_overrides(),
                    struct_storage: vec![
                        majit_translate::StructStorageDescriptor::raw(
                            "storage::linkedlist::ListBase",
                        ),
                        majit_translate::StructStorageDescriptor::headerless(
                            "storage::linkedlist::Node",
                        ),
                    ],
                    ..Default::default()
                },
                //   driver = jit.JitDriver(
                //       greens=['pc','stackok','is_queue','program'],
                //       reds=['stacksize','storage','selected'])
                register_trait_families: Vec::new(),
                jit_drivers: vec![majit_translate::JitDriverSpec {
                    // `mainloop` is a crate-root (`lib.rs`) function, so its
                    // module-qualified identity is the bare name.
                    portal: majit_translate::CallPath::from_segments(["mainloop"]),
                    // No synthetic runner wraps the portal here: the driver is
                    // entered from the `#[jit_interp]` merge-point hook inside
                    // `mainloop` itself, so there is no separate function whose
                    // direct calls `guess_call_kind` should classify as
                    // recursive.
                    portal_runner: None,
                    greens: vec![
                        "pc".to_string(),
                        "is_queue".to_string(),
                        "program".to_string(),
                    ],
                    reds: vec![
                        "stacksize".to_string(),
                        "storage".to_string(),
                        "selected".to_string(),
                    ],
                    // Empty leaves the positional-kind check disabled, the
                    // documented default for a driver that does not carry a
                    // portal signature here.
                    green_kinds: Vec::new(),
                    red_kinds: Vec::new(),
                    autoreds: false,
                    virtualizables: Vec::new(),
                    red_types: Vec::new(),
                    // The marker's own graph is the one to register against:
                    // `mainloop` reaches its `jit_merge_point` on the same
                    // graph the dispatch loop runs, so no split copy is made.
                    split_portal: false,
                }],
            },
        },
        None,
        vinfo_factory,
        &[],
        majit_translate::HostStaticAddrs::default(),
    );

    // aheui drives the JIT from the `#[jit_interp]` proc macro, not from
    // the pyre-oriented trace helpers. The `Minimal` flavor emits only
    // the generic metadata tables so the included file compiles without
    // `pyre_object` / `pyre_interpreter` in scope.
    let code = majit_translate::generate_trace_code_from_pipeline_with_flavor(
        &pipeline,
        majit_translate::CodegenFlavor::Minimal,
    );

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    let artifacts = majit_translate::artifacts::EmbeddedArtifacts::from_pipeline(&pipeline)
        .expect("encode Aheui pipeline artifacts");
    std::fs::write(
        format!("{out_dir}/jitcode_artifacts.bin"),
        artifacts.encode().unwrap(),
    )
    .unwrap();

    eprintln!(
        "[aheui-jit build.rs] canonical analysis: {} jitcodes, {} functions, {} blocks, {} flat ops, generated {} bytes",
        pipeline.jitcodes.len(),
        pipeline.functions.len(),
        pipeline.total_blocks,
        pipeline.total_ops,
        code.len(),
    );

    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
    println!("cargo::rerun-if-changed=src/jit/virtualizable_spec.rs");
    println!("cargo::rerun-if-changed=src/jit/call_spec.rs");
}

fn build_call_effect_overrides() -> Vec<majit_translate::CallEffectOverride> {
    call_spec::AHEUI_CALL_EFFECTS
        .iter()
        .map(|spec| {
            let target = match spec.target {
                call_spec::CallTargetSpec::Method {
                    name,
                    receiver_root,
                } => majit_translate::CallTarget::method(name, Some(receiver_root.to_string())),
                call_spec::CallTargetSpec::FunctionPath(segments) => {
                    majit_translate::CallTarget::function_path(segments.iter().copied())
                }
            };
            let effect = match spec.effect {
                call_spec::CallEffectKind::Elidable => majit_translate::CallEffectKind::Elidable,
                call_spec::CallEffectKind::Residual => majit_translate::CallEffectKind::Residual,
            };
            majit_translate::CallEffectOverride::new(target, effect)
        })
        .collect()
}
