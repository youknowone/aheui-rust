/// Build script for aheui-jit: analyzes the Aheui interpreter via the
/// majit-translate graph pipeline and emits the generated trace code.
///
/// This uses the same graph-based analysis path as pyre-jit.

#[path = "src/jit/call_spec.rs"]
mod call_spec;
#[path = "src/jit/virtualizable_spec.rs"]
mod virtualizable_spec;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let base = format!("{manifest_dir}/..");

    // The majit-translate graph pipeline lowers Charon-extracted MIR
    // (`.ullbc`), not the syn-parsed source strings below. The shared
    // front-end auto-discovers only the parent repo's pyre artefact pair,
    // so point it at aheui's own crate LLBC (extracted by
    // `scripts/extract-llbc.sh` into `<aheui>/build/llbc/`) via
    // `PYRE_MIR_FRONTEND_LLBC`, which the front-end honours ahead of
    // auto-discovery. An explicit env override still wins.
    if std::env::var_os("PYRE_MIR_FRONTEND_LLBC").is_none() {
        let llbc_dir = std::path::Path::new(&base).join("build").join("llbc");
        let rt = llbc_dir.join("aheui-runtime.ullbc");
        let interp = llbc_dir.join("aheuinterpreter.ullbc");
        if rt.exists() && interp.exists() {
            let joined = std::env::join_paths([rt, interp])
                .expect("aheui LLBC paths contain no path separator");
            // SAFETY: build scripts are single-threaded; no other thread
            // observes the environment during this set.
            unsafe { std::env::set_var("PYRE_MIR_FRONTEND_LLBC", joined) };
        } else {
            panic!(
                "aheui LLBC missing under {}.\n\
                 Run `aheui/scripts/extract-llbc.sh` to produce \
                 `aheui-runtime.ullbc` + `aheuinterpreter.ullbc` \
                 (install with the parent repo's \
                 `python3 scripts/install-charon.py`), or set \
                 `PYRE_MIR_FRONTEND_LLBC` explicitly.",
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
    // (`PYRE_MIR_FRONTEND_LLBC` above). `module_paths[i]` is the
    // crate-stripped module path of the i-th source file, derived by the
    // translator that consumes it — the spelling is its invariant, not a
    // label this build script gets to choose.
    // aheui carries no host vinfo / fnaddr / static-singleton bindings —
    // the `#[jit_interp]` macro plus the `Minimal` flavor cover those.
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
                // rpaheui/aheui/aheui.py:28-31
                //   driver = jit.JitDriver(
                //       greens=['pc','stackok','is_queue','program'],
                //       reds=['stacksize','storage','selected'])
                register_trait_families: Vec::new(),
                jit_drivers: vec![majit_translate::JitDriverSpec {
                    // `mainloop` is a crate-root (`lib.rs`) function, so its
                    // module-qualified identity is the bare name.
                    portal: majit_translate::CallPath::from_segments(["mainloop"]),
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

    let json = serde_json::to_string_pretty(&pipeline).unwrap();
    std::fs::write(format!("{out_dir}/jit_metadata.json"), &json).unwrap();

    // Persist `pipeline.jitcodes` (the codewriter's `make_jitcodes()` output:
    // the `mainloop` portal + per-storage-method sub-jitcodes, inter-linked via
    // the shared descr pool) as a bincode blob the runtime deserializes into
    // `Vec<Arc<JitCode>>`. Same single-store model as
    // `pyre-jit-trace/build.rs` `opcode_jitcodes.bin`.
    let jitcodes_bin = bincode::serialize(&pipeline.jitcodes).unwrap();
    std::fs::write(format!("{out_dir}/opcode_jitcodes.bin"), &jitcodes_bin).unwrap();

    // Persist the shared descr pool (`Assembler.descrs`, assembler.py:23 /
    // blackhole.py:59 `setup_descrs`). Each 'd'/'j' argcode in a
    // `JitCode.code` byte stream is a 2-byte index into this pool; the 'j'
    // (BC_INLINE_CALL) entries carry the inter-jitcode links. Same as
    // `pyre-jit-trace/build.rs` `opcode_descrs.bin`.
    let descrs_bin = bincode::serialize(&pipeline.descrs).unwrap();
    std::fs::write(format!("{out_dir}/opcode_descrs.bin"), &descrs_bin).unwrap();

    let symbolic_fnaddrs_bin = bincode::serialize(&pipeline.symbolic_fnaddr_paths).unwrap();
    std::fs::write(
        format!("{out_dir}/opcode_symbolic_fnaddrs.bin"),
        &symbolic_fnaddrs_bin,
    )
    .unwrap();

    std::fs::write(
        format!("{out_dir}/opcode_liveness.bin"),
        &pipeline.all_liveness,
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
