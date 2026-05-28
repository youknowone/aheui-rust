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

    let source_dirs = [
        format!("{base}/aheuinterpreter/src"),
        format!("{base}/aheui-runtime/src"),
    ];

    let mut sources = Vec::new();
    let mut source_paths = Vec::new();
    for dir in &source_dirs {
        collect_rs_files(dir, &mut sources, &mut source_paths);
    }

    eprintln!(
        "[aheui-jit build.rs] reading {} source files from {} dirs",
        sources.len(),
        source_dirs.len(),
    );

    let source_refs: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();
    let pipeline = majit_translate::analyze_multiple_pipeline_with_config(
        &source_refs,
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
                    ..Default::default()
                },
                // rpaheui/aheui/aheui.py:28-31
                //   driver = jit.JitDriver(
                //       greens=['pc','stackok','is_queue','program'],
                //       reds=['stacksize','storage','selected'])
                portal: Some(majit_translate::PortalSpec {
                    name: "mainloop".to_string(),
                    greens: vec![
                        "pc".to_string(),
                        "stackok".to_string(),
                        "is_queue".to_string(),
                        "program".to_string(),
                    ],
                    reds: vec![
                        "stacksize".to_string(),
                        "storage".to_string(),
                        "selected".to_string(),
                    ],
                    virtualizables: Vec::new(),
                    red_types: Vec::new(),
                }),
            },
        },
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

    eprintln!(
        "[aheui-jit build.rs] canonical analysis: {} opcode arms, {} functions, {} blocks, {} flat ops, generated {} bytes",
        pipeline.opcode_dispatch.len(),
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

fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[aheui-jit build.rs] warning: cannot read {dir}");
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path.to_string_lossy(), sources, paths);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let path_str = path.to_string_lossy().to_string();
            match std::fs::read_to_string(&path) {
                Ok(content) => {
                    paths.push(path_str);
                    sources.push(content);
                }
                Err(e) => {
                    eprintln!("[aheui-jit build.rs] warning: cannot read {path_str}: {e}");
                }
            }
        }
    }
}
