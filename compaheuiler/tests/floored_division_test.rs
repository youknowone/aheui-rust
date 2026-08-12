mod common;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const DIV_TWO_INPUTS: &str = "방방나망하";
const MOD_TWO_INPUTS: &str = "방방라망하";
const DIV_THREE_INPUTS: &str = "방방따방나망하";
const MOD_THREE_INPUTS: &str = "방방따방라망하";

struct Case {
    input: &'static str,
    wide_input: Option<&'static str>,
    div: &'static str,
    rem: &'static str,
}

const CASES: &[Case] = &[
    Case {
        input: "7\n2\n",
        wide_input: None,
        div: "3",
        rem: "1",
    },
    Case {
        input: "-7\n2\n",
        wide_input: None,
        div: "-4",
        rem: "1",
    },
    Case {
        input: "7\n-2\n",
        wide_input: None,
        div: "-4",
        rem: "-1",
    },
    Case {
        input: "-7\n-2\n",
        wide_input: None,
        div: "3",
        rem: "-1",
    },
    Case {
        input: "13\n10\n",
        wide_input: None,
        div: "1",
        rem: "3",
    },
    Case {
        input: "-13\n10\n",
        wide_input: None,
        div: "-2",
        rem: "7",
    },
    Case {
        input: "13\n-10\n",
        wide_input: None,
        div: "-2",
        rem: "-7",
    },
    Case {
        input: "-13\n-10\n",
        wide_input: None,
        div: "1",
        rem: "-3",
    },
    Case {
        input: "-6\n3\n",
        wide_input: None,
        div: "-2",
        rem: "0",
    },
    Case {
        input: "6\n-3\n",
        wide_input: None,
        div: "-2",
        rem: "0",
    },
    Case {
        input: "-16000000000000000000\n7\n",
        wide_input: Some("-4000000000\n4000000000\n7\n"),
        div: "-2285714285714285715",
        rem: "5",
    },
    Case {
        input: "16000000000000000000\n-7\n",
        wide_input: Some("4000000000\n4000000000\n-7\n"),
        div: "-2285714285714285715",
        rem: "-5",
    },
];

fn run_binary(path: &Path, input: &str) -> String {
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", path.display()));
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{} failed: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn assert_cases(backend: &str, div: &Path, rem: &Path, wide_div: &Path, wide_rem: &Path) {
    for case in CASES {
        let (input, div_bin, rem_bin) = match case.wide_input {
            Some(input) => (input, wide_div, wide_rem),
            None => (case.input, div, rem),
        };
        assert_eq!(
            run_binary(div_bin, input),
            case.div,
            "{backend} div {input:?}"
        );
        assert_eq!(
            run_binary(rem_bin, input),
            case.rem,
            "{backend} rem {input:?}"
        );
    }
}

#[test]
fn rust_aot_uses_floored_division_and_remainder() {
    let scratch = common::scratch_dir("floor_rust");
    let src = scratch.path().join("src/bin");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        scratch.path().join("Cargo.toml"),
        r#"[package]
name = "compaheuiler-floor-test"
version = "0.0.0"
edition = "2021"

[dependencies]
malachite-bigint = "0.9"
num-traits = "0.2"

[profile.release]
opt-level = 1
"#,
    )
    .unwrap();
    for (name, program) in [
        ("div", DIV_TWO_INPUTS),
        ("rem", MOD_TWO_INPUTS),
        ("wide_div", DIV_THREE_INPUTS),
        ("wide_rem", MOD_THREE_INPUTS),
    ] {
        std::fs::write(
            src.join(format!("{name}.rs")),
            compaheuiler::compile_to_rs_bigint(program),
        )
        .unwrap();
    }
    let output = Command::new("cargo")
        .args(["build", "--release", "--bins", "--quiet"])
        .current_dir(scratch.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "generated Rust failed to compile: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bins = scratch.path().join("target/release");
    assert_cases(
        "rust",
        &bins.join("div"),
        &bins.join("rem"),
        &bins.join("wide_div"),
        &bins.join("wide_rem"),
    );
}

fn compile_c(scratch: &Path, name: &str, program: &str) -> PathBuf {
    common::compile_c_bigint(scratch, name, &compaheuiler::compile_to_c_bigint(program))
}

#[test]
fn c_aot_uses_floored_division_and_remainder() {
    let scratch = common::scratch_dir("floor_c");
    let div = compile_c(scratch.path(), "div", DIV_TWO_INPUTS);
    let rem = compile_c(scratch.path(), "rem", MOD_TWO_INPUTS);
    let wide_div = compile_c(scratch.path(), "wide_div", DIV_THREE_INPUTS);
    let wide_rem = compile_c(scratch.path(), "wide_rem", MOD_THREE_INPUTS);
    assert_cases("c", &div, &rem, &wide_div, &wide_rem);
}

fn compile_wat(scratch: &Path, name: &str, program: &str) -> PathBuf {
    let wasm_path = scratch.join(format!("{name}.wasm"));
    let wasm = wat::parse_str(compaheuiler::compile_to_wat(program)).unwrap();
    std::fs::write(&wasm_path, wasm).unwrap();
    wasm_path
}

#[test]
fn wat_aot_uses_floored_division_and_remainder_for_i64_inputs() {
    let scratch = common::scratch_dir("floor_wat");
    let runner = scratch.path().join("run.js");
    std::fs::write(
        &runner,
        r#"const fs = require('fs');
const input = fs.readFileSync(0);
let pos = 0;
const output = [];
WebAssembly.instantiate(fs.readFileSync(process.argv[2]), {env: {
  read_byte: () => pos < input.length ? input[pos++] : -1,
  write_byte: value => output.push(value & 255),
}}).then(({instance}) => {
  instance.exports.run();
  process.stdout.write(Buffer.from(output));
}).catch(error => { console.error(error); process.exit(1); });
"#,
    )
    .unwrap();
    let div = compile_wat(scratch.path(), "div", DIV_TWO_INPUTS);
    let rem = compile_wat(scratch.path(), "rem", MOD_TWO_INPUTS);
    for case in CASES.iter().filter(|case| case.wide_input.is_none()) {
        let run = |wasm: &Path| {
            let mut child = Command::new("node")
                .arg(&runner)
                .arg(wasm)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            child
                .stdin
                .take()
                .unwrap()
                .write_all(case.input.as_bytes())
                .unwrap();
            let output = child.wait_with_output().unwrap();
            assert!(
                output.status.success(),
                "Node failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap()
        };
        assert_eq!(run(&div), case.div, "wat div {:?}", case.input);
        assert_eq!(run(&rem), case.rem, "wat rem {:?}", case.input);
    }
}

#[test]
fn wat_wasi_reuses_buffered_input_across_number_reads() {
    let scratch = common::scratch_dir("floor_wat_wasi");
    let wasm_path = scratch.path().join("div.wasm");
    let wasm = wat::parse_str(compaheuiler::compile_to_wat_wasi(DIV_TWO_INPUTS)).unwrap();
    std::fs::write(&wasm_path, wasm).unwrap();
    let runner = scratch.path().join("run.js");
    std::fs::write(
        &runner,
        r#"const fs = require('fs');
const { WASI } = require('wasi');
const wasi = new WASI({ version: 'preview1' });
WebAssembly.instantiate(fs.readFileSync(process.argv[2]), {
  wasi_snapshot_preview1: wasi.wasiImport,
}).then(({ instance }) => wasi.start(instance)).catch(error => {
  console.error(error);
  process.exit(1);
});
"#,
    )
    .unwrap();

    let mut child = Command::new("node")
        .arg(&runner)
        .arg(&wasm_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"7\t-2\r\n").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "Node WASI failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "-4");
}

#[cfg(feature = "cranelift")]
mod cranelift {
    use super::*;
    use compaheuiler::jit::{JitFunction, SpecialStorage};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    static INPUT: Mutex<VecDeque<i64>> = Mutex::new(VecDeque::new());
    static OUTPUT: Mutex<String> = Mutex::new(String::new());
    static RUN_LOCK: Mutex<()> = Mutex::new(());

    extern "C" fn write_char(value: i64) {
        if let Some(ch) = char::from_u32(value as u32) {
            OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).push(ch);
        }
    }
    extern "C" fn write_num(value: i64) {
        OUTPUT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_str(&value.to_string());
    }
    extern "C" fn read_char() -> i64 {
        -1
    }
    extern "C" fn read_num() -> i64 {
        INPUT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop_front()
            .unwrap_or(0)
    }

    fn compile(program: &str) -> JitFunction {
        let cfg = compaheuiler::pipeline::optimize(program, ahsembler::OptimizationLevel::O3);
        compaheuiler::jit::compile_cfg(&cfg).unwrap()
    }

    fn run(jit: &JitFunction, input: &str) -> String {
        let _run = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        *INPUT.lock().unwrap_or_else(|e| e.into_inner()) = input
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let mut data: Vec<Vec<i64>> = (0..28).map(|_| vec![0; 65536]).collect();
        let mut bases = [std::ptr::null_mut(); 28];
        for (base, storage) in bases.iter_mut().zip(&mut data) {
            *base = storage.as_mut_ptr();
        }
        let mut lengths = [0; 28];
        let mut sp = SpecialStorage::new();
        unsafe {
            jit.execute(
                &mut bases,
                &mut lengths,
                write_char,
                write_num,
                read_char,
                read_num,
                &mut sp as *mut SpecialStorage as *mut u8,
                compaheuiler::jit::sp_push,
                compaheuiler::jit::sp_pop,
                compaheuiler::jit::sp_depth,
                compaheuiler::jit::sp_dup,
                compaheuiler::jit::sp_swap,
            );
        }
        OUTPUT.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    #[test]
    fn cranelift_aot_uses_floored_division_and_remainder() {
        let div = compile(DIV_TWO_INPUTS);
        let rem = compile(MOD_TWO_INPUTS);
        let wide_div = compile(DIV_THREE_INPUTS);
        let wide_rem = compile(MOD_THREE_INPUTS);
        for case in CASES {
            #[cfg(not(feature = "bigint"))]
            if case.wide_input.is_some() {
                continue;
            }
            let (input, div_jit, rem_jit) = match case.wide_input {
                Some(input) => (input, &wide_div, &wide_rem),
                None => (case.input, &div, &rem),
            };
            assert_eq!(run(div_jit, input), case.div, "cranelift div {input:?}");
            assert_eq!(run(rem_jit, input), case.rem, "cranelift rem {input:?}");
        }
    }

    #[cfg(feature = "bigint")]
    #[test]
    fn cranelift_promotes_overflowing_min_division() {
        let div = compile(DIV_TWO_INPUTS);
        let rem = compile(MOD_TWO_INPUTS);
        assert_eq!(
            run(&div, "-9223372036854775808\n-1\n"),
            "9223372036854775808"
        );
        assert_eq!(run(&rem, "-9223372036854775808\n-1\n"), "0");
    }

    #[cfg(feature = "bigint")]
    #[test]
    fn cranelift_boxes_folded_literals_after_promotion() {
        let mut source = String::from("방방나박");
        for _ in 0..61 {
            source.push_str("박따");
        }
        source.push_str("망하");

        let cfg = compaheuiler::pipeline::optimize(&source, ahsembler::OptimizationLevel::O3);
        assert!(cfg.blocks.iter().any(|block| {
            block
                .instructions
                .contains(&ahsembler::cfg::Inst::Push(1_i64 << 62))
        }));
        let program = compaheuiler::jit::compile_cfg(&cfg).unwrap();
        assert_eq!(
            run(&program, "-9223372036854775808\n-1\n"),
            "4611686018427387904"
        );
    }
}
