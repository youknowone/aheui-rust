use std::path::PathBuf;
use std::process::Command;

use clap::{Parser, ValueEnum};

#[derive(Clone, Copy, ValueEnum)]
enum Emit {
    /// 바희너리로 컴파흴 (기본값)
    Link,
    /// 생성된 소스를 stdout으로 출력
    Asm,
}

#[derive(Clone, Copy, ValueEnum)]
enum Codegen {
    /// Rust 코드 생성 후 cargo/rustc로 컴파흴
    Rust,
    /// C 코드 생성 후 cc로 컴파흴
    C,
    /// Cranelift으로 AOT컴파흴 후 실행
    Cranelift,
    /// WAT 코드 생성 후 wat2wasm으로 컴파흴
    Wat,
}

fn parse_opt_level(s: &str) -> Result<ahsembler::OptimizationLevel, String> {
    ahsembler::OptimizationLevel::from_str(s)
}

/// 아희 컴파흴러
#[derive(Parser)]
#[command(name = "compaheuiler")]
struct Cli {
    /// 입력 .aheui 소스 파일
    input: PathBuf,

    /// 출력 파일 경로
    #[arg(short = 'o')]
    output: Option<PathBuf>,

    /// 최적화 수준 (0-3)
    #[arg(short = 'O', default_value = "3", value_parser = parse_opt_level)]
    opt_level: ahsembler::OptimizationLevel,

    /// 출력 형식
    #[arg(long, value_enum, default_value_t = Emit::Link)]
    emit: Emit,

    /// 코드 생성 백엔드
    #[arg(long, value_enum, default_value_t = Codegen::Rust)]
    codegen: Codegen,

    /// WAT codegen: WASI 모드로 생성 (기본: env import)
    #[arg(long)]
    wasi: bool,
}

fn main() {
    let cli = Cli::parse();

    let source = std::fs::read_to_string(&cli.input).unwrap_or_else(|e| {
        eprintln!("error: {}: {e}", cli.input.display());
        std::process::exit(1);
    });

    match (cli.codegen, cli.emit) {
        (Codegen::Rust, Emit::Asm) => {
            print!("{}", generate_rs(&source, cli.opt_level));
        }
        (Codegen::C, Emit::Asm) => {
            print!("{}", compaheuiler::compile_to_c(&source));
        }
        (Codegen::Cranelift, Emit::Asm) => {
            eprintln!("error: --codegen cranelift does not support --emit asm");
            std::process::exit(1);
        }
        (Codegen::Rust, Emit::Link) => {
            compile_rs(&generate_rs(&source, cli.opt_level), &output_path(&cli));
        }
        (Codegen::C, Emit::Link) => {
            compile_c(&compaheuiler::compile_to_c(&source), &output_path(&cli));
        }
        (Codegen::Cranelift, Emit::Link) => {
            #[cfg(feature = "cranelift")]
            {
                run_cranelift(&source);
            }
            #[cfg(not(feature = "cranelift"))]
            {
                eprintln!("error: cranelift feature not enabled");
                std::process::exit(1);
            }
        }
        (Codegen::Wat, Emit::Asm) => {
            let wat = if cli.wasi {
                compaheuiler::compile_to_wat_wasi(&source)
            } else {
                compaheuiler::compile_to_wat(&source)
            };
            print!("{wat}");
        }
        (Codegen::Wat, Emit::Link) => {
            compile_wat(&compaheuiler::compile_to_wat_wasi(&source), &output_path(&cli));
        }
    }
}

fn generate_rs(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    #[cfg(feature = "bigint")]
    { compaheuiler::compile_to_rs_bigint_opt(source, opt) }
    #[cfg(not(feature = "bigint"))]
    { compaheuiler::compile_to_rs_opt(source, opt) }
}

fn output_path(cli: &Cli) -> PathBuf {
    if let Some(ref p) = cli.output {
        return p.clone();
    }
    let mut stem = cli.input.file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    stem = stem.replace('.', "_").replace('-', "_");
    std::env::current_dir().unwrap().join(stem)
}

fn compile_rs(code: &str, bin_path: &PathBuf) {
    #[cfg(feature = "bigint")]
    {
        let stem = bin_path.file_name().unwrap().to_string_lossy();
        let dir = format!("/tmp/compaheuiler_{stem}");
        std::fs::create_dir_all(format!("{dir}/src")).ok();
        std::fs::write(format!("{dir}/src/main.rs"), code).unwrap();

        #[cfg(feature = "num-bigint")]
        let bigint_dep = r#"num-bigint = "0.4""#;
        #[cfg(not(feature = "num-bigint"))]
        let bigint_dep = r#"malachite-bigint = "0.9""#;

        std::fs::write(
            format!("{dir}/Cargo.toml"),
            format!(
                r#"[package]
name = "compaheuiler-output"
version = "0.0.1"
edition = "2021"

[dependencies]
{bigint_dep}
num-traits = "0.2"

[profile.release]
opt-level = 2
lto = "fat"
"#
            ),
        )
        .unwrap();

        let status = Command::new("cargo")
            .args(["build", "--release", "--quiet"])
            .current_dir(&dir)
            .status()
            .unwrap_or_else(|e| {
                eprintln!("error: cargo: {e}");
                std::process::exit(1);
            });
        if !status.success() {
            std::process::exit(1);
        }
        std::fs::copy(format!("{dir}/target/release/compaheuiler-output"), bin_path).unwrap();
    }

    #[cfg(not(feature = "bigint"))]
    {
        let rs_path = format!("/tmp/compaheuiler_{}.rs", std::process::id());
        std::fs::write(&rs_path, code).unwrap();

        let output = Command::new("rustc")
            .args(["-C", "opt-level=2", "-o", bin_path.to_str().unwrap(), &rs_path])
            .output()
            .unwrap_or_else(|e| {
                eprintln!("error: rustc: {e}");
                std::process::exit(1);
            });
        if !output.status.success() {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            std::process::exit(1);
        }
        let _ = std::fs::remove_file(&rs_path);
    }
}

fn compile_wat(code: &str, bin_path: &PathBuf) {
    let wat_path = format!("/tmp/compaheuiler_{}.wat", std::process::id());
    std::fs::write(&wat_path, code).unwrap();

    let wasm_path = bin_path.with_extension("wasm");
    let output = Command::new("wat2wasm")
        .args([&wat_path, "-o", wasm_path.to_str().unwrap()])
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: wat2wasm: {e}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
    let _ = std::fs::remove_file(&wat_path);
}

fn compile_c(code: &str, bin_path: &PathBuf) {
    let c_path = format!("/tmp/compaheuiler_{}.c", std::process::id());
    std::fs::write(&c_path, code).unwrap();

    let output = Command::new("cc")
        .args(["-O2", "-o", bin_path.to_str().unwrap(), &c_path, "-lm"])
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: cc: {e}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
    let _ = std::fs::remove_file(&c_path);
}

#[cfg(feature = "cranelift")]
fn run_cranelift(source: &str) {
    use compaheuiler::cranelift_backend::jit;
    use std::io::Write;

    let mut cfg = ahsembler::compile_to_cfg_aot(source);
    for _ in 0..10 {
        let before = cfg.num_blocks();
        let states = ahsembler::cfg_optimize::analyze_stack_depths(&cfg);
        ahsembler::cfg_optimize::eliminate_guards(&mut cfg, &states);
        ahsembler::cfg_optimize::eliminate_guard_depth(&mut cfg, &states);
        ahsembler::cfg_optimize::thread_jumps(&mut cfg);
        ahsembler::cfg_optimize::merge_guard_ok(&mut cfg);
        ahsembler::cfg_optimize::merge_blocks(&mut cfg);
        ahsembler::cfg_optimize::eliminate_dead_blocks(&mut cfg);
        if cfg.num_blocks() == before { break; }
    }

    let jit_fn = jit::JitCompiler::compile_cfg(&cfg).unwrap_or_else(|e| {
        eprintln!("error: cranelift: {e}");
        std::process::exit(1);
    });

    let mut bases = [std::ptr::null_mut::<i64>(); 28];
    let mut lengths = [0i32; 28];
    let mut data: Vec<Vec<i64>> = (0..28).map(|_| vec![0i64; 65536]).collect();
    for i in 0..28 { bases[i] = data[i].as_mut_ptr(); }

    extern "C" fn write_char(v: i64) {
        if let Some(ch) = char::from_u32(v as u32) {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let _ = std::io::stdout().write_all(s.as_bytes());
        }
    }
    extern "C" fn write_num(v: i64) {
        let _ = write!(std::io::stdout(), "{v}");
    }
    extern "C" fn read_char() -> i64 {
        use std::io::Read;
        let mut buf = [0u8; 1];
        match std::io::stdin().read(&mut buf) {
            Ok(1) => buf[0] as i64,
            _ => -1,
        }
    }
    extern "C" fn read_num() -> i64 {
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).ok();
        s.trim().parse().unwrap_or(0)
    }
    extern "C" fn sp_push(_: *mut u8, _: usize, _: i64) {}
    extern "C" fn sp_pop(_: *mut u8, _: usize) -> i64 { 0 }
    extern "C" fn sp_depth(_: *mut u8, _: usize) -> i64 { 0 }
    extern "C" fn sp_dup(_: *mut u8, _: usize) {}
    extern "C" fn sp_swap(_: *mut u8, _: usize) {}

    let exit_code = unsafe {
        jit_fn.execute(
            &mut bases, &mut lengths,
            write_char, write_num, read_char, read_num,
            std::ptr::null_mut(),
            sp_push, sp_pop, sp_depth, sp_dup, sp_swap,
        )
    };
    let _ = std::io::stdout().flush();
    std::process::exit(exit_code as i32);
}
