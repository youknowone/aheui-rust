use std::path::PathBuf;
use std::process::Command;

use clap::{Args, ValueEnum};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Emit {
    /// 바이너리로 컴파일 (기본값)
    Link,
    /// 생성된 소스를 stdout으로 출력
    Asm,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Codegen {
    /// Rust 코드 생성 후 cargo/rustc로 컴파일
    Rust,
    /// C 코드 생성 후 cc로 컴파일
    C,
    /// Cranelift으로 AOT컴파일 후 실행
    Cranelift,
    /// WASM (WASI): WAT 생성, fd_write/fd_read 사용
    #[value(name = "wasm32-wasi")]
    Wasm32Wasi,
    /// WASM (Web): WAT 생성, env.write_byte/read_byte import
    #[value(name = "wasm32-web")]
    Wasm32Web,
}

fn parse_opt_level(s: &str) -> Result<ahsembler::OptimizationLevel, String> {
    ahsembler::OptimizationLevel::from_str(s)
}

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// 입력 .aheui 소스 파일
    pub input: PathBuf,

    /// 출력 파일 경로
    #[arg(short = 'o')]
    pub output: Option<PathBuf>,

    /// 최적화 수준 (0-3)
    #[arg(short = 'O', default_value = "3", value_parser = parse_opt_level)]
    pub opt_level: ahsembler::OptimizationLevel,

    /// 출력 형식
    #[arg(long, value_enum, default_value_t = Emit::Link)]
    pub emit: Emit,

    /// 코드 생성 백엔드
    #[arg(long, value_enum, default_value_t = Codegen::Rust)]
    pub codegen: Codegen,
}

impl BuildArgs {
    pub fn output_path(&self) -> PathBuf {
        if let Some(ref p) = self.output {
            return p.clone();
        }
        let mut stem = self
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        stem = stem.replace('.', "_").replace('-', "_");
        std::env::current_dir().unwrap().join(stem)
    }
}

pub fn run_build(args: &BuildArgs) {
    let source = std::fs::read_to_string(&args.input).unwrap_or_else(|e| {
        eprintln!("error: {}: {e}", args.input.display());
        std::process::exit(1);
    });

    match (args.codegen, args.emit) {
        (Codegen::Rust, Emit::Asm) => {
            print!("{}", generate_rs(&source, args.opt_level));
        }
        (Codegen::C, Emit::Asm) => {
            print!("{}", generate_c(&source, args.opt_level));
        }
        (Codegen::Cranelift, Emit::Asm) => {
            eprintln!("error: --codegen cranelift does not support --emit asm");
            std::process::exit(1);
        }
        (Codegen::Wasm32Wasi, Emit::Asm) => {
            print!(
                "{}",
                crate::compile_to_wat_wasi_opt(&source, args.opt_level)
            );
        }
        (Codegen::Wasm32Web, Emit::Asm) => {
            print!("{}", crate::compile_to_wat_opt(&source, args.opt_level));
        }
        (Codegen::Rust, Emit::Link) => {
            compile_rs(&generate_rs(&source, args.opt_level), &args.output_path());
        }
        (Codegen::C, Emit::Link) => {
            #[cfg(feature = "bigint")]
            compile_c_bigint(&generate_c(&source, args.opt_level), &args.output_path());
            #[cfg(not(feature = "bigint"))]
            compile_c(&generate_c(&source, args.opt_level), &args.output_path());
        }
        (Codegen::Cranelift, Emit::Link) => {
            #[cfg(feature = "cranelift")]
            {
                run_cranelift(&source, args.opt_level);
            }
            #[cfg(not(feature = "cranelift"))]
            {
                eprintln!("error: cranelift feature not enabled");
                std::process::exit(1);
            }
        }
        (Codegen::Wasm32Wasi, Emit::Link) => {
            compile_wat(
                &crate::compile_to_wat_wasi_opt(&source, args.opt_level),
                &args.output_path(),
            );
        }
        (Codegen::Wasm32Web, Emit::Link) => {
            compile_wat(
                &crate::compile_to_wat_opt(&source, args.opt_level),
                &args.output_path(),
            );
        }
    }
}

fn generate_rs(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    #[cfg(feature = "bigint")]
    {
        crate::compile_to_rs_bigint_opt(source, opt)
    }
    #[cfg(not(feature = "bigint"))]
    {
        crate::compile_to_rs_opt(source, opt)
    }
}

fn generate_c(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    #[cfg(feature = "bigint")]
    {
        crate::compile_to_c_bigint_opt(source, opt)
    }
    #[cfg(not(feature = "bigint"))]
    {
        crate::compile_to_c_opt(source, opt)
    }
}

/// `--remap-path-prefix` arguments that keep this machine out of the compiled
/// binary.
///
/// rustc records the path of every file it compiles, so without these the
/// output names the generated crate under `/tmp`, the dependency sources under
/// `$CARGO_HOME` and the standard library under the toolchain sysroot — every
/// one of them per-machine. Two people compiling the same Aheui program got
/// two different binaries, each carrying the other's home directory.
///
/// `cc` and `wat2wasm` get no equivalent: they are invoked without debug info,
/// their temporary path does not reach their output, and the flag that would
/// do this (`-ffile-prefix-map`) is not portable across the compilers `cc` may
/// resolve to.
fn remap_path_prefixes(generated_dir: &str) -> Vec<String> {
    let mut flags = Vec::new();
    // The generated crate. Canonicalized because `/tmp` is a symlink on macOS
    // and rustc records the resolved path, which a `/tmp` prefix never matches.
    let generated = std::fs::canonicalize(generated_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| generated_dir.to_string());
    flags.push(format!("--remap-path-prefix={generated}=/compaheuiler"));

    if let Some(cargo_home) = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cargo")))
    {
        flags.push(format!(
            "--remap-path-prefix={}=/cargo",
            cargo_home.display()
        ));
    }

    if let Ok(out) = Command::new("rustc").args(["--print", "sysroot"]).output() {
        if out.status.success() {
            if let Ok(sysroot) = String::from_utf8(out.stdout) {
                flags.push(format!(
                    "--remap-path-prefix={}=/rust",
                    sysroot.trim_end_matches('\n')
                ));
            }
        }
    }
    flags
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

        // `.cargo/config.toml` beside the generated crate rather than
        // `RUSTFLAGS`, so an existing `RUSTFLAGS` in the caller's environment
        // still wins instead of being silently replaced.
        std::fs::create_dir_all(format!("{dir}/.cargo")).ok();
        let rustflags = remap_path_prefixes(&dir)
            .iter()
            .map(|f| format!("{:?}", f))
            .collect::<Vec<_>>()
            .join(", ");
        std::fs::write(
            format!("{dir}/.cargo/config.toml"),
            format!("[build]\nrustflags = [{rustflags}]\n"),
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
        std::fs::copy(
            format!("{dir}/target/release/compaheuiler-output"),
            bin_path,
        )
        .unwrap();
    }

    #[cfg(not(feature = "bigint"))]
    {
        let dir = format!("/tmp/compaheuiler_{}", std::process::id());
        std::fs::create_dir_all(&dir).ok();
        let rs_path = format!("{dir}/main.rs");
        std::fs::write(&rs_path, code).unwrap();

        let output = Command::new("rustc")
            .args([
                "-C",
                "opt-level=2",
                "-o",
                bin_path.to_str().unwrap(),
                &rs_path,
            ])
            .args(remap_path_prefixes(&dir))
            .output()
            .unwrap_or_else(|e| {
                eprintln!("error: rustc: {e}");
                std::process::exit(1);
            });
        if !output.status.success() {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            std::process::exit(1);
        }
        let _ = std::fs::remove_dir_all(&dir);
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

#[cfg(not(feature = "bigint"))]
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

#[cfg(feature = "bigint")]
fn compile_c_bigint(code: &str, bin_path: &PathBuf) {
    let stem = bin_path.file_name().unwrap().to_string_lossy();
    let suffix: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    let suffix = suffix.trim_matches('-');
    let suffix = if suffix.is_empty() { "output" } else { suffix };
    let package = format!("compaheuiler-c-output-{suffix}");
    let dir = format!("/tmp/compaheuiler_c_{stem}");
    let target_dir = "/tmp/compaheuiler_c_cargo_target";
    std::fs::create_dir_all(format!("{dir}/src")).ok();
    let c_path = format!("{dir}/program.c");
    let object_path = format!("{dir}/program.o");
    // Older compaheuiler versions put a per-output link arg here, which also
    // forced Cargo to rebuild every BigInt dependency for every C program.
    let _ = std::fs::remove_file(format!("{dir}/.cargo/config.toml"));
    std::fs::write(&c_path, code).unwrap();
    std::fs::write(format!("{dir}/src/main.rs"), crate::c_bigint_bridge_rs()).unwrap();
    std::fs::write(
        format!("{dir}/Cargo.toml"),
        format!(
            r#"[package]
name = {package:?}
version = "0.0.1"
edition = "2024"

[dependencies]
{bigint_dependency}
num-traits = "0.2"

[profile.release]
opt-level = 3
"#,
            bigint_dependency = crate::c_bigint_dependency_toml(),
        ),
    )
    .unwrap();

    let output = Command::new("cc")
        .args(["-O2", "-std=c99", "-DCOMPAHEUILER_RUST_BIGINT", "-c"])
        .arg(&c_path)
        .args(["-o", &object_path])
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: cc: {e}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }

    std::fs::write(
        format!("{dir}/build.rs"),
        format!(
            "fn main() {{\n  let object = {object_path:?};\n  println!(\"cargo:rerun-if-changed={{object}}\");\n  println!(\"cargo:rustc-link-arg-bin={package}={{object}}\");\n}}\n",
        ),
    )
    .unwrap();

    let status = Command::new("cargo")
        .args(["rustc", "--release", "--quiet"])
        .current_dir(&dir)
        .env("CARGO_TARGET_DIR", target_dir)
        .arg("--")
        .args(remap_path_prefixes(&dir))
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error: cargo: {e}");
            std::process::exit(1);
        });
    if !status.success() {
        std::process::exit(1);
    }
    std::fs::copy(format!("{target_dir}/release/{package}"), bin_path).unwrap();
}

#[cfg(feature = "cranelift")]
fn run_cranelift(source: &str, opt: ahsembler::OptimizationLevel) {
    use crate::cranelift_backend::jit;
    use std::io::Write;

    let cfg = crate::pipeline::optimize(source, opt);

    let jit_fn = jit::compile_cfg(&cfg).unwrap_or_else(|e| {
        eprintln!("error: cranelift: {e}");
        std::process::exit(1);
    });

    let mut bases = [std::ptr::null_mut::<i64>(); 28];
    let mut lengths = [0i32; 28];
    let mut data: Vec<Vec<i64>> = (0..28).map(|_| vec![0i64; 65536]).collect();
    for i in 0..28 {
        bases[i] = data[i].as_mut_ptr();
    }

    extern "C" fn write_char(v: i64) {
        if let Some(ch) = char::from_u32(v as u32) {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            let _ = std::io::stdout().write_all(s.as_bytes());
        }
    }
    extern "C" fn write_bytes(data: *const u8, len: usize) {
        let bytes = unsafe { std::slice::from_raw_parts(data, len) };
        let _ = std::io::stdout().write_all(bytes);
    }
    extern "C" fn write_num(v: i64) {
        let _ = write!(std::io::stdout(), "{v}");
    }
    /// One character, not one byte: the continuation bytes of a multi-byte
    /// sequence are consumed here and the code point is returned. Same decoding
    /// the rust and C backends emit.
    extern "C" fn read_char() -> i64 {
        use std::io::Read;
        let mut buf = [0u8; 4];
        match std::io::stdin().read(&mut buf[..1]) {
            Ok(0) | Err(_) => -1,
            Ok(_) => {
                let b = buf[0];
                if b < 0x80 {
                    return b as i64;
                }
                let (n, mut val) = if b >> 5 == 6 {
                    (1, (b & 0x1F) as i32)
                } else if b >> 4 == 14 {
                    (2, (b & 0x0F) as i32)
                } else if b >> 3 == 30 {
                    (3, (b & 0x07) as i32)
                } else {
                    return -1;
                };
                for _ in 0..n {
                    if std::io::stdin().read(&mut buf[..1]).unwrap_or(0) == 0 {
                        return -1;
                    }
                    val = (val << 6) | (buf[0] & 0x3F) as i32;
                }
                val as i64
            }
        }
    }
    extern "C" fn read_num() -> i64 {
        let mut s = String::new();
        std::io::stdin().read_line(&mut s).ok();
        s.trim().parse().unwrap_or(0)
    }
    // The queue and the port live outside `bases`/`lengths`, so they are reached
    // through these callbacks. Stubbing them out drops every queue push and
    // answers every pop and depth with 0.
    let mut sp = jit::SpecialStorage::new();
    let sp_ctx = &mut sp as *mut jit::SpecialStorage as *mut u8;

    let exit_code = unsafe {
        jit_fn.execute_buffered(
            &mut bases,
            &mut lengths,
            write_char,
            write_bytes,
            write_num,
            read_char,
            read_num,
            sp_ctx,
            jit::sp_push,
            jit::sp_pop,
            jit::sp_depth,
            jit::sp_dup,
            jit::sp_swap,
        )
    };
    let _ = std::io::stdout().flush();
    std::process::exit(exit_code as i32);
}
