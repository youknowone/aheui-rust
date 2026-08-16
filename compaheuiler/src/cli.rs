use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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
    pub fn output_path(&self) -> std::io::Result<PathBuf> {
        if let Some(ref p) = self.output {
            return Ok(p.clone());
        }
        let mut stem = self
            .input
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        stem = stem.replace(['.', '-'], "_");
        Ok(std::env::current_dir()?.join(stem))
    }
}

pub fn run_build(args: &BuildArgs) {
    let source = std::fs::read_to_string(&args.input).unwrap_or_else(|e| {
        eprintln!("error: {}: {e}", args.input.display());
        std::process::exit(1);
    });

    let output_path = match args.emit {
        Emit::Asm => None,
        Emit::Link => Some(args.output_path().unwrap_or_else(|e| {
            eprintln!("error: cannot determine output path: {e}");
            std::process::exit(1);
        })),
    };

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
            compile_rs(
                &generate_rs(&source, args.opt_level),
                output_path.as_ref().unwrap(),
            );
        }
        (Codegen::C, Emit::Link) => {
            #[cfg(feature = "bigint")]
            compile_c_bigint(
                &generate_c(&source, args.opt_level),
                output_path.as_ref().unwrap(),
            );
            #[cfg(not(feature = "bigint"))]
            compile_c(
                &generate_c(&source, args.opt_level),
                output_path.as_ref().unwrap(),
            );
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
                output_path.as_ref().unwrap(),
            );
        }
        (Codegen::Wasm32Web, Emit::Link) => {
            compile_wat(
                &crate::compile_to_wat_opt(&source, args.opt_level),
                output_path.as_ref().unwrap(),
            );
        }
    }
}

struct ScratchDir(PathBuf);

impl ScratchDir {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch_dir(label: &str) -> std::io::Result<ScratchDir> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "compaheuiler_{label}_{}_{}",
            std::process::id(),
            serial
        ));
        match std::fs::create_dir(&path) {
            Ok(()) => return Ok(ScratchDir(path)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

fn scratch_dir_or_exit(label: &str) -> ScratchDir {
    scratch_dir(label).unwrap_or_else(|e| {
        eprintln!("error: cannot create temporary directory: {e}");
        std::process::exit(1);
    })
}

/// Cargo target directory for generated bigint executables.
///
/// A caller compiling a corpus can set one directory for every generated
/// crate. Cargo then builds the bigint dependency once while each snippet's
/// tiny root crate is still compiled and linked independently. Ordinary CLI
/// use keeps the scratch-local target and leaves no persistent artefacts.
fn generated_target_dir(scratch: &Path) -> PathBuf {
    std::env::var_os("COMPAHEUILER_GENERATED_TARGET_DIR")
        .map(PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                std::env::current_dir().unwrap_or_default().join(path)
            }
        })
        .unwrap_or_else(|| scratch.join("target"))
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
fn remap_path_prefixes(generated_dir: &Path) -> Vec<String> {
    let mut flags = Vec::new();
    // The generated crate. Canonicalized because `/tmp` is a symlink on macOS
    // and rustc records the resolved path, which a `/tmp` prefix never matches.
    let generated = std::fs::canonicalize(generated_dir)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| generated_dir.to_string_lossy().into_owned());
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

    if let Ok(out) = Command::new("rustc").args(["--print", "sysroot"]).output()
        && out.status.success()
        && let Ok(sysroot) = String::from_utf8(out.stdout)
    {
        flags.push(format!(
            "--remap-path-prefix={}=/rust",
            sysroot.trim_end_matches('\n')
        ));
    }
    flags
}

fn compile_rs(code: &str, bin_path: &Path) {
    #[cfg(feature = "bigint")]
    {
        let scratch = scratch_dir_or_exit("rust");
        let dir = scratch.path();
        std::fs::create_dir(dir.join("src")).unwrap();
        std::fs::write(dir.join("src/main.rs"), code).unwrap();

        let bigint_dep = crate::c_bigint_dependency_toml();

        std::fs::write(
            dir.join("Cargo.toml"),
            format!(
                r#"[package]
name = "compaheuiler-output"
version = "0.0.1"
edition = "2021"
rust-version = "1.85"

[dependencies]
{bigint_dep}

[profile.release]
opt-level = 2
lto = "fat"
"#
            ),
        )
        .unwrap();

        let target_dir = generated_target_dir(dir);

        let status = Command::new("cargo")
            .args(["rustc", "--release", "--quiet"])
            .current_dir(dir)
            .env("CARGO_TARGET_DIR", &target_dir)
            .arg("--")
            .args(remap_path_prefixes(dir))
            .status()
            .unwrap_or_else(|e| {
                eprintln!("error: cargo: {e}");
                std::process::exit(1);
            });
        if !status.success() {
            std::process::exit(1);
        }
        std::fs::copy(target_dir.join("release/compaheuiler-output"), bin_path).unwrap();
    }

    #[cfg(not(feature = "bigint"))]
    {
        let scratch = scratch_dir_or_exit("rust");
        let dir = scratch.path();
        let rs_path = dir.join("main.rs");
        std::fs::write(&rs_path, code).unwrap();

        let output = Command::new("rustc")
            .args(["-C", "opt-level=2", "-o"])
            .arg(bin_path)
            .arg(&rs_path)
            .args(remap_path_prefixes(dir))
            .output()
            .unwrap_or_else(|e| {
                eprintln!("error: rustc: {e}");
                std::process::exit(1);
            });
        if !output.status.success() {
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            std::process::exit(1);
        }
    }
}

fn compile_wat(code: &str, bin_path: &Path) {
    let scratch = scratch_dir_or_exit("wat");
    let wat_path = scratch.path().join("program.wat");
    std::fs::write(&wat_path, code).unwrap();

    let wasm_path = bin_path.with_extension("wasm");
    let output = Command::new("wat2wasm")
        .arg(&wat_path)
        .arg("-o")
        .arg(&wasm_path)
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: wat2wasm: {e}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
}

#[cfg(not(feature = "bigint"))]
fn compile_c(code: &str, bin_path: &Path) {
    let scratch = scratch_dir_or_exit("c");
    let c_path = scratch.path().join("program.c");
    std::fs::write(&c_path, code).unwrap();

    let output = Command::new("cc")
        .args(["-O2", "-o"])
        .arg(bin_path)
        .arg(&c_path)
        .arg("-lm")
        .output()
        .unwrap_or_else(|e| {
            eprintln!("error: cc: {e}");
            std::process::exit(1);
        });
    if !output.status.success() {
        eprint!("{}", String::from_utf8_lossy(&output.stderr));
        std::process::exit(1);
    }
}

#[cfg(feature = "bigint")]
fn compile_c_bigint(code: &str, bin_path: &Path) {
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
    let scratch = scratch_dir_or_exit("c_bigint");
    let dir = scratch.path();
    let target_dir = generated_target_dir(dir);
    std::fs::create_dir(dir.join("src")).unwrap();
    let c_path = dir.join("program.c");
    let object_path = dir.join("program.o");
    std::fs::write(&c_path, code).unwrap();
    std::fs::write(dir.join("src/main.rs"), crate::c_bigint_bridge_rs()).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            r#"[package]
name = {package:?}
version = "0.0.1"
edition = "2024"
rust-version = "1.85"

[dependencies]
{bigint_dependency}

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
        .arg("-o")
        .arg(&object_path)
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
        dir.join("build.rs"),
        format!(
            "fn main() {{\n  let object = {object_path:?};\n  println!(\"cargo:rerun-if-changed={{object}}\");\n  println!(\"cargo:rustc-link-arg-bin={package}={{object}}\");\n}}\n",
            object_path = object_path.to_string_lossy(),
        ),
    )
    .unwrap();

    let status = Command::new("cargo")
        .args(["rustc", "--release", "--quiet"])
        .current_dir(dir)
        .env("CARGO_TARGET_DIR", &target_dir)
        .arg("--")
        .args(remap_path_prefixes(dir))
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error: cargo: {e}");
            std::process::exit(1);
        });
    if !status.success() {
        std::process::exit(1);
    }
    std::fs::copy(target_dir.join("release").join(package), bin_path).unwrap();
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
