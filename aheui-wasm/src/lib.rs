//! 아희 인터프리터 / 컴파일러의 wasm-bindgen 라이브러리.
//!
//! 브라우저에서 다음 두 방식으로 아희 코드를 실행할 수 있다:
//! 1. `interpret(source, input)` — 네이티브 인터프리터로 직접 실행
//! 2. `compile_to_wat_web(source) → wat_to_wasm(wat) → WebAssembly.instantiate`
//!
//! wasm32-unknown-unknown 타겟에는 실제 stdin/stdout 이 없으므로
//! `aheui_runtime::io::wasm_buf` 의 thread-local 버퍼를 통해 I/O 를 주고받는다.
//! `interpret` 와 버퍼는 이 타겟에서만 컴파일되고, 타겟과 무관한 컴파일러
//! API 는 호스트 빌드에서도 사용할 수 있다.

use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use aheui_runtime::io::wasm_buf;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use aheuinterpreter::interp;
use ahsembler::OptimizationLevel;

fn parse_opt(level: u8) -> OptimizationLevel {
    match level {
        0 => OptimizationLevel::O0,
        1 => OptimizationLevel::O1,
        2 => OptimizationLevel::O2,
        _ => OptimizationLevel::O3,
    }
}

/// 아희 소스를 네이티브 인터프리터로 실행한다.
///
/// `input` 은 stdin 으로 주입될 바이트열 (UTF-8),
/// 반환값은 stdout 에 쓰인 바이트열을 UTF-8 문자열로 해석한 것.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[wasm_bindgen]
pub fn interpret(source: &str, input: &str) -> String {
    wasm_buf::set_input(input.as_bytes());
    let program = ahsembler::compile(source, OptimizationLevel::O2);
    let _exit = interp::mainloop(&program);
    let out = wasm_buf::take_output();
    String::from_utf8_lossy(&out).into_owned()
}

/// Web 브라우저용 WAT (env.write_byte / env.read_byte import) 을 생성한다.
#[wasm_bindgen]
pub fn compile_to_wat_web(source: &str) -> String {
    compaheuiler::compile_to_wat(source)
}

/// WASI 용 WAT (fd_write / fd_read) 을 생성한다.
#[wasm_bindgen]
pub fn compile_to_wat_wasi(source: &str) -> String {
    compaheuiler::compile_to_wat_wasi(source)
}

/// C 소스 코드를 생성한다.
#[wasm_bindgen]
pub fn compile_to_c(source: &str) -> String {
    compaheuiler::compile_to_c(source)
}

/// Rust 소스 코드를 생성한다 (smallint 모드).
#[wasm_bindgen]
pub fn compile_to_rust(source: &str, opt_level: u8) -> String {
    compaheuiler::compile_to_rs_opt(source, parse_opt(opt_level))
}

/// Rust 소스 코드를 생성한다 (BigInt 모드).
#[wasm_bindgen]
pub fn compile_to_rust_bigint(source: &str, opt_level: u8) -> String {
    compaheuiler::compile_to_rs_bigint_opt(source, parse_opt(opt_level))
}

/// 아희 소스를 ahsembly (주석이 붙은 어셈블리) 로 출력한다.
#[wasm_bindgen]
pub fn compile_to_asm(source: &str, opt_level: u8) -> String {
    let mut out: Vec<u8> = Vec::new();
    ahsembler::assemble(source, parse_opt(opt_level), &mut out)
        .expect("writing to Vec cannot fail");
    String::from_utf8_lossy(&out).into_owned()
}

/// WAT 텍스트를 wasm 바이너리로 변환한다. 실패하면 에러 문자열을 JsValue 로 던진다.
#[wasm_bindgen]
pub fn wat_to_wasm(wat: &str) -> Result<Vec<u8>, JsValue> {
    wat::parse_str(wat).map_err(|e| JsValue::from_str(&format!("{e}")))
}

/// 소스 → WAT → wasm 까지 한 번에 수행한다 (web 백엔드).
#[wasm_bindgen]
pub fn compile_to_wasm_web(source: &str) -> Result<Vec<u8>, JsValue> {
    let wat = compaheuiler::compile_to_wat(source);
    wat::parse_str(&wat).map_err(|e| JsValue::from_str(&format!("{e}")))
}
