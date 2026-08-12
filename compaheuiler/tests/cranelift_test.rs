#![cfg(feature = "cranelift")]

mod common;

use compaheuiler::jit::SpecialStorage;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Instant;

static STDOUT_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());
static RUN_LOCK: Mutex<()> = Mutex::new(());
static READ_NUM: AtomicI64 = AtomicI64::new(0);

extern "C" fn write_char(v: i64) {
    let c = v as u32;
    let mut buf = STDOUT_BUF.lock().unwrap_or_else(|e| e.into_inner());
    if c <= 0x7f {
        buf.push(c as u8);
    } else if let Some(ch) = char::from_u32(c) {
        let mut tmp = [0u8; 4];
        buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
    }
}
extern "C" fn write_bytes(data: *const u8, len: usize) {
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    STDOUT_BUF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend_from_slice(bytes);
}
extern "C" fn write_num(v: i64) {
    STDOUT_BUF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .extend_from_slice(format!("{}", v).as_bytes());
}
extern "C" fn read_char() -> i64 {
    -1
}
extern "C" fn read_num() -> i64 {
    READ_NUM.swap(0, Ordering::Relaxed)
}

fn run_cranelift(source: &str) -> (String, i64, f64) {
    run_cranelift_with_num(source, 0)
}

fn run_cranelift_with_num(source: &str, input: i64) -> (String, i64, f64) {
    let _run = RUN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    READ_NUM.store(input, Ordering::Relaxed);
    let t = Instant::now();
    // The same pipeline `aheui build --codegen cranelift` runs, so this test
    // measures the CFG the CLI actually hands the backend.
    let cfg = compaheuiler::pipeline::optimize(source, ahsembler::OptimizationLevel::O3);

    let cfg_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let jit = compaheuiler::jit::compile_cfg(&cfg).expect("Cranelift compilation failed");
    let cl_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let mut data: Vec<Vec<i64>> = (0..28).map(|_| vec![0i64; 65536]).collect();
    let mut bases: [*mut i64; 28] = {
        let mut b = [std::ptr::null_mut(); 28];
        for i in 0..28 {
            b[i] = data[i].as_mut_ptr();
        }
        b
    };
    let mut lengths = [0i32; 28];
    let mut sp = SpecialStorage::new();
    let sp_ctx = &mut sp as *mut SpecialStorage as *mut u8;
    STDOUT_BUF.lock().unwrap_or_else(|e| e.into_inner()).clear();
    let t3 = Instant::now();
    let exit = unsafe {
        jit.execute_buffered(
            &mut bases,
            &mut lengths,
            write_char,
            write_bytes,
            write_num,
            read_char,
            read_num,
            sp_ctx,
            compaheuiler::jit::sp_push,
            compaheuiler::jit::sp_pop,
            compaheuiler::jit::sp_depth,
            compaheuiler::jit::sp_dup,
            compaheuiler::jit::sp_swap,
        )
    };
    let run_ms = t3.elapsed().as_secs_f64() * 1000.0;
    let out =
        String::from_utf8_lossy(&STDOUT_BUF.lock().unwrap_or_else(|e| e.into_inner())).to_string();
    let total_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "  breakdown: cfg={cfg_ms:.1}ms cranelift={cl_ms:.1}ms run={run_ms:.1}ms total={total_ms:.1}ms"
    );
    (out, exit, total_ms)
}

#[test]
fn test_add_cranelift() {
    let (out, exit, compile_ms) = run_cranelift("반받다망희");
    eprintln!("cranelift add: {compile_ms:.1}ms exit={exit} out={out:?}");
    assert_eq!(out, "5");
}

#[cfg(feature = "bigint")]
#[test]
fn test_2e65_cranelift_bigint() {
    let source = common::require_snippet("integer/2e65-print.aheui");
    let expected = common::require_snippet("integer/2e65-print.out");
    let (out, exit, compile_ms) = run_cranelift(&source);
    eprintln!("cranelift 2e65 bigint: {compile_ms:.1}ms exit={exit} out={out:?}");
    assert_eq!(out.trim_end(), expected.trim_end());
}

#[cfg(feature = "bigint")]
#[test]
fn test_post_promotion_ops_cranelift_bigint() {
    let (out, exit, _) = run_cranelift(common::DUAL_MODE_POST_PROMOTION_OPS);
    assert_eq!(exit, 0);
    assert_eq!(out, common::DUAL_MODE_POST_PROMOTION_OUTPUT);
}

#[cfg(feature = "bigint")]
#[test]
fn test_read_num_boxes_large_i64_after_cranelift_promotion() {
    let source = "반빠따빠따빠따빠따빠따빠따마방망하";
    let (out, exit, _) = run_cranelift_with_num(source, 5_000_000_000_000_000_000);
    assert_eq!(out, "5000000000000000000");
    assert_eq!(exit, 0);
}

#[cfg(feature = "bigint")]
#[test]
fn test_loop_carried_storage_survives_cranelift_promotion() {
    let source = common::require_snippet("factorial/factorial.aheui");
    let (out, exit, _) = run_cranelift_with_num(&source, 21);
    assert_eq!(exit, 0);
    assert_eq!(out, "51090942171709440000");
}

#[test]
fn test_hello_cranelift() {
    let source = common::require_snippet("hello-world/hello.puzzlet.aheui");
    let (out, _exit, compile_ms) = run_cranelift(&source);
    eprintln!(
        "cranelift hello: {compile_ms:.1}ms out={:?}",
        &out[..out.len().min(30)]
    );
    assert!(
        out.contains("안녕하세요"),
        "expected 안녕하세요, got {:?}",
        &out[..out.len().min(30)]
    );
}

#[test]
fn test_logo_cranelift() {
    let source = common::require_snippet("logo/logo.aheui");
    let (out, _exit, compile_ms) = run_cranelift(&source);
    eprintln!(
        "cranelift logo: total={compile_ms:.1}ms output={}bytes",
        out.len()
    );
    assert!(out.starts_with("P1 615 810"));
    assert_eq!(out.len(), 996310);
}

#[test]
fn test_queue_cranelift() {
    // standard/queue.aheui: tests queue (sel=21) operations
    let source = common::require_snippet("standard/queue.aheui");
    let (out, _exit, compile_ms) = run_cranelift(&source);
    eprintln!("cranelift queue: {compile_ms:.1}ms out={out:?}");
    assert_eq!(out, "235223");
}

#[test]
fn test_99bottles_cranelift() {
    let source = common::require_snippet("99bottles/99bottles.aheui");
    let (out, exit, compile_ms) = run_cranelift(&source);
    eprintln!(
        "cranelift 99bottles: {compile_ms:.1}ms exit={exit} out={}bytes",
        out.len()
    );
    assert_eq!(out.len(), 11782);
}

#[test]
#[ignore] // aheui.aheui has 407 blocks — Cranelift compilation is slow
fn test_aheui_interp_cranelift() {
    let Some(source) = common::read_self_interp() else {
        common::skip(
            "test_aheui_interp_cranelift",
            "aheui.aheui (set AHEUI_SELF_INTERP)",
        );
        return;
    };
    let (_out, exit, _) = run_cranelift(&source);
    eprintln!("cranelift aheui(밝희): exit={exit}");
    assert_eq!(exit, 7);
}
