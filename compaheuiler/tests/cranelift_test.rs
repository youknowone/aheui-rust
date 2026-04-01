#![cfg(feature = "cranelift")]

use compaheuiler::jit::SpecialStorage;
use std::sync::Mutex;
use std::time::Instant;

static STDOUT_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());

extern "C" fn write_char(v: i64) {
    let c = v as u32;
    let mut buf = STDOUT_BUF.lock().unwrap();
    if c <= 0x7f {
        buf.push(c as u8);
    } else if let Some(ch) = char::from_u32(c) {
        let mut tmp = [0u8; 4];
        buf.extend_from_slice(ch.encode_utf8(&mut tmp).as_bytes());
    }
}
extern "C" fn write_num(v: i64) {
    STDOUT_BUF
        .lock()
        .unwrap()
        .extend_from_slice(format!("{}", v).as_bytes());
}
extern "C" fn read_char() -> i64 {
    -1
}
extern "C" fn read_num() -> i64 {
    0
}

fn run_cranelift(source: &str) -> (String, i64, f64) {
    let t = Instant::now();
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
        if cfg.num_blocks() == before {
            break;
        }
    }
    let removed = ahsembler::cfg_optimize::eliminate_loop_guards(&mut cfg);
    if removed > 0 {
        for _ in 0..10 {
            let before = cfg.num_blocks();
            let states = ahsembler::cfg_optimize::analyze_stack_depths(&cfg);
            ahsembler::cfg_optimize::eliminate_guard_depth(&mut cfg, &states);
            ahsembler::cfg_optimize::thread_jumps(&mut cfg);
            ahsembler::cfg_optimize::merge_blocks(&mut cfg);
            ahsembler::cfg_optimize::eliminate_dead_blocks(&mut cfg);
            if cfg.num_blocks() == before {
                break;
            }
        }
        let states = ahsembler::cfg_optimize::analyze_stack_depths(&cfg);
        ahsembler::cfg_optimize::fold_constants(&mut cfg, &states);
        ahsembler::cfg_optimize::peephole_cleanup(&mut cfg, &states);
    }
    let chain_removed = ahsembler::cfg_optimize::eliminate_chain_guards(&mut cfg);
    if chain_removed > 0 {
        for _ in 0..10 {
            let before = cfg.num_blocks();
            ahsembler::cfg_optimize::thread_jumps(&mut cfg);
            ahsembler::cfg_optimize::merge_blocks(&mut cfg);
            ahsembler::cfg_optimize::eliminate_dead_blocks(&mut cfg);
            if cfg.num_blocks() == before {
                break;
            }
        }
    }

    let cfg_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t2 = Instant::now();
    let jit = compaheuiler::jit::compile_cfg(&cfg).expect("Cranelift compilation failed");
    let cl_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let compile_ms = t.elapsed().as_secs_f64() * 1000.0;

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
    STDOUT_BUF.lock().unwrap().clear();
    let t3 = Instant::now();
    let exit = unsafe {
        jit.execute(
            &mut bases,
            &mut lengths,
            write_char,
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
    let out = String::from_utf8_lossy(&STDOUT_BUF.lock().unwrap()).to_string();
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

#[test]
fn test_hello_cranelift() {
    let source = std::fs::read_to_string(
        "snippets/hello-world/hello.puzzlet.aheui",
    )
    .unwrap();
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
    let source =
        std::fs::read_to_string("snippets/logo/logo.aheui")
            .unwrap();
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
    let source = std::fs::read_to_string(
        "snippets/standard/queue.aheui",
    )
    .unwrap();
    let (out, _exit, compile_ms) = run_cranelift(&source);
    eprintln!("cranelift queue: {compile_ms:.1}ms out={out:?}");
    assert_eq!(out, "235223");
}

#[test]
fn test_99bottles_cranelift() {
    let source = std::fs::read_to_string(
        "snippets/99bottles/99bottles.aheui",
    )
    .unwrap();
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
    let source = std::fs::read_to_string(
        "tests/aheui.aheui",
    )
    .unwrap();
    let (out, exit, _) = run_cranelift(&source);
    eprintln!("cranelift aheui(밝희): exit={exit}");
    assert_eq!(exit, 7);
}
