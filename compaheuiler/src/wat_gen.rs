//! Generate WAT (WebAssembly Text format) code from an Aheui CFG.
//!
//! Produces a standalone WASI module. Compile with:
//!   wat2wasm output.wat -o output.wasm
//!   wasmtime output.wasm

use ahsembler::cfg::*;
use std::collections::{BTreeSet, HashMap};

const QUEUE: usize = 21;
const PORT: usize = 27;

/// Memory layout constants (byte offsets).
const IO_BUF: i32 = 0x0000; // 256 bytes I/O buffer
const IOVEC: i32 = 0x0100; // 8 bytes: ptr(i32) + len(i32)
const NWRITTEN: i32 = 0x0108; // 4 bytes: nwritten result
const DIGIT_SCRATCH: i32 = 0x0110; // 24 bytes: digit conversion scratch area
const TOPS: i32 = 0x0200; // 28 * 4 = 112 bytes: tops array
const Q_HEAD: i32 = 0x0300; // 4 bytes
const Q_LEN: i32 = 0x0304; // 4 bytes
const P_LEN: i32 = 0x0308; // 4 bytes
const _PAD: i32 = 0x030C; // 4 bytes padding
const PORT_LAST: i32 = 0x0310; // 8 bytes
const STORAGE_BASE: i32 = 0x0400;
const STORAGE_SIZE: i32 = 524288; // 65536 entries * 8 bytes
const QUEUE_CAP: i32 = 65536;

fn tops_offset(s: usize) -> i32 {
    TOPS + (s as i32) * 4
}

fn storage_base(s: usize) -> i32 {
    STORAGE_BASE + (s as i32) * STORAGE_SIZE
}

/// Generate WAT with env.write_byte/read_byte imports (pure module, no WASI).
pub fn compile_to_wat(source: &str) -> String {
    compile_to_wat_opt(source, ahsembler::OptimizationLevel::O3)
}

pub fn compile_to_wat_opt(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    generate_wat_dispatch(&crate::pipeline::optimize(source, opt), false)
}

/// Generate WAT with WASI glue: self-contained _start entry using fd_write/fd_read.
pub fn compile_to_wat_wasi(source: &str) -> String {
    compile_to_wat_wasi_opt(source, ahsembler::OptimizationLevel::O3)
}

pub fn compile_to_wat_wasi_opt(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    generate_wat_dispatch(&crate::pipeline::optimize(source, opt), true)
}

fn generate_wat_dispatch(cfg: &Cfg, wasi: bool) -> String {
    let mut out = String::with_capacity(65536);

    let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);

    // Collect used storages
    let mut used: BTreeSet<usize> = BTreeSet::new();
    used.insert(0);
    for b in &cfg.blocks {
        for i in &b.instructions {
            match i {
                Inst::Sel(s) | Inst::Mov(s) => {
                    used.insert(*s);
                }
                _ => {}
            }
        }
    }
    let has_queue = used.contains(&QUEUE);
    let has_port = used.contains(&PORT);

    // Collect live blocks
    let mut live_blocks: Vec<BlockId> = Vec::new();
    for block_id in 0..cfg.num_blocks() as BlockId {
        let entry_state = states.get(block_id as usize);
        if entry_state.map_or(true, |s| s.is_bottom()) {
            continue;
        }
        live_blocks.push(block_id);
    }
    let block_order: HashMap<BlockId, usize> = live_blocks
        .iter()
        .enumerate()
        .map(|(i, &b)| (b, i))
        .collect();

    let n = live_blocks.len();

    // Check for I/O, dynamic sel, special storage usage
    let has_io = live_blocks.iter().any(|&bid| {
        let block = cfg.block(bid);
        block.instructions.iter().any(|i| {
            matches!(
                i,
                Inst::PopNum | Inst::PopChar | Inst::PushNum | Inst::PushChar
            )
        })
    });
    let has_dyn_sel = live_blocks.iter().any(|&bid| {
        states
            .get(bid as usize)
            .map_or(false, |s| s.selected.is_none())
    });

    // Module header
    out.push_str(WAT_HEADER);
    out.push_str("(module\n");

    // Imports (must come before memory/func definitions)
    if wasi {
        out.push_str("  (import \"wasi_snapshot_preview1\" \"fd_write\"\n");
        out.push_str("    (func $fd_write (param i32 i32 i32 i32) (result i32)))\n");
        out.push_str("  (import \"wasi_snapshot_preview1\" \"fd_read\"\n");
        out.push_str("    (func $fd_read (param i32 i32 i32 i32) (result i32)))\n");
        out.push_str("  (import \"wasi_snapshot_preview1\" \"proc_exit\"\n");
        out.push_str("    (func $proc_exit (param i32)))\n\n");
    } else {
        out.push_str("  (import \"env\" \"write_byte\" (func $write_byte_import (param i32)))\n");
        out.push_str("  (import \"env\" \"read_byte\" (func $read_byte_import (result i32)))\n\n");
    }

    // Memory: 224 pages = ~14.6MB
    out.push_str("  (memory (export \"memory\") 224)\n\n");

    // Helper functions
    emit_helpers(&mut out, has_queue, has_port, has_io, wasi);

    // Main function
    if wasi {
        out.push_str("  (func $main (export \"_start\")\n");
    } else {
        out.push_str("  (func $main (export \"run\") (result i32)\n");
    }
    out.push_str("    (local $pc i32)\n");
    out.push_str("    (local $sel i32)\n");
    out.push_str("    (local $v0 i64) (local $v1 i64) (local $v2 i64)\n");
    out.push_str("    (local $addr i32) (local $tmp i32) (local $exit_code i32)\n\n");

    // Initialize tops array: each storage top = base address of that storage
    out.push_str("    ;; Initialize tops\n");
    for &s in &used {
        let base = storage_base(s);
        let offset = tops_offset(s);
        out.push_str(&format!(
            "    (i32.store (i32.const {offset}) (i32.const {base}))\n"
        ));
    }
    // Initialize queue/port metadata
    if has_queue {
        out.push_str(&format!(
            "    (i32.store (i32.const {Q_HEAD}) (i32.const 0))\n"
        ));
        out.push_str(&format!(
            "    (i32.store (i32.const {Q_LEN}) (i32.const 0))\n"
        ));
    }
    if has_port {
        out.push_str(&format!(
            "    (i32.store (i32.const {P_LEN}) (i32.const 0))\n"
        ));
        out.push_str(&format!(
            "    (i64.store (i32.const {PORT_LAST}) (i64.const 0))\n"
        ));
    }

    // Set initial pc
    let entry_seq = block_order.get(&cfg.entry).copied().unwrap_or(0);
    out.push_str(&format!("    (local.set $pc (i32.const {entry_seq}))\n"));
    out.push_str("    (local.set $sel (i32.const 0))\n\n");

    // Dispatch loop using br_table
    //   (block $exit
    //     (loop $dispatch
    //       (block $bN ... (block $b0
    //         (br_table $b0 $b1 ... $bN $exit (local.get $pc))
    //       ) ;; $b0 body
    //       ) ;; $b1 body
    //       ...
    //     )
    //   )
    out.push_str("    (block $exit\n");
    out.push_str("      (loop $dispatch\n");

    // Open all nested blocks
    for i in (0..n).rev() {
        out.push_str(&format!("        (block $b{i}\n"));
    }

    // br_table
    out.push_str("          (br_table ");
    for i in 0..n {
        out.push_str(&format!("$b{i} "));
    }
    out.push_str("$exit (local.get $pc))\n");

    // Close innermost block and emit code for each block
    for (seq_idx, &block_id) in live_blocks.iter().enumerate() {
        out.push_str(&format!("        ) ;; $b{seq_idx}\n"));
        out.push_str(&format!("        ;; block {block_id}\n"));

        let block = cfg.block(block_id);
        let entry_state = states.get(block_id as usize);
        let entry_sel: Option<usize> = entry_state.and_then(|s| s.selected);
        let mut sel = entry_sel;

        // Instructions
        for inst in &block.instructions {
            emit_instruction(&mut out, inst, &mut sel, has_dyn_sel, &used, &block_order);
        }

        // Terminator
        emit_terminator(
            &mut out,
            &block.terminator,
            sel,
            has_queue || has_port,
            &used,
            &block_order,
            wasi,
        );
    }

    // Close loop and exit block
    out.push_str("      ) ;; loop $dispatch\n");
    out.push_str("    ) ;; block $exit\n");
    if wasi {
        out.push_str("    (call $proc_exit (local.get $exit_code))\n");
    } else {
        out.push_str("    (local.get $exit_code)\n");
    }
    out.push_str("  ) ;; func $main\n\n");

    out.push_str(")\n");
    out
}

/// Emit WAT for a single instruction.
fn emit_instruction(
    out: &mut String,
    inst: &Inst,
    sel: &mut Option<usize>,
    _has_dyn_sel: bool,
    _used: &BTreeSet<usize>,
    block_order: &HashMap<BlockId, usize>,
) {
    let ind = "        ";
    let is_special = |s: usize| s == QUEUE || s == PORT;
    let sp_known = sel.is_some_and(|s| is_special(s));
    let ds = sel.is_none();

    match inst {
        Inst::Push(v) => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(call $sp_push (i32.const {s}) (i64.const {v}))\n"
                ));
            } else if ds {
                // Dynamic: check if special
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (call $sp_push (local.get $sel) (i64.const {v})))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_push_dyn(out, &format!("{ind}    "), &format!("(i64.const {v})"));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_push(out, &format!("{ind}"), s, &format!("(i64.const {v})"));
            }
        }
        Inst::Pop => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!("{ind}(drop (call $sp_pop (i32.const {s})))\n"));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (drop (call $sp_pop (local.get $sel))))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_pop_dyn_drop(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_pop_drop(out, &format!("{ind}"), s);
            }
        }
        Inst::Dup => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!("{ind}(call $sp_dup (i32.const {s}))\n"));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!("{ind}  (then (call $sp_dup (local.get $sel)))\n"));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_dup_dyn(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_dup(out, &format!("{ind}"), s);
            }
        }
        Inst::Swap => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!("{ind}(call $sp_swap (i32.const {s}))\n"));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!("{ind}  (then (call $sp_swap (local.get $sel)))\n"));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_swap_dyn(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_swap(out, &format!("{ind}"), s);
            }
        }
        Inst::BinOp(kind) => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(local.set $v0 (call $sp_pop (i32.const {s})))\n"
                ));
                out.push_str(&format!(
                    "{ind}(local.set $v1 (call $sp_pop (i32.const {s})))\n"
                ));
                let expr = binop_expr(kind, "(local.get $v1)", "(local.get $v0)");
                out.push_str(&format!("{ind}(call $sp_push (i32.const {s}) {expr})\n"));
            } else if ds {
                // Dynamic storage binop
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!("{ind}  (then\n"));
                out.push_str(&format!(
                    "{ind}    (local.set $v0 (call $sp_pop (local.get $sel)))\n"
                ));
                out.push_str(&format!(
                    "{ind}    (local.set $v1 (call $sp_pop (local.get $sel)))\n"
                ));
                let expr = binop_expr(kind, "(local.get $v1)", "(local.get $v0)");
                out.push_str(&format!(
                    "{ind}    (call $sp_push (local.get $sel) {expr})\n"
                ));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_binop_dyn(out, &format!("{ind}    "), kind);
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_binop(out, &format!("{ind}"), s, kind);
            }
        }
        Inst::Sel(new_sel) => {
            out.push_str(&format!("{ind}(local.set $sel (i32.const {new_sel}))\n"));
            *sel = Some(*new_sel);
        }
        Inst::Mov(target) => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(local.set $v0 (call $sp_pop (i32.const {s})))\n"
                ));
                if is_special(*target) {
                    out.push_str(&format!(
                        "{ind}(call $sp_push (i32.const {target}) (local.get $v0))\n"
                    ));
                } else {
                    emit_stack_push(out, &format!("{ind}"), *target, "(local.get $v0)");
                }
            } else if ds {
                // Pop from dynamic source
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (local.set $v0 (call $sp_pop (local.get $sel))))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_pop_dyn_to_v0(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
                // Push to target
                if is_special(*target) {
                    out.push_str(&format!(
                        "{ind}(call $sp_push (i32.const {target}) (local.get $v0))\n"
                    ));
                } else {
                    emit_stack_push(out, &format!("{ind}"), *target, "(local.get $v0)");
                }
            } else {
                let s = sel.unwrap();
                emit_stack_pop_to_v0(out, &format!("{ind}"), s);
                if is_special(*target) {
                    out.push_str(&format!(
                        "{ind}(call $sp_push (i32.const {target}) (local.get $v0))\n"
                    ));
                } else {
                    emit_stack_push(out, &format!("{ind}"), *target, "(local.get $v0)");
                }
            }
        }
        Inst::PopNum => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(call $write_num (call $sp_pop (i32.const {s})))\n"
                ));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (call $write_num (call $sp_pop (local.get $sel))))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_pop_dyn_to_v0(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}    (call $write_num (local.get $v0))\n"));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_pop_to_v0(out, &format!("{ind}"), s);
                out.push_str(&format!("{ind}(call $write_num (local.get $v0))\n"));
            }
        }
        Inst::PopChar => {
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(call $write_char (call $sp_pop (i32.const {s})))\n"
                ));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (call $write_char (call $sp_pop (local.get $sel))))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_pop_dyn_to_v0(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}    (call $write_char (local.get $v0))\n"));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_pop_to_v0(out, &format!("{ind}"), s);
                out.push_str(&format!("{ind}(call $write_char (local.get $v0))\n"));
            }
        }
        Inst::PushNum => {
            out.push_str(&format!("{ind}(call $flush_out)\n"));
            out.push_str(&format!("{ind}(local.set $v0 (call $read_num))\n"));
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(call $sp_push (i32.const {s}) (local.get $v0))\n"
                ));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (call $sp_push (local.get $sel) (local.get $v0)))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_push_dyn(out, &format!("{ind}    "), "(local.get $v0)");
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_push(out, &format!("{ind}"), s, "(local.get $v0)");
            }
        }
        Inst::PushChar => {
            out.push_str(&format!("{ind}(call $flush_out)\n"));
            out.push_str(&format!("{ind}(local.set $v0 (call $read_char))\n"));
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(call $sp_push (i32.const {s}) (local.get $v0))\n"
                ));
            } else if ds {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (call $sp_push (local.get $sel) (local.get $v0)))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_push_dyn(out, &format!("{ind}    "), "(local.get $v0)");
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_push(out, &format!("{ind}"), s, "(local.get $v0)");
            }
        }
        Inst::GuardDepth { min_depth, fail } => {
            let fail_seq = block_order.get(fail).copied().unwrap_or(0);
            if sp_known {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(if (i32.lt_s (call $sp_depth (i32.const {s})) (i32.const {min_depth}))\n"
                ));
            } else if ds {
                // Dynamic: check special vs regular
                out.push_str(&format!("{ind}(if (i32.lt_s (call $depth_dyn (local.get $sel)) (i32.const {min_depth}))\n"));
            } else {
                let s = sel.unwrap();
                let toff = tops_offset(s);
                let base = storage_base(s);
                // depth = (tops[s] - base) / 8
                out.push_str(&format!(
                    "{ind}(if (i32.lt_s (i32.shr_u (i32.sub (i32.load (i32.const {toff})) (i32.const {base})) (i32.const 3)) (i32.const {min_depth}))\n"
                ));
            }
            out.push_str(&format!(
                "{ind}  (then (local.set $pc (i32.const {fail_seq})) (br $dispatch))\n"
            ));
            out.push_str(&format!("{ind})\n"));
        }
    }
}

/// Emit WAT for a terminator.
fn emit_terminator(
    out: &mut String,
    term: &Terminator,
    sel: Option<usize>,
    has_special: bool,
    _used: &BTreeSet<usize>,
    block_order: &HashMap<BlockId, usize>,
    _wasi: bool,
) {
    let ind = "        ";
    let is_special = |s: usize| s == QUEUE || s == PORT;

    match term {
        Terminator::Goto(target) => {
            let target_seq = block_order.get(target).copied().unwrap_or(0);
            out.push_str(&format!("{ind}(local.set $pc (i32.const {target_seq}))\n"));
            out.push_str(&format!("{ind}(br $dispatch)\n"));
        }
        Terminator::StackGuard {
            min_depth,
            ok,
            fail,
        } => {
            let ok_seq = block_order.get(ok).copied().unwrap_or(0);
            let fail_seq = block_order.get(fail).copied().unwrap_or(0);

            // Compute depth
            if sel.is_some_and(|s| is_special(s)) {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(if (i32.ge_s (call $sp_depth (i32.const {s})) (i32.const {min_depth}))\n"
                ));
            } else if sel.is_none() && has_special {
                out.push_str(&format!(
                    "{ind}(if (i32.ge_s (call $depth_dyn (local.get $sel)) (i32.const {min_depth}))\n"
                ));
            } else {
                let s = sel.unwrap_or(0);
                if sel.is_some() {
                    let toff = tops_offset(s);
                    let base = storage_base(s);
                    out.push_str(&format!(
                        "{ind}(if (i32.ge_s (i32.shr_u (i32.sub (i32.load (i32.const {toff})) (i32.const {base})) (i32.const 3)) (i32.const {min_depth}))\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "{ind}(if (i32.ge_s (call $depth_dyn (local.get $sel)) (i32.const {min_depth}))\n"
                    ));
                }
            }
            out.push_str(&format!(
                "{ind}  (then (local.set $pc (i32.const {ok_seq})) (br $dispatch))\n"
            ));
            out.push_str(&format!(
                "{ind}  (else (local.set $pc (i32.const {fail_seq})) (br $dispatch))\n"
            ));
            out.push_str(&format!("{ind})\n"));
        }
        Terminator::BranchZero {
            on_zero,
            on_nonzero,
        } => {
            let zero_seq = block_order.get(on_zero).copied().unwrap_or(0);
            let nonzero_seq = block_order.get(on_nonzero).copied().unwrap_or(0);

            // Pop value
            if sel.is_some_and(|s| is_special(s)) {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(local.set $v0 (call $sp_pop (i32.const {s})))\n"
                ));
            } else if sel.is_none() {
                out.push_str(&format!("{ind}(if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!(
                    "{ind}  (then (local.set $v0 (call $sp_pop (local.get $sel))))\n"
                ));
                out.push_str(&format!("{ind}  (else\n"));
                emit_stack_pop_dyn_to_v0(out, &format!("{ind}    "));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                emit_stack_pop_to_v0(out, &format!("{ind}"), s);
            }

            // Branch
            out.push_str(&format!("{ind}(if (i64.eqz (local.get $v0))\n"));
            out.push_str(&format!(
                "{ind}  (then (local.set $pc (i32.const {zero_seq})))\n"
            ));
            out.push_str(&format!(
                "{ind}  (else (local.set $pc (i32.const {nonzero_seq})))\n"
            ));
            out.push_str(&format!("{ind})\n"));
            out.push_str(&format!("{ind}(br $dispatch)\n"));
        }
        Terminator::Halt => {
            out.push_str(&format!("{ind}(call $flush_out)\n"));
            // Get exit code from current storage top (or 0 if empty)
            if sel.is_some_and(|s| is_special(s)) {
                let s = sel.unwrap();
                out.push_str(&format!(
                    "{ind}(if (i32.gt_s (call $sp_depth (i32.const {s})) (i32.const 0))\n"
                ));
                out.push_str(&format!(
                    "{ind}  (then (local.set $exit_code (i32.wrap_i64 (call $sp_pop (i32.const {s})))))\n"
                ));
                out.push_str(&format!(
                    "{ind}  (else (local.set $exit_code (i32.const 0)))\n"
                ));
                out.push_str(&format!("{ind})\n"));
            } else if sel.is_none() {
                // Dynamic
                out.push_str(&format!(
                    "{ind}(if (i32.gt_s (call $depth_dyn (local.get $sel)) (i32.const 0))\n"
                ));
                out.push_str(&format!("{ind}  (then\n"));
                out.push_str(&format!("{ind}    (if (i32.or (i32.eq (local.get $sel) (i32.const {QUEUE})) (i32.eq (local.get $sel) (i32.const {PORT})))\n"));
                out.push_str(&format!("{ind}      (then (local.set $exit_code (i32.wrap_i64 (call $sp_pop (local.get $sel)))))\n"));
                out.push_str(&format!("{ind}      (else\n"));
                // peek top of regular storage
                out.push_str(&format!("{ind}        (local.set $exit_code (i32.wrap_i64 (i64.load (i32.sub (call $tops_get_dyn (local.get $sel)) (i32.const 8)))))\n"));
                out.push_str(&format!("{ind}      )\n"));
                out.push_str(&format!("{ind}    )\n"));
                out.push_str(&format!("{ind}  )\n"));
                out.push_str(&format!(
                    "{ind}  (else (local.set $exit_code (i32.const 0)))\n"
                ));
                out.push_str(&format!("{ind})\n"));
            } else {
                let s = sel.unwrap();
                let toff = tops_offset(s);
                let base = storage_base(s);
                out.push_str(&format!(
                    "{ind}(if (i32.gt_s (i32.load (i32.const {toff})) (i32.const {base}))\n"
                ));
                out.push_str(&format!(
                    "{ind}  (then (local.set $exit_code (i32.wrap_i64 (i64.load (i32.sub (i32.load (i32.const {toff})) (i32.const 8))))))\n"
                ));
                out.push_str(&format!(
                    "{ind}  (else (local.set $exit_code (i32.const 0)))\n"
                ));
                out.push_str(&format!("{ind})\n"));
            }
            out.push_str(&format!("{ind}(br $exit)\n"));
        }
    }
}

// --- Stack operation helpers for known regular storage ---

/// Push a value onto storage `s`.
fn emit_stack_push(out: &mut String, ind: &str, s: usize, val_expr: &str) {
    let toff = tops_offset(s);
    // addr = tops[s]; store val at addr; tops[s] += 8
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.load (i32.const {toff})))\n"
    ));
    out.push_str(&format!("{ind}(i64.store (local.get $addr) {val_expr})\n"));
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.add (local.get $addr) (i32.const 8)))\n"
    ));
}

/// Push onto dynamic (non-special) storage via $sel.
fn emit_stack_push_dyn(out: &mut String, ind: &str, val_expr: &str) {
    // addr = tops[sel]; store val; tops[sel] += 8
    out.push_str(&format!(
        "{ind}(local.set $addr (call $tops_get_dyn (local.get $sel)))\n"
    ));
    out.push_str(&format!("{ind}(i64.store (local.get $addr) {val_expr})\n"));
    out.push_str(&format!(
        "{ind}(call $tops_set_dyn (local.get $sel) (i32.add (local.get $addr) (i32.const 8)))\n"
    ));
}

/// Pop and discard from storage `s`.
fn emit_stack_pop_drop(out: &mut String, ind: &str, s: usize) {
    let toff = tops_offset(s);
    // tops[s] -= 8
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.sub (i32.load (i32.const {toff})) (i32.const 8)))\n"
    ));
}

/// Pop and discard from dynamic storage.
fn emit_stack_pop_dyn_drop(out: &mut String, ind: &str) {
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.sub (call $tops_get_dyn (local.get $sel)) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(call $tops_set_dyn (local.get $sel) (local.get $addr))\n"
    ));
}

/// Pop from storage `s` into $v0.
fn emit_stack_pop_to_v0(out: &mut String, ind: &str, s: usize) {
    let toff = tops_offset(s);
    // tops[s] -= 8; v0 = load(tops[s])
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.sub (i32.load (i32.const {toff})) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.load (i32.const {toff}))))\n"
    ));
}

/// Pop from dynamic storage into $v0.
fn emit_stack_pop_dyn_to_v0(out: &mut String, ind: &str) {
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.sub (call $tops_get_dyn (local.get $sel)) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(call $tops_set_dyn (local.get $sel) (local.get $addr))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (local.get $addr)))\n"
    ));
}

/// Dup on storage `s`.
fn emit_stack_dup(out: &mut String, ind: &str, s: usize) {
    let toff = tops_offset(s);
    // v0 = load(tops[s] - 8); store v0 at tops[s]; tops[s] += 8
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.sub (i32.load (i32.const {toff})) (i32.const 8))))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.load (i32.const {toff})))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (local.get $addr) (local.get $v0))\n"
    ));
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.add (local.get $addr) (i32.const 8)))\n"
    ));
}

/// Dup on dynamic storage.
fn emit_stack_dup_dyn(out: &mut String, ind: &str) {
    out.push_str(&format!(
        "{ind}(local.set $addr (call $tops_get_dyn (local.get $sel)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.sub (local.get $addr) (i32.const 8))))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (local.get $addr) (local.get $v0))\n"
    ));
    out.push_str(&format!(
        "{ind}(call $tops_set_dyn (local.get $sel) (i32.add (local.get $addr) (i32.const 8)))\n"
    ));
}

/// Swap on storage `s`.
fn emit_stack_swap(out: &mut String, ind: &str, s: usize) {
    let toff = tops_offset(s);
    // addr1 = tops[s] - 8; addr2 = tops[s] - 16
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.load (i32.const {toff})))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.sub (local.get $addr) (i32.const 8))))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v1 (i64.load (i32.sub (local.get $addr) (i32.const 16))))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (i32.sub (local.get $addr) (i32.const 8)) (local.get $v1))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (i32.sub (local.get $addr) (i32.const 16)) (local.get $v0))\n"
    ));
}

/// Swap on dynamic storage.
fn emit_stack_swap_dyn(out: &mut String, ind: &str) {
    out.push_str(&format!(
        "{ind}(local.set $addr (call $tops_get_dyn (local.get $sel)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.sub (local.get $addr) (i32.const 8))))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v1 (i64.load (i32.sub (local.get $addr) (i32.const 16))))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (i32.sub (local.get $addr) (i32.const 8)) (local.get $v1))\n"
    ));
    out.push_str(&format!(
        "{ind}(i64.store (i32.sub (local.get $addr) (i32.const 16)) (local.get $v0))\n"
    ));
}

/// BinOp on known regular storage.
fn emit_stack_binop(out: &mut String, ind: &str, s: usize, kind: &BinOpKind) {
    let toff = tops_offset(s);
    // Pop two
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.sub (i32.load (i32.const {toff})) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (i32.load (i32.const {toff}))))\n"
    ));
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.sub (i32.load (i32.const {toff})) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v1 (i64.load (i32.load (i32.const {toff}))))\n"
    ));
    // Compute result and push
    let expr = binop_expr(kind, "(local.get $v1)", "(local.get $v0)");
    out.push_str(&format!(
        "{ind}(i64.store (i32.load (i32.const {toff})) {expr})\n"
    ));
    out.push_str(&format!(
        "{ind}(i32.store (i32.const {toff}) (i32.add (i32.load (i32.const {toff})) (i32.const 8)))\n"
    ));
}

/// BinOp on dynamic (non-special) storage.
fn emit_stack_binop_dyn(out: &mut String, ind: &str, kind: &BinOpKind) {
    // Pop two from dynamic storage
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.sub (call $tops_get_dyn (local.get $sel)) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v0 (i64.load (local.get $addr)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $addr (i32.sub (local.get $addr) (i32.const 8)))\n"
    ));
    out.push_str(&format!(
        "{ind}(local.set $v1 (i64.load (local.get $addr)))\n"
    ));
    // Compute and push result
    let expr = binop_expr(kind, "(local.get $v1)", "(local.get $v0)");
    out.push_str(&format!("{ind}(i64.store (local.get $addr) {expr})\n"));
    out.push_str(&format!(
        "{ind}(call $tops_set_dyn (local.get $sel) (i32.add (local.get $addr) (i32.const 8)))\n"
    ));
}

/// Generate a WAT expression for a binary operation.
fn binop_expr(kind: &BinOpKind, a: &str, b: &str) -> String {
    match kind {
        BinOpKind::Add => format!("(call $wrap_add {a} {b})"),
        BinOpKind::Sub => format!("(call $wrap_sub {a} {b})"),
        BinOpKind::Mul => format!("(call $wrap_mul {a} {b})"),
        BinOpKind::Div => format!("(call $safe_div {a} {b})"),
        BinOpKind::Mod => format!("(call $safe_rem {a} {b})"),
        BinOpKind::Cmp => {
            format!("(select (i64.const 1) (i64.const 0) (i64.ge_s {a} {b}))")
        }
    }
}

/// Emit all helper functions.
fn emit_helpers(out: &mut String, has_queue: bool, has_port: bool, _has_io: bool, wasi: bool) {
    // --- Wrapping arithmetic ---
    out.push_str(
        r#"  ;; Wrapping arithmetic (i64 add/sub/mul just wrap naturally in WASM)
  (func $wrap_add (param $a i64) (param $b i64) (result i64)
    (i64.add (local.get $a) (local.get $b))
  )
  (func $wrap_sub (param $a i64) (param $b i64) (result i64)
    (i64.sub (local.get $a) (local.get $b))
  )
  (func $wrap_mul (param $a i64) (param $b i64) (result i64)
    (i64.mul (local.get $a) (local.get $b))
  )

  ;; Floored division: returns 0 on div-by-zero, wraps INT64_MIN / -1
  (func $safe_div (param $a i64) (param $b i64) (result i64)
    (local $q i64) (local $r i64)
    (if (result i64) (i64.eqz (local.get $b))
      (then (i64.const 0))
      (else
        (if (result i64) (i32.and
            (i64.eq (local.get $a) (i64.const -9223372036854775808))
            (i64.eq (local.get $b) (i64.const -1)))
          (then (i64.const -9223372036854775808))
          (else
            (local.set $q (i64.div_s (local.get $a) (local.get $b)))
            (local.set $r (i64.rem_s (local.get $a) (local.get $b)))
            (if (result i64) (i32.and
                (i64.ne (local.get $r) (i64.const 0))
                (i32.ne (i64.lt_s (local.get $r) (i64.const 0))
                        (i64.lt_s (local.get $b) (i64.const 0))))
              (then (i64.sub (local.get $q) (i64.const 1)))
              (else (local.get $q)))
          )
        )
      )
    )
  )

  ;; Floored remainder: returns 0 on div-by-zero, 0 for INT64_MIN % -1
  (func $safe_rem (param $a i64) (param $b i64) (result i64)
    (local $q i64) (local $r i64)
    (if (result i64) (i64.eqz (local.get $b))
      (then (i64.const 0))
      (else
        (if (result i64) (i32.and
            (i64.eq (local.get $a) (i64.const -9223372036854775808))
            (i64.eq (local.get $b) (i64.const -1)))
          (then (i64.const 0))
          (else
            (local.set $q (i64.div_s (local.get $a) (local.get $b)))
            (local.set $r (i64.rem_s (local.get $a) (local.get $b)))
            (if (result i64) (i32.and
                (i64.ne (local.get $r) (i64.const 0))
                (i32.ne (i64.lt_s (local.get $r) (i64.const 0))
                        (i64.lt_s (local.get $b) (i64.const 0))))
              (then (i64.add (local.get $r) (local.get $b)))
              (else (local.get $r)))
          )
        )
      )
    )
  )

"#,
    );

    // --- Dynamic tops access (for unknown sel) ---
    out.push_str(&format!(
        r#"  ;; Dynamic tops array access
  (func $tops_get_dyn (param $s i32) (result i32)
    (i32.load (i32.add (i32.const {TOPS}) (i32.mul (local.get $s) (i32.const 4))))
  )
  (func $tops_set_dyn (param $s i32) (param $v i32)
    (i32.store (i32.add (i32.const {TOPS}) (i32.mul (local.get $s) (i32.const 4))) (local.get $v))
  )

  ;; Dynamic depth: for special storages use sp_depth, else compute from tops
  (func $depth_dyn (param $s i32) (result i32)
    (if (result i32) (i32.or (i32.eq (local.get $s) (i32.const {QUEUE})) (i32.eq (local.get $s) (i32.const {PORT})))
      (then (call $sp_depth (local.get $s)))
      (else
        ;; (tops[s] - base(s)) / 8
        ;; base(s) = 0x0400 + s * 524288
        (i32.shr_u
          (i32.sub
            (call $tops_get_dyn (local.get $s))
            (i32.add (i32.const {STORAGE_BASE}) (i32.mul (local.get $s) (i32.const {STORAGE_SIZE})))
          )
          (i32.const 3)
        )
      )
    )
  )

"#
    ));

    // --- Special storage (queue + port) ---
    if has_queue || has_port {
        out.push_str(&format!(
            r#"  ;; Queue/Port: push
  (func $sp_push (param $s i32) (param $v i64)
    (local $idx i32)
    (if (i32.eq (local.get $s) (i32.const {QUEUE}))
      (then
        ;; queue[(q_head + q_len) % CAP] = v; q_len++
        (local.set $idx
          (i32.rem_u
            (i32.add (i32.load (i32.const {Q_HEAD})) (i32.load (i32.const {Q_LEN})))
            (i32.const {QUEUE_CAP})
          )
        )
        (i64.store
          (i32.add (i32.const {STORAGE_BASE_Q}) (i32.mul (local.get $idx) (i32.const 8)))
          (local.get $v)
        )
        (i32.store (i32.const {Q_LEN}) (i32.add (i32.load (i32.const {Q_LEN})) (i32.const 1)))
      )
      (else
        ;; port[p_len++] = v; port_last = v
        (i64.store
          (i32.add (i32.const {STORAGE_BASE_P}) (i32.mul (i32.load (i32.const {P_LEN})) (i32.const 8)))
          (local.get $v)
        )
        (i32.store (i32.const {P_LEN}) (i32.add (i32.load (i32.const {P_LEN})) (i32.const 1)))
        (i64.store (i32.const {PORT_LAST}) (local.get $v))
      )
    )
  )

  ;; Queue/Port: pop
  (func $sp_pop (param $s i32) (result i64)
    (local $v i64)
    (if (i32.eq (local.get $s) (i32.const {QUEUE}))
      (then
        (if (i32.eqz (i32.load (i32.const {Q_LEN})))
          (then (return (i64.const 0)))
        )
        (local.set $v
          (i64.load
            (i32.add (i32.const {STORAGE_BASE_Q}) (i32.mul (i32.load (i32.const {Q_HEAD})) (i32.const 8)))
          )
        )
        (i32.store (i32.const {Q_HEAD})
          (i32.rem_u (i32.add (i32.load (i32.const {Q_HEAD})) (i32.const 1)) (i32.const {QUEUE_CAP}))
        )
        (i32.store (i32.const {Q_LEN}) (i32.sub (i32.load (i32.const {Q_LEN})) (i32.const 1)))
      )
      (else
        (if (i32.eqz (i32.load (i32.const {P_LEN})))
          (then (return (i64.const 0)))
        )
        (i32.store (i32.const {P_LEN}) (i32.sub (i32.load (i32.const {P_LEN})) (i32.const 1)))
        (local.set $v
          (i64.load
            (i32.add (i32.const {STORAGE_BASE_P}) (i32.mul (i32.load (i32.const {P_LEN})) (i32.const 8)))
          )
        )
      )
    )
    (local.get $v)
  )

  ;; Queue/Port: depth
  (func $sp_depth (param $s i32) (result i32)
    (if (result i32) (i32.eq (local.get $s) (i32.const {QUEUE}))
      (then (i32.load (i32.const {Q_LEN})))
      (else (i32.load (i32.const {P_LEN})))
    )
  )

  ;; Queue/Port: dup
  (func $sp_dup (param $s i32)
    (local $v i64)
    (if (i32.eq (local.get $s) (i32.const {QUEUE}))
      (then
        (if (i32.eqz (i32.load (i32.const {Q_LEN}))) (then (return)))
        ;; peek front
        (local.set $v
          (i64.load (i32.add (i32.const {STORAGE_BASE_Q}) (i32.mul (i32.load (i32.const {Q_HEAD})) (i32.const 8))))
        )
        ;; push_front: move head back, store, len++
        (i32.store (i32.const {Q_HEAD})
          (i32.rem_u
            (i32.add (i32.sub (i32.load (i32.const {Q_HEAD})) (i32.const 1)) (i32.const {QUEUE_CAP}))
            (i32.const {QUEUE_CAP})
          )
        )
        (i64.store
          (i32.add (i32.const {STORAGE_BASE_Q}) (i32.mul (i32.load (i32.const {Q_HEAD})) (i32.const 8)))
          (local.get $v)
        )
        (i32.store (i32.const {Q_LEN}) (i32.add (i32.load (i32.const {Q_LEN})) (i32.const 1)))
      )
      (else
        ;; port: push port_last
        (i64.store
          (i32.add (i32.const {STORAGE_BASE_P}) (i32.mul (i32.load (i32.const {P_LEN})) (i32.const 8)))
          (i64.load (i32.const {PORT_LAST}))
        )
        (i32.store (i32.const {P_LEN}) (i32.add (i32.load (i32.const {P_LEN})) (i32.const 1)))
      )
    )
  )

  ;; Queue/Port: swap
  (func $sp_swap (param $s i32)
    (local $i0 i32) (local $i1 i32) (local $tmp i64)
    (if (i32.eq (local.get $s) (i32.const {QUEUE}))
      (then
        (if (i32.lt_u (i32.load (i32.const {Q_LEN})) (i32.const 2)) (then (return)))
        (local.set $i0 (i32.add (i32.const {STORAGE_BASE_Q}) (i32.mul (i32.load (i32.const {Q_HEAD})) (i32.const 8))))
        (local.set $i1 (i32.add (i32.const {STORAGE_BASE_Q})
          (i32.mul (i32.rem_u (i32.add (i32.load (i32.const {Q_HEAD})) (i32.const 1)) (i32.const {QUEUE_CAP})) (i32.const 8))
        ))
        (local.set $tmp (i64.load (local.get $i0)))
        (i64.store (local.get $i0) (i64.load (local.get $i1)))
        (i64.store (local.get $i1) (local.get $tmp))
      )
      (else
        (if (i32.lt_u (i32.load (i32.const {P_LEN})) (i32.const 2)) (then (return)))
        (local.set $i0 (i32.add (i32.const {STORAGE_BASE_P}) (i32.mul (i32.sub (i32.load (i32.const {P_LEN})) (i32.const 1)) (i32.const 8))))
        (local.set $i1 (i32.add (i32.const {STORAGE_BASE_P}) (i32.mul (i32.sub (i32.load (i32.const {P_LEN})) (i32.const 2)) (i32.const 8))))
        (local.set $tmp (i64.load (local.get $i0)))
        (i64.store (local.get $i0) (i64.load (local.get $i1)))
        (i64.store (local.get $i1) (local.get $tmp))
      )
    )
  )

"#,
            STORAGE_BASE_Q = storage_base(QUEUE),
            STORAGE_BASE_P = storage_base(PORT),
        ));
    } else {
        // Stubs for when special storages are unused but still called dynamically
        out.push_str(&format!(
            r#"  (func $sp_push (param $s i32) (param $v i64))
  (func $sp_pop (param $s i32) (result i64) (i64.const 0))
  (func $sp_depth (param $s i32) (result i32) (i32.const 0))
  (func $sp_dup (param $s i32))
  (func $sp_swap (param $s i32))

"#
        ));
    }

    // --- I/O: output ---
    if wasi {
        out.push_str(&format!(
            r#"  (global $out_pos (mut i32) (i32.const 0))

  (func $flush_out
    (if (i32.gt_s (global.get $out_pos) (i32.const 0))
      (then
        (i32.store (i32.const {IOVEC}) (i32.const {IO_BUF}))
        (i32.store (i32.const {IOVEC_LEN}) (global.get $out_pos))
        (drop (call $fd_write (i32.const 1) (i32.const {IOVEC}) (i32.const 1) (i32.const {NWRITTEN})))
        (global.set $out_pos (i32.const 0))
      )
    )
  )

  (func $emit_byte (param $b i32)
    (i32.store8 (i32.add (i32.const {IO_BUF}) (global.get $out_pos)) (local.get $b))
    (global.set $out_pos (i32.add (global.get $out_pos) (i32.const 1)))
    (if (i32.ge_u (global.get $out_pos) (i32.const 256))
      (then (call $flush_out))
    )
  )

"#,
            IOVEC_LEN = IOVEC + 4,
        ));
    } else {
        out.push_str(
            r#"  (func $flush_out)
  (func $emit_byte (param $b i32) (call $write_byte_import (local.get $b)))

"#,
        );
    }

    // --- write_num: convert i64 to decimal, emit bytes ---
    out.push_str(&format!(
        r#"  ;; Write i64 as decimal to stdout
  (func $write_num (param $n i64)
    (local $u i64) (local $buf_end i32) (local $pos i32) (local $neg i32)
    ;; Use scratch area separate from output buffer
    (local.set $buf_end (i32.const {DIGIT_BUF_END}))
    (local.set $pos (local.get $buf_end))
    (local.set $neg (i32.const 0))
    (if (i64.lt_s (local.get $n) (i64.const 0))
      (then
        (local.set $neg (i32.const 1))
        ;; u = (uint64_t)(-(n+1)) + 1
        (local.set $u (i64.add (i64.sub (i64.const 0) (i64.add (local.get $n) (i64.const 1))) (i64.const 1)))
      )
      (else
        (local.set $u (local.get $n))
      )
    )
    (block $done
      (loop $digit_loop
        (local.set $pos (i32.sub (local.get $pos) (i32.const 1)))
        (i32.store8 (local.get $pos)
          (i32.add (i32.const 48) (i32.wrap_i64 (i64.rem_u (local.get $u) (i64.const 10))))
        )
        (local.set $u (i64.div_u (local.get $u) (i64.const 10)))
        (br_if $done (i64.eqz (local.get $u)))
        (br $digit_loop)
      )
    )
    (if (local.get $neg) (then (call $emit_byte (i32.const 45)))) ;; '-'
    (block $emit_done
      (loop $emit_loop
        (br_if $emit_done (i32.ge_u (local.get $pos) (local.get $buf_end)))
        (call $emit_byte (i32.load8_u (local.get $pos)))
        (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
        (br $emit_loop)
      )
    )
  )

"#,
        DIGIT_BUF_END = DIGIT_SCRATCH + 24, // 20 digits max for i64 + padding
    ));

    // --- write_char: UTF-8 encode and emit ---
    out.push_str(
        r#"  ;; Write i64 as UTF-8 char to stdout
  (func $write_char (param $v i64)
    (local $c i32)
    (local.set $c (i32.wrap_i64 (local.get $v)))
    (if (i32.le_u (local.get $c) (i32.const 0x7F))
      (then (call $emit_byte (local.get $c)))
      (else (if (i32.le_u (local.get $c) (i32.const 0x7FF))
        (then
          (call $emit_byte (i32.or (i32.const 0xC0) (i32.shr_u (local.get $c) (i32.const 6))))
          (call $emit_byte (i32.or (i32.const 0x80) (i32.and (local.get $c) (i32.const 0x3F))))
        )
        (else (if (i32.le_u (local.get $c) (i32.const 0xFFFF))
          (then
            (call $emit_byte (i32.or (i32.const 0xE0) (i32.shr_u (local.get $c) (i32.const 12))))
            (call $emit_byte (i32.or (i32.const 0x80) (i32.and (i32.shr_u (local.get $c) (i32.const 6)) (i32.const 0x3F))))
            (call $emit_byte (i32.or (i32.const 0x80) (i32.and (local.get $c) (i32.const 0x3F))))
          )
          (else (if (i32.le_u (local.get $c) (i32.const 0x10FFFF))
            (then
              (call $emit_byte (i32.or (i32.const 0xF0) (i32.shr_u (local.get $c) (i32.const 18))))
              (call $emit_byte (i32.or (i32.const 0x80) (i32.and (i32.shr_u (local.get $c) (i32.const 12)) (i32.const 0x3F))))
              (call $emit_byte (i32.or (i32.const 0x80) (i32.and (i32.shr_u (local.get $c) (i32.const 6)) (i32.const 0x3F))))
              (call $emit_byte (i32.or (i32.const 0x80) (i32.and (local.get $c) (i32.const 0x3F))))
            )
          ))
        ))
      ))
    )
  )

"#,
    );

    // --- read_num / read_char ---
    if wasi {
        out.push_str(&format!(
            r#"  (func $read_num (result i64)
    (local $nread i32) (local $pos i32) (local $b i32)
    (local $result i64) (local $neg i32)
    (i32.store (i32.const {IOVEC}) (i32.const {IO_BUF}))
    (i32.store (i32.const {IOVEC_LEN}) (i32.const 64))
    (drop (call $fd_read (i32.const 0) (i32.const {IOVEC}) (i32.const 1) (i32.const {NWRITTEN})))
    (local.set $nread (i32.load (i32.const {NWRITTEN})))
    (if (i32.eqz (local.get $nread)) (then (return (i64.const 0))))
    (block $ws_done (loop $ws_loop
      (br_if $ws_done (i32.ge_u (local.get $pos) (local.get $nread)))
      (local.set $b (i32.load8_u (i32.add (i32.const {IO_BUF}) (local.get $pos))))
      (br_if $ws_done (i32.and (i32.ne (local.get $b) (i32.const 32)) (i32.ne (local.get $b) (i32.const 10))))
      (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
      (br $ws_loop)))
    (if (i32.lt_u (local.get $pos) (local.get $nread))
      (then (if (i32.eq (i32.load8_u (i32.add (i32.const {IO_BUF}) (local.get $pos))) (i32.const 45))
        (then (local.set $neg (i32.const 1)) (local.set $pos (i32.add (local.get $pos) (i32.const 1)))))))
    (block $pd (loop $pl
      (br_if $pd (i32.ge_u (local.get $pos) (local.get $nread)))
      (local.set $b (i32.load8_u (i32.add (i32.const {IO_BUF}) (local.get $pos))))
      (br_if $pd (i32.or (i32.lt_u (local.get $b) (i32.const 48)) (i32.gt_u (local.get $b) (i32.const 57))))
      (local.set $result (i64.add (i64.mul (local.get $result) (i64.const 10)) (i64.extend_i32_u (i32.sub (local.get $b) (i32.const 48)))))
      (local.set $pos (i32.add (local.get $pos) (i32.const 1)))
      (br $pl)))
    (if (result i64) (local.get $neg) (then (i64.sub (i64.const 0) (local.get $result))) (else (local.get $result)))
  )

  (func $read_char (result i64)
    (local $b i32) (local $n i32) (local $val i32) (local $i i32)
    (i32.store (i32.const {IOVEC}) (i32.const {IO_BUF}))
    (i32.store (i32.const {IOVEC_LEN}) (i32.const 1))
    (drop (call $fd_read (i32.const 0) (i32.const {IOVEC}) (i32.const 1) (i32.const {NWRITTEN})))
    (if (i32.eqz (i32.load (i32.const {NWRITTEN}))) (then (return (i64.const -1))))
    (local.set $b (i32.load8_u (i32.const {IO_BUF})))
    (if (i32.lt_u (local.get $b) (i32.const 0x80)) (then (return (i64.extend_i32_u (local.get $b)))))
    (if (i32.eq (i32.shr_u (local.get $b) (i32.const 5)) (i32.const 6))
      (then (local.set $n (i32.const 1)) (local.set $val (i32.and (local.get $b) (i32.const 0x1F))))
      (else (if (i32.eq (i32.shr_u (local.get $b) (i32.const 4)) (i32.const 14))
        (then (local.set $n (i32.const 2)) (local.set $val (i32.and (local.get $b) (i32.const 0x0F))))
        (else (if (i32.eq (i32.shr_u (local.get $b) (i32.const 3)) (i32.const 30))
          (then (local.set $n (i32.const 3)) (local.set $val (i32.and (local.get $b) (i32.const 0x07))))
          (else (return (i64.const -1))))))))
    (block $cd (loop $cl
      (br_if $cd (i32.ge_u (local.get $i) (local.get $n)))
      (i32.store (i32.const {IOVEC}) (i32.const {IO_BUF}))
      (i32.store (i32.const {IOVEC_LEN}) (i32.const 1))
      (drop (call $fd_read (i32.const 0) (i32.const {IOVEC}) (i32.const 1) (i32.const {NWRITTEN})))
      (if (i32.eqz (i32.load (i32.const {NWRITTEN}))) (then (return (i64.const -1))))
      (local.set $val (i32.or (i32.shl (local.get $val) (i32.const 6)) (i32.and (i32.load8_u (i32.const {IO_BUF})) (i32.const 0x3F))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $cl)))
    (i64.extend_i32_u (local.get $val))
  )

"#,
            IOVEC_LEN = IOVEC + 4,
        ));
    } else {
        // env mode: read_byte returns one byte (-1 for EOF)
        out.push_str(
            r#"  (func $read_num (result i64)
    (local $b i32) (local $result i64) (local $neg i32) (local $started i32)
    ;; Skip whitespace
    (block $ws_done (loop $ws_loop
      (local.set $b (call $read_byte_import))
      (br_if $ws_done (i32.lt_s (local.get $b) (i32.const 0)))
      (br_if $ws_done (i32.and (i32.ne (local.get $b) (i32.const 32)) (i32.ne (local.get $b) (i32.const 10)) ))
      (br $ws_loop)))
    (if (i32.lt_s (local.get $b) (i32.const 0)) (then (return (i64.const 0))))
    (if (i32.eq (local.get $b) (i32.const 45))
      (then (local.set $neg (i32.const 1)) (local.set $b (call $read_byte_import))))
    ;; Parse digits
    (block $pd (loop $pl
      (br_if $pd (i32.or (i32.lt_u (local.get $b) (i32.const 48)) (i32.gt_u (local.get $b) (i32.const 57))))
      (local.set $result (i64.add (i64.mul (local.get $result) (i64.const 10)) (i64.extend_i32_u (i32.sub (local.get $b) (i32.const 48)))))
      (local.set $b (call $read_byte_import))
      (br $pl)))
    (if (result i64) (local.get $neg) (then (i64.sub (i64.const 0) (local.get $result))) (else (local.get $result)))
  )

  (func $read_char (result i64)
    (local $b i32) (local $n i32) (local $val i32) (local $i i32)
    (local.set $b (call $read_byte_import))
    (if (i32.lt_s (local.get $b) (i32.const 0)) (then (return (i64.const -1))))
    (if (i32.lt_u (local.get $b) (i32.const 0x80)) (then (return (i64.extend_i32_u (local.get $b)))))
    (if (i32.eq (i32.shr_u (local.get $b) (i32.const 5)) (i32.const 6))
      (then (local.set $n (i32.const 1)) (local.set $val (i32.and (local.get $b) (i32.const 0x1F))))
      (else (if (i32.eq (i32.shr_u (local.get $b) (i32.const 4)) (i32.const 14))
        (then (local.set $n (i32.const 2)) (local.set $val (i32.and (local.get $b) (i32.const 0x0F))))
        (else (if (i32.eq (i32.shr_u (local.get $b) (i32.const 3)) (i32.const 30))
          (then (local.set $n (i32.const 3)) (local.set $val (i32.and (local.get $b) (i32.const 0x07))))
          (else (return (i64.const -1))))))))
    (block $cd (loop $cl
      (br_if $cd (i32.ge_u (local.get $i) (local.get $n)))
      (local.set $b (call $read_byte_import))
      (if (i32.lt_s (local.get $b) (i32.const 0)) (then (return (i64.const -1))))
      (local.set $val (i32.or (i32.shl (local.get $val) (i32.const 6)) (i32.and (local.get $b) (i32.const 0x3F))))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $cl)))
    (i64.extend_i32_u (local.get $val))
  )

"#,
        );
    }
}

const WAT_HEADER: &str = "\
;; Generated by compaheuiler — https://github.com/youknowone/aheui-rust
;; This code includes runtime components licensed under AGPL-3.0-or-later.
;; Distributing binaries compiled from this code requires compliance with the AGPL.

";
