//! Generate C99 code from an Aheui CFG using the same optimization pipeline as rust_gen.
//!
//! The i64 output compiles directly with `cc`. Dual-mode output is compiled to
//! an object and linked with [`c_bigint_bridge_rs`] by Cargo.

use ahsembler::cfg::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

const QUEUE: usize = 21;
const PORT: usize = 27;
fn is_special(s: usize) -> bool {
    s == QUEUE || s == PORT
}

pub fn compile_to_c(source: &str) -> String {
    compile_to_c_opt(source, ahsembler::OptimizationLevel::O3)
}

pub fn compile_to_c_opt(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    let cfg = crate::pipeline::optimize(source, opt);
    generate_c_dispatch(&cfg, false)
}

/// Compile to C with the same raw-i64-to-tagged-bigint transition used by the
/// Rust backend. The emitted C99 calls the Rust ABI provided by
/// [`c_bigint_bridge_rs`]; compile it as an object and let Cargo link both.
pub fn compile_to_c_bigint(source: &str) -> String {
    compile_to_c_bigint_opt(source, ahsembler::OptimizationLevel::O3)
}

pub fn compile_to_c_bigint_opt(source: &str, opt: ahsembler::OptimizationLevel) -> String {
    let cfg = crate::pipeline::optimize(source, opt);
    generate_c_dispatch(&cfg, true)
}

/// Rust half of the C BigInt runtime. It supplies the C ABI and the final
/// `main`, using the same BigInt crate selected for compaheuiler itself.
pub fn c_bigint_bridge_rs() -> String {
    let bigint_import = if cfg!(feature = "num-bigint") {
        "use num_bigint::BigInt;\n"
    } else {
        "use malachite_bigint::BigInt;\n"
    };
    format!("{bigint_import}{}", include_str!("c_bigint_bridge.rs"))
}

/// Cargo dependency declaration matching [`c_bigint_bridge_rs`].
pub fn c_bigint_dependency_toml() -> &'static str {
    if cfg!(feature = "num-bigint") {
        r#"num-bigint = "0.4""#
    } else {
        r#"malachite-bigint = "0.9""#
    }
}

fn generate_c_dispatch(cfg: &Cfg, bigint: bool) -> String {
    let mut out = String::with_capacity(32768);
    out.push_str(if bigint { C_BIGINT_PRELUDE } else { C_PRELUDE });

    let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);
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
    let has_special = has_queue || has_port;

    // Collect live block IDs in order
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

    let use_match = live_blocks.len() > 16;
    // Function signature
    out.push_str("static int64_t aheui_main(int64_t** bases, int32_t* lengths) {\n");
    out.push_str("  SpecialStorage sp;\n  sp_init(&sp);\n");
    for &s in &used {
        out.push_str(&format!("  int64_t* t{s} = bases[{s}] + lengths[{s}];\n"));
    }
    out.push_str("  int64_t* top = t0;\n  size_t sel = 0;\n  /*VARDECL*/\n");
    if bigint {
        out.push_str("  int _bm = 0, _bm_prev = 0;\n");
        out.push_str("  int64_t* tops_snapshot[STORAGE_COUNT] = {0};\n");
        // Keep promotion metadata coherent when a top changes. This avoids
        // copying every storage top before every checked arithmetic operation.
        for &s in &used {
            if !is_special(s) {
                out.push_str(&format!("  tops_snapshot[{s}] = t{s};\n"));
            }
        }
    }
    let mut max_var: usize = 0;
    let ind = "      ";

    out.push_str(&format!(
        "  uint32_t _pc = {};\n",
        block_order.get(&cfg.entry).copied().unwrap_or(0)
    ));
    out.push_str("  for (;;) {\n");
    if use_match {
        out.push_str("    switch (_pc) {\n");
    }

    for (seq_idx, &block_id) in live_blocks.iter().enumerate() {
        let block = cfg.block(block_id);
        let entry_state = states.get(block_id as usize);
        let entry_sel: Option<usize> = entry_state.and_then(|s| s.selected);
        let mut sel = entry_sel;
        let next_seq = seq_idx + 1;

        // Detect self-loop BranchZero
        let is_self_loop = matches!(&block.terminator,
            Terminator::BranchZero { on_zero, on_nonzero }
            if *on_zero == block_id || *on_nonzero == block_id);

        // Detect scan-to-zero
        let is_scan_to_zero = is_self_loop
            && entry_sel == Some(QUEUE)
            && matches!(&block.terminator, Terminator::BranchZero { .. })
            && {
                let insts = &block.instructions;
                (insts.len() == 2
                    && matches!(&insts[0], Inst::Dup)
                    && matches!(&insts[1], Inst::Mov(t) if *t == QUEUE))
                    || (insts.len() == 3
                        && matches!(&insts[0], Inst::GuardDepth { min_depth: 1, .. })
                        && matches!(&insts[1], Inst::Dup)
                        && matches!(&insts[2], Inst::Mov(t) if *t == QUEUE))
            };
        let scan_guard_fail = if is_scan_to_zero && block.instructions.len() == 3 {
            if let Inst::GuardDepth { fail, .. } = &block.instructions[0] {
                Some(block_order.get(fail).copied().unwrap_or(0))
            } else {
                None
            }
        } else {
            None
        };

        if use_match {
            out.push_str(&format!("    case {seq_idx}: {{\n"));
        } else {
            out.push_str(&format!("    if (_pc <= {seq_idx}) {{\n"));
        }

        if is_scan_to_zero {
            if let Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } = &block.terminator
            {
                let exit_target = if *on_nonzero == block_id {
                    on_zero
                } else {
                    on_nonzero
                };
                let exit_seq = block_order.get(exit_target).copied().unwrap_or(0);
                if let Some(guard_fail_seq) = scan_guard_fail {
                    out.push_str(&format!(
                        "{ind}  if (sp.q_len == 0) {{ _pc = {guard_fail_seq}; continue; }}\n"
                    ));
                }
                out.push_str(&format!("{ind}  sp_scan_to_zero(&sp);\n"));
                out.push_str(&format!("{ind}  _pc = {exit_seq}; continue;\n"));
            }
            if use_match {
                out.push_str("    } break;\n");
            } else {
                out.push_str("    }\n");
            }
            continue;
        }

        if is_self_loop {
            out.push_str(&format!("{ind}  for (;;) {{\n"));
        }

        // Entry sync
        let has_dyn_sel = live_blocks.iter().any(|&bid| {
            states
                .get(bid as usize)
                .map_or(false, |s| s.selected.is_none())
        });
        let skip_entry_sync = entry_sel.is_none() && has_dyn_sel;
        if entry_sel.is_none() && !skip_entry_sync {
            out.push_str(&format!("{ind}  switch (sel) {{ "));
            for &s in &used {
                if !is_special(s) {
                    out.push_str(&format!("case {s}: top = t{s}; break; "));
                }
            }
            out.push_str("default: break; }\n");
        }

        // Abs register allocation
        let mut abs = Abs {
            stacks: BTreeMap::new(),
            active: entry_sel.unwrap_or(0),
            sel_known: entry_sel.is_some(),
            next_var: max_var,
            bigint,
        };
        let insts = &block.instructions;
        let mut ii = 0;
        let mut top_modified = false;

        let op_str_fn = |kind: &BinOpKind, a: &str, b: &str| -> String {
            if bigint {
                match kind {
                    BinOpKind::Add => {
                        format!("dual_add({a}, {b}, &_bm, bases, tops_snapshot, &sp)")
                    }
                    BinOpKind::Sub => {
                        format!("dual_sub({a}, {b}, &_bm, bases, tops_snapshot, &sp)")
                    }
                    BinOpKind::Mul => {
                        format!("dual_mul({a}, {b}, &_bm, bases, tops_snapshot, &sp)")
                    }
                    BinOpKind::Div => format!("int_div({a}, {b}, _bm)"),
                    BinOpKind::Mod => format!("int_rem({a}, {b}, _bm)"),
                    BinOpKind::Cmp => {
                        format!("int_ge({a}, {b}, _bm) ? int_lit(1, _bm) : int_lit(0, _bm)")
                    }
                }
            } else {
                match kind {
                    BinOpKind::Add => format!("((int64_t)((uint64_t)({a}) + (uint64_t)({b})))"),
                    BinOpKind::Sub => format!("((int64_t)((uint64_t)({a}) - (uint64_t)({b})))"),
                    BinOpKind::Mul => format!("((int64_t)((uint64_t)({a}) * (uint64_t)({b})))"),
                    BinOpKind::Div => format!("(({b}) != 0 ? wrapping_div({a}, {b}) : 0)"),
                    BinOpKind::Mod => format!("(({b}) != 0 ? wrapping_rem({a}, {b}) : 0)"),
                    BinOpKind::Cmp => format!("(({a}) >= ({b}) ? 1 : 0)"),
                }
            }
        };

        while ii < insts.len() {
            let inst = &insts[ii];
            let sp_known = sel.is_some_and(is_special);
            let ds = sel.is_none();
            let reg = sel.is_some() && !sp_known;
            match inst {
                Inst::Push(v) => {
                    let cv = if bigint {
                        format!("int_lit((int64_t){v}LL, _bm)")
                    } else {
                        format!("(int64_t){v}LL")
                    };
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_push(&sp, {}, {cv});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, {cv}); }} else {{ *top = {cv}; top++; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else {
                        let var = abs.fresh();
                        out.push_str(&format!("{ind}  {var} = {cv};\n"));
                        abs.push(var);
                    }
                }
                Inst::Pop => {
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_pop(&sp, {});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_pop(&sp, sel); }} else {{ top--; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else if abs.pop().is_none() {
                        let t = abs.top_var();
                        out.push_str(&format!("{ind}  {t}--;\n"));
                        if bigint {
                            out.push_str(&format!(
                                "{ind}  tops_snapshot[{}] = {t};\n",
                                sel.unwrap()
                            ));
                        }
                    }
                }
                Inst::Dup => {
                    if sp_known {
                        let s = sel.unwrap();
                        let next = insts.get(ii + 1);
                        if s == QUEUE {
                            if let Some(Inst::Mov(t)) = next {
                                if *t == s {
                                    out.push_str(&format!("{ind}  if (sp.q_len > 0) {{ int64_t _v = sp.queue[sp.q_head]; sp.queue[(sp.q_head + sp.q_len) % QUEUE_CAP] = _v; sp.q_len++; }}\n"));
                                    ii += 2;
                                    continue;
                                }
                            }
                        }
                        out.push_str(&format!("{ind}  sp_dup(&sp, {s});\n"));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_dup(&sp, sel); }} else {{ int64_t _v = *(top-1); *top = _v; top++; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let v = abs.peek().unwrap().clone();
                        abs.push(v);
                    }
                }
                Inst::Swap => {
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_swap(&sp, {});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_swap(&sp, sel); }} else {{ int64_t _a = *(top-1), _b = *(top-2); *(top-1) = _b; *(top-2) = _a; }}\n"));
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 2, &format!("{ind}  "));
                        let stk = abs.stacks.entry(abs.active).or_default();
                        let n = stk.len();
                        stk.swap(n - 1, n - 2);
                    }
                }
                Inst::BinOp(kind) => {
                    if bigint && matches!(kind, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul) {
                        out.push_str(&format!("{ind}  _bm_prev = _bm;\n"));
                    }
                    if sp_known {
                        let s = sel.unwrap();
                        let v1 = abs.fresh();
                        let v2 = abs.fresh();
                        let vr = abs.fresh();
                        out.push_str(&format!(
                            "{ind}  {v1} = sp_pop(&sp, {s}); {v2} = sp_pop(&sp, {s});\n"
                        ));
                        out.push_str(&format!("{ind}  {vr} = {};\n", op_str_fn(kind, &v2, &v1)));
                        out.push_str(&format!("{ind}  sp_push(&sp, {s}, {vr});\n"));
                        if bigint
                            && matches!(kind, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul)
                        {
                            emit_promote_live_vars_c(&mut out, &abs, ind);
                        }
                    } else if ds {
                        top_modified = true;
                        let v1 = abs.fresh();
                        let v2 = abs.fresh();
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{\n"));
                        out.push_str(&format!(
                            "{ind}    {v1} = sp_pop(&sp, sel); {v2} = sp_pop(&sp, sel);\n"
                        ));
                        out.push_str(&format!("{ind}    {vr} = {};\n", op_str_fn(kind, &v2, &v1)));
                        out.push_str(&format!("{ind}    sp_push(&sp, sel, {vr});\n"));
                        out.push_str(&format!("{ind}  }} else {{\n"));
                        out.push_str(&format!(
                            "{ind}    top -= 2; {v2} = *top; {v1} = *(top+1);\n"
                        ));
                        out.push_str(&format!("{ind}    {vr} = {};\n", op_str_fn(kind, &v2, &v1)));
                        out.push_str(&format!("{ind}    *top = {vr}; top++;\n"));
                        out.push_str(&format!("{ind}  }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                        if bigint
                            && matches!(kind, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul)
                        {
                            emit_promote_live_vars_c(&mut out, &abs, ind);
                        }
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 2, &format!("{ind}  "));
                        let r1 = abs.pop().unwrap();
                        let r2 = abs.pop().unwrap();
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = {};\n", op_str_fn(kind, &r2, &r1)));
                        if bigint
                            && matches!(kind, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul)
                        {
                            emit_promote_live_vars_c(&mut out, &abs, ind);
                        }
                        abs.push(vr);
                    }
                }
                Inst::Sel(new_sel) => {
                    if ds && top_modified {
                        out.push_str(&format!("{ind}  switch (sel) {{ "));
                        for &s in &used {
                            if !is_special(s) {
                                out.push_str(&format!("case {s}: t{s} = top; break; "));
                            }
                        }
                        out.push_str("default: break; }\n");
                        top_modified = false;
                    }
                    out.push_str(&format!("{ind}  sel = {new_sel};\n"));
                    abs.active = *new_sel;
                    abs.sel_known = true;
                    sel = Some(*new_sel);
                }
                Inst::Mov(target) => {
                    if sp_known {
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = sp_pop(&sp, {});\n", sel.unwrap()));
                        if is_special(*target) {
                            out.push_str(&format!("{ind}  sp_push(&sp, {target}, {vr});\n"));
                        } else {
                            out.push_str(&format!("{ind}  *t{target} = {vr}; t{target}++;\n"));
                            if bigint {
                                out.push_str(&format!(
                                    "{ind}  tops_snapshot[{target}] = t{target};\n"
                                ));
                            }
                        }
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                        if is_special(*target) {
                            out.push_str(&format!("{ind}  sp_push(&sp, {target}, {vr});\n"));
                        } else {
                            out.push_str(&format!("{ind}  *t{target} = {vr}; t{target}++;\n"));
                            if bigint {
                                out.push_str(&format!(
                                    "{ind}  tops_snapshot[{target}] = t{target};\n"
                                ));
                            }
                        }
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let vr = abs.pop().unwrap();
                        if is_special(*target) {
                            out.push_str(&format!("{ind}  sp_push(&sp, {target}, {vr});\n"));
                        } else {
                            abs.stacks.entry(*target).or_default().push(vr);
                        }
                    }
                }
                Inst::PopNum => {
                    if sp_known {
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = sp_pop(&sp, {});\n", sel.unwrap()));
                        out.push_str(&format!(
                            "{ind}  write_num({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                        out.push_str(&format!(
                            "{ind}  write_num({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let vr = abs.pop().unwrap();
                        out.push_str(&format!(
                            "{ind}  write_num({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    }
                }
                Inst::PopChar => {
                    if sp_known {
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = sp_pop(&sp, {});\n", sel.unwrap()));
                        out.push_str(&format!(
                            "{ind}  write_char({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                        out.push_str(&format!(
                            "{ind}  write_char({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let vr = abs.pop().unwrap();
                        out.push_str(&format!(
                            "{ind}  write_char({vr}{});\n",
                            if bigint { ", _bm" } else { "" }
                        ));
                    }
                }
                Inst::PushNum => {
                    if reg {
                        abs.flush_all_c(&mut out, &format!("{ind}  "));
                    }
                    let vr = abs.fresh();
                    out.push_str(&format!(
                        "{ind}  fflush(stdout);\n{ind}  {vr} = read_num{};\n",
                        if bigint { "(_bm)" } else { "()" }
                    ));
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_push(&sp, {}, {vr});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, {vr}); }} else {{ *top = {vr}; top++; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else {
                        abs.push(vr);
                    }
                }
                Inst::PushChar => {
                    if reg {
                        abs.flush_all_c(&mut out, &format!("{ind}  "));
                    }
                    let vr = abs.fresh();
                    out.push_str(&format!(
                        "{ind}  fflush(stdout);\n{ind}  {vr} = read_char_{};\n",
                        if bigint { "(_bm)" } else { "()" }
                    ));
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_push(&sp, {}, {vr});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, {vr}); }} else {{ *top = {vr}; top++; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else {
                        abs.push(vr);
                    }
                }
                Inst::GuardDepth { min_depth, fail } => {
                    if reg {
                        abs.flush_all_c(&mut out, &format!("{ind}  "));
                    }
                    let t = if let Some(s) = sel {
                        format!("t{s}")
                    } else {
                        "top".into()
                    };
                    let depth_expr = if sp_known {
                        format!("(int64_t)sp_depth(&sp, {})", sel.unwrap())
                    } else if ds {
                        format!(
                            "(sel == {QUEUE} || sel == {PORT}) ? (int64_t)sp_depth(&sp, sel) : (int64_t)({t} - bases[sel])"
                        )
                    } else {
                        let base = format!("bases[{}]", sel.unwrap());
                        format!("(int64_t)({t} - {base})")
                    };
                    if ds && top_modified {
                        out.push_str(&format!("{ind}  switch (sel) {{ "));
                        for &s in &used {
                            if !is_special(s) {
                                out.push_str(&format!("case {s}: t{s} = top; break; "));
                            }
                        }
                        out.push_str("default: break; }\n");
                    }
                    let fail_seq = block_order.get(fail).copied().unwrap_or(0);
                    out.push_str(&format!("{ind}  if (({depth_expr}) < (int64_t){min_depth}) {{ _pc = {fail_seq}; continue; }}\n"));
                }
            }
            ii += 1;
        }
        max_var = max_var.max(abs.next_var);

        // BranchZero: pop from abs BEFORE flush
        let brz_var = if let Terminator::BranchZero { .. } = &block.terminator {
            let reg = sel.is_some() && !sel.is_some_and(is_special);
            if reg {
                ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                Some(abs.pop().unwrap())
            } else {
                None
            }
        } else {
            None
        };
        max_var = max_var.max(abs.next_var);

        // Flush abs + save-back before terminator
        abs.flush_all_c(&mut out, &format!("{ind}  "));
        let t = if let Some(s) = sel {
            format!("t{s}")
        } else {
            "top".into()
        };
        if sel.is_none() && top_modified {
            out.push_str(&format!("{ind}  switch (sel) {{ "));
            for &s in &used {
                if !is_special(s) {
                    out.push_str(&format!("case {s}: t{s} = top; break; "));
                }
            }
            out.push_str("default: break; }\n");
        }

        // sel-known exit: sync top
        if has_dyn_sel && sel.is_some() && !sel.is_some_and(is_special) {
            out.push_str(&format!("{ind}  top = {t};\n"));
        }

        // Terminator
        match &block.terminator {
            Terminator::Goto(target) => {
                let target_seq = block_order.get(target).copied().unwrap_or(0);
                if !use_match && target_seq == next_seq {
                    out.push_str(&format!("{ind}  _pc = {next_seq};\n"));
                } else {
                    out.push_str(&format!("{ind}  _pc = {target_seq}; continue;\n"));
                }
            }
            Terminator::StackGuard {
                min_depth,
                ok,
                fail,
            } => {
                let sp_active = sel.is_some_and(is_special);
                let depth_expr = if sp_active {
                    format!("(int64_t)sp_depth(&sp, {})", sel.unwrap())
                } else if sel.is_none() && has_special {
                    format!(
                        "(sel == {QUEUE} || sel == {PORT}) ? (int64_t)sp_depth(&sp, sel) : (int64_t)({t} - bases[sel])"
                    )
                } else {
                    let base = match sel {
                        Some(s) => format!("bases[{s}]"),
                        None => "bases[sel]".into(),
                    };
                    format!("(int64_t)({t} - {base})")
                };
                let ok_seq = block_order.get(ok).copied().unwrap_or(0);
                let fail_seq = block_order.get(fail).copied().unwrap_or(0);
                if ok_seq == next_seq {
                    out.push_str(&format!("{ind}  if (({depth_expr}) < (int64_t){min_depth}) {{ _pc = {fail_seq}; continue; }}\n"));
                    out.push_str(&format!("{ind}  _pc = {next_seq};\n"));
                } else {
                    out.push_str(&format!("{ind}  _pc = (({depth_expr}) >= (int64_t){min_depth}) ? {ok_seq} : {fail_seq}; continue;\n"));
                }
            }
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } => {
                let vr = if let Some(v) = brz_var.clone() {
                    v
                } else {
                    let sp_active = sel.is_some_and(is_special);
                    let vr = abs.fresh();
                    max_var = max_var.max(abs.next_var);
                    if sp_active {
                        out.push_str(&format!("{ind}  {vr} = sp_pop(&sp, {});\n", sel.unwrap()));
                    } else if sel.is_none() && has_special {
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  if (sel != {QUEUE} && sel != {PORT}) tops_snapshot[sel] = top;\n"));
                        }
                    } else if sel.is_none() {
                        out.push_str(&format!("{ind}  top--; {vr} = *top;\n"));
                        if bigint {
                            out.push_str(&format!("{ind}  tops_snapshot[sel] = top;\n"));
                        }
                    } else {
                        out.push_str(&format!("{ind}  {t}--; {vr} = *{t};\n"));
                        if bigint {
                            out.push_str(&format!(
                                "{ind}  tops_snapshot[{}] = {t};\n",
                                sel.unwrap()
                            ));
                        }
                    }
                    vr
                };
                // Re-save after pop
                if sel.is_none() {
                    out.push_str(&format!("{ind}  switch (sel) {{ "));
                    for &s in &used {
                        if !is_special(s) {
                            out.push_str(&format!("case {s}: t{s} = top; break; "));
                        }
                    }
                    out.push_str("default: break; }\n");
                }
                let zero_seq = block_order.get(on_zero).copied().unwrap_or(0);
                let nonzero_seq = block_order.get(on_nonzero).copied().unwrap_or(0);
                if is_self_loop && *on_nonzero == block_id {
                    out.push_str(&format!(
                        "{ind}  if ({} ) {{ _pc = {zero_seq}; break; }}\n",
                        if bigint {
                            format!("int_is_zero({vr}, _bm)")
                        } else {
                            format!("{vr} == 0")
                        }
                    ));
                } else if is_self_loop && *on_zero == block_id {
                    out.push_str(&format!(
                        "{ind}  if (!({})) {{ _pc = {nonzero_seq}; break; }}\n",
                        if bigint {
                            format!("int_is_zero({vr}, _bm)")
                        } else {
                            format!("{vr} == 0")
                        }
                    ));
                } else {
                    let zero = if bigint {
                        format!("int_is_zero({vr}, _bm)")
                    } else {
                        format!("{vr} == 0")
                    };
                    out.push_str(&format!(
                        "{ind}  _pc = ({zero}) ? {zero_seq} : {nonzero_seq}; continue;\n"
                    ));
                }
            }
            Terminator::Halt => {
                out.push_str(&format!("{ind}  fflush(stdout);\n"));
                let base = match sel {
                    Some(s) => format!("bases[{s}]"),
                    None => "bases[sel]".into(),
                };
                if bigint {
                    out.push_str(&format!(
                        "{ind}  return ({t} > {base}) ? int_to_i64(*({t}-1), _bm) : 0;\n"
                    ));
                } else {
                    out.push_str(&format!("{ind}  return ({t} > {base}) ? *({t}-1) : 0;\n"));
                }
            }
        }

        // Close inner loop
        if is_self_loop {
            out.push_str(&format!("{ind}  }} /* end inner loop */\n"));
            out.push_str(&format!("{ind}  continue;\n"));
        }
        // Close block
        if use_match {
            out.push_str("    } break;\n");
        } else {
            out.push_str("    }\n");
        }
    }

    if use_match {
        out.push_str("    default: goto done;\n    }\n  }\ndone:\n  return 0;\n}\n\n");
    } else {
        out.push_str("    break;\n  }\n  return 0;\n}\n\n");
    }

    // Insert variable declarations
    let mut var_decl = String::new();
    for i in 0..max_var {
        var_decl.push_str(&format!("  int64_t v{i} = 0;\n"));
    }
    out = out.replacen("  /*VARDECL*/\n", &var_decl, 1);

    out.push_str(C_MAIN);
    out
}

fn emit_promote_live_vars_c(out: &mut String, abs: &Abs, indent: &str) {
    let mut live_vars: Vec<&str> = Vec::new();
    for stack in abs.stacks.values() {
        for value in stack {
            if !live_vars.contains(&value.as_str()) {
                live_vars.push(value);
            }
        }
    }
    if live_vars.is_empty() {
        return;
    }
    out.push_str(&format!("{indent}  if (_bm && !_bm_prev) {{ "));
    for value in live_vars {
        out.push_str(&format!("{value} = promote_val({value}); "));
    }
    out.push_str("}\n");
}

struct Abs {
    stacks: BTreeMap<usize, Vec<String>>,
    active: usize,
    sel_known: bool,
    next_var: usize,
    bigint: bool,
}

impl Abs {
    fn fresh(&mut self) -> String {
        let n = self.next_var;
        self.next_var += 1;
        format!("v{n}")
    }
    fn top_var(&self) -> String {
        if self.sel_known {
            format!("t{}", self.active)
        } else {
            "top".into()
        }
    }
    fn push(&mut self, var: String) {
        self.stacks.entry(self.active).or_default().push(var);
    }
    fn pop(&mut self) -> Option<String> {
        self.stacks.entry(self.active).or_default().pop()
    }
    fn len(&self) -> usize {
        self.stacks.get(&self.active).map_or(0, |v| v.len())
    }
    fn peek(&self) -> Option<&String> {
        self.stacks.get(&self.active).and_then(|v| v.last())
    }
    fn flush_all_c(&mut self, out: &mut String, indent: &str) {
        let active = self.active;
        let active_top = self.top_var();
        for (&s, stack) in self.stacks.iter_mut() {
            if stack.is_empty() {
                continue;
            }
            let t = if s == active {
                active_top.clone()
            } else {
                format!("t{s}")
            };
            for var in stack.drain(..) {
                out.push_str(&format!("{indent}*{t} = {var}; {t}++;\n"));
            }
            if self.bigint {
                out.push_str(&format!("{indent}tops_snapshot[{s}] = {t};\n"));
            }
        }
    }
}

fn ensure_depth_c(out: &mut String, abs: &mut Abs, depth: usize, indent: &str) {
    if abs.len() >= depth {
        return;
    }
    let need = depth - abs.len();
    let t = abs.top_var();
    let active = abs.active;
    let mut loaded = Vec::with_capacity(need);
    for i in 0..need {
        let var = abs.fresh();
        out.push_str(&format!("{indent}{var} = *({t}-{});\n", need - i));
        loaded.push(var);
    }
    out.push_str(&format!("{indent}{t} -= {need};\n"));
    if abs.bigint {
        if abs.sel_known {
            out.push_str(&format!("{indent}tops_snapshot[{active}] = {t};\n"));
        } else {
            out.push_str(&format!("{indent}tops_snapshot[sel] = {t};\n"));
        }
    }
    let stack = abs.stacks.entry(active).or_default();
    let mut existing = std::mem::take(stack);
    stack.extend(loaded);
    stack.append(&mut existing);
}

const C_PRELUDE: &str = r#"/* Generated by compaheuiler — https://github.com/youknowone/aheui-rust
 * This code includes runtime components licensed under AGPL-3.0-or-later.
 * Distributing binaries compiled from this code requires compliance with the AGPL.
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define STORAGE_COUNT 28
#define MAX_STACK 65536
#define QUEUE_CAP 65536
#define PORT_CAP 65536

static inline int64_t wrapping_div(int64_t a, int64_t b) {
    if (a == INT64_MIN && b == -1) return INT64_MIN;
    int64_t q = a / b, r = a % b;
    if (r != 0 && (r < 0) != (b < 0)) q -= 1;
    return q;
}
static inline int64_t wrapping_rem(int64_t a, int64_t b) {
    if (a == INT64_MIN && b == -1) return 0;
    int64_t q = a / b, r = a % b;
    (void)q;
    if (r != 0 && (r < 0) != (b < 0)) r += b;
    return r;
}

typedef struct {
    int64_t queue[QUEUE_CAP];
    size_t q_head;
    size_t q_len;
    int64_t port[PORT_CAP];
    size_t p_len;
    int64_t port_last;
} SpecialStorage;

static inline void sp_init(SpecialStorage* s) {
    s->q_head = 0; s->q_len = 0; s->p_len = 0; s->port_last = 0;
}

static inline void sp_push(SpecialStorage* s, size_t sel, int64_t v) {
    if (sel == 21) {
        s->queue[(s->q_head + s->q_len) % QUEUE_CAP] = v;
        s->q_len++;
    } else {
        s->port_last = v;
        s->port[s->p_len++] = v;
    }
}

static inline int64_t sp_pop(SpecialStorage* s, size_t sel) {
    if (sel == 21) {
        if (s->q_len == 0) return 0;
        int64_t v = s->queue[s->q_head];
        s->q_head = (s->q_head + 1) % QUEUE_CAP;
        s->q_len--;
        return v;
    } else {
        if (s->p_len == 0) return 0;
        return s->port[--s->p_len];
    }
}

static inline size_t sp_depth(SpecialStorage* s, size_t sel) {
    return (sel == 21) ? s->q_len : s->p_len;
}

static inline void sp_dup(SpecialStorage* s, size_t sel) {
    if (sel == 21) {
        if (s->q_len > 0) {
            int64_t v = s->queue[s->q_head];
            /* push_front: move head back */
            s->q_head = (s->q_head + QUEUE_CAP - 1) % QUEUE_CAP;
            s->queue[s->q_head] = v;
            s->q_len++;
        }
    } else {
        s->port[s->p_len] = s->port_last;
        s->p_len++;
    }
}

static inline void sp_swap(SpecialStorage* s, size_t sel) {
    if (sel == 21 && s->q_len >= 2) {
        size_t i0 = s->q_head;
        size_t i1 = (s->q_head + 1) % QUEUE_CAP;
        int64_t tmp = s->queue[i0];
        s->queue[i0] = s->queue[i1];
        s->queue[i1] = tmp;
    } else if (sel == 27 && s->p_len >= 2) {
        int64_t tmp = s->port[s->p_len - 1];
        s->port[s->p_len - 1] = s->port[s->p_len - 2];
        s->port[s->p_len - 2] = tmp;
    }
}

static inline void sp_scan_to_zero(SpecialStorage* s) {
    for (size_t i = 0; i < s->q_len; i++) {
        size_t idx = (s->q_head + i) % QUEUE_CAP;
        if (s->queue[idx] == 0) {
            /* rotate_left(pos+1): element at pos+1 becomes new front, length unchanged */
            s->q_head = (s->q_head + i + 1) % QUEUE_CAP;
            return;
        }
    }
}

static char _outbuf[16384];
static size_t _outpos = 0;
static inline void _flush(void) { if (_outpos > 0) { fwrite(_outbuf, 1, _outpos, stdout); _outpos = 0; } }
static inline void _emit(char c) { _outbuf[_outpos++] = c; if (_outpos >= 16384) _flush(); }

static inline void write_num(int64_t n) {
    if (n < 0) {
        _emit('-');
        uint64_t u = (uint64_t)(-(n + 1)) + 1;
        char buf[20]; int i = 20;
        do { buf[--i] = '0' + (u % 10); u /= 10; } while (u > 0);
        for (; i < 20; i++) _emit(buf[i]);
    } else {
        uint64_t u = (uint64_t)n;
        char buf[20]; int i = 20;
        do { buf[--i] = '0' + (u % 10); u /= 10; } while (u > 0);
        for (; i < 20; i++) _emit(buf[i]);
    }
}

static inline void write_char(int64_t v) {
    uint32_t c = (uint32_t)v;
    if (c <= 0x7F) { _emit((char)c); }
    else if (c <= 0x7FF) { _emit(0xC0 | (c >> 6)); _emit(0x80 | (c & 0x3F)); }
    else if (c <= 0xFFFF) { _emit(0xE0 | (c >> 12)); _emit(0x80 | ((c >> 6) & 0x3F)); _emit(0x80 | (c & 0x3F)); }
    else if (c <= 0x10FFFF) { _emit(0xF0 | (c >> 18)); _emit(0x80 | ((c >> 12) & 0x3F)); _emit(0x80 | ((c >> 6) & 0x3F)); _emit(0x80 | (c & 0x3F)); }
}

static inline int64_t read_num(void) {
    char buf[64];
    if (fgets(buf, sizeof(buf), stdin) == NULL) return 0;
    return (int64_t)atoll(buf);
}

static inline int64_t read_char_(void) {
    int b = getchar();
    if (b == EOF) return -1;
    if (b < 0x80) return (int64_t)b;
    int n; int32_t val;
    if ((b >> 5) == 6) { n = 1; val = b & 0x1F; }
    else if ((b >> 4) == 14) { n = 2; val = b & 0x0F; }
    else if ((b >> 3) == 30) { n = 3; val = b & 0x07; }
    else return -1;
    for (int i = 0; i < n; i++) {
        int c = getchar();
        if (c == EOF) return -1;
        val = (val << 6) | (c & 0x3F);
    }
    return (int64_t)val;
}

"#;

const C_BIGINT_PRELUDE: &str = include_str!("c_bigint_runtime.c");

const C_MAIN: &str = r#"
int64_t compaheuiler_c_entry(void) {
    int64_t* data[STORAGE_COUNT];
    int64_t* bases[STORAGE_COUNT];
    int32_t lengths[STORAGE_COUNT];
    for (int i = 0; i < STORAGE_COUNT; i++) {
        data[i] = (int64_t*)calloc(MAX_STACK, sizeof(int64_t));
        bases[i] = data[i];
        lengths[i] = 0;
    }
    int64_t result = aheui_main(bases, lengths);
    _flush();
    fflush(stdout);
    for (int i = 0; i < STORAGE_COUNT; i++) free(data[i]);
    return result;
}

#ifndef COMPAHEUILER_RUST_BIGINT
int main(void) { return (int)compaheuiler_c_entry(); }
#endif
"#;
