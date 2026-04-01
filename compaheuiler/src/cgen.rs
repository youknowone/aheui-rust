//! Generate C99 code from an Aheui CFG using the same optimization pipeline as rgen.
//!
//! Compile with: cc -O2 -o program output.c

use ahsembler::cfg::*;
use std::collections::{BTreeSet, HashMap};

const QUEUE: usize = 21;
const PORT: usize = 27;
fn is_special(s: usize) -> bool {
    s == QUEUE || s == PORT
}

pub fn compile_to_c(source: &str) -> String {
    let mut cfg = ahsembler::compile_to_cfg_aot(source);
    // Same optimization pipeline as compile_to_rs_inner
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
    generate_c_dispatch(&cfg)
}

fn generate_c_dispatch(cfg: &Cfg) -> String {
    let mut out = String::with_capacity(32768);
    out.push_str(C_PRELUDE);

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
    let has_dyn_sel = live_blocks.iter().any(|&bid| {
        states
            .get(bid as usize)
            .map_or(false, |s| s.selected.is_none())
    });

    // Function signature
    out.push_str("static int64_t aheui_main(int64_t** bases, int32_t* lengths) {\n");
    out.push_str("  SpecialStorage sp;\n  sp_init(&sp);\n");
    for &s in &used {
        out.push_str(&format!("  int64_t* t{s} = bases[{s}] + lengths[{s}];\n"));
    }
    out.push_str("  int64_t* top = t0;\n  size_t sel = 0;\n  /*VARDECL*/\n");
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
            stacks: HashMap::new(),
            active: entry_sel.unwrap_or(0),
            sel_known: entry_sel.is_some(),
            next_var: max_var,
        };
        let insts = &block.instructions;
        let mut ii = 0;
        let mut top_modified = false;

        let op_str_fn = |kind: &BinOpKind, a: &str, b: &str| -> String {
            match kind {
                BinOpKind::Add => format!("((int64_t)((uint64_t)({a}) + (uint64_t)({b})))"),
                BinOpKind::Sub => format!("((int64_t)((uint64_t)({a}) - (uint64_t)({b})))"),
                BinOpKind::Mul => format!("((int64_t)((uint64_t)({a}) * (uint64_t)({b})))"),
                BinOpKind::Div => format!("(({b}) != 0 ? wrapping_div({a}, {b}) : 0)"),
                BinOpKind::Mod => format!("(({b}) != 0 ? wrapping_rem({a}, {b}) : 0)"),
                BinOpKind::Cmp => format!("(({a}) >= ({b}) ? 1 : 0)"),
            }
        };

        while ii < insts.len() {
            let inst = &insts[ii];
            let sp_known = sel.is_some_and(is_special);
            let ds = sel.is_none();
            let reg = sel.is_some() && !sp_known;
            match inst {
                Inst::Push(v) => {
                    if sp_known {
                        out.push_str(&format!(
                            "{ind}  sp_push(&sp, {}, (int64_t){v}LL);\n",
                            sel.unwrap()
                        ));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, (int64_t){v}LL); }} else {{ *top = (int64_t){v}LL; top++; }}\n"));
                    } else {
                        let var = abs.fresh();
                        out.push_str(&format!("{ind}  {var} = (int64_t){v}LL;\n"));
                        abs.push(var);
                    }
                }
                Inst::Pop => {
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_pop(&sp, {});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_pop(&sp, sel); }} else {{ top--; }}\n"));
                    } else if abs.pop().is_none() {
                        let t = abs.top_var();
                        out.push_str(&format!("{ind}  {t}--;\n"));
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
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 2, &format!("{ind}  "));
                        let r1 = abs.pop().unwrap();
                        let r2 = abs.pop().unwrap();
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = {};\n", op_str_fn(kind, &r2, &r1)));
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
                        }
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        if is_special(*target) {
                            out.push_str(&format!("{ind}  sp_push(&sp, {target}, {vr});\n"));
                        } else {
                            out.push_str(&format!("{ind}  *t{target} = {vr}; t{target}++;\n"));
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
                        out.push_str(&format!("{ind}  write_num({vr});\n"));
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        out.push_str(&format!("{ind}  write_num({vr});\n"));
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let vr = abs.pop().unwrap();
                        out.push_str(&format!("{ind}  write_num({vr});\n"));
                    }
                }
                Inst::PopChar => {
                    if sp_known {
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  {vr} = sp_pop(&sp, {});\n", sel.unwrap()));
                        out.push_str(&format!("{ind}  write_char({vr});\n"));
                    } else if ds {
                        top_modified = true;
                        let vr = abs.fresh();
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ {vr} = sp_pop(&sp, sel); }} else {{ top--; {vr} = *top; }}\n"));
                        out.push_str(&format!("{ind}  write_char({vr});\n"));
                    } else {
                        ensure_depth_c(&mut out, &mut abs, 1, &format!("{ind}  "));
                        let vr = abs.pop().unwrap();
                        out.push_str(&format!("{ind}  write_char({vr});\n"));
                    }
                }
                Inst::PushNum => {
                    if reg {
                        abs.flush_all_c(&mut out, &format!("{ind}  "));
                    }
                    let vr = abs.fresh();
                    out.push_str(&format!(
                        "{ind}  fflush(stdout);\n{ind}  {vr} = read_num();\n"
                    ));
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_push(&sp, {}, {vr});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, {vr}); }} else {{ *top = {vr}; top++; }}\n"));
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
                        "{ind}  fflush(stdout);\n{ind}  {vr} = read_char_();\n"
                    ));
                    if sp_known {
                        out.push_str(&format!("{ind}  sp_push(&sp, {}, {vr});\n", sel.unwrap()));
                    } else if ds {
                        top_modified = true;
                        out.push_str(&format!("{ind}  if (sel == {QUEUE} || sel == {PORT}) {{ sp_push(&sp, sel, {vr}); }} else {{ *top = {vr}; top++; }}\n"));
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
                    } else if sel.is_none() {
                        out.push_str(&format!("{ind}  top--; {vr} = *top;\n"));
                    } else {
                        out.push_str(&format!("{ind}  {t}--; {vr} = *{t};\n"));
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
                        "{ind}  if ({vr} == 0) {{ _pc = {zero_seq}; break; }}\n"
                    ));
                } else if is_self_loop && *on_zero == block_id {
                    out.push_str(&format!(
                        "{ind}  if ({vr} != 0) {{ _pc = {nonzero_seq}; break; }}\n"
                    ));
                } else {
                    out.push_str(&format!(
                        "{ind}  _pc = ({vr} == 0) ? {zero_seq} : {nonzero_seq}; continue;\n"
                    ));
                }
            }
            Terminator::Halt => {
                out.push_str(&format!("{ind}  fflush(stdout);\n"));
                let base = match sel {
                    Some(s) => format!("bases[{s}]"),
                    None => "bases[sel]".into(),
                };
                out.push_str(&format!("{ind}  return ({t} > {base}) ? *({t}-1) : 0;\n"));
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

struct Abs {
    stacks: HashMap<usize, Vec<String>>,
    active: usize,
    sel_known: bool,
    next_var: usize,
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
        let mut keys: Vec<usize> = self.stacks.keys().copied().collect();
        keys.sort();
        for s in keys {
            let stack = self.stacks.get_mut(&s).unwrap();
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
    /* C99 division truncates toward zero, which matches wrapping_div for most cases.
       Special case: INT64_MIN / -1 would overflow. */
    if (a == INT64_MIN && b == -1) return INT64_MIN;
    return a / b;
}
static inline int64_t wrapping_rem(int64_t a, int64_t b) {
    if (a == INT64_MIN && b == -1) return 0;
    return a % b;
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

const C_MAIN: &str = r#"
int main(void) {
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
    return (int)result;
}
"#;
