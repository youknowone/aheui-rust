//! Generate Rust code from an Aheui CFG using the Stackifier algorithm.
//!
//! All control flow expressed as loop/break/continue — no match dispatch.
//! Compile with: rustc -C opt-level=3 -o program output.rs

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::cfg::*;

pub fn compile_to_rs(source: &str) -> String {
    let cfg = crate::compile_to_cfg_aot(source);
    generate_rs(&cfg)
}

// ── Abstract stack ──────────────────────────────────────────────────

struct Abs {
    stacks: HashMap<usize, Vec<String>>,
    active: usize,
    sel_known: bool,
    next_var: usize,
}

impl Abs {
    fn new(sel: Option<usize>) -> Self {
        Abs { stacks: HashMap::new(), active: sel.unwrap_or(0), sel_known: sel.is_some(), next_var: 0 }
    }
    fn from_snap(stacks: HashMap<usize, Vec<String>>, nv: usize, sel: Option<usize>) -> Self {
        Abs { stacks, active: sel.unwrap_or(0), sel_known: sel.is_some(), next_var: nv }
    }
    fn fresh(&mut self) -> String { let n = self.next_var; self.next_var += 1; format!("v{n}") }
    fn top_var(&self) -> String { if self.sel_known { format!("t{}", self.active) } else { "top".into() } }
    fn push(&mut self, var: String) { self.stacks.entry(self.active).or_default().push(var); }
    fn pop(&mut self) -> Option<String> { self.stacks.entry(self.active).or_default().pop() }
    fn len(&self) -> usize { self.stacks.get(&self.active).map_or(0, |v| v.len()) }
    fn peek(&self) -> Option<&String> { self.stacks.get(&self.active).and_then(|v| v.last()) }
    fn snapshot(&self) -> (HashMap<usize, Vec<String>>, usize) { (self.stacks.clone(), self.next_var) }
    fn flush_all(&mut self, out: &mut String, indent: &str) {
        let active = self.active;
        let active_top = self.top_var();
        let mut keys: Vec<usize> = self.stacks.keys().copied().collect();
        keys.sort();
        for s in keys {
            let stack = self.stacks.get_mut(&s).unwrap();
            if stack.is_empty() { continue; }
            let t = if s == active { active_top.clone() } else { format!("t{s}") };
            for var in stack.drain(..) {
                out.push_str(&format!("{indent}*{t} = {var}; {t} = {t}.add(1);\n"));
            }
        }
    }
}

fn ensure_depth(out: &mut String, abs: &mut Abs, depth: usize, indent: &str) {
    if abs.len() >= depth { return; }
    let need = depth - abs.len();
    let t = abs.top_var();
    let active = abs.active;
    let mut loaded = Vec::with_capacity(need);
    for i in 0..need {
        let var = abs.fresh();
        out.push_str(&format!("{indent}{var} = *{t}.sub({});\n", need - i));
        loaded.push(var);
    }
    out.push_str(&format!("{indent}{t} = {t}.sub({need});\n"));
    let stack = abs.stacks.entry(active).or_default();
    let existing: Vec<_> = stack.drain(..).collect();
    stack.extend(loaded);
    stack.extend(existing);
}

// ── Stackifier scope analysis ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeKind { Loop, Forward }

/// A scope [open_rpo, close_rpo) with a label.
#[derive(Debug, Clone)]
struct Scope {
    kind: ScopeKind,
    label: BlockId, // loop header or forward target
    open: usize,    // RPO index where scope opens
    close: usize,   // RPO index where scope closes
}

fn compute_scopes(cfg: &Cfg, rpo: &[BlockId]) -> Vec<Scope> {
    let n = rpo.len();
    let mut rpo_pos: HashMap<BlockId, usize> = HashMap::new();
    for (i, &bid) in rpo.iter().enumerate() { rpo_pos.insert(bid, i); }

    let mut scopes = Vec::new();

    // Loop scopes: back edge target is loop header.
    let mut loop_extent: HashMap<BlockId, usize> = HashMap::new(); // header → max back-edge source RPO
    for (idx, &bid) in rpo.iter().enumerate() {
        for &succ in &cfg.block(bid).all_successors() {
            if let Some(&succ_pos) = rpo_pos.get(&succ) {
                if succ_pos <= idx { // back edge
                    let e = loop_extent.entry(succ).or_insert(idx);
                    *e = (*e).max(idx);
                }
            }
        }
    }
    // Extend loop extents to ensure proper nesting: if loop A contains
    // loop B's header, A must extend at least to B's extent.
    let mut changed = true;
    while changed {
        changed = false;
        let headers: Vec<BlockId> = loop_extent.keys().copied().collect();
        for &h1 in &headers {
            for &h2 in &headers {
                if h1 == h2 { continue; }
                let h1_pos = rpo_pos[&h1];
                let h2_pos = rpo_pos[&h2];
                // If h1 contains h2 (h1 opens before h2), extend h1 to cover h2's extent.
                if h1_pos < h2_pos && loop_extent[&h1] >= h2_pos && loop_extent[&h1] < loop_extent[&h2] {
                    *loop_extent.get_mut(&h1).unwrap() = loop_extent[&h2];
                    changed = true;
                }
            }
        }
    }

    for (&header, &last_src) in &loop_extent {
        scopes.push(Scope {
            kind: ScopeKind::Loop,
            label: header,
            open: rpo_pos[&header],
            close: last_src + 1,
        });
    }

    // Forward block scopes: non-fallthrough, non-back-edge jumps.
    let mut fwd_sources: HashMap<BlockId, usize> = HashMap::new(); // target → earliest source RPO
    for (idx, &bid) in rpo.iter().enumerate() {
        let next_rpo = if idx + 1 < n { Some(rpo[idx + 1]) } else { None };
        for &succ in &cfg.block(bid).all_successors() {
            if let Some(&succ_pos) = rpo_pos.get(&succ) {
                // Forward skip: not fallthrough, not back edge
                if succ_pos > idx && Some(succ) != next_rpo && succ_pos > idx + 1 {
                    let e = fwd_sources.entry(succ).or_insert(idx);
                    *e = (*e).min(idx);
                }
            }
        }
    }
    for (&target, &earliest_src) in &fwd_sources {
        scopes.push(Scope {
            kind: ScopeKind::Forward,
            label: target,
            open: earliest_src,
            close: rpo_pos[&target],
        });
    }

    // Ensure proper nesting: no scope may partially overlap another.
    // If scope A starts before B and ends inside B, extend A to end after B.
    let mut changed2 = true;
    while changed2 {
        changed2 = false;
        for i in 0..scopes.len() {
            for j in 0..scopes.len() {
                if i == j { continue; }
                let (a_open, a_close) = (scopes[i].open, scopes[i].close);
                let (b_open, b_close) = (scopes[j].open, scopes[j].close);
                // A starts before B, A ends inside B → extend A to cover B.
                if a_open <= b_open && a_close > b_open && a_close < b_close {
                    scopes[i].close = b_close;
                    changed2 = true;
                }
            }
        }
    }

    // Sort: by open position, then wider scopes first (larger close).
    scopes.sort_by(|a, b| a.open.cmp(&b.open).then(b.close.cmp(&a.close)));
    scopes
}

// ── Code generation ─────────────────────────────────────────────────

fn generate_rs(cfg: &Cfg) -> String {
    let mut out = String::with_capacity(16384);
    out.push_str(PRELUDE);

    let states = crate::cfg_optimize::analyze_stack_depths(cfg);
    let rpo = cfg.reverse_postorder();
    let preds = cfg.predecessors();

    let mut used: BTreeSet<usize> = BTreeSet::new();
    used.insert(0);
    for b in &cfg.blocks {
        for i in &b.instructions {
            match i { Inst::Sel(s) | Inst::Mov(s) => { used.insert(*s); } _ => {} }
        }
    }

    let mut rpo_pos: HashMap<BlockId, usize> = HashMap::new();
    for (i, &bid) in rpo.iter().enumerate() { rpo_pos.insert(bid, i); }

    // Compute structured scopes.
    let scopes = compute_scopes(cfg, &rpo);

    // Debug: dump scopes
    for s in &scopes {
        let kind = if s.kind == ScopeKind::Loop { "loop" } else { "fwd" };
        eprintln!("  scope {kind} l{}: open={} close={}", s.label, s.open, s.close);
    }

    // Pre-compute: which scopes open/close at each RPO index.
    let mut opens_at: HashMap<usize, Vec<usize>> = HashMap::new(); // rpo_idx → scope indices
    let mut closes_at: HashMap<usize, Vec<usize>> = HashMap::new();
    for (si, scope) in scopes.iter().enumerate() {
        opens_at.entry(scope.open).or_default().push(si);
        closes_at.entry(scope.close).or_default().push(si);
    }

    // Identify loop headers and forward targets for jump emission.
    let loop_headers: HashSet<BlockId> = scopes.iter()
        .filter(|s| s.kind == ScopeKind::Loop).map(|s| s.label).collect();
    let fwd_targets: HashSet<BlockId> = scopes.iter()
        .filter(|s| s.kind == ScopeKind::Forward).map(|s| s.label).collect();

    // Can-inherit analysis.
    let can_inherit = {
        let mut ci = vec![false; cfg.num_blocks()];
        for block_id in 0..cfg.num_blocks() {
            let bp = &preds[block_id];
            if bp.len() == 1 {
                let pred = bp[0];
                let ok = match &cfg.block(pred).terminator {
                    Terminator::Goto(_) => true,
                    Terminator::StackGuard { ok, .. } if *ok == block_id as BlockId => true,
                    Terminator::BranchZero { .. } => true,
                    _ => false,
                };
                if ok && pred < block_id as BlockId { ci[block_id] = true; }
            }
        }
        ci
    };

    // Function header.
    out.push_str("pub unsafe fn aheui_main(bases: &mut [*mut i64; 28], lengths: &mut [i32; 28], w: &mut impl std::io::Write) -> i64 {\n");
    for &s in &used {
        out.push_str(&format!("  let mut t{s}: *mut i64 = bases[{s}].add(lengths[{s}] as usize);\n"));
    }
    out.push_str("  let mut top: *mut i64 = t0;\n");
    out.push_str("  let mut sel: usize = 0;\n");
    out.push_str("  /*VARDECL*/\n");

    let mut max_var: usize = 0;
    let mut block_exit: HashMap<BlockId, (HashMap<usize, Vec<String>>, usize)> = HashMap::new();
    let mut depth: usize = 1; // indentation depth

    let indent = |d: usize| -> String { "  ".repeat(d) };

    for (order_idx, &block_id) in rpo.iter().enumerate() {
        // Close scopes ending at this position (in reverse order — innermost first).
        if let Some(closing) = closes_at.get(&order_idx) {
            for &si in closing.iter().rev() {
                depth -= 1;
                let s = &scopes[si];
                match s.kind {
                    ScopeKind::Loop => out.push_str(&format!("{}}} // end loop l{}\n", indent(depth), s.label)),
                    ScopeKind::Forward => out.push_str(&format!("{}}} // end fwd f{}\n", indent(depth), s.label)),
                }
            }
        }

        // Open scopes starting at this position.
        if let Some(opening) = opens_at.get(&order_idx) {
            for &si in opening {
                let s = &scopes[si];
                match s.kind {
                    ScopeKind::Loop => out.push_str(&format!("{}'l{}: loop {{\n", indent(depth), s.label)),
                    ScopeKind::Forward => out.push_str(&format!("{}'f{}: {{\n", indent(depth), s.label)),
                }
                depth += 1;
            }
        }

        let ind = indent(depth);
        let block = cfg.block(block_id);
        let entry_sel: Option<usize> = states.get(block_id as usize).and_then(|s| s.selected);
        let mut sel = entry_sel;

        out.push_str(&format!("{ind}// B{block_id}\n"));

        let mut abs = if can_inherit[block_id as usize] {
            let pred = preds[block_id as usize][0];
            if let Some((stacks, nv)) = block_exit.remove(&pred) {
                Abs::from_snap(stacks, nv, entry_sel)
            } else { Abs::new(entry_sel) }
        } else {
            if let Some(s) = sel {
                out.push_str(&format!("{ind}top = t{s}; sel = {s};\n"));
            }
            Abs::new(entry_sel)
        };

        // Emit instructions.
        for inst in &block.instructions {
            let t = abs.top_var();
            match inst {
                Inst::Push(v) => {
                    let var = abs.fresh();
                    out.push_str(&format!("{ind}{var} = {v};\n"));
                    abs.push(var);
                }
                Inst::Pop => {
                    if abs.pop().is_none() { out.push_str(&format!("{ind}{t} = {t}.sub(1);\n")); }
                }
                Inst::Dup => {
                    ensure_depth(&mut out, &mut abs, 1, &ind);
                    let top_val = abs.peek().unwrap().clone();
                    let var = abs.fresh();
                    out.push_str(&format!("{ind}{var} = {top_val};\n"));
                    abs.push(var);
                }
                Inst::Swap => {
                    ensure_depth(&mut out, &mut abs, 2, &ind);
                    let len = abs.len();
                    let active = abs.active;
                    abs.stacks.get_mut(&active).unwrap().swap(len - 1, len - 2);
                }
                Inst::BinOp(kind) => {
                    ensure_depth(&mut out, &mut abs, 2, &ind);
                    let r1 = abs.pop().unwrap();
                    let r2 = abs.pop().unwrap();
                    let var = abs.fresh();
                    let expr = match kind {
                        BinOpKind::Add => format!("{r2}.wrapping_add({r1})"),
                        BinOpKind::Sub => format!("{r2}.wrapping_sub({r1})"),
                        BinOpKind::Mul => format!("{r2}.wrapping_mul({r1})"),
                        BinOpKind::Div => format!("if {r1} != 0 {{ {r2}.wrapping_div({r1}) }} else {{ 0 }}"),
                        BinOpKind::Mod => format!("if {r1} != 0 {{ {r2}.wrapping_rem({r1}) }} else {{ 0 }}"),
                        BinOpKind::Cmp => format!("if {r2} >= {r1} {{ 1 }} else {{ 0 }}"),
                    };
                    out.push_str(&format!("{ind}{var} = {expr};\n"));
                    abs.push(var);
                }
                Inst::Sel(new_sel) => {
                    if sel.is_none() {
                        abs.flush_all(&mut out, &ind);
                        out.push_str(&format!("{ind}match sel {{ "));
                        for &s in &used { out.push_str(&format!("{s} => t{s} = top, ")); }
                        out.push_str("_ => {} }\n");
                    }
                    abs.active = *new_sel;
                    abs.sel_known = true;
                    sel = Some(*new_sel);
                }
                Inst::Mov(target) => {
                    ensure_depth(&mut out, &mut abs, 1, &ind);
                    let val = abs.pop().unwrap();
                    out.push_str(&format!("{ind}*t{target} = {val}; t{target} = t{target}.add(1);\n"));
                }
                Inst::PopNum => {
                    ensure_depth(&mut out, &mut abs, 1, &ind);
                    let val = abs.pop().unwrap();
                    out.push_str(&format!("{ind}write_num(w, {val});\n"));
                }
                Inst::PopChar => {
                    ensure_depth(&mut out, &mut abs, 1, &ind);
                    let val = abs.pop().unwrap();
                    out.push_str(&format!("{ind}write_char(w, {val} as u32);\n"));
                }
                Inst::PushNum => {
                    abs.flush_all(&mut out, &ind);
                    let var = abs.fresh();
                    out.push_str(&format!("{ind}{var} = read_num();\n"));
                    abs.push(var);
                }
                Inst::PushChar => {
                    abs.flush_all(&mut out, &ind);
                    let var = abs.fresh();
                    out.push_str(&format!("{ind}{var} = read_char() as i64;\n"));
                    abs.push(var);
                }
            }
        }

        if abs.next_var > max_var { max_var = abs.next_var; }
        let t = abs.top_var();
        let next_rpo = rpo.get(order_idx + 1).copied();

        // ── Structured terminator ───────────────────────────────────
        // Jump types:
        //   fallthrough = target is next RPO block → no code
        //   back edge   = target is loop header at/before us → continue 'lH
        //   forward     = target is a forward-block target → break 'fT

        // Check if any scope closes between order_idx and target_rpo_idx.
        let scope_closes_before = |target_rpo: usize| -> bool {
            for s in &scopes {
                if s.close > order_idx && s.close <= target_rpo {
                    return true;
                }
            }
            false
        };

        // Find the innermost loop or forward scope that contains order_idx
        // and whose close >= target_rpo. Break out of that scope.
        let find_break_label = |target_rpo: usize| -> Option<(ScopeKind, BlockId)> {
            // Find the innermost scope that:
            //   - contains this block (open <= order_idx)
            //   - whose break lands at or after target (close >= target_rpo)
            // Among those, pick the OUTERMOST one whose close == target_rpo,
            // so we break past all nested scopes.
            // If no exact match, pick the smallest scope covering the target.
            let mut exact: Option<(usize, ScopeKind, BlockId)> = None; // widest exact
            let mut wider: Option<(usize, ScopeKind, BlockId)> = None; // smallest wider
            for s in &scopes {
                if s.open <= order_idx && s.close >= target_rpo {
                    let size = s.close - s.open;
                    if s.close == target_rpo {
                        // Exact: break lands right at target. Pick widest.
                        if exact.is_none() || size > exact.unwrap().0 {
                            exact = Some((size, s.kind, s.label));
                        }
                    } else {
                        // Wider: break lands after target. Pick smallest.
                        if wider.is_none() || size < wider.unwrap().0 {
                            wider = Some((size, s.kind, s.label));
                        }
                    }
                }
            }
            exact.or(wider).map(|(_, k, l)| (k, l))
        };

        let emit_jump = |out: &mut String, target: BlockId, ind: &str| {
            if let Some(&tp) = rpo_pos.get(&target) {
                if tp <= order_idx {
                    // Back edge → continue.
                    out.push_str(&format!("{ind}continue 'l{target};\n"));
                    return;
                }
                // Forward: check if we can fallthrough.
                if Some(target) == next_rpo && !scope_closes_before(tp) {
                    return; // true fallthrough — no scope boundary
                }
                // Find the outermost scope that:
                //   1. Contains us (open <= order_idx)
                //   2. Closes at exactly the target (close == tp)
                // This ensures we exit all nested scopes to reach the target.
                // If no exact match: break the outermost scope closing >= tp.
                let mut best: Option<&Scope> = None;
                for s in &scopes {
                    if s.open > order_idx { continue; }
                    if s.close < tp { continue; }
                    // s contains us and its break lands at or after target.
                    // Prefer exact close == tp, then widest scope.
                    match best {
                        None => best = Some(s),
                        Some(b) => {
                            if s.close == tp && b.close != tp {
                                best = Some(s); // prefer exact
                            } else if s.close == tp && b.close == tp {
                                // Both exact: prefer wider (earlier open)
                                if s.open < b.open { best = Some(s); }
                            } else if b.close > tp && s.close <= b.close {
                                // Both overshoot: prefer tighter
                                if s.close < b.close || (s.close == b.close && s.open > b.open) {
                                    best = Some(s);
                                }
                            }
                        }
                    }
                }
                if let Some(s) = best {
                    let prefix = if s.kind == ScopeKind::Loop { "l" } else { "f" };
                    out.push_str(&format!("{ind}break '{prefix}{};\n", s.label));
                    return;
                }
            }
            // Unreachable or Halt — shouldn't happen.
        };

        match &block.terminator {
            Terminator::Goto(target) => {
                if can_inherit[*target as usize] {
                    block_exit.insert(block_id, abs.snapshot());
                } else {
                    abs.flush_all(&mut out, &ind);
                }
                emit_jump(&mut out, *target, &ind);
            }
            Terminator::StackGuard { min_depth, ok, fail } => {
                abs.flush_all(&mut out, &ind);
                let base = match sel {
                    Some(s) => format!("bases[{s}]"),
                    None => "bases[sel]".into(),
                };
                // Fail path first (less common).
                out.push_str(&format!("{ind}if (({t} as usize).wrapping_sub({base} as usize) / 8) < {min_depth} {{\n"));
                emit_jump(&mut out, *fail, &format!("{ind}  "));
                out.push_str(&format!("{ind}}}\n"));
                // Ok is fallthrough or jump.
                emit_jump(&mut out, *ok, &ind);
            }
            Terminator::BranchZero { on_zero, on_nonzero } => {
                ensure_depth(&mut out, &mut abs, 1, &ind);
                let val = abs.pop().unwrap();
                abs.flush_all(&mut out, &ind);
                out.push_str(&format!("{ind}if {val} == 0 {{\n"));
                emit_jump(&mut out, *on_zero, &format!("{ind}  "));
                out.push_str(&format!("{ind}}}\n"));
                emit_jump(&mut out, *on_nonzero, &ind);
            }
            Terminator::Halt => {
                abs.flush_all(&mut out, &ind);
                out.push_str(&format!(
                    "{ind}w.flush().ok();\n{ind}let _t = {t}; let _b = {};\n{ind}return if _t > _b {{ *_t.sub(1) }} else {{ 0 }};\n",
                    match sel { Some(s) => format!("bases[{s}]"), None => "bases[sel]".into() },
                ));
            }
        }
    }

    // Close any remaining scopes.
    while depth > 1 { depth -= 1; out.push_str(&format!("{}}}\n", indent(depth))); }
    out.push_str("}\n\n");

    // Variable declarations.
    let var_decl = if max_var > 0 {
        let mut d = String::new();
        for i in 0..max_var { d.push_str(&format!("  let mut v{i}: i64 = 0;\n")); }
        d
    } else { String::new() };
    out = out.replacen("  /*VARDECL*/\n", &var_decl, 1);

    out.push_str(MAIN_FN);
    out
}

const PRELUDE: &str = r#"#![allow(unused, unused_assignments, unused_mut)]
use std::io::{Read, Write};

#[inline(always)]
fn write_num(w: &mut impl Write, v: i64) {
    let mut buf = [0u8; 20];
    let mut n = if v < 0 { w.write_all(b"-").ok(); (v as u64).wrapping_neg() } else { v as u64 };
    let mut i = 20;
    loop { i -= 1; buf[i] = b'0' + (n % 10) as u8; n /= 10; if n == 0 { break; } }
    w.write_all(&buf[i..]).ok();
}

#[inline(always)]
fn write_char(w: &mut impl Write, c: u32) {
    if c <= 0x7F { w.write_all(&[c as u8]).ok(); return; }
    let mut buf = [0u8; 4];
    if let Some(ch) = char::from_u32(c) {
        let s = ch.encode_utf8(&mut buf);
        w.write_all(s.as_bytes()).ok();
    }
}

fn read_num() -> i64 {
    let mut s = String::new();
    std::io::stdin().read_line(&mut s).ok();
    s.trim().parse().unwrap_or(0)
}

fn read_char() -> i32 {
    let mut buf = [0u8; 4];
    match std::io::stdin().read(&mut buf[..1]) {
        Ok(0) | Err(_) => -1,
        Ok(_) => {
            let b = buf[0];
            if b < 0x80 { return b as i32; }
            let (n, mut val) = if b >> 5 == 6 { (1, (b & 0x1F) as i32) }
                else if b >> 4 == 14 { (2, (b & 0x0F) as i32) }
                else if b >> 3 == 30 { (3, (b & 0x07) as i32) }
                else { return -1 };
            for _ in 0..n {
                if std::io::stdin().read(&mut buf[..1]).unwrap_or(0) == 0 { return -1; }
                val = (val << 6) | (buf[0] & 0x3F) as i32;
            }
            val
        }
    }
}

const STORAGE_COUNT: usize = 28;
const MAX_STACK: usize = 65536;

"#;

const MAIN_FN: &str = r#"
fn main() {
    let mut data: Vec<Vec<i64>> = (0..STORAGE_COUNT).map(|_| vec![0i64; MAX_STACK]).collect();
    let mut bases: [*mut i64; 28] = {
        let mut b = [std::ptr::null_mut(); 28];
        for i in 0..28 { b[i] = data[i].as_mut_ptr(); }
        b
    };
    let mut lengths = [0i32; 28];
    let stdout = std::io::stdout();
    let mut w = std::io::BufWriter::with_capacity(65536, stdout.lock());
    let result = unsafe { aheui_main(&mut bases, &mut lengths, &mut w) };
    w.flush().ok();
    std::process::exit(result as i32);
}
"#;
