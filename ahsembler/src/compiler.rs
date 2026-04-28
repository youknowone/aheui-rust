use std::{
    collections::{HashMap, HashSet, VecDeque},
    io::{self, Write},
};

use crate::consts::*;

/// A compiled Aheui program: linear sequence of opcodes with operands and labels.
pub struct Program {
    pub opcodes: Vec<u8>,
    pub values: Vec<i32>,
    pub labels: HashMap<i32, usize>,
    pub size: usize,
}

impl Program {
    #[inline(always)]
    pub fn get_op(&self, pc: usize) -> u8 {
        self.opcodes[pc]
    }

    #[inline(always)]
    pub fn get_operand(&self, pc: usize) -> i32 {
        self.values[pc]
    }

    #[inline(always)]
    pub fn get_req_size(&self, pc: usize) -> usize {
        OP_REQSIZE[self.opcodes[pc] as usize]
    }

    /// rpaheui/aheui/aheui.py:175-189 `Program.get_label` resolves a label
    /// id through the `labels` dict on every jump.  After
    /// `resolve_jump_targets` rewrites `values[pc]` for jump ops to point
    /// at the target PC directly the lookup collapses to a single array
    /// load — RPython's `@jit.elidable` plus immutable `labels[**]`
    /// achieves the same effect at JIT time, but the naive interpreter
    /// path needs the rewrite to avoid the per-jump `HashMap` probe.
    #[inline(always)]
    pub fn get_label(&self, pc: usize) -> usize {
        self.values[pc] as usize
    }

    /// Rewrite `values[pc]` for every jump op so it carries the target PC
    /// instead of a label id.  Idempotent — once resolved, target PCs are
    /// already valid PCs (in `0..size`) and re-running the pass keeps
    /// them unchanged.
    pub fn resolve_jump_targets(&mut self) {
        for pc in 0..self.size {
            if is_jump_op(self.opcodes[pc]) {
                let label_id = self.values[pc];
                if let Some(&target_pc) = self.labels.get(&label_id) {
                    self.values[pc] = target_pc as i32;
                }
            }
        }
    }
}

/// 2D representation of an Aheui program.
struct PrimitiveProgram {
    pane: HashMap<(i32, i32), char>,
    max_row: i32,
    max_col: i32,
}

impl PrimitiveProgram {
    fn new(text: &str) -> Self {
        let mut pane = HashMap::new();
        let mut pc_row: i32 = 0;
        let mut pc_col: i32 = 0;
        let mut max_col: i32 = 0;

        for ch in text.chars() {
            if ch == '\n' {
                pc_row += 1;
                max_col = max_col.max(pc_col);
                pc_col = 0;
                continue;
            }
            if ('가'..='힣').contains(&ch) {
                pane.insert((pc_row, pc_col), ch);
            } else {
                pane.insert((pc_row, pc_col), '\0');
            }
            pc_col += 1;
        }
        max_col = max_col.max(pc_col);

        PrimitiveProgram {
            pane,
            max_row: pc_row,
            max_col,
        }
    }

    fn decode(&self, position: (i32, i32)) -> (u8, i32, i32) {
        let code = self.pane.get(&position).copied().unwrap_or('\0');
        if code == '\0' {
            return (OP_NONE, MV_NONE, -1);
        }
        decompose(code).unwrap_or((OP_NONE, MV_NONE, -1))
    }

    fn advance_position(&self, position: (i32, i32), direction: i32, step: i32) -> (i32, i32) {
        let (mut r, mut c) = position;
        match direction {
            DIR_DOWN => {
                r += step;
                if r > self.max_row {
                    r = 0;
                    while !self.pane.contains_key(&(r, c)) && r <= self.max_row {
                        r += 1;
                    }
                    if r > self.max_row {
                        r = 0;
                    }
                }
            }
            DIR_RIGHT => {
                c += step;
                if c > self.max_col {
                    c = 0;
                }
            }
            DIR_UP => {
                r -= step;
                if r < 0 {
                    r = self.max_row;
                    while !self.pane.contains_key(&(r, c)) && r >= 0 {
                        r -= 1;
                    }
                    if r < 0 {
                        r = self.max_row;
                    }
                }
            }
            DIR_LEFT => {
                c -= step;
                if c < 0 {
                    c = self.max_col;
                    while !self.pane.contains_key(&(r, c)) && c >= 0 {
                        c -= 1;
                    }
                    if c < 0 {
                        c = self.max_col;
                    }
                }
            }
            _ => panic!("invalid direction: {direction}"),
        }
        (r, c)
    }
}

/// The compiler: parses and serializes Aheui programs.
pub struct Compiler {
    pub lines: Vec<(u8, i32)>,
    pub label_map: HashMap<i32, usize>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        Compiler {
            lines: Vec::new(),
            label_map: HashMap::new(),
        }
    }

    /// Remove labels that no jump instruction references. This allows
    /// subsequent peephole passes to fold constants across former block
    /// boundaries.
    pub(crate) fn remove_dead_labels(&mut self) {
        let used_labels: HashSet<i32> = self
            .lines
            .iter()
            .filter(|(op, _)| is_jump_op(*op))
            .map(|(_, v)| *v)
            .collect();
        self.label_map.retain(|k, _| used_labels.contains(k));
    }

    /// Reconstruct a Compiler from a linearized Program for further optimization.
    pub fn from_program(program: &super::compiler::Program) -> Self {
        let lines: Vec<(u8, i32)> = program
            .opcodes
            .iter()
            .zip(program.values.iter())
            .map(|(&op, &val)| (op, val))
            .collect();
        Compiler {
            lines,
            label_map: program.labels.clone(),
        }
    }

    /// Compile Aheui source text into linear bytecode.
    pub fn compile(&mut self, program: &str) {
        let primitive = PrimitiveProgram::new(program);
        let (lines, label_map) = self.serialize(&primitive);
        self.lines = lines;
        self.label_map = label_map;
    }

    /// Serialize 2D Aheui program into linear opcode sequence.
    fn serialize(&self, primitive: &PrimitiveProgram) -> (Vec<(u8, i32)>, HashMap<i32, usize>) {
        let mut job_queue: VecDeque<((i32, i32), i32, i32, i32)> =
            VecDeque::from([((0, 0), DIR_DOWN, 1, -1)]);

        let mut lines: Vec<(u8, i32)> = Vec::new();
        let mut label_map: HashMap<i32, usize> = HashMap::new();
        let mut code_map: HashMap<((i32, i32), i32, i32), usize> = HashMap::new();

        while let Some((mut position, mut direction, mut step, marker)) = job_queue.pop_front() {
            if marker >= 0 {
                label_map.insert(marker, lines.len());
            }

            loop {
                if !primitive.pane.contains_key(&position) {
                    position = primitive.advance_position(position, direction, step);
                    continue;
                }

                let (op, mv, val) = primitive.decode(position);
                let (new_direction, new_step) = dir_from_mv(mv, direction, step);
                if MV_DETERMINISTICS.contains(&mv) {
                    direction = new_direction;
                    step = new_step;
                }

                let key = (position, direction, step);
                if let Some(&index) = code_map.get(&key) {
                    let posdir_key = (position, direction + 10, step);
                    code_map.insert((position, direction + 5, step), lines.len());
                    let target = code_map.get(&posdir_key).copied().unwrap_or(index);
                    let label_id = lines.len() as i32;
                    label_map.insert(label_id, target);
                    lines.push((OP_JMP, label_id));
                    break;
                }

                code_map.insert(key, lines.len());

                direction = new_direction;
                step = new_step;

                if OP_HASOP[op as usize] {
                    let mut actual_op = op;
                    let actual_val = val;

                    if op == OP_POP {
                        if val == VAL_NUMBER {
                            actual_op = OP_POPNUM;
                        } else if val == VAL_UNICODE {
                            actual_op = OP_POPCHAR;
                        }
                    } else if op == OP_PUSH {
                        if val == VAL_NUMBER {
                            actual_op = OP_PUSHNUM;
                        } else if val == VAL_UNICODE {
                            actual_op = OP_PUSHCHAR;
                        }
                    }

                    if actual_op == OP_PUSH {
                        lines.push((actual_op, VAL_CONSTS[actual_val as usize] as i32));
                    } else {
                        let req_size = OP_REQSIZE[actual_op as usize];
                        if req_size > 0 {
                            let brop = if req_size == 1 { OP_BRPOP1 } else { OP_BRPOP2 };
                            let idx = lines.len() as i32;
                            code_map.insert((position, direction + 10, step), lines.len());
                            lines.push((brop, idx));
                            code_map.insert((position, direction, step), lines.len());

                            if OP_USEVAL[actual_op as usize] {
                                if actual_op == OP_BRZ {
                                    lines.push((actual_op, idx));
                                    let alt_position =
                                        primitive.advance_position(position, -direction, step);
                                    job_queue.push_back((alt_position, -direction, step, idx));
                                } else {
                                    lines.push((actual_op, actual_val));
                                }
                            } else {
                                lines.push((actual_op, -1));
                            }

                            let alt_position =
                                primitive.advance_position(position, -direction, step);
                            job_queue.push_back((alt_position, -direction, step, idx));
                        } else if OP_USEVAL[actual_op as usize] {
                            lines.push((actual_op, actual_val));
                        } else {
                            lines.push((actual_op, -1));
                            if actual_op == OP_HALT {
                                break;
                            }
                        }
                    }
                }

                position = primitive.advance_position(position, direction, step);
            }
        }

        (lines, label_map)
    }

    pub fn optimize1(&mut self) {
        self.optimize_jump();
        let reachability = self.optimize_deadcode1();
        self.optimize_adjust(&reachability);

        let reachability = self.optimize_operation();
        self.optimize_jump();
        self.optimize_adjust(&reachability);
    }

    pub fn optimize2(&mut self) {
        self.optimize1();
    }

    pub(crate) fn optimize_jump(&mut self) {
        loop {
            let mut changed = false;
            let labels: Vec<i32> = self.label_map.keys().cloned().collect();
            for label in labels {
                let direct_target = self.label_map[&label];
                if direct_target < self.lines.len() {
                    let (op, operand) = self.lines[direct_target];
                    if op == OP_JMP
                        && let Some(&indirect_target) = self.label_map.get(&operand)
                        && indirect_target != direct_target
                    {
                        self.label_map.insert(label, indirect_target);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn optimize_deadcode1(&self) -> Vec<bool> {
        let mut min_stacksize_map = vec![-1i32; self.lines.len()];
        let mut job_queue: VecDeque<(usize, i32)> = VecDeque::from([(0, 0)]);

        while let Some((mut pc, mut stacksize)) = job_queue.pop_front() {
            while pc < self.lines.len() {
                let (op, val) = self.lines[pc];
                let min_stacksize = min_stacksize_map[pc];
                if min_stacksize >= 0 {
                    let min_diff = min_stacksize - stacksize;
                    let stack_delta = OP_STACKADD[op as usize] - OP_STACKDEL[op as usize];
                    if min_diff <= 0 && min_diff <= stack_delta {
                        break;
                    }
                }
                if op == OP_BRPOP1 || op == OP_BRPOP2 {
                    let reqsize = OP_REQSIZE[op as usize] as i32;
                    if stacksize >= reqsize {
                        pc += 1;
                        continue;
                    } else {
                        min_stacksize_map[pc] = stacksize;
                        if let Some(&target) = self.label_map.get(&val) {
                            job_queue.push_back((target, stacksize));
                        }
                    }
                } else {
                    min_stacksize_map[pc] = stacksize;
                    stacksize -= OP_STACKDEL[op as usize];
                    if stacksize < 0 {
                        stacksize = 0;
                    }
                    stacksize += OP_STACKADD[op as usize];
                    if op == OP_BRZ {
                        if let Some(&target) = self.label_map.get(&val) {
                            job_queue.push_back((target, stacksize));
                        }
                    } else if op == OP_JMP {
                        if let Some(&target) = self.label_map.get(&val) {
                            pc = target;
                            continue;
                        } else {
                            break;
                        }
                    } else if op == OP_SEL {
                        min_stacksize_map[pc] = stacksize;
                        stacksize = 0;
                    } else if op == OP_HALT {
                        break;
                    }
                }
                pc += 1;
            }
        }

        min_stacksize_map.iter().map(|&s| s >= 0).collect()
    }

    pub(crate) fn optimize_deadcode2(&self) -> Vec<bool> {
        let n = self.lines.len();
        let mut min_map: Vec<[i32; STORAGE_COUNT]> = vec![[-1i32; STORAGE_COUNT]; n];
        let label_targets: HashSet<usize> = self.label_map.values().copied().collect();

        fn min_merge(a: &[i32; STORAGE_COUNT], b: &[i32; STORAGE_COUNT]) -> [i32; STORAGE_COUNT] {
            if a[0] == -1 {
                return *b;
            }
            let mut out = [0i32; STORAGE_COUNT];
            for i in 0..STORAGE_COUNT {
                out[i] = a[i].min(b[i]);
            }
            out
        }

        let mut job_queue: VecDeque<(usize, usize, [i32; STORAGE_COUNT])> =
            VecDeque::from([(0, 0, [0i32; STORAGE_COUNT])]);

        while let Some((mut pc, mut selected, mut stacksizes)) = job_queue.pop_front() {
            while pc < n {
                let stacksize = stacksizes[selected];
                debug_assert!(stacksize >= 0);
                let (op, val) = self.lines[pc];
                let prev = min_map[pc];
                if prev[selected] >= 0 {
                    let min_diff = prev[selected] - stacksizes[selected];
                    let stack_delta = OP_STACKADD[op as usize] - OP_STACKDEL[op as usize];
                    let merged = min_merge(&prev, &stacksizes);
                    if min_diff <= stack_delta && merged == prev {
                        break;
                    }
                }
                if op == OP_BRPOP1 || op == OP_BRPOP2 {
                    let reqsize = OP_REQSIZE[op as usize] as i32;
                    if stacksize >= reqsize && !label_targets.contains(&pc) {
                        pc += 1;
                        continue;
                    } else {
                        min_map[pc] = min_merge(&prev, &stacksizes);
                        if let Some(&target) = self.label_map.get(&val) {
                            job_queue.push_back((target, selected, stacksizes));
                        }
                    }
                } else {
                    min_map[pc] = min_merge(&prev, &stacksizes);
                    let mut ss = stacksizes[selected] - OP_STACKDEL[op as usize];
                    if ss < 0 {
                        ss = 0;
                    }
                    ss += OP_STACKADD[op as usize];
                    stacksizes[selected] = ss;
                    if op == OP_BRZ {
                        if let Some(&target) = self.label_map.get(&val) {
                            job_queue.push_back((target, selected, stacksizes));
                        }
                    } else if op == OP_JMP {
                        if let Some(&target) = self.label_map.get(&val) {
                            pc = target;
                            continue;
                        } else {
                            break;
                        }
                    } else if op == OP_SEL {
                        min_map[pc] = min_merge(&min_map[pc], &stacksizes);
                        selected = val as usize;
                    } else if op == OP_MOV {
                        stacksizes[val as usize] += 1;
                    } else if op == OP_HALT {
                        break;
                    }
                }
                pc += 1;
            }
        }

        min_map.iter().map(|sizes| sizes[0] >= 0).collect()
    }

    pub(crate) fn optimize_adjust(&mut self, reachability: &[bool]) {
        let mut useless_map = vec![0usize; reachability.len()];
        let mut count = 0usize;
        for (i, &reachable) in reachability.iter().enumerate() {
            useless_map[i] = count;
            if !reachable {
                count += 1;
            }
        }

        let mut new_lines = Vec::new();
        let mut new_label_map = HashMap::new();
        for (i, &(op, val)) in self.lines.iter().enumerate() {
            if !reachability[i] {
                continue;
            }
            new_lines.push((op, val));
            if is_jump_op(op)
                && let Some(&target_idx) = self.label_map.get(&val)
            {
                let new_target = target_idx - useless_map[target_idx];
                new_label_map.insert(val, new_target);
            }
        }

        self.lines = new_lines;
        self.label_map = new_label_map;
    }

    pub(crate) fn optimize_operation(&mut self) -> Vec<bool> {
        let mut queue_map = vec![-1i8; self.lines.len()];
        let mut bfs: VecDeque<(usize, i8)> = VecDeque::from([(0, 0)]);
        while let Some((mut pc, mut in_queue)) = bfs.pop_front() {
            while pc < self.lines.len() {
                let (op, val) = self.lines[pc];
                if queue_map[pc] >= 0 && (in_queue == 0 || queue_map[pc] == 1) {
                    break;
                }
                queue_map[pc] = in_queue;
                if matches!(op, OP_BRZ | OP_BRPOP1 | OP_BRPOP2) {
                    bfs.push_back((pc + 1, in_queue));
                    if let Some(&target) = self.label_map.get(&val) {
                        bfs.push_back((target, in_queue));
                    }
                    break;
                } else if op == OP_JMP {
                    if let Some(&target) = self.label_map.get(&val) {
                        bfs.push_back((target, in_queue));
                    }
                    break;
                } else {
                    pc += 1;
                    if op == OP_SEL {
                        in_queue = if val == VAL_QUEUE as i32 || val == VAL_PORT as i32 {
                            1
                        } else {
                            0
                        };
                    } else if op == OP_HALT {
                        break;
                    }
                }
            }
        }

        let mut removed = vec![false; self.lines.len()];
        for i in 1..self.lines.len() {
            let (op, _) = self.lines[i];
            if op != OP_DUP {
                continue;
            }
            let mut i1 = i - 1;
            while i1 > 0 && self.lines[i1].0 == OP_NONE {
                i1 -= 1;
            }
            if queue_map[i] != 0 || queue_map[i1] != 0 {
                continue;
            }
            if self.label_map.values().any(|&target| target == i) {
                continue;
            }
            let (op1, _) = self.lines[i1];
            if op1 == OP_PUSH {
                self.lines[i] = self.lines[i1];
            }
        }

        let label_targets: Vec<usize> = self.label_map.values().cloned().collect();
        let mut label_rmap: HashMap<usize, i32> = HashMap::new();
        for (&label, &target) in &self.label_map {
            match label_rmap.entry(target) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    entry.insert(-1);
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(label);
                }
            }
        }

        for i in 2..self.lines.len() {
            let (op, v) = self.lines[i];

            let mut i1 = i - 1;
            while i1 >= 1 && self.lines[i1].0 == OP_NONE {
                i1 -= 1;
            }
            let mut i2 = if i1 > 0 { i1 - 1 } else { continue };
            while i2 > 0 && self.lines[i2].0 == OP_NONE {
                i2 -= 1;
            }

            let (op1, v1) = self.lines[i1];
            let (op2, v2) = self.lines[i2];

            if op2 != OP_PUSH {
                continue;
            }

            if queue_map[i] != 0 || queue_map[i1] != 0 || queue_map[i2] != 0 {
                continue;
            }

            let effective_v1 = if op1 == OP_PUSH {
                v1
            } else if op1 == OP_DUP {
                v2
            } else {
                continue;
            };

            if label_targets.contains(&i1) || label_targets.contains(&i) {
                continue;
            }

            let actual_op = if op == OP_JMP {
                if let Some(&target) = self.label_map.get(&v) {
                    if target < self.lines.len() && label_rmap.get(&target) == Some(&v) {
                        self.lines[target].0
                    } else {
                        continue;
                    }
                } else {
                    continue;
                }
            } else {
                op
            };

            if !is_binary_op(actual_op) {
                continue;
            }

            let result = match actual_op {
                OP_ADD => match v2.checked_add(effective_v1) {
                    Some(r) => r,
                    None => continue, // overflow — skip folding
                },
                OP_SUB => match v2.checked_sub(effective_v1) {
                    Some(r) => r,
                    None => continue,
                },
                OP_MUL => match v2.checked_mul(effective_v1) {
                    Some(r) => r,
                    None => continue,
                },
                OP_DIV => {
                    if effective_v1 != 0 {
                        match v2.checked_div(effective_v1) {
                            Some(r) => r,
                            None => continue,
                        }
                    } else {
                        0
                    }
                }
                OP_MOD => {
                    if effective_v1 != 0 {
                        v2 % effective_v1
                    } else {
                        0
                    }
                }
                OP_CMP => {
                    if v2 >= effective_v1 {
                        1
                    } else {
                        0
                    }
                }
                _ => continue,
            };

            if op == OP_JMP {
                self.lines[i1] = (OP_PUSH, result);
                self.lines[i2] = (OP_NONE, -1);
                removed[i2] = true;
                if let Some(target) = self.label_map.get_mut(&v) {
                    let old_target = *target;
                    *target += 1;
                    label_rmap.remove(&old_target);
                    match label_rmap.entry(old_target + 1) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            entry.insert(-1);
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(v);
                        }
                    }
                }
            } else {
                self.lines[i] = (OP_PUSH, result);
                self.lines[i1] = (OP_NONE, -1);
                self.lines[i2] = (OP_NONE, -1);
                removed[i1] = true;
                removed[i2] = true;
            }
        }

        for i in 1..self.lines.len() {
            let (op, v) = self.lines[i];
            if op != OP_BRZ {
                continue;
            }
            let mut i1 = i - 1;
            while i1 > 0 && self.lines[i1].0 == OP_NONE {
                i1 -= 1;
            }
            let (op1, v1) = self.lines[i1];
            if op1 != OP_PUSH || queue_map[i] != 0 || queue_map[i1] != 0 {
                continue;
            }
            if label_targets.contains(&i1) || label_targets.contains(&i) {
                continue;
            }

            self.lines[i1] = (OP_NONE, -1);
            removed[i1] = true;
            if v1 == 0 {
                self.lines[i] = (OP_JMP, v);
            } else {
                self.lines[i] = (OP_NONE, -1);
                removed[i] = true;
            }
        }

        let control_reachability = self.compute_control_reachability();
        control_reachability
            .into_iter()
            .zip(removed)
            .map(|(reachable, removed)| reachable && !removed)
            .collect()
    }

    fn compute_control_reachability(&self) -> Vec<bool> {
        let mut reachability = vec![false; self.lines.len()];
        let mut queue: VecDeque<usize> = VecDeque::from([0]);

        while let Some(mut pc) = queue.pop_front() {
            while pc < self.lines.len() && !reachability[pc] {
                reachability[pc] = true;
                let (op, val) = self.lines[pc];
                match op {
                    OP_BRZ | OP_BRPOP1 | OP_BRPOP2 => {
                        queue.push_back(pc + 1);
                        if let Some(&target) = self.label_map.get(&val) {
                            queue.push_back(target);
                        }
                        break;
                    }
                    OP_JMP => {
                        if let Some(&target) = self.label_map.get(&val) {
                            queue.push_back(target);
                        }
                        break;
                    }
                    OP_HALT => break,
                    _ => pc += 1,
                }
            }
        }

        reachability
    }

    pub fn into_program(self) -> Program {
        let size = self.lines.len();
        let mut opcodes = Vec::with_capacity(size);
        let mut values = Vec::with_capacity(size);
        for (op, val) in &self.lines {
            opcodes.push(*op);
            values.push(*val);
        }
        Program {
            opcodes,
            values,
            labels: self.label_map,
            size,
        }
    }
}

pub fn dump_asm(program: &Program, writer: &mut impl Write) -> io::Result<()> {
    // After `resolve_jump_targets`, `values[pc]` for jump ops carries the
    // target PC directly.  Derive label markers from the unique set of
    // jump targets so `JMP L<val>` matches an `L<pc>:` line at the same
    // PC.
    let mut targets: HashSet<usize> = HashSet::new();
    for pc in 0..program.size {
        if is_jump_op(program.opcodes[pc]) {
            targets.insert(program.values[pc] as usize);
        }
    }

    for pc in 0..program.size {
        if targets.contains(&pc) {
            writeln!(writer, "L{pc}:")?;
        }
        let op = program.opcodes[pc];
        let val = program.values[pc];
        let name = if (op as usize) < NUM_OPS {
            OP_NAMES[op as usize].unwrap_or("???")
        } else {
            "???"
        };
        if is_jump_op(op) {
            writeln!(writer, "  {pc:4}: {name:<10} L{val}")?;
        } else if op == OP_PUSH || op == OP_SEL || op == OP_MOV {
            writeln!(writer, "  {pc:4}: {name:<10} {val}")?;
        } else {
            writeln!(writer, "  {pc:4}: {name}")?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_halt() {
        let mut c = Compiler::new();
        c.compile("희");
        assert!(!c.lines.is_empty());
        assert!(c.lines.iter().any(|(op, _)| *op == OP_HALT));
    }

    #[test]
    fn test_compile_push_add_halt() {
        let mut c = Compiler::new();
        c.compile("반반다희");
        let program = c.into_program();
        assert!(program.size > 0);
    }

    #[test]
    fn test_decompose_in_compile() {
        let mut c = Compiler::new();
        c.compile("밤희");
        let (op, val) = c.lines[0];
        assert_eq!(op, OP_PUSH);
        assert_eq!(val, 4);
    }

    #[test]
    fn test_optimize1() {
        let mut c = Compiler::new();
        c.compile("반반다희");
        c.optimize1();
        let program = c.into_program();
        let pushes: Vec<_> = program
            .opcodes
            .iter()
            .zip(program.values.iter())
            .filter(|&(&op, _)| op == OP_PUSH)
            .collect();
        assert!(!pushes.is_empty());
    }

    #[test]
    fn test_optimize2_eliminates_static_brpop() {
        let mut c = Compiler::new();
        c.compile("반반다희");
        c.optimize2();
        let program = c.into_program();
        assert_eq!(program.size, 2);
        assert_eq!(program.opcodes, vec![OP_PUSH, OP_HALT]);
        assert_eq!(program.values, vec![4, -1]);
    }

    #[test]
    fn test_optimize_jump_resolves_chains() {
        let mut c = Compiler::new();
        c.lines = vec![(OP_JMP, 1), (OP_JMP, 2), (OP_JMP, 3), (OP_HALT, -1)];
        c.label_map = HashMap::from([(1, 1), (2, 2), (3, 3)]);
        c.optimize_jump();
        assert_eq!(c.label_map[&1], 3);
        assert_eq!(c.label_map[&2], 3);
    }

    #[test]
    fn test_optimize_operation_folds_zero_brz() {
        let mut c = Compiler::new();
        c.lines = vec![
            (OP_PUSH, 0),
            (OP_BRZ, 1),
            (OP_PUSH, 9),
            (OP_HALT, -1),
            (OP_PUSH, 7),
            (OP_HALT, -1),
        ];
        c.label_map = HashMap::from([(1, 4)]);

        let reachability = c.optimize_operation();
        c.optimize_jump();
        c.optimize_adjust(&reachability);
        let program = c.into_program();

        assert_eq!(program.opcodes, vec![OP_JMP, OP_PUSH, OP_HALT]);
        assert_eq!(program.values, vec![1, 7, -1]);
        assert_eq!(program.labels[&1], 1);
    }

    #[test]
    fn test_deadcode2_keeps_block_entry_brpop() {
        let mut c = Compiler::new();
        c.lines = vec![
            (OP_PUSH, 1),
            (OP_JMP, 10),
            (OP_BRPOP1, 11),
            (OP_DUP, -1),
            (OP_HALT, -1),
            (OP_HALT, -1),
        ];
        c.label_map = HashMap::from([(10, 2), (11, 5)]);

        let reachability = c.optimize_deadcode2();
        c.optimize_adjust(&reachability);

        assert!(c.lines.iter().any(|&(op, _)| op == OP_BRPOP1));
    }

    #[test]
    fn test_hello_puzzlet_has_halt() {
        let source =
            include_str!("../../../../../../rpaheui/snippets/hello-world/hello.puzzlet.aheui");
        let mut c = Compiler::new();
        c.compile(source);
        assert!(c.lines.iter().any(|&(op, _)| op == OP_HALT));
    }
}
