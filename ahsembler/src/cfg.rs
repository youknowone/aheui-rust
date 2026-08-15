//! Control Flow Graph representation for Aheui bytecode.

use crate::consts::STORAGE_COUNT;

pub type BlockId = u32;

pub const ENTRY_BLOCK: BlockId = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Cmp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Inst {
    Push(i64),
    Pop,
    Dup,
    Swap,
    BinOp(BinOpKind),
    Sel(usize),
    Mov(usize),
    PopNum,
    PopChar,
    PushNum,
    PushChar,
    /// Inline depth guard: if depth < min_depth, jump to fail_block.
    /// Created by merge_guard_ok to fold StackGuard ok-paths.
    GuardDepth {
        min_depth: usize,
        fail: BlockId,
    },
}

impl Inst {
    /// Stack elements consumed by this instruction.
    pub fn stack_del(&self) -> i32 {
        match self {
            Inst::Push(_)
            | Inst::Sel(_)
            | Inst::PushNum
            | Inst::PushChar
            | Inst::GuardDepth { .. } => 0,
            Inst::Pop | Inst::Dup | Inst::Mov(_) | Inst::PopNum | Inst::PopChar => 1,
            Inst::Swap | Inst::BinOp(_) => 2,
        }
    }

    /// Stack elements produced by this instruction.
    pub fn stack_add(&self) -> i32 {
        match self {
            Inst::Pop
            | Inst::Sel(_)
            | Inst::Mov(_)
            | Inst::PopNum
            | Inst::PopChar
            | Inst::GuardDepth { .. } => 0,
            Inst::Push(_) | Inst::PushNum | Inst::PushChar | Inst::BinOp(_) => 1,
            Inst::Dup | Inst::Swap => 2,
        }
    }

    /// Net stack depth change.
    pub fn stack_delta(&self) -> i32 {
        self.stack_add() - self.stack_del()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Unconditional jump to target block.
    Goto(BlockId),
    /// Stack underflow guard: if depth < min_depth, jump to fail; otherwise ok.
    StackGuard {
        min_depth: usize,
        ok: BlockId,
        fail: BlockId,
    },
    /// Conditional branch: pop top, if zero jump to on_zero, otherwise on_nonzero.
    BranchZero {
        on_zero: BlockId,
        on_nonzero: BlockId,
    },
    /// Program termination.
    Halt,
}

impl Terminator {
    /// All successor block IDs.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Goto(b) => vec![*b],
            Terminator::StackGuard { ok, fail, .. } => vec![*ok, *fail],
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } => vec![*on_zero, *on_nonzero],
            Terminator::Halt => vec![],
        }
    }

    /// Replace a successor block ID.
    pub fn replace_successor(&mut self, old: BlockId, new: BlockId) {
        match self {
            Terminator::Goto(b) => {
                if *b == old {
                    *b = new;
                }
            }
            Terminator::StackGuard { ok, fail, .. } => {
                if *ok == old {
                    *ok = new;
                }
                if *fail == old {
                    *fail = new;
                }
            }
            Terminator::BranchZero {
                on_zero,
                on_nonzero,
            } => {
                if *on_zero == old {
                    *on_zero = new;
                }
                if *on_nonzero == old {
                    *on_nonzero = new;
                }
            }
            Terminator::Halt => {}
        }
    }
}

#[derive(Debug, Clone)]
pub struct CfgBlock {
    pub id: BlockId,
    pub instructions: Vec<Inst>,
    pub terminator: Terminator,
}

impl CfgBlock {
    /// All successor block IDs including GuardDepth fail targets.
    pub fn all_successors(&self) -> Vec<BlockId> {
        let mut succs = self.terminator.successors();
        for inst in &self.instructions {
            if let Inst::GuardDepth { fail, .. } = inst {
                succs.push(*fail);
            }
        }
        succs
    }
}

/// Lattice element for abstract stack depth analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// Not yet analyzed (bottom of lattice).
    Bottom,
    /// Known lower bound on stack depth.
    AtLeast(i32),
}

impl Depth {
    /// Meet operation: conservative merge of two depth estimates.
    pub fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Depth::Bottom, x) | (x, Depth::Bottom) => x,
            (Depth::AtLeast(a), Depth::AtLeast(b)) => Depth::AtLeast(a.min(b)),
        }
    }

    /// Check if proven >= threshold.
    pub fn is_at_least(self, threshold: i32) -> bool {
        matches!(self, Depth::AtLeast(d) if d >= threshold)
    }
}

/// Abstract state at a CFG block entry point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbstractState {
    pub depths: [Depth; STORAGE_COUNT],
    pub selected: Option<usize>,
}

impl AbstractState {
    pub fn bottom() -> Self {
        AbstractState {
            depths: [Depth::Bottom; STORAGE_COUNT],
            selected: None,
        }
    }

    pub fn initial() -> Self {
        AbstractState {
            depths: [Depth::AtLeast(0); STORAGE_COUNT],
            selected: Some(0),
        }
    }

    pub fn is_bottom(&self) -> bool {
        self.depths.iter().all(|d| *d == Depth::Bottom)
    }

    /// Meet two abstract states.
    pub fn meet(&self, other: &Self) -> Self {
        let mut depths = [Depth::Bottom; STORAGE_COUNT];
        for i in 0..STORAGE_COUNT {
            depths[i] = self.depths[i].meet(other.depths[i]);
        }
        let selected = match (self.selected, other.selected) {
            (Some(a), Some(b)) if a == b => Some(a),
            (None, x) | (x, None) if self.is_bottom() || other.is_bottom() => x,
            _ => None,
        };
        AbstractState { depths, selected }
    }
}

#[derive(Debug)]
pub struct Cfg {
    pub blocks: Vec<CfgBlock>,
    pub entry: BlockId,
}

impl Default for Cfg {
    fn default() -> Self {
        Self::new()
    }
}

impl Cfg {
    pub fn new() -> Self {
        Cfg {
            blocks: Vec::new(),
            entry: ENTRY_BLOCK,
        }
    }

    pub fn add_block(&mut self, instructions: Vec<Inst>, terminator: Terminator) -> BlockId {
        let id = self.blocks.len() as BlockId;
        self.blocks.push(CfgBlock {
            id,
            instructions,
            terminator,
        });
        id
    }

    pub fn block(&self, id: BlockId) -> &CfgBlock {
        &self.blocks[id as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut CfgBlock {
        &mut self.blocks[id as usize]
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Compute predecessor map.
    pub fn predecessors(&self) -> Vec<Vec<BlockId>> {
        let mut preds = vec![vec![]; self.blocks.len()];
        for block in &self.blocks {
            for &succ in &block.all_successors() {
                if (succ as usize) < preds.len() {
                    preds[succ as usize].push(block.id);
                }
            }
        }
        preds
    }

    /// Compute reverse postorder for iteration.
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let n = self.blocks.len();
        let mut visited = vec![false; n];
        let mut postorder = Vec::with_capacity(n);

        fn dfs(
            cfg: &Cfg,
            block_id: BlockId,
            visited: &mut Vec<bool>,
            postorder: &mut Vec<BlockId>,
        ) {
            let idx = block_id as usize;
            if idx >= visited.len() || visited[idx] {
                return;
            }
            visited[idx] = true;
            for &succ in &cfg.blocks[idx].all_successors() {
                dfs(cfg, succ, visited, postorder);
            }
            postorder.push(block_id);
        }

        dfs(self, self.entry, &mut visited, &mut postorder);
        postorder.reverse();
        postorder
    }
}
