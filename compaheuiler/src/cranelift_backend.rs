//! Cranelift JIT backend for Aheui AOT compiler.
//!
//! Per-storage Cranelift Variables for sel-known blocks enable register promotion.
//! The Variable for the CURRENT sel tracks the top pointer in a register.
//! On sel change or block boundary, the Variable is flushed to / loaded from tops_slot.

#[cfg(feature = "cranelift")]
pub mod jit {
    use ahsembler::cfg::*;
    use ahsembler::consts::STORAGE_COUNT;
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::types::*;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::collections::{BTreeSet, HashMap, VecDeque};

    const QUEUE: usize = 21;
    const PORT: usize = 27;
    fn is_special(s: usize) -> bool {
        s == QUEUE || s == PORT
    }

    /// Runtime special storage for queue (sel=21) and port (sel=27).
    /// Passed as opaque `*mut u8` to the JIT function, accessed via callbacks.
    pub struct SpecialStorage {
        pub queue: VecDeque<i64>,
        pub port: Vec<i64>,
        pub port_last: i64,
    }

    impl SpecialStorage {
        pub fn new() -> Self {
            SpecialStorage {
                queue: VecDeque::new(),
                port: Vec::new(),
                port_last: 0,
            }
        }
        pub fn push(&mut self, sel: usize, v: i64) {
            if sel == QUEUE {
                self.queue.push_back(v);
            } else {
                self.port_last = v;
                self.port.push(v);
            }
        }
        pub fn pop(&mut self, sel: usize) -> i64 {
            if sel == QUEUE {
                self.queue.pop_front().unwrap_or(0)
            } else {
                self.port.pop().unwrap_or(0)
            }
        }
        pub fn depth(&self, sel: usize) -> i64 {
            (if sel == QUEUE {
                self.queue.len()
            } else {
                self.port.len()
            }) as i64
        }
        pub fn dup(&mut self, sel: usize) {
            if sel == QUEUE {
                if let Some(&v) = self.queue.front() {
                    self.queue.push_front(v);
                }
            } else {
                self.port.push(self.port_last);
            }
        }
        pub fn swap(&mut self, sel: usize) {
            if sel == QUEUE && self.queue.len() >= 2 {
                let a = self.queue.pop_front().unwrap();
                let b = self.queue.pop_front().unwrap();
                self.queue.push_front(a);
                self.queue.push_front(b);
            } else if sel == PORT && self.port.len() >= 2 {
                let n = self.port.len();
                self.port.swap(n - 1, n - 2);
            }
        }
    }

    /// Callback: push value to special storage.
    pub extern "C" fn sp_push(ctx: *mut u8, sel: usize, v: i64) {
        let sp = unsafe { &mut *(ctx as *mut SpecialStorage) };
        sp.push(sel, v);
    }
    /// Callback: pop value from special storage.
    pub extern "C" fn sp_pop(ctx: *mut u8, sel: usize) -> i64 {
        let sp = unsafe { &mut *(ctx as *mut SpecialStorage) };
        sp.pop(sel)
    }
    /// Callback: get depth of special storage.
    pub extern "C" fn sp_depth(ctx: *mut u8, sel: usize) -> i64 {
        let sp = unsafe { &*(ctx as *const SpecialStorage) };
        sp.depth(sel)
    }
    /// Callback: duplicate top of special storage.
    pub extern "C" fn sp_dup(ctx: *mut u8, sel: usize) {
        let sp = unsafe { &mut *(ctx as *mut SpecialStorage) };
        sp.dup(sel);
    }
    /// Callback: swap top two of special storage.
    pub extern "C" fn sp_swap(ctx: *mut u8, sel: usize) {
        let sp = unsafe { &mut *(ctx as *mut SpecialStorage) };
        sp.swap(sel);
    }

    pub struct JitFunction {
        _module: JITModule,
        code_ptr: *const u8,
    }
    unsafe impl Send for JitFunction {}
    unsafe impl Sync for JitFunction {}

    impl JitFunction {
        /// Execute the JIT-compiled function.
        ///
        /// `sp_ctx`: pointer to `SpecialStorage` (or null if unused).
        /// `sp_push_fn` .. `sp_swap_fn`: callback function pointers for special storage ops.
        pub unsafe fn execute(
            &self,
            bases: &mut [*mut i64; STORAGE_COUNT],
            lengths: &mut [i32; STORAGE_COUNT],
            write_char_fn: extern "C" fn(i64),
            write_num_fn: extern "C" fn(i64),
            read_char_fn: extern "C" fn() -> i64,
            read_num_fn: extern "C" fn() -> i64,
            sp_ctx: *mut u8,
            sp_push_fn: extern "C" fn(*mut u8, usize, i64),
            sp_pop_fn: extern "C" fn(*mut u8, usize) -> i64,
            sp_depth_fn: extern "C" fn(*mut u8, usize) -> i64,
            sp_dup_fn: extern "C" fn(*mut u8, usize),
            sp_swap_fn: extern "C" fn(*mut u8, usize),
        ) -> i64 {
            type Fn = unsafe extern "C" fn(
                *mut *mut i64,
                *mut i32,
                extern "C" fn(i64),
                extern "C" fn(i64),
                extern "C" fn() -> i64,
                extern "C" fn() -> i64,
                *mut u8,
                extern "C" fn(*mut u8, usize, i64),
                extern "C" fn(*mut u8, usize) -> i64,
                extern "C" fn(*mut u8, usize) -> i64,
                extern "C" fn(*mut u8, usize),
                extern "C" fn(*mut u8, usize),
            ) -> i64;
            let func: Fn = unsafe { std::mem::transmute(self.code_ptr) };
            unsafe {
                func(
                    bases.as_mut_ptr(),
                    lengths.as_mut_ptr(),
                    write_char_fn,
                    write_num_fn,
                    read_char_fn,
                    read_num_fn,
                    sp_ctx,
                    sp_push_fn,
                    sp_pop_fn,
                    sp_depth_fn,
                    sp_dup_fn,
                    sp_swap_fn,
                )
            }
        }
    }

    struct VarAlloc(u32);
    impl VarAlloc {
        fn next(&mut self) -> Variable {
            let v = Variable::from_u32(self.0);
            self.0 += 1;
            v
        }
    }

    /// Load the top pointer for the CURRENT sel using its Variable (register).
    /// Falls back to slot access for special/unknown sel.
    fn load_top(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_sel: Variable,
        v_top: &HashMap<usize, Variable>,
        ptr_type: cranelift_codegen::ir::Type,
    ) -> cranelift_codegen::ir::Value {
        if let Some(s) = sel {
            if !is_special(s) {
                if let Some(&var) = v_top.get(&s) {
                    return builder.use_var(var);
                }
                return builder
                    .ins()
                    .stack_load(ptr_type, tops_slot, (s * 8) as i32);
            }
        }
        // Dynamic: tops_slot[v_sel * 8]
        let sel_val = builder.use_var(v_sel);
        let byte_off = builder.ins().imul_imm(sel_val, 8);
        let slot_addr = builder.ins().stack_addr(ptr_type, tops_slot, 0);
        let addr = builder.ins().iadd(slot_addr, byte_off);
        builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0)
    }

    /// Store the top pointer for the CURRENT sel using its Variable (register).
    /// Falls back to slot access for special/unknown sel.
    fn store_top(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_sel: Variable,
        v_top: &HashMap<usize, Variable>,
        val: cranelift_codegen::ir::Value,
        ptr_type: cranelift_codegen::ir::Type,
    ) {
        if let Some(s) = sel {
            if !is_special(s) {
                if let Some(&var) = v_top.get(&s) {
                    builder.def_var(var, val);
                    return;
                }
                builder.ins().stack_store(val, tops_slot, (s * 8) as i32);
                return;
            }
        }
        let sel_val = builder.use_var(v_sel);
        let byte_off = builder.ins().imul_imm(sel_val, 8);
        let slot_addr = builder.ins().stack_addr(ptr_type, tops_slot, 0);
        let addr = builder.ins().iadd(slot_addr, byte_off);
        builder.ins().store(MemFlags::trusted(), val, addr, 0);
    }

    /// Load the top pointer from slot (bypassing Variable). For use after flush.
    fn load_top_slot(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_sel: Variable,
        ptr_type: cranelift_codegen::ir::Type,
    ) -> cranelift_codegen::ir::Value {
        if let Some(s) = sel {
            if !is_special(s) {
                return builder
                    .ins()
                    .stack_load(ptr_type, tops_slot, (s * 8) as i32);
            }
        }
        let sel_val = builder.use_var(v_sel);
        let byte_off = builder.ins().imul_imm(sel_val, 8);
        let slot_addr = builder.ins().stack_addr(ptr_type, tops_slot, 0);
        let addr = builder.ins().iadd(slot_addr, byte_off);
        builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0)
    }

    /// Store to slot directly (bypassing Variable). For terminators after flush.
    fn store_top_slot(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_sel: Variable,
        val: cranelift_codegen::ir::Value,
        ptr_type: cranelift_codegen::ir::Type,
    ) {
        if let Some(s) = sel {
            if !is_special(s) {
                builder.ins().stack_store(val, tops_slot, (s * 8) as i32);
                return;
            }
        }
        let sel_val = builder.use_var(v_sel);
        let byte_off = builder.ins().imul_imm(sel_val, 8);
        let slot_addr = builder.ins().stack_addr(ptr_type, tops_slot, 0);
        let addr = builder.ins().iadd(slot_addr, byte_off);
        builder.ins().store(MemFlags::trusted(), val, addr, 0);
    }

    /// Flush current sel's Variable to its tops_slot entry.
    fn flush_top_var(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_top: &HashMap<usize, Variable>,
    ) {
        if let Some(s) = sel {
            if !is_special(s) {
                if let Some(&var) = v_top.get(&s) {
                    let val = builder.use_var(var);
                    builder.ins().stack_store(val, tops_slot, (s * 8) as i32);
                }
            }
        }
    }

    /// Load current sel's Variable from its tops_slot entry.
    fn reload_top_var(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        sel: Option<usize>,
        v_top: &HashMap<usize, Variable>,
        ptr_type: cranelift_codegen::ir::Type,
    ) {
        if let Some(s) = sel {
            if !is_special(s) {
                if let Some(&var) = v_top.get(&s) {
                    let val = builder
                        .ins()
                        .stack_load(ptr_type, tops_slot, (s * 8) as i32);
                    builder.def_var(var, val);
                }
            }
        }
    }

    pub fn compile_cfg(cfg: &Cfg) -> Result<JitFunction, String> {
        let states = ahsembler::cfg_optimize::analyze_stack_depths(cfg);
        let mut used = BTreeSet::new();
        used.insert(0usize);
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

        // Cranelift setup
        let mut flag_builder = settings::builder();
        flag_builder.set("opt_level", "speed").unwrap();
        let isa = cranelift_native::builder()
            .map_err(|e| e.to_string())?
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| e.to_string())?;
        let ptr_type = isa.pointer_type();
        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let mut module = JITModule::new(jit_builder);

        let mut sig = module.make_signature();
        // 12 params: bases, lengths, write_char, write_num, read_char, read_num,
        //            sp_ctx, sp_push, sp_pop, sp_depth, sp_dup, sp_swap
        for _ in 0..12 {
            sig.params.push(AbiParam::new(ptr_type));
        }
        sig.returns.push(AbiParam::new(I64));
        let func_id = module
            .declare_function("aheui_main", Linkage::Local, &sig)
            .map_err(|e| e.to_string())?;

        let mut ctx = module.make_context();
        ctx.func.signature = sig.clone();
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let mut va = VarAlloc(0);
        let v_bases = va.next();
        builder.declare_var(v_bases, ptr_type);
        let v_lengths = va.next();
        builder.declare_var(v_lengths, ptr_type);
        let v_wc = va.next();
        builder.declare_var(v_wc, ptr_type);
        let v_wn = va.next();
        builder.declare_var(v_wn, ptr_type);
        let v_rc = va.next();
        builder.declare_var(v_rc, ptr_type);
        let v_rn = va.next();
        builder.declare_var(v_rn, ptr_type);
        let v_sel = va.next();
        builder.declare_var(v_sel, I64);
        // Special storage callback variables
        let v_sp_ctx = va.next();
        builder.declare_var(v_sp_ctx, ptr_type);
        let v_sp_push = va.next();
        builder.declare_var(v_sp_push, ptr_type);
        let v_sp_pop = va.next();
        builder.declare_var(v_sp_pop, ptr_type);
        let v_sp_depth = va.next();
        builder.declare_var(v_sp_depth, ptr_type);
        let v_sp_dup = va.next();
        builder.declare_var(v_sp_dup, ptr_type);
        let v_sp_swap = va.next();
        builder.declare_var(v_sp_swap, ptr_type);

        // Per-storage Variables for register promotion
        let mut v_top: HashMap<usize, Variable> = HashMap::new();
        for &s in &used {
            if is_special(s) {
                continue;
            }
            let v = va.next();
            builder.declare_var(v, ptr_type);
            v_top.insert(s, v);
        }

        // tops_slot: array of 28 pointers on stack
        let tops_slot = builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (STORAGE_COUNT * 8) as u32,
            3,
        ));

        let sig_vi64 = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        let sig_i64v = {
            let mut s = module.make_signature();
            s.returns.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        // sp_push(ctx, sel, val)
        let sig_sp_push = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            s.params.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        // sp_pop(ctx, sel) -> i64
        let sig_sp_pop = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            s.returns.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        // sp_depth(ctx, sel) -> i64
        let sig_sp_depth = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            s.returns.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        // sp_dup(ctx, sel)
        let sig_sp_dup = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        // sp_swap(ctx, sel)
        let sig_sp_swap = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            builder.import_signature(s)
        };

        // Blocks
        let rpo = cfg.reverse_postorder();
        let live: Vec<BlockId> = rpo
            .iter()
            .filter(|&&b| states.get(b as usize).map_or(false, |s| !s.is_bottom()))
            .copied()
            .collect();
        let mut cl: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
        for &bid in &live {
            cl.insert(bid, builder.create_block());
        }
        let halt_block = builder.create_block();

        // Init block
        let init = builder.create_block();
        builder.append_block_params_for_function_params(init);
        builder.switch_to_block(init);
        builder.seal_block(init);
        let params: Vec<_> = builder.block_params(init).to_vec();
        builder.def_var(v_bases, params[0]);
        builder.def_var(v_lengths, params[1]);
        builder.def_var(v_wc, params[2]);
        builder.def_var(v_wn, params[3]);
        builder.def_var(v_rc, params[4]);
        builder.def_var(v_rn, params[5]);
        builder.def_var(v_sp_ctx, params[6]);
        builder.def_var(v_sp_push, params[7]);
        builder.def_var(v_sp_pop, params[8]);
        builder.def_var(v_sp_depth, params[9]);
        builder.def_var(v_sp_dup, params[10]);
        builder.def_var(v_sp_swap, params[11]);
        let z = builder.ins().iconst(I64, 0);
        builder.def_var(v_sel, z);

        // Init tops_slot: tops[s] = bases[s] + lengths[s] * 8
        // Also def_var each per-storage Variable
        let bv = builder.use_var(v_bases);
        let lv = builder.use_var(v_lengths);
        for &s in &used {
            if is_special(s) {
                continue;
            }
            let base = builder
                .ins()
                .load(ptr_type, MemFlags::trusted(), bv, (s * 8) as i32);
            let len = builder
                .ins()
                .load(I32, MemFlags::trusted(), lv, (s * 4) as i32);
            let len64 = builder.ins().sextend(I64, len);
            let off = builder.ins().imul_imm(len64, 8);
            let top = builder.ins().iadd(base, off);
            builder.ins().stack_store(top, tops_slot, (s * 8) as i32);
            if let Some(&var) = v_top.get(&s) {
                builder.def_var(var, top);
            }
        }
        builder.ins().jump(cl[&cfg.entry], &[]);

        // Block-splitting threshold: split after every N Aheui instructions
        // to keep Cranelift blocks small for the register allocator.
        const SPLIT_THRESHOLD: usize = 35;

        // Emit CFG blocks
        for &bid in &live {
            let block = cfg.block(bid);
            builder.switch_to_block(cl[&bid]);
            let entry_sel = states.get(bid as usize).and_then(|s| s.selected);
            let mut sel = entry_sel;

            // Block entry: load current sel's Variable from slot
            reload_top_var(&mut builder, tops_slot, sel, &v_top, ptr_type);

            // Instructions
            let mut inst_count: usize = 0;
            for inst in &block.instructions {
                // Block splitting: if sel is known and not special, and we've
                // emitted enough instructions, jump to a fresh block to keep
                // Cranelift blocks small for the register allocator.
                if inst_count > 0 && inst_count % SPLIT_THRESHOLD == 0 {
                    if let Some(s) = sel {
                        if !is_special(s) {
                            flush_top_var(&mut builder, tops_slot, sel, &v_top);
                            let cont = builder.create_block();
                            builder.ins().jump(cont, &[]);
                            builder.switch_to_block(cont);
                            builder.seal_block(cont);
                            reload_top_var(&mut builder, tops_slot, sel, &v_top, ptr_type);
                        }
                    }
                }
                inst_count += 1;
                match inst {
                    Inst::Push(v) => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let fn_ptr = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let cv = builder.ins().iconst(I64, *v);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, fn_ptr, &[ctx, sv, cv]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let cv = builder.ins().iconst(I64, *v);
                            builder.ins().store(MemFlags::trusted(), cv, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::Pop => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let fn_ptr = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_pop, fn_ptr, &[ctx, sv]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::Dup => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let fn_ptr = builder.use_var(v_sp_dup);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_dup, fn_ptr, &[ctx, sv]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let val = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::Swap => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let fn_ptr = builder.use_var(v_sp_swap);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_swap, fn_ptr, &[ctx, sv]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let a = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                            let b = builder.ins().load(I64, MemFlags::trusted(), top, -16);
                            builder.ins().store(MemFlags::trusted(), b, top, -8);
                            builder.ins().store(MemFlags::trusted(), a, top, -16);
                        }
                    }
                    Inst::BinOp(kind) => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci1 = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let b = builder.inst_results(ci1)[0];
                            let ci2 = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let a = builder.inst_results(ci2)[0];
                            let r = match kind {
                                BinOpKind::Add => builder.ins().iadd(a, b),
                                BinOpKind::Sub => builder.ins().isub(a, b),
                                BinOpKind::Mul => builder.ins().imul(a, b),
                                BinOpKind::Div => {
                                    let is_nz = builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                                    let dbb = builder.create_block();
                                    let mbb = builder.create_block();
                                    builder.append_block_param(mbb, I64);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().brif(is_nz, dbb, &[], mbb, &[zero]);
                                    builder.switch_to_block(dbb);
                                    builder.seal_block(dbb);
                                    let d = builder.ins().sdiv(a, b);
                                    builder.ins().jump(mbb, &[d]);
                                    builder.switch_to_block(mbb);
                                    builder.seal_block(mbb);
                                    builder.block_params(mbb)[0]
                                }
                                BinOpKind::Mod => {
                                    let is_nz = builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                                    let rbb = builder.create_block();
                                    let mbb = builder.create_block();
                                    builder.append_block_param(mbb, I64);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().brif(is_nz, rbb, &[], mbb, &[zero]);
                                    builder.switch_to_block(rbb);
                                    builder.seal_block(rbb);
                                    let rem = builder.ins().srem(a, b);
                                    builder.ins().jump(mbb, &[rem]);
                                    builder.switch_to_block(mbb);
                                    builder.seal_block(mbb);
                                    builder.block_params(mbb)[0]
                                }
                                BinOpKind::Cmp => {
                                    let cmp =
                                        builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
                                    let one = builder.ins().iconst(I64, 1);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().select(cmp, one, zero)
                                }
                            };
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, r]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -16);
                            let a = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            let b = builder.ins().load(I64, MemFlags::trusted(), nt, 8);
                            let r = match kind {
                                BinOpKind::Add => builder.ins().iadd(a, b),
                                BinOpKind::Sub => builder.ins().isub(a, b),
                                BinOpKind::Mul => builder.ins().imul(a, b),
                                BinOpKind::Div => {
                                    let is_nz = builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                                    let dbb = builder.create_block();
                                    let mbb = builder.create_block();
                                    builder.append_block_param(mbb, I64);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().brif(is_nz, dbb, &[], mbb, &[zero]);
                                    builder.switch_to_block(dbb);
                                    builder.seal_block(dbb);
                                    let d = builder.ins().sdiv(a, b);
                                    builder.ins().jump(mbb, &[d]);
                                    builder.switch_to_block(mbb);
                                    builder.seal_block(mbb);
                                    builder.block_params(mbb)[0]
                                }
                                BinOpKind::Mod => {
                                    let is_nz = builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                                    let rbb = builder.create_block();
                                    let mbb = builder.create_block();
                                    builder.append_block_param(mbb, I64);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().brif(is_nz, rbb, &[], mbb, &[zero]);
                                    builder.switch_to_block(rbb);
                                    builder.seal_block(rbb);
                                    let rem = builder.ins().srem(a, b);
                                    builder.ins().jump(mbb, &[rem]);
                                    builder.switch_to_block(mbb);
                                    builder.seal_block(mbb);
                                    builder.block_params(mbb)[0]
                                }
                                BinOpKind::Cmp => {
                                    let cmp =
                                        builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
                                    let one = builder.ins().iconst(I64, 1);
                                    let zero = builder.ins().iconst(I64, 0);
                                    builder.ins().select(cmp, one, zero)
                                }
                            };
                            builder.ins().store(MemFlags::trusted(), r, nt, 0);
                            let pushed = builder.ins().iadd_imm(nt, 8);
                            store_top(
                                &mut builder,
                                tops_slot,
                                sel,
                                v_sel,
                                &v_top,
                                pushed,
                                ptr_type,
                            );
                        }
                    }
                    Inst::Sel(new_sel) => {
                        // Flush old sel's Variable to slot
                        flush_top_var(&mut builder, tops_slot, sel, &v_top);
                        // Update v_sel
                        let sv = builder.ins().iconst(I64, *new_sel as i64);
                        builder.def_var(v_sel, sv);
                        sel = Some(*new_sel);
                        // Load new sel's Variable from slot
                        reload_top_var(&mut builder, tops_slot, sel, &v_top, ptr_type);
                    }
                    Inst::Mov(target) => {
                        if sel.is_some_and(is_special) {
                            // Pop from special source
                            let ctx = builder.use_var(v_sp_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let val = builder.inst_results(ci)[0];
                            // Push to target
                            if is_special(*target) {
                                let push_fn = builder.use_var(v_sp_push);
                                let tv = builder.ins().iconst(I64, *target as i64);
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_push, push_fn, &[ctx, tv, val]);
                            } else {
                                let tt = builder.ins().stack_load(
                                    ptr_type,
                                    tops_slot,
                                    (*target * 8) as i32,
                                );
                                builder.ins().store(MemFlags::trusted(), val, tt, 0);
                                let ntt = builder.ins().iadd_imm(tt, 8);
                                builder
                                    .ins()
                                    .stack_store(ntt, tops_slot, (*target * 8) as i32);
                                // Reload Variable if target has one
                                if let Some(&var) = v_top.get(target) {
                                    builder.def_var(var, ntt);
                                }
                            }
                        } else {
                            // Pop from current sel (via Variable)
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            // Push to target storage
                            if is_special(*target) {
                                let ctx = builder.use_var(v_sp_ctx);
                                let push_fn = builder.use_var(v_sp_push);
                                let tv = builder.ins().iconst(I64, *target as i64);
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_push, push_fn, &[ctx, tv, val]);
                            } else {
                                // If target == current sel, flush Variable first so slot is coherent.
                                if sel == Some(*target) {
                                    flush_top_var(&mut builder, tops_slot, sel, &v_top);
                                }
                                let tt = builder.ins().stack_load(
                                    ptr_type,
                                    tops_slot,
                                    (*target * 8) as i32,
                                );
                                builder.ins().store(MemFlags::trusted(), val, tt, 0);
                                let ntt = builder.ins().iadd_imm(tt, 8);
                                builder
                                    .ins()
                                    .stack_store(ntt, tops_slot, (*target * 8) as i32);
                                // If target == current sel, reload Variable from updated slot
                                if sel == Some(*target) {
                                    reload_top_var(&mut builder, tops_slot, sel, &v_top, ptr_type);
                                }
                            }
                        }
                    }
                    Inst::PopChar => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let val = builder.inst_results(ci)[0];
                            let wc = builder.use_var(v_wc);
                            builder.ins().call_indirect(sig_vi64, wc, &[val]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            let wc = builder.use_var(v_wc);
                            builder.ins().call_indirect(sig_vi64, wc, &[val]);
                        }
                    }
                    Inst::PopNum => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let val = builder.inst_results(ci)[0];
                            let wn = builder.use_var(v_wn);
                            builder.ins().call_indirect(sig_vi64, wn, &[val]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            let wn = builder.use_var(v_wn);
                            builder.ins().call_indirect(sig_vi64, wn, &[val]);
                        }
                    }
                    Inst::PushNum => {
                        let rn = builder.use_var(v_rn);
                        let ci = builder.ins().call_indirect(sig_i64v, rn, &[]);
                        let val = builder.inst_results(ci)[0];
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::PushChar => {
                        let rc = builder.use_var(v_rc);
                        let ci = builder.ins().call_indirect(sig_i64v, rc, &[]);
                        let val = builder.inst_results(ci)[0];
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::GuardDepth { min_depth, fail } => {
                        // Flush Variable to slot before branch (fail path needs slot)
                        flush_top_var(&mut builder, tops_slot, sel, &v_top);
                        let elems = if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_sp_ctx);
                            let depth_fn = builder.use_var(v_sp_depth);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci =
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv]);
                            builder.inst_results(ci)[0]
                        } else if sel.is_none() {
                            // Dynamic sel: branch on special vs normal
                            let sv = builder.use_var(v_sel);
                            let is_q = builder.ins().icmp_imm(IntCC::Equal, sv, QUEUE as i64);
                            let is_p = builder.ins().icmp_imm(IntCC::Equal, sv, PORT as i64);
                            let is_sp = builder.ins().bor(is_q, is_p);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.append_block_param(merge_bb, I64);
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            // Special path
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_sp_ctx);
                            let depth_fn = builder.use_var(v_sp_depth);
                            let sv2 = builder.use_var(v_sel);
                            let ci =
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv2]);
                            let sp_depth = builder.inst_results(ci)[0];
                            builder.ins().jump(merge_bb, &[sp_depth]);
                            // Normal path
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                            let bv = builder.use_var(v_bases);
                            let sv3 = builder.use_var(v_sel);
                            let off = builder.ins().imul_imm(sv3, 8);
                            let addr = builder.ins().iadd(bv, off);
                            let base = builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0);
                            let diff = builder.ins().isub(top, base);
                            let normal_elems = builder.ins().sshr_imm(diff, 3);
                            builder.ins().jump(merge_bb, &[normal_elems]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                            builder.block_params(merge_bb)[0]
                        } else {
                            let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                            let bv = builder.use_var(v_bases);
                            let base = builder.ins().load(
                                ptr_type,
                                MemFlags::trusted(),
                                bv,
                                (sel.unwrap() * 8) as i32,
                            );
                            let diff = builder.ins().isub(top, base);
                            builder.ins().sshr_imm(diff, 3)
                        };
                        let min = builder.ins().iconst(I64, *min_depth as i64);
                        let cond = builder.ins().icmp(IntCC::SignedLessThan, elems, min);
                        let fail_cl = cl.get(fail).copied().unwrap_or(halt_block);
                        let cont = builder.create_block();
                        builder.ins().brif(cond, fail_cl, &[], cont, &[]);
                        builder.switch_to_block(cont);
                        builder.seal_block(cont);
                        // Reload Variable after branch (fresh SSA def for cont block)
                        reload_top_var(&mut builder, tops_slot, sel, &v_top, ptr_type);
                    }
                }
            }

            // Block exit: flush current sel's Variable to slot before terminator
            flush_top_var(&mut builder, tops_slot, sel, &v_top);

            // Terminator — all slot access uses load_top_slot/store_top_slot (post-flush)
            match &block.terminator {
                Terminator::Goto(target) => {
                    builder
                        .ins()
                        .jump(cl.get(target).copied().unwrap_or(halt_block), &[]);
                }
                Terminator::StackGuard {
                    min_depth,
                    ok,
                    fail,
                } => {
                    let elems = if sel.is_some_and(is_special) {
                        let ctx = builder.use_var(v_sp_ctx);
                        let depth_fn = builder.use_var(v_sp_depth);
                        let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci = builder
                            .ins()
                            .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv]);
                        builder.inst_results(ci)[0]
                    } else if sel.is_none() {
                        // Dynamic sel: branch on special vs normal
                        let sv = builder.use_var(v_sel);
                        let is_q = builder.ins().icmp_imm(IntCC::Equal, sv, QUEUE as i64);
                        let is_p = builder.ins().icmp_imm(IntCC::Equal, sv, PORT as i64);
                        let is_sp = builder.ins().bor(is_q, is_p);
                        let sp_bb = builder.create_block();
                        let normal_bb = builder.create_block();
                        let merge_bb = builder.create_block();
                        builder.append_block_param(merge_bb, I64);
                        builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                        builder.switch_to_block(sp_bb);
                        builder.seal_block(sp_bb);
                        let ctx = builder.use_var(v_sp_ctx);
                        let depth_fn = builder.use_var(v_sp_depth);
                        let sv2 = builder.use_var(v_sel);
                        let ci = builder
                            .ins()
                            .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv2]);
                        let sp_depth = builder.inst_results(ci)[0];
                        builder.ins().jump(merge_bb, &[sp_depth]);
                        builder.switch_to_block(normal_bb);
                        builder.seal_block(normal_bb);
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let bv = builder.use_var(v_bases);
                        let sv3 = builder.use_var(v_sel);
                        let off = builder.ins().imul_imm(sv3, 8);
                        let addr = builder.ins().iadd(bv, off);
                        let base = builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0);
                        let diff = builder.ins().isub(top, base);
                        let normal_elems = builder.ins().sshr_imm(diff, 3);
                        builder.ins().jump(merge_bb, &[normal_elems]);
                        builder.switch_to_block(merge_bb);
                        builder.seal_block(merge_bb);
                        builder.block_params(merge_bb)[0]
                    } else {
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let bv = builder.use_var(v_bases);
                        let base = builder.ins().load(
                            ptr_type,
                            MemFlags::trusted(),
                            bv,
                            (sel.unwrap() * 8) as i32,
                        );
                        let diff = builder.ins().isub(top, base);
                        builder.ins().sshr_imm(diff, 3)
                    };
                    let min = builder.ins().iconst(I64, *min_depth as i64);
                    let ok_c = builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, elems, min);
                    builder.ins().brif(
                        ok_c,
                        cl.get(ok).copied().unwrap_or(halt_block),
                        &[],
                        cl.get(fail).copied().unwrap_or(halt_block),
                        &[],
                    );
                }
                Terminator::BranchZero {
                    on_zero,
                    on_nonzero,
                } => {
                    let val = if sel.is_some_and(is_special) {
                        let ctx = builder.use_var(v_sp_ctx);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                        builder.inst_results(ci)[0]
                    } else if sel.is_none() {
                        // Dynamic sel: branch on special vs normal
                        let sv = builder.use_var(v_sel);
                        let is_q = builder.ins().icmp_imm(IntCC::Equal, sv, QUEUE as i64);
                        let is_p = builder.ins().icmp_imm(IntCC::Equal, sv, PORT as i64);
                        let is_sp = builder.ins().bor(is_q, is_p);
                        let sp_bb = builder.create_block();
                        let normal_bb = builder.create_block();
                        let merge_bb = builder.create_block();
                        builder.append_block_param(merge_bb, I64);
                        builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                        builder.switch_to_block(sp_bb);
                        builder.seal_block(sp_bb);
                        let ctx = builder.use_var(v_sp_ctx);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let sv2 = builder.use_var(v_sel);
                        let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv2]);
                        let sp_val = builder.inst_results(ci)[0];
                        builder.ins().jump(merge_bb, &[sp_val]);
                        builder.switch_to_block(normal_bb);
                        builder.seal_block(normal_bb);
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let nt = builder.ins().iadd_imm(top, -8);
                        let normal_val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                        store_top_slot(&mut builder, tops_slot, sel, v_sel, nt, ptr_type);
                        builder.ins().jump(merge_bb, &[normal_val]);
                        builder.switch_to_block(merge_bb);
                        builder.seal_block(merge_bb);
                        builder.block_params(merge_bb)[0]
                    } else {
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let nt = builder.ins().iadd_imm(top, -8);
                        let v = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                        store_top_slot(&mut builder, tops_slot, sel, v_sel, nt, ptr_type);
                        v
                    };
                    let iz = builder.ins().icmp_imm(IntCC::Equal, val, 0);
                    builder.ins().brif(
                        iz,
                        cl.get(on_zero).copied().unwrap_or(halt_block),
                        &[],
                        cl.get(on_nonzero).copied().unwrap_or(halt_block),
                        &[],
                    );
                }
                Terminator::Halt => {
                    if sel.is_some_and(is_special) {
                        // For special storage, check depth > 0 then pop
                        let ctx = builder.use_var(v_sp_ctx);
                        let depth_fn = builder.use_var(v_sp_depth);
                        let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci = builder
                            .ins()
                            .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv]);
                        let depth = builder.inst_results(ci)[0];
                        let has = builder.ins().icmp_imm(IntCC::SignedGreaterThan, depth, 0);
                        let has_bb = builder.create_block();
                        let ret_bb = builder.create_block();
                        builder.append_block_param(ret_bb, I64);
                        let zero = builder.ins().iconst(I64, 0);
                        builder.ins().brif(has, has_bb, &[], ret_bb, &[zero]);
                        builder.switch_to_block(has_bb);
                        builder.seal_block(has_bb);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let ctx2 = builder.use_var(v_sp_ctx);
                        let sv2 = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci2 = builder
                            .ins()
                            .call_indirect(sig_sp_pop, pop_fn, &[ctx2, sv2]);
                        let val = builder.inst_results(ci2)[0];
                        builder.ins().jump(ret_bb, &[val]);
                        builder.switch_to_block(ret_bb);
                        builder.seal_block(ret_bb);
                        let ret = builder.block_params(ret_bb)[0];
                        builder.ins().return_(&[ret]);
                    } else {
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let bv = builder.use_var(v_bases);
                        let base = if let Some(s) = sel {
                            builder
                                .ins()
                                .load(ptr_type, MemFlags::trusted(), bv, (s * 8) as i32)
                        } else {
                            let sv = builder.use_var(v_sel);
                            let off = builder.ins().imul_imm(sv, 8);
                            let addr = builder.ins().iadd(bv, off);
                            builder.ins().load(ptr_type, MemFlags::trusted(), addr, 0)
                        };
                        let has = builder.ins().icmp(IntCC::UnsignedGreaterThan, top, base);
                        let has_bb = builder.create_block();
                        let ret_bb = builder.create_block();
                        builder.append_block_param(ret_bb, I64);
                        let zero = builder.ins().iconst(I64, 0);
                        builder.ins().brif(has, has_bb, &[], ret_bb, &[zero]);
                        builder.switch_to_block(has_bb);
                        builder.seal_block(has_bb);
                        let val = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                        builder.ins().jump(ret_bb, &[val]);
                        builder.switch_to_block(ret_bb);
                        builder.seal_block(ret_bb);
                        let ret = builder.block_params(ret_bb)[0];
                        builder.ins().return_(&[ret]);
                    }
                }
            }
        }

        // Halt fallback
        builder.switch_to_block(halt_block);
        builder.seal_block(halt_block);
        let zr = builder.ins().iconst(I64, 0);
        builder.ins().return_(&[zr]);

        builder.seal_all_blocks();
        builder.finalize();

        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| e.to_string())?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().map_err(|e| e.to_string())?;
        let code_ptr = module.get_finalized_function(func_id);
        Ok(JitFunction {
            _module: module,
            code_ptr,
        })
    }
}
