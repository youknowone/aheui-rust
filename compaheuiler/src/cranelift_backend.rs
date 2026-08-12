//! Cranelift JIT backend for Aheui AOT compiler.
//!
//! Per-storage Cranelift Variables for sel-known blocks enable register promotion.
//! The Variable for the CURRENT sel tracks the top pointer in a register.
//! On sel change or block boundary, the Variable is flushed to / loaded from tops_slot.

#[cfg(feature = "cranelift")]
pub mod jit {
    use ahsembler::cfg::*;
    use ahsembler::consts::STORAGE_COUNT;
    #[cfg(feature = "bigint")]
    use ahsembler::consts::{floor_div_i64, floor_mod_i64};
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::ir::types::*;
    use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{Linkage, Module};
    use std::collections::{BTreeSet, HashMap, VecDeque};

    #[cfg(all(feature = "bigint", not(feature = "num-bigint")))]
    use malachite_bigint::BigInt;
    #[cfg(all(feature = "bigint", feature = "num-bigint"))]
    use num_bigint::BigInt;
    #[cfg(feature = "bigint")]
    use num_traits::ToPrimitive;

    const QUEUE: usize = 21;
    const PORT: usize = 27;
    const OUTPUT_BUFFER_CAPACITY: usize = 16 * 1024;
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

    /// Per-execution owner for the dual-mode state.  The generated function
    /// receives this pointer in its existing `sp_ctx` ABI slot; the remaining
    /// callback slots point at the adapters below.  This keeps bigint mode and
    /// all user callbacks execution-local without TLS or global registries.
    #[repr(C)]
    struct ExecutionContext {
        big_mode: i64,
        write_char: extern "C" fn(i64),
        write_num: extern "C" fn(i64),
        read_char: extern "C" fn() -> i64,
        read_num: extern "C" fn() -> i64,
        sp_ctx: *mut u8,
        sp_push: extern "C" fn(*mut u8, usize, i64),
        sp_pop: extern "C" fn(*mut u8, usize) -> i64,
        sp_depth: extern "C" fn(*mut u8, usize) -> i64,
        sp_dup: extern "C" fn(*mut u8, usize),
        sp_swap: extern "C" fn(*mut u8, usize),
        write_bytes: Option<extern "C" fn(*const u8, usize)>,
        buffer_output: i64,
        output_len: usize,
        output: [u8; OUTPUT_BUFFER_CAPACITY],
        #[cfg(feature = "bigint")]
        bigints: Vec<Box<BigInt>>,
    }

    #[inline]
    unsafe fn execution_context<'a>(ctx: *mut u8) -> &'a mut ExecutionContext {
        unsafe { &mut *(ctx as *mut ExecutionContext) }
    }

    extern "C" fn runtime_flush_output(raw_ctx: *mut u8) {
        let ctx = unsafe { execution_context(raw_ctx) };
        let len = ctx.output_len;
        if len == 0 {
            return;
        }
        ctx.output_len = 0;
        if let Some(write_bytes) = ctx.write_bytes {
            write_bytes(ctx.output.as_ptr(), len);
        } else {
            let write_char = ctx.write_char;
            for &byte in &ctx.output[..len] {
                write_char(byte as i64);
            }
        }
    }

    const SMALL_MIN: i64 = -(1_i64 << 62);
    const SMALL_MAX: i64 = (1_i64 << 62) - 1;

    #[cfg(feature = "bigint")]
    #[inline]
    fn promote_value(ctx: &mut ExecutionContext, value: i64) -> i64 {
        if (SMALL_MIN..=SMALL_MAX).contains(&value) {
            (value << 1) | 1
        } else {
            store_bigint(ctx, BigInt::from(value))
        }
    }

    #[cfg(feature = "bigint")]
    #[inline]
    fn store_bigint(ctx: &mut ExecutionContext, value: BigInt) -> i64 {
        let value = Box::new(value);
        let ptr = (&*value as *const BigInt) as i64;
        ctx.bigints.push(value);
        ptr
    }

    #[cfg(feature = "bigint")]
    #[inline]
    fn tagged_to_big(value: i64) -> BigInt {
        if value & 1 != 0 {
            BigInt::from(value >> 1)
        } else {
            unsafe { &*(value as *const BigInt) }.clone()
        }
    }

    #[cfg(feature = "bigint")]
    #[inline]
    fn normalize_big(ctx: &mut ExecutionContext, value: BigInt) -> i64 {
        match value.to_i64() {
            Some(value) if (SMALL_MIN..=SMALL_MAX).contains(&value) => (value << 1) | 1,
            _ => store_bigint(ctx, value),
        }
    }

    #[cfg(feature = "bigint")]
    #[inline]
    fn floor_bigint_divmod(a: BigInt, b: BigInt) -> (BigInt, BigInt) {
        let mut q = a.clone() / b.clone();
        let mut r = a % b.clone();
        if r != BigInt::from(0) && (r < BigInt::from(0)) != (b < BigInt::from(0)) {
            q -= BigInt::from(1);
            r += b;
        }
        (q, r)
    }

    #[cfg(feature = "bigint")]
    unsafe fn promote_execution(
        ctx: &mut ExecutionContext,
        bases: *mut *mut i64,
        tops: *mut *mut i64,
    ) {
        ctx.big_mode = 1;
        for storage in 0..STORAGE_COUNT {
            let mut value = unsafe { *bases.add(storage) };
            let top = unsafe { *tops.add(storage) };
            while value < top {
                unsafe { *value = promote_value(ctx, *value) };
                value = unsafe { value.add(1) };
            }
        }
        // `execute` documents and enforces that this is the backend's own
        // SpecialStorage. Queue order and port_last are semantic state too.
        if !ctx.sp_ctx.is_null() {
            let special = unsafe { &mut *(ctx.sp_ctx as *mut SpecialStorage) };
            for value in &mut special.queue {
                *value = promote_value(ctx, *value);
            }
            for value in &mut special.port {
                *value = promote_value(ctx, *value);
            }
            special.port_last = promote_value(ctx, special.port_last);
        }
    }

    /// Bigint slow path used only after the in-IR checked i64 path overflows,
    /// or after an earlier operation has already put the execution in tagged
    /// mode. `opcode` follows BinOpKind's stable order below.
    #[cfg(feature = "bigint")]
    extern "C" fn runtime_bigint_binop(
        raw_ctx: *mut u8,
        opcode: i64,
        a: i64,
        b: i64,
        bases: *mut *mut i64,
        tops: *mut *mut i64,
    ) -> i64 {
        let ctx = unsafe { execution_context(raw_ctx) };
        if ctx.big_mode != 0 && a & b & 1 != 0 {
            let (small_a, small_b) = (a >> 1, b >> 1);
            let small = match opcode {
                0 => small_a.checked_add(small_b),
                1 => small_a.checked_sub(small_b),
                2 => small_a.checked_mul(small_b),
                3 if small_b != 0 => Some(floor_div_i64(small_a, small_b)),
                4 if small_b != 0 => Some(floor_mod_i64(small_a, small_b)),
                3 | 4 => Some(0),
                5 => return if small_a >= small_b { 3 } else { 1 },
                _ => unreachable!("unknown Aheui bigint opcode"),
            };
            if let Some(value) = small.filter(|value| (SMALL_MIN..=SMALL_MAX).contains(value)) {
                return (value << 1) | 1;
            }
        }
        let (a, b) = if ctx.big_mode != 0 {
            (tagged_to_big(a), tagged_to_big(b))
        } else {
            unsafe { promote_execution(ctx, bases, tops) };
            (BigInt::from(a), BigInt::from(b))
        };
        match opcode {
            0 => normalize_big(ctx, a + b),
            1 => normalize_big(ctx, a - b),
            2 => normalize_big(ctx, a * b),
            3 => {
                if b == BigInt::from(0) {
                    1
                } else {
                    normalize_big(ctx, floor_bigint_divmod(a, b).0)
                }
            }
            4 => {
                if b == BigInt::from(0) {
                    1
                } else {
                    normalize_big(ctx, floor_bigint_divmod(a, b).1)
                }
            }
            5 => {
                if a >= b {
                    3
                } else {
                    1
                }
            }
            _ => unreachable!("unknown Aheui bigint opcode"),
        }
    }

    #[cfg(not(feature = "bigint"))]
    extern "C" fn runtime_bigint_binop(
        _raw_ctx: *mut u8,
        _opcode: i64,
        _a: i64,
        _b: i64,
        _bases: *mut *mut i64,
        _tops: *mut *mut i64,
    ) -> i64 {
        unreachable!("bigint slow path emitted without the bigint feature")
    }

    extern "C" fn runtime_write_char(raw_ctx: *mut u8, value: i64) {
        runtime_flush_output(raw_ctx);
        let ctx = unsafe { execution_context(raw_ctx) };
        #[cfg(feature = "bigint")]
        let value = if ctx.big_mode != 0 {
            if value & 1 != 0 {
                value >> 1
            } else {
                unsafe { &*(value as *const BigInt) }.to_i64().unwrap_or(0)
            }
        } else {
            value
        };
        (ctx.write_char)(value);
    }

    extern "C" fn runtime_write_num(raw_ctx: *mut u8, value: i64) {
        runtime_flush_output(raw_ctx);
        let ctx = unsafe { execution_context(raw_ctx) };
        #[cfg(feature = "bigint")]
        if ctx.big_mode != 0 {
            if value & 1 != 0 {
                (ctx.write_num)(value >> 1);
            } else {
                for byte in unsafe { &*(value as *const BigInt) }.to_string().bytes() {
                    (ctx.write_char)(byte as i64);
                }
            }
            return;
        }
        (ctx.write_num)(value);
    }

    extern "C" fn runtime_read_char(raw_ctx: *mut u8) -> i64 {
        runtime_flush_output(raw_ctx);
        let ctx = unsafe { execution_context(raw_ctx) };
        let value = (ctx.read_char)();
        #[cfg(feature = "bigint")]
        if ctx.big_mode != 0 {
            return promote_value(ctx, value);
        }
        value
    }

    extern "C" fn runtime_read_num(raw_ctx: *mut u8) -> i64 {
        runtime_flush_output(raw_ctx);
        let ctx = unsafe { execution_context(raw_ctx) };
        let value = (ctx.read_num)();
        #[cfg(feature = "bigint")]
        if ctx.big_mode != 0 {
            return promote_value(ctx, value);
        }
        value
    }

    extern "C" fn runtime_literal(raw_ctx: *mut u8, value: i64) -> i64 {
        #[cfg(feature = "bigint")]
        {
            let ctx = unsafe { execution_context(raw_ctx) };
            if ctx.big_mode != 0 {
                return promote_value(ctx, value);
            }
        }
        #[cfg(not(feature = "bigint"))]
        let _ = raw_ctx;
        value
    }

    extern "C" fn runtime_to_i64(raw_ctx: *mut u8, value: i64) -> i64 {
        #[cfg(feature = "bigint")]
        let ctx = unsafe { execution_context(raw_ctx) };
        #[cfg(not(feature = "bigint"))]
        let _ = raw_ctx;
        #[cfg(feature = "bigint")]
        if ctx.big_mode != 0 {
            if value == 0 {
                // Empty-storage Halt contributes the ABI's raw zero rather
                // than a value word; it is not a tagged bigint pointer.
                return 0;
            }
            return if value & 1 != 0 {
                value >> 1
            } else {
                unsafe { &*(value as *const BigInt) }.to_i64().unwrap_or(0)
            };
        }
        value
    }

    extern "C" fn runtime_sp_push(raw_ctx: *mut u8, sel: usize, value: i64) {
        let ctx = unsafe { execution_context(raw_ctx) };
        (ctx.sp_push)(ctx.sp_ctx, sel, value);
    }
    extern "C" fn runtime_sp_pop(raw_ctx: *mut u8, sel: usize) -> i64 {
        let ctx = unsafe { execution_context(raw_ctx) };
        let value = (ctx.sp_pop)(ctx.sp_ctx, sel);
        #[cfg(feature = "bigint")]
        if ctx.big_mode != 0 && value == 0 {
            return 1;
        }
        value
    }
    extern "C" fn runtime_sp_depth(raw_ctx: *mut u8, sel: usize) -> i64 {
        let ctx = unsafe { execution_context(raw_ctx) };
        (ctx.sp_depth)(ctx.sp_ctx, sel)
    }
    extern "C" fn runtime_sp_dup(raw_ctx: *mut u8, sel: usize) {
        let ctx = unsafe { execution_context(raw_ctx) };
        (ctx.sp_dup)(ctx.sp_ctx, sel);
    }
    extern "C" fn runtime_sp_swap(raw_ctx: *mut u8, sel: usize) {
        let ctx = unsafe { execution_context(raw_ctx) };
        (ctx.sp_swap)(ctx.sp_ctx, sel);
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
        /// # Safety
        ///
        /// `bases` must point to writable storage for the whole execution. When
        /// bigint support is enabled, a non-null `sp_ctx` must point to this
        /// crate's [`SpecialStorage`], and the special-storage callbacks must be
        /// compatible with it: promotion traverses that concrete type directly.
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
            unsafe {
                self.execute_inner(
                    bases,
                    lengths,
                    write_char_fn,
                    None,
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

        /// Execute with an additional byte-slice sink. Raw ASCII output is
        /// buffered inside the execution context and delivered in 16 KiB
        /// batches; non-ASCII and post-promotion output retains the ordinary
        /// character callback semantics.
        ///
        /// # Safety
        ///
        /// `bases` must point to writable storage for the whole execution. When
        /// bigint support is enabled, a non-null `sp_ctx` must point to this
        /// crate's [`SpecialStorage`], and the special-storage callbacks must be
        /// compatible with it: promotion traverses that concrete type directly.
        pub unsafe fn execute_buffered(
            &self,
            bases: &mut [*mut i64; STORAGE_COUNT],
            lengths: &mut [i32; STORAGE_COUNT],
            write_char_fn: extern "C" fn(i64),
            write_bytes_fn: extern "C" fn(*const u8, usize),
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
            unsafe {
                self.execute_inner(
                    bases,
                    lengths,
                    write_char_fn,
                    Some(write_bytes_fn),
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

        #[allow(clippy::too_many_arguments)]
        unsafe fn execute_inner(
            &self,
            bases: &mut [*mut i64; STORAGE_COUNT],
            lengths: &mut [i32; STORAGE_COUNT],
            write_char_fn: extern "C" fn(i64),
            write_bytes_fn: Option<extern "C" fn(*const u8, usize)>,
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
                extern "C" fn(*mut u8, i64),
                extern "C" fn(*mut u8, i64),
                extern "C" fn(*mut u8) -> i64,
                extern "C" fn(*mut u8) -> i64,
                *mut u8,
                extern "C" fn(*mut u8, usize, i64),
                extern "C" fn(*mut u8, usize) -> i64,
                extern "C" fn(*mut u8, usize) -> i64,
                extern "C" fn(*mut u8, usize),
                extern "C" fn(*mut u8, usize),
                extern "C" fn(*mut u8),
            ) -> i64;
            let func: Fn = unsafe { std::mem::transmute(self.code_ptr) };
            let mut runtime = ExecutionContext {
                big_mode: 0,
                write_char: write_char_fn,
                write_num: write_num_fn,
                read_char: read_char_fn,
                read_num: read_num_fn,
                sp_ctx,
                sp_push: sp_push_fn,
                sp_pop: sp_pop_fn,
                sp_depth: sp_depth_fn,
                sp_dup: sp_dup_fn,
                sp_swap: sp_swap_fn,
                write_bytes: write_bytes_fn,
                buffer_output: i64::from(write_bytes_fn.is_some()),
                output_len: 0,
                output: [0; OUTPUT_BUFFER_CAPACITY],
                #[cfg(feature = "bigint")]
                bigints: Vec::new(),
            };
            let runtime_ctx = &mut runtime as *mut ExecutionContext as *mut u8;
            let result = unsafe {
                func(
                    bases.as_mut_ptr(),
                    lengths.as_mut_ptr(),
                    runtime_write_char,
                    runtime_write_num,
                    runtime_read_char,
                    runtime_read_num,
                    runtime_ctx,
                    runtime_sp_push,
                    runtime_sp_pop,
                    runtime_sp_depth,
                    runtime_sp_dup,
                    runtime_sp_swap,
                    runtime_flush_output,
                )
            };
            runtime_flush_output(runtime_ctx);
            result
        }
    }

    /// Raw i64 operation. Division and modulo answer 0 for a zero divisor,
    /// which needs a branch, so this cannot be a plain expression.
    fn emit_raw_binop(
        builder: &mut FunctionBuilder,
        kind: &BinOpKind,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
    ) -> cranelift_codegen::ir::Value {
        match kind {
            BinOpKind::Add => builder.ins().iadd(a, b),
            BinOpKind::Sub => builder.ins().isub(a, b),
            BinOpKind::Mul => builder.ins().imul(a, b),
            BinOpKind::Div | BinOpKind::Mod => {
                let is_nz = builder.ins().icmp_imm(IntCC::NotEqual, b, 0);
                let opbb = builder.create_block();
                let mbb = builder.create_block();
                builder.append_block_param(mbb, I64);
                let zero = builder.ins().iconst(I64, 0);
                let (can_operate, fallback) = if matches!(kind, BinOpKind::Div) {
                    let is_min = builder.ins().icmp_imm(IntCC::Equal, a, i64::MIN);
                    let is_neg_one = builder.ins().icmp_imm(IntCC::Equal, b, -1);
                    let overflows = builder.ins().band(is_min, is_neg_one);
                    let does_not_overflow = builder.ins().icmp_imm(IntCC::Equal, overflows, 0);
                    let can_divide = builder.ins().band(is_nz, does_not_overflow);
                    let wrapped = builder.ins().iconst(I64, i64::MIN);
                    (can_divide, builder.ins().select(overflows, wrapped, zero))
                } else {
                    (is_nz, zero)
                };
                builder
                    .ins()
                    .brif(can_operate, opbb, &[], mbb, &[fallback.into()]);
                builder.switch_to_block(opbb);
                builder.seal_block(opbb);
                let r = builder.ins().srem(a, b);
                let r_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, r, 0);
                let r_negative = builder.ins().icmp_imm(IntCC::SignedLessThan, r, 0);
                let b_negative = builder.ins().icmp_imm(IntCC::SignedLessThan, b, 0);
                let signs_differ = builder.ins().bxor(r_negative, b_negative);
                let needs_floor = builder.ins().band(r_nonzero, signs_differ);
                let v = if matches!(kind, BinOpKind::Div) {
                    let q = builder.ins().sdiv(a, b);
                    let floor_q = builder.ins().iadd_imm(q, -1);
                    builder.ins().select(needs_floor, floor_q, q)
                } else {
                    let floor_r = builder.ins().iadd(r, b);
                    builder.ins().select(needs_floor, floor_r, r)
                };
                builder.ins().jump(mbb, &[v.into()]);
                builder.switch_to_block(mbb);
                builder.seal_block(mbb);
                builder.block_params(mbb)[0]
            }
            BinOpKind::Cmp => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, a, b);
                let one = builder.ins().iconst(I64, 1);
                let zero = builder.ins().iconst(I64, 0);
                builder.ins().select(cmp, one, zero)
            }
        }
    }

    fn flush_all_top_vars(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        v_top: &HashMap<usize, Variable>,
    ) {
        for (&storage, &var) in v_top {
            let top = builder.use_var(var);
            builder
                .ins()
                .stack_store(top, tops_slot, (storage * 8) as i32);
        }
    }

    fn reload_all_top_vars(
        builder: &mut FunctionBuilder,
        tops_slot: cranelift_codegen::ir::StackSlot,
        v_top: &HashMap<usize, Variable>,
        ptr_type: cranelift_codegen::ir::Type,
    ) {
        for (&storage, &var) in v_top {
            let top = builder
                .ins()
                .stack_load(ptr_type, tops_slot, (storage * 8) as i32);
            builder.def_var(var, top);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_binop(
        builder: &mut FunctionBuilder,
        kind: &BinOpKind,
        a: cranelift_codegen::ir::Value,
        b: cranelift_codegen::ir::Value,
        bigint: bool,
        v_runtime_ctx: Variable,
        v_bases: Variable,
        tops_slot: cranelift_codegen::ir::StackSlot,
        v_top: &HashMap<usize, Variable>,
        ptr_type: cranelift_codegen::ir::Type,
        sig_bigint: Option<cranelift_codegen::ir::SigRef>,
    ) -> cranelift_codegen::ir::Value {
        let Some(sig_bigint) = sig_bigint.filter(|_| bigint) else {
            return emit_raw_binop(builder, kind, a, b);
        };

        let ctx = builder.use_var(v_runtime_ctx);
        let mode = builder.ins().load(
            I64,
            MemFlags::trusted(),
            ctx,
            std::mem::offset_of!(ExecutionContext, big_mode) as i32,
        );
        let tagged = builder.ins().icmp_imm(IntCC::NotEqual, mode, 0);
        let big_block = builder.create_block();
        let raw_block = builder.create_block();
        let merge = builder.create_block();
        builder.append_block_param(merge, I64);
        builder.ins().brif(tagged, big_block, &[], raw_block, &[]);

        let emit_slow = |builder: &mut FunctionBuilder| {
            let ctx = builder.use_var(v_runtime_ctx);
            let bases = builder.use_var(v_bases);
            let tops = builder.ins().stack_addr(ptr_type, tops_slot, 0);
            let opcode = builder.ins().iconst(
                I64,
                match kind {
                    BinOpKind::Add => 0,
                    BinOpKind::Sub => 1,
                    BinOpKind::Mul => 2,
                    BinOpKind::Div => 3,
                    BinOpKind::Mod => 4,
                    BinOpKind::Cmp => 5,
                },
            );
            let address = builder
                .ins()
                .iconst(ptr_type, runtime_bigint_binop as *const () as usize as i64);
            let call =
                builder
                    .ins()
                    .call_indirect(sig_bigint, address, &[ctx, opcode, a, b, bases, tops]);
            builder.inst_results(call)[0]
        };

        builder.switch_to_block(big_block);
        builder.seal_block(big_block);
        let result = emit_slow(builder);
        builder.ins().jump(merge, &[result.into()]);

        builder.switch_to_block(raw_block);
        builder.seal_block(raw_block);
        if matches!(kind, BinOpKind::Add | BinOpKind::Sub | BinOpKind::Mul) {
            let (result, overflow) = match kind {
                BinOpKind::Add => builder.ins().sadd_overflow(a, b),
                BinOpKind::Sub => builder.ins().ssub_overflow(a, b),
                BinOpKind::Mul => builder.ins().smul_overflow(a, b),
                _ => unreachable!(),
            };
            let overflow_block = builder.create_block();
            builder
                .ins()
                .brif(overflow, overflow_block, &[], merge, &[result.into()]);
            builder.switch_to_block(overflow_block);
            builder.seal_block(overflow_block);
            flush_all_top_vars(builder, tops_slot, v_top);
            let result = emit_slow(builder);
            reload_all_top_vars(builder, tops_slot, v_top, ptr_type);
            builder.ins().jump(merge, &[result.into()]);
        } else if matches!(kind, BinOpKind::Div) {
            let is_min = builder.ins().icmp_imm(IntCC::Equal, a, i64::MIN);
            let is_neg_one = builder.ins().icmp_imm(IntCC::Equal, b, -1);
            let overflow = builder.ins().band(is_min, is_neg_one);
            let overflow_block = builder.create_block();
            let normal_block = builder.create_block();
            builder
                .ins()
                .brif(overflow, overflow_block, &[], normal_block, &[]);

            builder.switch_to_block(overflow_block);
            builder.seal_block(overflow_block);
            flush_all_top_vars(builder, tops_slot, v_top);
            let result = emit_slow(builder);
            reload_all_top_vars(builder, tops_slot, v_top, ptr_type);
            builder.ins().jump(merge, &[result.into()]);

            builder.switch_to_block(normal_block);
            builder.seal_block(normal_block);
            let result = emit_raw_binop(builder, kind, a, b);
            builder.ins().jump(merge, &[result.into()]);
        } else {
            let result = emit_raw_binop(builder, kind, a, b);
            builder.ins().jump(merge, &[result.into()]);
        }

        builder.switch_to_block(merge);
        builder.seal_block(merge);
        builder.block_params(merge)[0]
    }

    fn emit_literal(
        builder: &mut FunctionBuilder,
        value: i64,
        bigint: bool,
        v_runtime_ctx: Variable,
        ptr_type: cranelift_codegen::ir::Type,
        signature: cranelift_codegen::ir::SigRef,
    ) -> cranelift_codegen::ir::Value {
        let raw = builder.ins().iconst(I64, value);
        if !bigint {
            return raw;
        }
        if !(SMALL_MIN..=SMALL_MAX).contains(&value) {
            let ctx = builder.use_var(v_runtime_ctx);
            let address = builder
                .ins()
                .iconst(ptr_type, runtime_literal as *const () as usize as i64);
            let call = builder.ins().call_indirect(signature, address, &[ctx, raw]);
            return builder.inst_results(call)[0];
        }
        let tagged = builder
            .ins()
            .iconst(I64, ((value as u64).wrapping_shl(1) | 1) as i64);
        let ctx = builder.use_var(v_runtime_ctx);
        let mode = builder.ins().load(I64, MemFlags::trusted(), ctx, 0);
        let is_big = builder.ins().icmp_imm(IntCC::NotEqual, mode, 0);
        builder.ins().select(is_big, tagged, raw)
    }

    fn emit_is_zero(
        builder: &mut FunctionBuilder,
        value: cranelift_codegen::ir::Value,
        bigint: bool,
        v_runtime_ctx: Variable,
        ptr_type: cranelift_codegen::ir::Type,
        signature: cranelift_codegen::ir::SigRef,
    ) -> cranelift_codegen::ir::Value {
        let zero = emit_literal(builder, 0, bigint, v_runtime_ctx, ptr_type, signature);
        builder.ins().icmp(IntCC::Equal, value, zero)
    }

    fn emit_to_i64(
        builder: &mut FunctionBuilder,
        value: cranelift_codegen::ir::Value,
        v_runtime_ctx: Variable,
        ptr_type: cranelift_codegen::ir::Type,
        signature: cranelift_codegen::ir::SigRef,
    ) -> cranelift_codegen::ir::Value {
        let ctx = builder.use_var(v_runtime_ctx);
        let address = builder
            .ins()
            .iconst(ptr_type, runtime_to_i64 as *const () as usize as i64);
        let call = builder
            .ins()
            .call_indirect(signature, address, &[ctx, value]);
        builder.inst_results(call)[0]
    }

    /// Keep the overwhelmingly common pre-promotion ASCII path inside the JIT.
    /// The old path crossed two indirect Rust callbacks for every character;
    /// logo emits nearly one million ASCII bytes.
    fn emit_write_char(
        builder: &mut FunctionBuilder,
        value: cranelift_codegen::ir::Value,
        v_runtime_ctx: Variable,
        v_write_char: Variable,
        v_flush_output: Variable,
        ptr_type: cranelift_codegen::ir::Type,
        sig_write_char: cranelift_codegen::ir::SigRef,
        sig_flush_output: cranelift_codegen::ir::SigRef,
    ) {
        let ctx = builder.use_var(v_runtime_ctx);
        let buffered = builder.ins().load(
            I64,
            MemFlags::trusted(),
            ctx,
            std::mem::offset_of!(ExecutionContext, buffer_output) as i32,
        );
        let enabled = builder.ins().icmp_imm(IntCC::NotEqual, buffered, 0);
        let mode = builder.ins().load(
            I64,
            MemFlags::trusted(),
            ctx,
            std::mem::offset_of!(ExecutionContext, big_mode) as i32,
        );
        let raw = builder.ins().icmp_imm(IntCC::Equal, mode, 0);
        let ascii = builder
            .ins()
            .icmp_imm(IntCC::UnsignedLessThanOrEqual, value, 0x7f);
        let fast = builder.ins().band(enabled, raw);
        let fast = builder.ins().band(fast, ascii);

        let fast_block = builder.create_block();
        let slow_block = builder.create_block();
        let flush_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.ins().brif(fast, fast_block, &[], slow_block, &[]);

        builder.switch_to_block(fast_block);
        builder.seal_block(fast_block);
        let len = builder.ins().load(
            ptr_type,
            MemFlags::trusted(),
            ctx,
            std::mem::offset_of!(ExecutionContext, output_len) as i32,
        );
        let output = builder
            .ins()
            .iadd_imm(ctx, std::mem::offset_of!(ExecutionContext, output) as i64);
        let address = builder.ins().iadd(output, len);
        let byte = builder.ins().ireduce(I8, value);
        builder.ins().store(MemFlags::trusted(), byte, address, 0);
        let next_len = builder.ins().iadd_imm(len, 1);
        builder.ins().store(
            MemFlags::trusted(),
            next_len,
            ctx,
            std::mem::offset_of!(ExecutionContext, output_len) as i32,
        );
        let full = builder
            .ins()
            .icmp_imm(IntCC::Equal, next_len, OUTPUT_BUFFER_CAPACITY as i64);
        builder.ins().brif(full, flush_block, &[], merge_block, &[]);

        builder.switch_to_block(flush_block);
        builder.seal_block(flush_block);
        let flush = builder.use_var(v_flush_output);
        builder.ins().call_indirect(sig_flush_output, flush, &[ctx]);
        builder.ins().jump(merge_block, &[]);

        builder.switch_to_block(slow_block);
        builder.seal_block(slow_block);
        let write_char = builder.use_var(v_write_char);
        builder
            .ins()
            .call_indirect(sig_write_char, write_char, &[ctx, value]);
        builder.ins().jump(merge_block, &[]);

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
    }

    /// `sel == QUEUE || sel == PORT`, for the ops whose selected storage is only
    /// known at run time.
    ///
    /// Those ops cannot take the direct-memory path unconditionally: the queue
    /// and the port have no backing array, so their `tops_slot` entries are never
    /// initialized, and reading one yields whatever the stack held.
    fn is_special_at_runtime(
        builder: &mut FunctionBuilder,
        v_sel: Variable,
    ) -> cranelift_codegen::ir::Value {
        let sv = builder.use_var(v_sel);
        let is_q = builder.ins().icmp_imm(IntCC::Equal, sv, QUEUE as i64);
        let is_p = builder.ins().icmp_imm(IntCC::Equal, sv, PORT as i64);
        builder.ins().bor(is_q, is_p)
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
        // 13 params: bases, lengths, write_char, write_num, read_char, read_num,
        //            runtime ctx, special-storage callbacks, buffered-output flush
        for _ in 0..13 {
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

        let v_bases = builder.declare_var(ptr_type);
        let v_lengths = builder.declare_var(ptr_type);
        let v_wc = builder.declare_var(ptr_type);
        let v_wn = builder.declare_var(ptr_type);
        let v_rc = builder.declare_var(ptr_type);
        let v_rn = builder.declare_var(ptr_type);
        let v_sel = builder.declare_var(I64);
        let bigint = cfg!(feature = "bigint");
        // Special storage callback variables
        let v_runtime_ctx = builder.declare_var(ptr_type);
        let v_sp_push = builder.declare_var(ptr_type);
        let v_sp_pop = builder.declare_var(ptr_type);
        let v_sp_depth = builder.declare_var(ptr_type);
        let v_sp_dup = builder.declare_var(ptr_type);
        let v_sp_swap = builder.declare_var(ptr_type);
        let v_flush_output = builder.declare_var(ptr_type);

        // Per-storage Variables for register promotion
        let mut v_top: HashMap<usize, Variable> = HashMap::new();
        for &s in &used {
            if is_special(s) {
                continue;
            }
            let v = builder.declare_var(ptr_type);
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
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        let sig_i64v = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.returns.push(AbiParam::new(I64));
            builder.import_signature(s)
        };
        let sig_flush_output = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            builder.import_signature(s)
        };
        #[cfg(feature = "bigint")]
        let sig_bigint = {
            let mut s = module.make_signature();
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(I64));
            s.params.push(AbiParam::new(I64));
            s.params.push(AbiParam::new(I64));
            s.params.push(AbiParam::new(ptr_type));
            s.params.push(AbiParam::new(ptr_type));
            s.returns.push(AbiParam::new(I64));
            Some(builder.import_signature(s))
        };
        #[cfg(not(feature = "bigint"))]
        let sig_bigint: Option<cranelift_codegen::ir::SigRef> = None;
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
        builder.def_var(v_runtime_ctx, params[6]);
        builder.def_var(v_sp_push, params[7]);
        builder.def_var(v_sp_pop, params[8]);
        builder.def_var(v_sp_depth, params[9]);
        builder.def_var(v_sp_dup, params[10]);
        builder.def_var(v_sp_swap, params[11]);
        builder.def_var(v_flush_output, params[12]);
        let z = builder.ins().iconst(I64, 0);
        builder.def_var(v_sel, z);

        // Init tops_slot: tops[s] = bases[s] + lengths[s] * 8
        // Also def_var each per-storage Variable
        let bv = builder.use_var(v_bases);
        let lv = builder.use_var(v_lengths);
        for s in 0..STORAGE_COUNT {
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
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let cv = emit_literal(
                                &mut builder,
                                *v,
                                bigint,
                                v_runtime_ctx,
                                ptr_type,
                                sig_sp_pop,
                            );
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, fn_ptr, &[ctx, sv, cv]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_push);
                            let sv = builder.use_var(v_sel);
                            let cv = emit_literal(
                                &mut builder,
                                *v,
                                bigint,
                                v_runtime_ctx,
                                ptr_type,
                                sig_sp_pop,
                            );
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, fn_ptr, &[ctx, sv, cv]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let cv = emit_literal(
                                &mut builder,
                                *v,
                                bigint,
                                v_runtime_ctx,
                                ptr_type,
                                sig_sp_pop,
                            );
                            builder.ins().store(MemFlags::trusted(), cv, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let cv = emit_literal(
                                &mut builder,
                                *v,
                                bigint,
                                v_runtime_ctx,
                                ptr_type,
                                sig_sp_pop,
                            );
                            builder.ins().store(MemFlags::trusted(), cv, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::Pop => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_pop, fn_ptr, &[ctx, sv]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_pop);
                            let sv = builder.use_var(v_sel);
                            builder.ins().call_indirect(sig_sp_pop, fn_ptr, &[ctx, sv]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::Dup => {
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_dup);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_dup, fn_ptr, &[ctx, sv]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_dup);
                            let sv = builder.use_var(v_sel);
                            builder.ins().call_indirect(sig_sp_dup, fn_ptr, &[ctx, sv]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let val = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
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
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_swap);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder.ins().call_indirect(sig_sp_swap, fn_ptr, &[ctx, sv]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let fn_ptr = builder.use_var(v_sp_swap);
                            let sv = builder.use_var(v_sel);
                            builder.ins().call_indirect(sig_sp_swap, fn_ptr, &[ctx, sv]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let a = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                            let b = builder.ins().load(I64, MemFlags::trusted(), top, -16);
                            builder.ins().store(MemFlags::trusted(), b, top, -8);
                            builder.ins().store(MemFlags::trusted(), a, top, -16);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
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
                        // Special: pop both through the callback, push the result back.
                        // Normal: both operands are already adjacent under the top.
                        let emit_special =
                            |builder: &mut FunctionBuilder, sv: cranelift_codegen::ir::Value| {
                                let ctx = builder.use_var(v_runtime_ctx);
                                let pop_fn = builder.use_var(v_sp_pop);
                                let push_fn = builder.use_var(v_sp_push);
                                let ci1 =
                                    builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                                let b = builder.inst_results(ci1)[0];
                                let ci2 =
                                    builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                                let a = builder.inst_results(ci2)[0];
                                let r = emit_binop(
                                    builder,
                                    kind,
                                    a,
                                    b,
                                    bigint,
                                    v_runtime_ctx,
                                    v_bases,
                                    tops_slot,
                                    &v_top,
                                    ptr_type,
                                    sig_bigint,
                                );
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_push, push_fn, &[ctx, sv, r]);
                            };
                        if sel.is_some_and(is_special) {
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            emit_special(&mut builder, sv);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let sv = builder.use_var(v_sel);
                            emit_special(&mut builder, sv);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -16);
                            let a = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            let b = builder.ins().load(I64, MemFlags::trusted(), nt, 8);
                            let r = emit_binop(
                                &mut builder,
                                kind,
                                a,
                                b,
                                bigint,
                                v_runtime_ctx,
                                v_bases,
                                tops_slot,
                                &v_top,
                                ptr_type,
                                sig_bigint,
                            );
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
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -16);
                            let a = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            let b = builder.ins().load(I64, MemFlags::trusted(), nt, 8);
                            let r = emit_binop(
                                &mut builder,
                                kind,
                                a,
                                b,
                                bigint,
                                v_runtime_ctx,
                                v_bases,
                                tops_slot,
                                &v_top,
                                ptr_type,
                                sig_bigint,
                            );
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
                            let ctx = builder.use_var(v_runtime_ctx);
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
                            // Pop from current sel (via Variable). With the sel only
                            // known at run time this has to branch: the queue and the
                            // port have no backing array to read a top pointer from.
                            let val = if sel.is_none() {
                                let is_sp = is_special_at_runtime(&mut builder, v_sel);
                                let sp_bb = builder.create_block();
                                let normal_bb = builder.create_block();
                                let merge_bb = builder.create_block();
                                builder.append_block_param(merge_bb, I64);
                                builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                                builder.switch_to_block(sp_bb);
                                builder.seal_block(sp_bb);
                                let ctx = builder.use_var(v_runtime_ctx);
                                let pop_fn = builder.use_var(v_sp_pop);
                                let sv = builder.use_var(v_sel);
                                let ci =
                                    builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                                let sp_val = builder.inst_results(ci)[0];
                                builder.ins().jump(merge_bb, &[sp_val.into()]);
                                builder.switch_to_block(normal_bb);
                                builder.seal_block(normal_bb);
                                let top =
                                    load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                                let nt = builder.ins().iadd_imm(top, -8);
                                let n_val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                                store_top(
                                    &mut builder,
                                    tops_slot,
                                    sel,
                                    v_sel,
                                    &v_top,
                                    nt,
                                    ptr_type,
                                );
                                builder.ins().jump(merge_bb, &[n_val.into()]);
                                builder.switch_to_block(merge_bb);
                                builder.seal_block(merge_bb);
                                builder.block_params(merge_bb)[0]
                            } else {
                                let top =
                                    load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                                let nt = builder.ins().iadd_imm(top, -8);
                                let v = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                                store_top(
                                    &mut builder,
                                    tops_slot,
                                    sel,
                                    v_sel,
                                    &v_top,
                                    nt,
                                    ptr_type,
                                );
                                v
                            };
                            // Push to target storage
                            if is_special(*target) {
                                let ctx = builder.use_var(v_runtime_ctx);
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
                                // The slot and its SSA cache have co-ownership. Keep the
                                // target Variable coherent even when it is not the current
                                // selection: overflow promotion flushes every Variable and
                                // would otherwise overwrite this slot with a stale top.
                                if let Some(&target_top) = v_top.get(target) {
                                    builder.def_var(target_top, ntt);
                                }
                            }
                        }
                    }
                    Inst::PopChar => {
                        let out_fn =
                            |builder: &mut FunctionBuilder, val: cranelift_codegen::ir::Value| {
                                emit_write_char(
                                    builder,
                                    val,
                                    v_runtime_ctx,
                                    v_wc,
                                    v_flush_output,
                                    ptr_type,
                                    sig_vi64,
                                    sig_flush_output,
                                );
                            };
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let val = builder.inst_results(ci)[0];
                            out_fn(&mut builder, val);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.append_block_param(merge_bb, I64);
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.use_var(v_sel);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let sp_val = builder.inst_results(ci)[0];
                            builder.ins().jump(merge_bb, &[sp_val.into()]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let n_val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[n_val.into()]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                            let val = builder.block_params(merge_bb)[0];
                            out_fn(&mut builder, val);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            out_fn(&mut builder, val);
                        }
                    }
                    Inst::PopNum => {
                        let out_fn =
                            |builder: &mut FunctionBuilder, val: cranelift_codegen::ir::Value| {
                                let w = builder.use_var(v_wn);
                                let ctx = builder.use_var(v_runtime_ctx);
                                builder.ins().call_indirect(sig_vi64, w, &[ctx, val]);
                            };
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let val = builder.inst_results(ci)[0];
                            out_fn(&mut builder, val);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.append_block_param(merge_bb, I64);
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let pop_fn = builder.use_var(v_sp_pop);
                            let sv = builder.use_var(v_sel);
                            let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                            let sp_val = builder.inst_results(ci)[0];
                            builder.ins().jump(merge_bb, &[sp_val.into()]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let n_val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[n_val.into()]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                            let val = builder.block_params(merge_bb)[0];
                            out_fn(&mut builder, val);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            let nt = builder.ins().iadd_imm(top, -8);
                            let val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            out_fn(&mut builder, val);
                        }
                    }
                    Inst::PushNum => {
                        let r = builder.use_var(v_rn);
                        let ctx = builder.use_var(v_runtime_ctx);
                        let ci = builder.ins().call_indirect(sig_i64v, r, &[ctx]);
                        let val = builder.inst_results(ci)[0];
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.use_var(v_sel);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
                        } else {
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                        }
                    }
                    Inst::PushChar => {
                        let r = builder.use_var(v_rc);
                        let ctx = builder.use_var(v_runtime_ctx);
                        let ci = builder.ins().call_indirect(sig_i64v, r, &[ctx]);
                        let val = builder.inst_results(ci)[0];
                        if sel.is_some_and(is_special) {
                            let ctx = builder.use_var(v_runtime_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                        } else if sel.is_none() {
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let push_fn = builder.use_var(v_sp_push);
                            let sv = builder.use_var(v_sel);
                            builder
                                .ins()
                                .call_indirect(sig_sp_push, push_fn, &[ctx, sv, val]);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(normal_bb);
                            builder.seal_block(normal_bb);
                            let top =
                                load_top(&mut builder, tops_slot, sel, v_sel, &v_top, ptr_type);
                            builder.ins().store(MemFlags::trusted(), val, top, 0);
                            let nt = builder.ins().iadd_imm(top, 8);
                            store_top(&mut builder, tops_slot, sel, v_sel, &v_top, nt, ptr_type);
                            builder.ins().jump(merge_bb, &[]);
                            builder.switch_to_block(merge_bb);
                            builder.seal_block(merge_bb);
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
                            let ctx = builder.use_var(v_runtime_ctx);
                            let depth_fn = builder.use_var(v_sp_depth);
                            let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                            let ci =
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv]);
                            builder.inst_results(ci)[0]
                        } else if sel.is_none() {
                            // Dynamic sel: branch on special vs normal
                            let is_sp = is_special_at_runtime(&mut builder, v_sel);
                            let sp_bb = builder.create_block();
                            let normal_bb = builder.create_block();
                            let merge_bb = builder.create_block();
                            builder.append_block_param(merge_bb, I64);
                            builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                            // Special path
                            builder.switch_to_block(sp_bb);
                            builder.seal_block(sp_bb);
                            let ctx = builder.use_var(v_runtime_ctx);
                            let depth_fn = builder.use_var(v_sp_depth);
                            let sv2 = builder.use_var(v_sel);
                            let ci =
                                builder
                                    .ins()
                                    .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv2]);
                            let sp_depth = builder.inst_results(ci)[0];
                            builder.ins().jump(merge_bb, &[sp_depth.into()]);
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
                            builder.ins().jump(merge_bb, &[normal_elems.into()]);
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
                        let ctx = builder.use_var(v_runtime_ctx);
                        let depth_fn = builder.use_var(v_sp_depth);
                        let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci = builder
                            .ins()
                            .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv]);
                        builder.inst_results(ci)[0]
                    } else if sel.is_none() {
                        // Dynamic sel: branch on special vs normal
                        let is_sp = is_special_at_runtime(&mut builder, v_sel);
                        let sp_bb = builder.create_block();
                        let normal_bb = builder.create_block();
                        let merge_bb = builder.create_block();
                        builder.append_block_param(merge_bb, I64);
                        builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                        builder.switch_to_block(sp_bb);
                        builder.seal_block(sp_bb);
                        let ctx = builder.use_var(v_runtime_ctx);
                        let depth_fn = builder.use_var(v_sp_depth);
                        let sv2 = builder.use_var(v_sel);
                        let ci = builder
                            .ins()
                            .call_indirect(sig_sp_depth, depth_fn, &[ctx, sv2]);
                        let sp_depth = builder.inst_results(ci)[0];
                        builder.ins().jump(merge_bb, &[sp_depth.into()]);
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
                        builder.ins().jump(merge_bb, &[normal_elems.into()]);
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
                        let ctx = builder.use_var(v_runtime_ctx);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let sv = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv]);
                        builder.inst_results(ci)[0]
                    } else if sel.is_none() {
                        // Dynamic sel: branch on special vs normal
                        let is_sp = is_special_at_runtime(&mut builder, v_sel);
                        let sp_bb = builder.create_block();
                        let normal_bb = builder.create_block();
                        let merge_bb = builder.create_block();
                        builder.append_block_param(merge_bb, I64);
                        builder.ins().brif(is_sp, sp_bb, &[], normal_bb, &[]);
                        builder.switch_to_block(sp_bb);
                        builder.seal_block(sp_bb);
                        let ctx = builder.use_var(v_runtime_ctx);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let sv2 = builder.use_var(v_sel);
                        let ci = builder.ins().call_indirect(sig_sp_pop, pop_fn, &[ctx, sv2]);
                        let sp_val = builder.inst_results(ci)[0];
                        builder.ins().jump(merge_bb, &[sp_val.into()]);
                        builder.switch_to_block(normal_bb);
                        builder.seal_block(normal_bb);
                        let top = load_top_slot(&mut builder, tops_slot, sel, v_sel, ptr_type);
                        let nt = builder.ins().iadd_imm(top, -8);
                        let normal_val = builder.ins().load(I64, MemFlags::trusted(), nt, 0);
                        store_top_slot(&mut builder, tops_slot, sel, v_sel, nt, ptr_type);
                        builder.ins().jump(merge_bb, &[normal_val.into()]);
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
                    let iz = emit_is_zero(
                        &mut builder,
                        val,
                        bigint,
                        v_runtime_ctx,
                        ptr_type,
                        sig_sp_pop,
                    );
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
                        let ctx = builder.use_var(v_runtime_ctx);
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
                        builder.ins().brif(has, has_bb, &[], ret_bb, &[zero.into()]);
                        builder.switch_to_block(has_bb);
                        builder.seal_block(has_bb);
                        let pop_fn = builder.use_var(v_sp_pop);
                        let ctx2 = builder.use_var(v_runtime_ctx);
                        let sv2 = builder.ins().iconst(I64, sel.unwrap() as i64);
                        let ci2 = builder
                            .ins()
                            .call_indirect(sig_sp_pop, pop_fn, &[ctx2, sv2]);
                        let val = builder.inst_results(ci2)[0];
                        builder.ins().jump(ret_bb, &[val.into()]);
                        builder.switch_to_block(ret_bb);
                        builder.seal_block(ret_bb);
                        let ret = builder.block_params(ret_bb)[0];
                        let ret =
                            emit_to_i64(&mut builder, ret, v_runtime_ctx, ptr_type, sig_sp_pop);
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
                        builder.ins().brif(has, has_bb, &[], ret_bb, &[zero.into()]);
                        builder.switch_to_block(has_bb);
                        builder.seal_block(has_bb);
                        let val = builder.ins().load(I64, MemFlags::trusted(), top, -8);
                        builder.ins().jump(ret_bb, &[val.into()]);
                        builder.switch_to_block(ret_bb);
                        builder.seal_block(ret_bb);
                        let ret = builder.block_params(ret_bb)[0];
                        let ret =
                            emit_to_i64(&mut builder, ret, v_runtime_ctx, ptr_type, sig_sp_pop);
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
