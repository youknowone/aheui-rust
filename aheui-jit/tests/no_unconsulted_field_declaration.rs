//! No field declaration in this crate may describe something nothing reaches.
//!
//! `int_fields` / `ref_fields` are keyed `"StructType::field"`, built from the
//! **declared** type of the base an access goes through — never from the
//! runtime object. A key naming a struct no access site is typed as therefore
//! matches nothing, and both halves of that are the problem: it contributes no
//! descr width, so it is dead, and the macro's `const _: fn(&S) -> T` witness
//! is emitted only on a match, so what it claims about the field is never
//! checked either. Measured here: 36 blocks declaring a `u32` field as `u8`
//! compiled clean, while the same misdeclaration on a consulted key produced
//! 13 `E0308`s. An unconsulted entry is not untidy — it is a false statement
//! the build agrees to.
//!
//! Deleting one is only safe because the opposite direction is now caught: an
//! access with no declaration emits a width witness, so removing an entry that
//! *is* consulted fails the build rather than silently registering the
//! eight-byte default over a `u32` and losing the sub-word `intbounds` range
//! the declaration exists to buy.
//!
//! The gate spans both macro surfaces on purpose. A `#[jit_interp]` machine
//! declares a handful of keys; the `#[jit_inline]` helpers repeat theirs at
//! every site, so a key matching nothing at one site matches nothing at dozens.
//! When this was first measured five of the six survivors were on helpers and
//! one on the portal — a portal-only assertion would have reported the one and
//! called the file clean.

use ahsembler::compiler::Program;
use ahsembler::consts::{OP_BRPOP1, OP_HALT, OP_JMP, OP_POP, OP_PUSH};
use std::collections::HashMap;

/// The `state = T` name this crate's dispatch arms are recorded under.
const AHEUI_INTERP: &str = "AheuiState";

const LABEL_LOOP: i32 = 1_000_001;
const LABEL_END: i32 = 1_000_002;

/// A hot loop over the storage ops, so the portal installs and the inline
/// helpers it splices in are built.
///
/// The recording happens when a jitcode is *built*, so reaching the install is
/// what this program is for; the loop is what makes the install happen through
/// the same path production does.
fn drain_loop(n: i32) -> Program {
    let mut opcodes: Vec<u8> = Vec::new();
    let mut values: Vec<i32> = Vec::new();

    for i in 0..n {
        opcodes.push(OP_PUSH);
        values.push(i);
    }

    let loop_pc = opcodes.len();
    // `-1` is the unused operand slot: only OP_PUSH and the branches read one.
    for (op, val) in [(OP_BRPOP1, LABEL_END), (OP_POP, -1), (OP_JMP, LABEL_LOOP)] {
        opcodes.push(op);
        values.push(val);
    }
    let end_pc = opcodes.len();
    opcodes.push(OP_HALT);
    values.push(-1);

    let size = opcodes.len();
    let mut labels: HashMap<i32, usize> = HashMap::new();
    labels.insert(LABEL_LOOP, loop_pc);
    labels.insert(LABEL_END, end_pc);

    let mut program = Program {
        opcodes,
        values,
        labels,
        size,
    };
    program.resolve_jump_targets();
    program
}

const THRESHOLD: u32 = 8;
const ITERATIONS: i32 = 200;

/// One test, not two: `mainloop` installs process-global state — the nursery,
/// the GC roots, the raw/tagged value mode — keyed to a `state` local on its
/// own frame, so two `#[test]`s calling it race over that and the second hangs.
#[test]
fn no_field_declaration_in_this_crate_is_unconsulted() {
    aheui_jit::init_gc_subsystem();
    let _exit = aheui_jit::mainloop(&drain_loop(ITERATIONS), THRESHOLD);

    // The denominator first. An empty unconsulted list means either "every
    // declaration is consulted" or "nothing was built", and only the first is a
    // pass — the second is the same shape as the failure this test exists to
    // catch, one level up.
    let stats = aheui_jit::last_jit_stats().expect("mainloop records its JitStats");
    assert!(
        stats.loops_compiled > 0,
        "nothing compiled (loops_compiled={}, loops_aborted={}), so no jitcode \
         was built and the assertions below would pass over an empty world",
        stats.loops_compiled,
        stats.loops_aborted,
    );

    // The portal, through the gate that carries its own census precondition and
    // prints the degraded arms beside any finding — a key used only in a
    // refused arm is unconsulted for a different reason than a stale one.
    majit_metainterp::assert_no_unconsulted_field_declarations(AHEUI_INTERP);

    // The helpers. `assert_no_unconsulted_field_declarations` is keyed on a
    // dispatch-arm census, which only `#[jit_interp]` machines have, so the
    // `#[jit_inline]` population — where this crate keeps most of its
    // declarations — needs its own assertion.
    let helpers: Vec<_> = majit_metainterp::unconsulted_field_declarations()
        .into_iter()
        .filter(|entry| entry.interp != AHEUI_INTERP)
        .collect();
    assert!(
        helpers.is_empty(),
        "{} inline helper field declaration(s) name a struct no access in that \
         helper is typed as, so each emitted no width and no witness:\n{}",
        helpers.len(),
        helpers
            .iter()
            .map(|entry| format!("  {} declares {}\n", entry.interp, entry.key))
            .collect::<String>(),
    );
}
