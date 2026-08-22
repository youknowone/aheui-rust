//! No field declaration in this crate may describe something nothing reaches.
//!
//! `int_fields` / `ref_fields` are keyed `"StructType::field"`, built from the
//! **declared** type of the base an access goes through — never from the
//! runtime object. A key naming a struct no access site is typed as therefore
//! matches nothing, and both halves of that are the problem: it contributes no
//! descr width, so it is dead, and the macro's `const _: fn(&S) -> T` witness
//! is emitted only on a match, so what it claims about the field is never
//! checked either. An unconsulted entry is not untidy — it is a false statement
//! the build agrees to.
//!
//! Deleting one is only safe because the opposite direction is caught too: an
//! access with no declaration emits a width witness, so removing an entry that
//! *is* consulted fails the build rather than silently registering the
//! eight-byte default over a `u32` and losing the sub-word `intbounds` range
//! the declaration exists to buy.
//!
//! The gate spans both macro surfaces on purpose. A `#[jit_interp]` machine
//! declares a handful of keys; the `#[jit_inline]` helpers repeat theirs at
//! every site, so a key matching nothing at one site matches nothing at dozens,
//! and a portal-only assertion would report the handful and call the file
//! clean.

mod common;

use ahsembler::consts::OP_POP;
use common::{ITERATIONS, NONE};

/// The `state = T` name this crate's dispatch arms are recorded under.
const AHEUI_INTERP: &str = "AheuiState";

#[test]
fn no_field_declaration_in_this_crate_is_unconsulted() {
    // The recording happens when a jitcode is built, so reaching the install is
    // what this program is for; a hot loop over the storage ops is what makes
    // the install happen through the same path production does.
    let _exit = common::run(&common::drain_loop(ITERATIONS, &[(OP_POP, NONE)]));

    // The denominator first. An empty unconsulted list means either "every
    // declaration is consulted" or "nothing was built", and only the first is a
    // pass — the second is the same shape as the failure this test exists to
    // catch, one level up.
    common::assert_compiled("the field declarations below");

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
