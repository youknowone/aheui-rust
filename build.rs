//! Link arguments the wasm32 JIT build needs from its host.
//!
//! A JIT-enabled wasm build does not run its own compiled traces: it hands the
//! emitted module to an embedder, which instantiates it against this module's
//! linear memory and function table and publishes the result as a new table
//! entry the guest then calls by index. Neither side of that is reachable
//! unless the table is exported and can grow, and wasm-ld does neither by
//! default — the failure is a missing export at instantiation, or a `table.grow`
//! that refuses at the first compile.
//!
//! Emitted for binaries only, so the browser library target (which carries its
//! own wasm-bindgen link step) is unaffected.
fn main() {
    let wasm32 = std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("wasm32");
    if wasm32 && std::env::var_os("CARGO_FEATURE_JIT").is_some() {
        println!("cargo::rustc-link-arg-bins=--export-table");
        println!("cargo::rustc-link-arg-bins=--growable-table");
    }
}
