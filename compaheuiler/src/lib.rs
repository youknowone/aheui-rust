pub mod c_gen;
pub mod cli;
pub mod cranelift_backend;
pub mod pipeline;
pub mod rust_gen;
pub mod wat_gen;

pub use c_gen::c_bigint_bridge_rs;
pub use c_gen::c_bigint_dependency_toml;
pub use c_gen::compile_to_c;
pub use c_gen::compile_to_c_bigint;
pub use c_gen::compile_to_c_bigint_opt;
pub use c_gen::compile_to_c_opt;
pub use rust_gen::compile_to_rs;
pub use rust_gen::compile_to_rs_bigint;
pub use rust_gen::compile_to_rs_bigint_opt;
pub use rust_gen::compile_to_rs_opt;
pub use wat_gen::compile_to_wat;
pub use wat_gen::compile_to_wat_opt;
pub use wat_gen::compile_to_wat_wasi;
pub use wat_gen::compile_to_wat_wasi_opt;

#[cfg(feature = "cranelift")]
pub use cranelift_backend::jit;
