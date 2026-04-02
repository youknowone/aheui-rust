pub mod c_gen;
pub mod cranelift_backend;
pub mod rust_gen;
pub mod wat_gen;

pub use c_gen::compile_to_c;
pub use rust_gen::compile_to_rs;
pub use rust_gen::compile_to_rs_bigint;
pub use rust_gen::compile_to_rs_opt;
pub use rust_gen::compile_to_rs_bigint_opt;
pub use wat_gen::compile_to_wat;

#[cfg(feature = "cranelift")]
pub use cranelift_backend::jit;
