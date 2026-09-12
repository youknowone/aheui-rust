use std::io;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args_os().skip(1);
    let source_path = args.next().map(PathBuf::from).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "usage: snippet_runner SOURCE")
    })?;
    if args.next().is_some() {
        return Err(
            io::Error::new(io::ErrorKind::InvalidInput, "usage: snippet_runner SOURCE").into(),
        );
    }

    let source = std::fs::read_to_string(source_path)?;
    let program = aheuinterpreter::ahsembler::compile(
        &source,
        aheuinterpreter::ahsembler::OptimizationLevel::O3,
    );
    let exit = aheuinterpreter::interp::mainloop(&program);
    std::process::exit(aheuinterpreter::value::val_to_i32_saturating(&exit));
}
