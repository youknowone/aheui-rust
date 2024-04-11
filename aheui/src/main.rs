use clap::Parser;
use num_traits::cast::ToPrimitive;
use rpaheui::{
    clap::{Source, Target},
    Aheui,
};
use std::io::Write;

fn main() -> anyhow::Result<()> {
    let args = rpaheui::clap::Args::parse();
    let input = args.input()?;

    let mut aheui = Aheui::new();
    let object = match args.source() {
        Source::Bytecode => aheui.load_bytecode(input),
        Source::Asm => aheui.compile_asm(std::str::from_utf8(input)?),
        Source::Text => aheui.compile(std::str::from_utf8(input)?),
    }
    .unwrap();

    let Some(target) = args.target else {
        let exit_code = aheui.run(&object).unwrap();
        let exit_code = exit_code.to_i32().unwrap_or(std::i32::MAX);
        std::process::exit(exit_code);
    };
    let output = match target {
        Target::Asm => {
            let asm = aheui.make_asm(&object, false).unwrap();
            asm.into_bytes()
        }
        Target::AsmComment => {
            let asm = aheui.make_asm(&object, true).unwrap();
            asm.into_bytes()
        }
        Target::Bytecode => aheui.make_bytecode(&object).unwrap(),
    };
    if let Some(output_path) = args.output {
        std::fs::write(output_path, &output)?;
    } else {
        let _ = std::io::stdout().write(&output)?;
    }

    Ok(())
}
