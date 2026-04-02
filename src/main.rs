#[cfg(not(any(feature = "naive", feature = "jit")))]
compile_error!("at least one of features `naive` or `jit` must be enabled");

#[cfg(all(feature = "naive", feature = "jit"))]
use clap::ArgAction;
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const ABOUT: &str = match (cfg!(feature = "jit"), cfg!(feature = "naive")) {
    (true, true) => "Aheui interpreter with optional JIT",
    (true, false) => "Aheui JIT interpreter",
    (false, true) => "Aheui interpreter",
    (false, false) => "Aheui (no backend)",
};

fn parse_opt_level(value: &str) -> Result<ahsembler::OptimizationLevel, String> {
    ahsembler::OptimizationLevel::from_str(value)
}

#[derive(ClapArgs, Debug)]
struct RunArgs {
    #[cfg(all(feature = "naive", feature = "jit"))]
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "no_jit",
        help = "Force JIT execution (default)")]
    jit: bool,

    #[cfg(all(feature = "naive", feature = "jit"))]
    #[arg(long, action = ArgAction::SetTrue, conflicts_with = "jit",
        help = "Run without JIT")]
    no_jit: bool,

    #[arg(short = 'b', long)]
    benchmark: bool,

    #[arg(short = 'O', long = "opt-level", default_value = "2", value_parser = parse_opt_level)]
    opt_level: ahsembler::OptimizationLevel,

    #[arg(short = 'c', long, value_name = "CODE")]
    cmd: Option<String>,

    #[arg(
        value_name = "FILE",
        conflicts_with = "cmd",
        required_unless_present = "cmd"
    )]
    file_path: Option<PathBuf>,
}

impl RunArgs {
    fn input(&self) -> std::io::Result<String> {
        if let Some(cmd) = &self.cmd {
            Ok(cmd.clone())
        } else {
            fs::read_to_string(self.file_path.as_ref().unwrap())
        }
    }

    fn use_jit(&self) -> bool {
        #[cfg(all(feature = "naive", feature = "jit"))]
        {
            self.jit || !self.no_jit
        }
        #[cfg(all(feature = "jit", not(feature = "naive")))]
        {
            true
        }
        #[cfg(not(feature = "jit"))]
        {
            false
        }
    }
}

#[derive(ClapArgs, Debug)]
struct AsmArgs {
    #[arg(short = 'O', long = "opt-level", default_value = "2", value_parser = parse_opt_level)]
    opt_level: ahsembler::OptimizationLevel,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(short = 'c', long, value_name = "CODE")]
    cmd: Option<String>,
    #[arg(
        value_name = "FILE",
        conflicts_with = "cmd",
        required_unless_present = "cmd"
    )]
    file_path: Option<PathBuf>,
}

impl AsmArgs {
    fn input(&self) -> std::io::Result<String> {
        if let Some(cmd) = &self.cmd {
            Ok(cmd.clone())
        } else {
            fs::read_to_string(self.file_path.as_ref().unwrap())
        }
    }
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compile Aheui source into commented ahsembly
    Asm(AsmArgs),
    /// Compile Aheui source into a native binary or other target
    Build(compaheuiler::cli::BuildArgs),
}

#[derive(Parser, Debug)]
#[command(version, about = ABOUT, long_about = None,
    subcommand_negates_reqs = true, args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    run: RunArgs,
}

fn compile_program(
    contents: &str,
    opt_level: ahsembler::OptimizationLevel,
) -> ahsembler::compiler::Program {
    ahsembler::compile(contents, opt_level)
}

#[cfg(feature = "jit")]
fn exit_code_to_i32(exit_code: aheui_jit::aheui_runtime::Val) -> i32 {
    aheui_jit::aheui_runtime::value::val_to_i32_saturating(&exit_code)
}

#[cfg(all(feature = "naive", not(feature = "jit")))]
fn exit_code_to_i32(exit_code: aheuinterpreter::Val) -> i32 {
    aheuinterpreter::value::val_to_i32_saturating(&exit_code)
}

fn run_program(args: RunArgs) -> Result<(), Box<dyn Error>> {
    let contents = args.input()?;
    let program = compile_program(&contents, args.opt_level);

    if args.benchmark {
        let program_o2 = compile_program(&contents, ahsembler::OptimizationLevel::O2);

        #[cfg(feature = "naive")]
        {
            eprintln!("--- interpreter ---");
            let start = Instant::now();
            let result = aheuinterpreter::interp::mainloop(&program_o2);
            let elapsed = start.elapsed();
            eprintln!("result = {result}");
            eprintln!("time   = {elapsed:?}");
        }

        #[cfg(feature = "jit")]
        if args.use_jit() {
            let program_jit = compile_program(&contents, ahsembler::OptimizationLevel::O2);
            eprintln!("\n--- JIT ---");
            let start = Instant::now();
            let result = aheui_jit::mainloop(&program_jit, aheui_jit::JIT_THRESHOLD);
            let elapsed = start.elapsed();
            eprintln!("result = {result}");
            eprintln!("time   = {elapsed:?}");
        }

        return Ok(());
    }

    #[cfg(feature = "jit")]
    if args.use_jit() {
        let exitcode = aheui_jit::mainloop(&program, aheui_jit::JIT_THRESHOLD);
        std::process::exit(exit_code_to_i32(exitcode));
    }

    #[cfg(feature = "naive")]
    {
        let exitcode = aheuinterpreter::interp::mainloop(&program);
        std::process::exit(exit_code_to_i32(exitcode));
    }

    #[cfg(not(feature = "naive"))]
    unreachable!()
}

fn run_asm(args: AsmArgs) -> Result<(), Box<dyn Error>> {
    let contents = args.input()?;
    if let Some(output_path) = args.output.as_ref() {
        let mut output = fs::File::create(output_path)?;
        ahsembler::assemble(&contents, args.opt_level, &mut output)?;
    } else {
        let stdout = std::io::stdout();
        let mut output = stdout.lock();
        ahsembler::assemble(&contents, args.opt_level, &mut output)?;
        output.flush()?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    match cli.command {
        Some(Command::Asm(args)) => run_asm(args),
        Some(Command::Build(args)) => {
            compaheuiler::cli::run_build(&args);
            Ok(())
        }
        None => run_program(cli.run),
    }
}
