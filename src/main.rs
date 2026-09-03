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
    #[cfg(feature = "jit")]
    #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "",
        value_name = "PARAMS",
        help = "Force JIT execution (default). `--jit=name=value,...` sets JIT \
                parameters (e.g. `--jit=stack_cap=8,trace_limit=20000`); \
                `--jit=off` disables the JIT")]
    #[cfg_attr(all(feature = "naive", feature = "jit"), arg(conflicts_with = "no_jit"))]
    jit: Option<String>,

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
            if self.jit.as_deref() == Some("off") {
                return false;
            }
            self.jit.is_some() || !self.no_jit
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

/// Largest status a WASI host accepts. `proc_exit` is specified over
/// `[0, 126)`, the range left after the shell conventions for 126 and 127.
#[cfg(target_os = "wasi")]
const WASI_MAX_EXIT: i32 = 125;

/// Leave with `code`, narrowed to what this platform's exit status can carry.
///
/// A POSIX status is eight bits, so `exit(810)` is observed by the parent as
/// `810 & 0xff` — 42, which is the value the corpus records for `logo`. Taking
/// the low byte here is therefore a no-op natively and states the rule once.
///
/// It is not a no-op under WASI, which narrows the range again to `[0, 126)`
/// and traps on anything outside it. That trap costs the status AND turns a
/// completed run into a failed one with its stdout already written, so 8 of
/// the 78 corpus programs exited 1 with a backtrace instead of the code they
/// computed. Four of them — `logo` among them — carry a low byte that fits,
/// and now report exactly what they report natively. For the rest the status
/// simply cannot be expressed, so it is named on stderr rather than replaced
/// by a number the program never produced.
fn exit_with(code: i32) -> ! {
    let low = (code as u32 & 0xff) as i32;
    #[cfg(target_os = "wasi")]
    if low > WASI_MAX_EXIT {
        eprintln!(
            "[exit] status {code} (low byte {low}) is outside the [0, 126) a \
             WASI host accepts; exiting {WASI_MAX_EXIT}"
        );
        std::process::exit(WASI_MAX_EXIT);
    }
    std::process::exit(low)
}

/// Print the one-line JIT statistics summary to stderr when `MAJIT_STATS` is
/// set, in the same `[jit-stats]` shape `pyre/pyrex` emits so one recorder and
/// one regression floor can read both.
///
/// `internal_compile_panics > 0` means an internal JIT bug silently disabled
/// compilation for some traces (graceful degradation in release). Must be
/// called before any `process::exit`, since exits skip destructors.
///
/// The descr-universe lines pyre also prints have no aheui counterpart — they
/// read the pyre descr tables — and a baseline simply carries no entry for
/// them, which the floor reads as 0 on both sides.
#[cfg(feature = "jit")]
fn maybe_print_jit_stats() {
    if std::env::var_os("MAJIT_STATS").is_none() {
        return;
    }
    let Some(stats) = aheui_jit::last_jit_stats() else {
        return;
    };
    eprintln!(
        "[jit-stats] mc_diag {}",
        aheui_jit::majit_metainterp::mc_diag_summary()
    );
    // `guard_failures` is one total, and a total cannot tell one guard failing
    // tens of thousands of times from tens of thousands of guards failing once.
    // Those want opposite fixes, so print the distribution the counter is a sum
    // of. Empty unless `MAJIT_GUARD_CENSUS` armed the recording.
    eprintln!(
        "[jit-stats] {}",
        aheui_jit::majit_metainterp::guard_census_summary(8)
    );
    #[cfg(target_arch = "wasm32")]
    {
        let (entries, modules, cache_hits) = aheui_jit::wasm_jit_counts();
        eprintln!(
            "[jit-stats] wasm_trace_entries={entries} wasm_host_modules={modules} \
             wasm_module_cache_hits={cache_hits}"
        );
        eprintln!(
            "[jit-stats] wasm_bridge_diag {}",
            aheui_jit::wasm_bridge_diag_summary()
        );
    }
    eprintln!(
        "[jit-stats] loops_compiled={} bridges_compiled={} loops_aborted={} \
         guard_failures={} internal_compile_panics={}",
        stats.loops_compiled,
        stats.bridges_compiled,
        stats.loops_aborted,
        stats.guard_failures,
        stats.internal_compile_panics,
    );
    // Why those traces were given up. `JitStats` carries only the total, and
    // the profiler's own `print_stats` is behind `MAJIT_LOG`, which is far too
    // slow to enable on a workload big enough to abort interestingly.
    if let Some(p) = aheui_jit::last_abort_reasons() {
        eprintln!(
            "[jit-stats] abort_too_long={} abort_bridge={} abort_bad_loop={} \
             abort_escape={} abort_force_quasiimmut={} abort_segmented_trace={}",
            p.abort_too_long,
            p.abort_bridge,
            p.abort_bad_loop,
            p.abort_escape,
            p.abort_force_quasiimmut,
            p.abort_segmented_trace,
        );
    }
}

fn run_program(args: RunArgs) -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "jit")]
    aheui_jit::init_gc_subsystem();

    #[cfg(feature = "jit")]
    match args.jit.as_deref() {
        Some("off") => {
            #[cfg(not(feature = "naive"))]
            return Err("--jit=off needs a binary built with the naive interpreter".into());
        }
        Some(params) if !params.is_empty() => aheui_jit::set_user_jit_params(params)?,
        _ => {}
    }

    let contents = args.input()?;
    let program = compile_program(&contents, args.opt_level);

    if args.benchmark {
        #[cfg(feature = "naive")]
        {
            let program_o2 = compile_program(&contents, ahsembler::OptimizationLevel::O2);
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
            let result = aheui_jit::mainloop(&program_jit, aheui_jit::jit_threshold());
            let elapsed = start.elapsed();
            eprintln!("result = {result}");
            eprintln!("time   = {elapsed:?}");
            maybe_print_jit_stats();
        }

        return Ok(());
    }

    #[cfg(feature = "jit")]
    if args.use_jit() {
        let exitcode = aheui_jit::mainloop(&program, aheui_jit::jit_threshold());
        maybe_print_jit_stats();
        exit_with(exit_code_to_i32(exitcode));
    }

    #[cfg(feature = "naive")]
    {
        let exitcode = aheuinterpreter::interp::mainloop(&program);
        exit_with(exit_code_to_i32(exitcode));
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
