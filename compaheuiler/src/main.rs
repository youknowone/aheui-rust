use clap::Parser;
use compaheuiler::cli::{BuildArgs, run_build};

/// 아희 컴파일러
#[derive(Parser)]
#[command(name = "compaheuiler")]
struct Cli {
    #[command(flatten)]
    build: BuildArgs,
}

fn main() {
    let cli = Cli::parse();
    run_build(&cli.build);
}
