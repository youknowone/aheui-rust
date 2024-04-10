use clap::{Parser, ValueEnum};
use std::{ffi::OsString, sync::OnceLock};

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opt {
    #[value(name = "0")]
    O0,
    #[value(name = "1")]
    O1,
    #[value(name = "2")]
    O2,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Bytecode,
    Asm,
    Text,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Bytecode,
    Asm,
    #[value(name = "asm+comment")]
    AsmComment,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[clap(short = 'O', long = "opt", default_value = "1")]
    pub opt: Opt,

    #[clap(short = 'S', long)]
    pub source: Option<Source>,

    #[clap(short = 'T', long)]
    pub target: Option<Target>,

    #[clap(long, requires("target"))]
    pub output: Option<OsString>,

    #[clap(short, long)]
    pub cmd: Option<String>,

    #[clap(conflicts_with = "cmd", required_unless_present("cmd"))]
    pub filename: Option<OsString>,
}

impl Args {
    pub fn input(&self) -> std::io::Result<&[u8]> {
        static INPUT: OnceLock<Vec<u8>> = OnceLock::new();
        if let Some(input) = INPUT.get() {
            return Ok(input);
        }
        let input = if let Some(cmd) = &self.cmd {
            cmd.as_bytes().to_vec()
        } else {
            let path = self.filename.as_ref().unwrap();
            std::fs::read(path)?
        };

        Ok(INPUT.get_or_init(|| input))
    }

    pub fn source(&self) -> Source {
        if let Some(source) = self.source {
            return source;
        }
        let Some(filename) = self.filename.as_ref() else {
            return Source::Text;
        };
        let filename = filename.to_string_lossy();
        if filename.ends_with(".아희") || filename.ends_with(".aheui") {
            Source::Text
        } else if filename.ends_with(".aheuis") {
            Source::Asm
        } else if filename.ends_with(".aheuic")
            || self
                .input()
                .unwrap()
                .windows(4)
                .any(|window| window == &b"\xff\xff\xff\xff"[..])
        {
            Source::Bytecode
        } else {
            Source::Text
        }
    }
}
