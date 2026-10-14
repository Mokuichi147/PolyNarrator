use std::process::ExitCode;

use clap::Parser;
use yomitone_cli::{run, Args, Factories};

fn main() -> ExitCode {
    let args = Args::parse();
    match run(&args, &Factories::default()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}
