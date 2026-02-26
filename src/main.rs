mod api;
mod cmd;
mod envconfig;
mod format;
mod parser;
mod progress;
mod runner;
mod server;
mod template;
mod sample;
mod fs;
mod tools;
mod models;
mod openai;
mod downloader;
mod discover;
mod assets;
mod middleware;
mod harmony;

#[allow(dead_code)]
fn init_all_models() {
    ollama::model::init_models();
}

use clap::{Parser, Subcommand};
use std::process;

#[derive(Parser)]
#[command(name = "ollama")]
#[command(version = "0.5.0")]
#[command(about = "Run large language models locally", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Serve,
    Run {
        model: String,
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },
    Create {
        model: String,
        #[arg(short, long)]
        file: Option<String>,
    },
    Show {
        model: String,
    },
    #[command(alias = "ls")]
    List,
    Ps,
    Pull {
        model: String,
        #[arg(short, long)]
        insecure: bool,
    },
    Push {
        model: String,
        #[arg(short, long)]
        insecure: bool,
    },
    #[command(alias = "cp")]
    Copy {
        source: String,
        destination: String,
    },
    #[command(alias = "rm")]
    Delete {
        model: String,
    },
    Stop {
        model: String,
    },
    Embed {
        model: String,
        input: String,
    },
    Version,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Run { model, args } => cmd::run(&model, args).await,
        Commands::Serve => cmd::serve().await,
        Commands::Create { model, file } => cmd::create(&model, file).await,
        Commands::Show { model } => cmd::show(&model).await,
        Commands::List => cmd::list().await,
        Commands::Ps => cmd::ps().await,
        Commands::Pull { model, insecure } => cmd::pull(&model, insecure).await,
        Commands::Push { model, insecure } => cmd::push(&model, insecure).await,
        Commands::Copy { source, destination } => cmd::copy(&source, &destination).await,
        Commands::Delete { model } => cmd::delete(&model).await,
        Commands::Stop { model } => cmd::stop(&model).await,
        Commands::Embed { model, input } => cmd::embed(&model, &input).await,
        Commands::Version => cmd::version().await,
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
