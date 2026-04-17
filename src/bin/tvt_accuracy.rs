//! CLI for the accuracy SQLite store (MEMA-26): init schema, list recent runs.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tvt::accuracy_store::AccuracyStore;

#[derive(Parser)]
#[command(name = "tvt-accuracy")]
#[command(about = "Inspect TVT detection accuracy benchmark databases", version)]
struct Cli {
    #[arg(long, default_value = "tests/fixtures/accuracy_runs.sqlite")]
    db: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create database file and apply schema (safe if it already exists).
    Init,
    /// List recent benchmark runs.
    List {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Init => {
            let _ = AccuracyStore::open(&cli.db).with_context(|| format!("open {:?}", cli.db))?;
            println!("OK: schema ready at {}", cli.db.display());
        }
        Commands::List { limit } => {
            let store =
                AccuracyStore::open(&cli.db).with_context(|| format!("open {:?}", cli.db))?;
            let rows = store.list_runs(limit)?;
            if rows.is_empty() {
                println!("No runs in {}", cli.db.display());
                return Ok(());
            }
            println!(
                "{:>6}  {:<20}  {:<24}  {:<14}  {}",
                "id", "created_at", "label", "video_algo", "audio_algo"
            );
            for r in rows {
                println!(
                    "{:>6}  {:<20}  {:<24}  {:<14}  {}",
                    r.id, r.created_at, r.label, r.similarity_algorithm, r.audio_algorithm
                );
            }
        }
    }
    Ok(())
}
