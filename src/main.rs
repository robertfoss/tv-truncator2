//! TVT (TV Truncator) - CLI entry point
//!
//! A command-line tool for removing repetitive segments from TV show episodes.

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use tvt::parallel::process_files_parallel;
use tvt::{Config, Result};

/// TVT - Remove repetitive segments from TV show episodes
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input directory containing episodes
    #[arg(short, long, value_name = "DIR")]
    input: PathBuf,

    /// Output directory for processed files
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Minimum episodes for common segment
    #[arg(short, long, default_value = "3")]
    threshold: usize,

    /// Minimum segment duration in seconds
    #[arg(short, long, default_value = "10.0")]
    min_duration: f64,

    /// Similarity threshold (0-100)
    #[arg(short, long, default_value = "90")]
    similarity: u8,

    /// Similarity threshold for detection (0.0-1.0)
    #[arg(long, default_value = "0.75")]
    similarity_threshold: f64,

    /// Algorithm to use for similarity detection
    #[arg(long, value_enum, default_value = "current")]
    algorithm: tvt::similarity::SimilarityAlgorithm,

    /// Number of parallel workers
    #[arg(short, long)]
    parallel: Option<usize>,

    /// Analyze only, don't process videos
    #[arg(long)]
    dry_run: bool,

    /// Quick mode - use 0.5fps sampling rate for faster testing
    #[arg(long)]
    quick: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Enable debug output (shows state transitions)
    #[arg(long)]
    debug: bool,

    /// Enable debug output for duplicate detection (shows similarity metrics)
    #[arg(long)]
    debug_dupes: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input directory
    if !args.input.exists() {
        anyhow::bail!("Input directory does not exist: {}", args.input.display());
    }

    if !args.input.is_dir() {
        anyhow::bail!("Input path is not a directory: {}", args.input.display());
    }

    // Set output directory (default to input/truncated)
    let output_dir = args.output.unwrap_or_else(|| {
        let mut path = args.input.clone();
        path.push("truncated");
        path
    });

    // Create configuration
    let config = Config {
        input_dir: args.input.clone(),
        output_dir: output_dir.clone(),
        threshold: args.threshold,
        min_duration: args.min_duration,
        similarity: args.similarity,
        similarity_threshold: args.similarity_threshold,
        similarity_algorithm: args.algorithm,
        dry_run: args.dry_run,
        quick: args.quick,
        verbose: args.verbose,
        debug: args.debug,
        debug_dupes: args.debug_dupes,
        parallel_workers: args
            .parallel
            .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1)),
    };

    if config.verbose {
        println!("Configuration: {:#?}", config);
    }

    println!("TVT - TV Truncator");
    println!("Input: {}", config.input_dir.display());
    println!("Output: {}", config.output_dir.display());
    println!("Threshold: {}", config.threshold);
    println!("Min duration: {}s", config.min_duration);
    println!("Similarity: {}%", config.similarity);
    
    // Check and display hardware acceleration status
    let (hw_available, hw_description) = tvt::gstreamer_extractor_v2::check_hardware_acceleration();
    if hw_available {
        println!("Hardware acceleration: {} (enabled)", hw_description);
    } else {
        println!("Hardware acceleration: {} (will use CPU)", hw_description);
    }
    
    println!("Parallel workers: {}", config.parallel_workers);
    println!("Dry run: {}", config.dry_run);
    println!("Debug: {}", config.debug);

    // Find video files
    let video_files = find_video_files(&config.input_dir)?;
    println!("\nFound {} video files", video_files.len());

    if video_files.is_empty() {
        println!("No video files found in input directory");
        return Ok(());
    }

    // Process files using state machine
    println!("\nStarting parallel processing with state machine...");
    let results = process_files_parallel(video_files, config)?;

    // Print summary
    use tvt::progress_display::print_summary;
    print_summary(&results);

    // Print segment analysis summary
    print_segment_summary(&results);

    Ok(())
}

/// Find all video files in the given directory
fn find_video_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut video_files = Vec::new();

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                if let Some(ext_str) = extension.to_str() {
                    let ext_lower = ext_str.to_lowercase();
                    if matches!(
                        ext_lower.as_str(),
                        "mp4" | "mkv" | "avi" | "mov" | "wmv" | "flv" | "webm"
                    ) {
                        video_files.push(path);
                    }
                }
            }
        }
    }

    Ok(video_files)
}

/// Print a summary of detected identical segments
fn print_segment_summary(processors: &[tvt::state_machine::FileProcessor]) {
    // Collect all common segments from all processors
    let mut all_segments = Vec::new();
    for processor in processors {
        if let Some(segments) = &processor.common_segments {
            all_segments.extend(segments.clone());
        }
    }

    if all_segments.is_empty() {
        println!("\n=== Segment Analysis Summary ===");
        println!("No identical segments found that meet the threshold criteria.");
        return;
    }

    // Remove duplicates (same segments might be stored in multiple processors)
    all_segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
    all_segments.dedup_by(|a, b| {
        (a.start_time - b.start_time).abs() < 0.1 && (a.end_time - b.end_time).abs() < 0.1
    });

    println!("\n=== Segment Analysis Summary ===");
    println!(
        "Found {} identical segment(s) across {} file(s):",
        all_segments.len(),
        processors.len()
    );
    println!();

    for (i, segment) in all_segments.iter().enumerate() {
        let duration = segment.end_time - segment.start_time;
        let confidence_percent = (segment.confidence * 100.0) as u8;

        println!("Segment {}:", i + 1);
        println!(
            "  Time: {:.2}s - {:.2}s (duration: {:.2}s)",
            segment.start_time, segment.end_time, duration
        );
        println!("  Confidence: {}%", confidence_percent);
        println!("  Found in {} episode(s):", segment.episode_list.len());

        for episode_name in &segment.episode_list {
            println!("    - {}", episode_name);
        }
        println!();
    }

    // Calculate total time saved
    let total_duration: f64 = all_segments.iter().map(|s| s.end_time - s.start_time).sum();

    println!(
        "Total time that will be removed: {:.2} seconds ({:.2} minutes)",
        total_duration,
        total_duration / 60.0
    );
}
