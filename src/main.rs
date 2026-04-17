//! TVT (TV Truncator) - CLI entry point
//!
//! A command-line tool for removing repetitive segments from TV show episodes.

use clap::Parser;
use std::path::PathBuf;
use tvt::parallel::process_files_parallel;
use tvt::progress_display::{
    build_json_run_summary, merged_common_segments, print_processing_summary,
};
use tvt::{discover_video_files, Config, Result};

/// TVT — remove repetitive segments (intros, outros, credits) from TV episodes.
#[derive(Parser, Debug)]
#[command(
    name = "tvt",
    author,
    version,
    about = "Remove repetitive segments from TV episodes (intro/outro/credits).",
    long_about = None,
    after_help = "Safety: use --dry-run to preview detection and cuts without writing outputs.\n\
                  Scripting: --json-summary prints one JSON object on stdout; progress is suppressed.\n\
                  Nested inputs: use -r / --recursive (skips subdirs named truncated).\n\
                  Logs: use -v for full configuration and per-file output paths."
)]
struct Args {
    /// Input directory containing episodes
    #[arg(short, long, value_name = "DIR", help_heading = "Input and output")]
    input: PathBuf,

    /// Recursively discover video files (skips subdirectories named `truncated`)
    #[arg(short, long, help_heading = "Input and output")]
    recursive: bool,

    /// Output directory for processed files (default: <input>/truncated)
    #[arg(short, long, value_name = "DIR", help_heading = "Input and output")]
    output: Option<PathBuf>,

    /// Analyze only — no files are modified or written (preview pipeline safely)
    #[arg(long, help_heading = "Safety")]
    dry_run: bool,

    /// Suppress banners, progress bars, and non-essential messages
    #[arg(short, long, help_heading = "Output and logging")]
    quiet: bool,

    /// Emit one JSON object on stdout at end with paths, counts, segments (implies quiet UI)
    #[arg(long = "json-summary", help_heading = "Output and logging")]
    json_summary: bool,

    /// Minimum episodes that must share a segment (capped to input file count when lower)
    #[arg(short, long, default_value = "3", help_heading = "Detection")]
    threshold: usize,

    /// Minimum segment duration in seconds
    #[arg(short, long, default_value = "10.0", help_heading = "Detection")]
    min_duration: f64,

    /// Similarity threshold (0-100)
    #[arg(short, long, default_value = "90", help_heading = "Detection")]
    similarity: u8,

    /// Similarity threshold for detection (0.0-1.0)
    #[arg(long, default_value = "0.75", help_heading = "Detection")]
    similarity_threshold: f64,

    /// Algorithm to use for video similarity detection
    #[arg(
        long,
        value_enum,
        default_value = "current",
        help_heading = "Detection"
    )]
    algorithm: tvt::similarity::SimilarityAlgorithm,

    /// Algorithm to use for audio matching
    #[arg(
        long,
        value_enum,
        default_value = "fingerprint",
        help_heading = "Detection"
    )]
    audio_algorithm: tvt::AudioAlgorithm,

    /// Number of parallel workers
    #[arg(short, long, help_heading = "Performance")]
    parallel: Option<usize>,

    /// Quick mode — lower sampling rate for faster runs
    #[arg(long, help_heading = "Performance")]
    quick: bool,

    /// Do not prefer hardware video decoders (skip GPU decoder rank boosts; typical autoplug order)
    #[arg(long = "no-hardware-video-decode", help_heading = "Performance")]
    no_hardware_video_decode: bool,

    /// Only detect audio segments (skip video analysis)
    #[arg(long, help_heading = "Detection")]
    audio_only: bool,

    /// Verbose output (full configuration echo and detailed summaries)
    #[arg(short, long, help_heading = "Output and logging")]
    verbose: bool,

    /// Enable debug output (shows state transitions)
    #[arg(long, help_heading = "Output and logging")]
    debug: bool,

    /// Enable debug output for duplicate detection (shows similarity metrics)
    #[arg(long, help_heading = "Output and logging")]
    debug_dupes: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let quiet = args.quiet || args.json_summary;

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
    let mut config = Config {
        input_dir: args.input.clone(),
        output_dir: output_dir.clone(),
        threshold: args.threshold,
        min_duration: args.min_duration,
        similarity: args.similarity,
        similarity_threshold: args.similarity_threshold,
        similarity_algorithm: args.algorithm,
        audio_algorithm: args.audio_algorithm,
        dry_run: args.dry_run,
        quick: args.quick,
        verbose: args.verbose,
        debug: args.debug,
        debug_dupes: args.debug_dupes,
        parallel_workers: args
            .parallel
            .unwrap_or_else(|| num_cpus::get().saturating_sub(1).max(1)),
        enable_audio_matching: true, // Always enabled
        audio_only: args.audio_only,
        quiet,
        json_summary: args.json_summary,
    };

    tvt::gstreamer_extractor_v2::set_prefer_hardware_video_decode(!args.no_hardware_video_decode);

    // Find video files
    let video_files = discover_video_files(&config.input_dir, args.recursive)?;

    if !video_files.is_empty() {
        let n = video_files.len();
        let effective = tvt::effective_episode_threshold(config.threshold, n);
        if effective != config.threshold {
            eprintln!(
                "Note: --threshold {} exceeds input file count ({}); using {}.",
                config.threshold, n, effective
            );
            config.threshold = effective;
        }
    }

    if config.verbose {
        println!("Configuration: {:#?}", config);
    }

    if !quiet && !config.json_summary {
        print_concise_startup(&config, video_files.len());
    } else if quiet && config.dry_run {
        eprintln!(
            "*** DRY RUN — no files will be written ({} video file(s) in {}) ***",
            video_files.len(),
            config.input_dir.display()
        );
    }

    if video_files.is_empty() {
        if config.json_summary {
            let summary = build_json_run_summary(&config, &[], 0);
            println!("{}", serde_json::to_string_pretty(&summary)?);
        } else {
            println!("No video files found in input directory");
        }
        return Ok(());
    }

    emit_video_decode_banner(&config);
    emit_optional_hw_decoder_hints(&config);

    // Process files using state machine
    if !quiet && !config.json_summary {
        println!("\nStarting parallel processing…");
    }
    let video_file_count = video_files.len();
    let config_clone = config.clone();
    let results = process_files_parallel(video_files, config)?;

    print_processing_summary(
        &results,
        config_clone.verbose,
        config_clone.quiet,
        config_clone.json_summary,
    );

    print_segment_summary(&results, &config_clone);

    if config_clone.json_summary {
        let summary = build_json_run_summary(&config_clone, &results, video_file_count);
        println!("{}", serde_json::to_string_pretty(&summary)?);
    }

    Ok(())
}

fn print_concise_startup(config: &Config, n_videos: usize) {
    println!(
        "tvt: {} → {} — {} video file(s), {} worker(s){}",
        config.input_dir.display(),
        config.output_dir.display(),
        n_videos,
        config.parallel_workers,
        if config.dry_run { ", dry-run" } else { "" }
    );

    if config.dry_run {
        println!();
        println!("*** DRY RUN — no output files will be written ***");
        println!();
    }
}

/// One line on whether video decode will prefer hardware (and if plugins are present). Skipped in
/// `--audio-only` mode. Uses stderr when `--quiet` / `--json-summary` so stdout stays clean.
fn emit_video_decode_banner(config: &Config) {
    if config.audio_only {
        return;
    }
    let line = format_video_decode_status();
    if config.quiet || config.json_summary {
        eprintln!("{}", line);
    } else {
        println!("{}", line);
    }
}

/// stderr-only: list optional GStreamer hardware decoder plugins that are absent (board: call out
/// specific missing elements that could improve performance). Always uses stderr so `--quiet` /
/// `--json-summary` keep stdout clean.
fn emit_optional_hw_decoder_hints(config: &Config) {
    if config.audio_only || !tvt::gstreamer_extractor_v2::prefer_hardware_video_decode_enabled() {
        return;
    }
    let (hw_available, _) = tvt::gstreamer_extractor_v2::check_hardware_acceleration();
    let hints = tvt::gstreamer_extractor_v2::missing_optional_hw_decoder_install_hints();
    if hints.is_empty() {
        return;
    }
    if hw_available {
        eprintln!(
            "Video decode: optional plugins from the same GPU stack are still missing (installing them may improve decode speed for matching codecs):"
        );
    } else {
        eprintln!(
            "Video decode: optional hardware decoder plugins are missing — software decode in use; installing any of the following may improve performance:"
        );
    }
    for line in hints.iter().take(6) {
        eprintln!("  {}", line);
    }
    if hints.len() > 6 {
        eprintln!("  … and {} more.", hints.len() - 6);
    }
}

fn format_video_decode_status() -> String {
    if !tvt::gstreamer_extractor_v2::prefer_hardware_video_decode_enabled() {
        return "Video decode: software preference (--no-hardware-video-decode).".to_string();
    }
    let (hw_available, hw_description) = tvt::gstreamer_extractor_v2::check_hardware_acceleration();
    if hw_available {
        format!(
            "Video decode: hardware preferred — {} decoder available.",
            hw_description
        )
    } else {
        "Video decode: software (no hardware decoder plugins found; install VA-API, NVDEC, or similar)."
            .to_string()
    }
}

/// Print a summary of detected identical segments
fn print_segment_summary(processors: &[tvt::state_machine::FileProcessor], config: &Config) {
    if config.json_summary {
        return;
    }

    let all_segments = merged_common_segments(processors);

    if all_segments.is_empty() {
        if config.quiet {
            println!("Segments: none over threshold.");
        } else {
            println!("\n=== Segment Analysis Summary ===");
            println!("No identical segments found that meet the threshold criteria.");
        }
        return;
    }

    if config.quiet {
        let total_duration: f64 = all_segments.iter().map(|s| s.end_time - s.start_time).sum();
        println!(
            "Segments: {} (~{:.1}s removable)",
            all_segments.len(),
            total_duration
        );
        return;
    }

    println!("\n=== Segment Analysis Summary ===");
    if config.audio_only {
        println!("(Audio-only mode)");
    } else if config.enable_audio_matching {
        println!("(Combined audio+video matching)");
    }
    println!(
        "Found {} identical segment(s) across {} file(s):",
        all_segments.len(),
        processors.len()
    );
    println!();

    for (i, segment) in all_segments.iter().enumerate() {
        let duration = segment.end_time - segment.start_time;

        println!("Segment {}:", i + 1);
        println!(
            "  Time: {} - {} (duration: {})",
            tvt::format_time(segment.start_time),
            tvt::format_time(segment.end_time),
            tvt::format_time(duration)
        );
        println!("  Match type: {}", segment.match_type);

        // Print separate confidence values
        match segment.match_type {
            tvt::segment_detector::MatchType::Video => {
                if let Some(v_conf) = segment.video_confidence {
                    println!("  Video confidence: {:.0}%", v_conf * 100.0);
                }
            }
            tvt::segment_detector::MatchType::Audio => {
                if let Some(a_conf) = segment.audio_confidence {
                    println!("  Audio confidence: {:.0}%", a_conf * 100.0);
                }
            }
            tvt::segment_detector::MatchType::AudioAndVideo => {
                if let Some(v_conf) = segment.video_confidence {
                    println!("  Video confidence: {:.0}%", v_conf * 100.0);
                }
                if let Some(a_conf) = segment.audio_confidence {
                    println!("  Audio confidence: {:.0}%", a_conf * 100.0);
                }
            }
        }

        println!("  Found in {} episode(s):", segment.episode_list.len());

        // Show per-episode timing if available (for time-shifted segments)
        // Otherwise just show file names with segment timing
        if let Some(ref timings) = segment.episode_timings {
            for timing in timings {
                let offset = timing.start_time - segment.start_time;
                if offset.abs() < 0.5 {
                    println!(
                        "    - {} at {}-{}",
                        timing.episode_name,
                        tvt::format_time(timing.start_time),
                        tvt::format_time(timing.end_time)
                    );
                } else {
                    println!(
                        "    - {} at {}-{} (time shift: {:+.1}s)",
                        timing.episode_name,
                        tvt::format_time(timing.start_time),
                        tvt::format_time(timing.end_time),
                        offset
                    );
                }
            }
        } else {
            // No per-episode timing, show files with segment's reference timing
            for episode_name in &segment.episode_list {
                println!(
                    "    - {} at {}-{}",
                    episode_name,
                    tvt::format_time(segment.start_time),
                    tvt::format_time(segment.end_time)
                );
            }
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
