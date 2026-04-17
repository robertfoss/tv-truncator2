//! Benchmark tool for comparing frame extractor performance
//!
//! This tool measures and compares the performance of different frame extraction
//! implementations to identify bottlenecks and verify optimization improvements.

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tvt::gstreamer_extractor_v2::{extract_frames_gstreamer_v2, get_video_duration_gstreamer};
use tvt::{Config, Result};

/// Benchmark frame extractors
#[derive(Parser, Debug)]
#[command(author, version, about = "Benchmark frame extractor performance", long_about = None)]
struct Args {
    /// Input video file or directory
    #[arg(short, long, value_name = "PATH")]
    input: PathBuf,

    /// Extractor to benchmark (legacy, optimized)
    #[arg(long, default_value = "legacy")]
    extractor: String,

    /// Sample rate for frame extraction (fps)
    #[arg(long, default_value = "1.0")]
    sample_rate: f64,

    /// Number of iterations to run
    #[arg(long, default_value = "1")]
    iterations: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Frame Extractor Benchmark ===");
    println!("Input: {}", args.input.display());
    println!("Extractor: Optimized V2 (with hardware acceleration if available)");
    println!("Sample rate: {} fps", args.sample_rate);
    println!("Iterations: {}", args.iterations);
    println!();

    // Find video files
    let video_files = if args.input.is_dir() {
        find_video_files(&args.input)?
    } else if args.input.is_file() {
        vec![args.input.clone()]
    } else {
        anyhow::bail!("Input path does not exist: {}", args.input.display());
    };

    if video_files.is_empty() {
        anyhow::bail!("No video files found");
    }

    println!("Found {} video file(s) to benchmark", video_files.len());
    println!();

    // Run benchmarks
    let mut total_results = Vec::new();

    for video_file in &video_files {
        println!("--- Benchmarking: {} ---", video_file.display());

        let results = benchmark_video(video_file, args.sample_rate, args.iterations)?;
        total_results.push((video_file.clone(), results));

        println!();
    }

    // Print summary
    print_summary(&total_results, args.iterations);

    Ok(())
}

struct BenchmarkResult {
    duration_query_time: std::time::Duration,
    video_duration: f64,
    expected_frames: usize,
    extraction_time: std::time::Duration,
    actual_frames: usize,
    frames_per_second: f64,
}

fn benchmark_video(
    video_path: &PathBuf,
    sample_rate: f64,
    iterations: usize,
) -> Result<Vec<BenchmarkResult>> {
    let config = Config {
        debug: false,
        verbose: false,
        ..Default::default()
    };

    let mut results = Vec::new();

    for i in 0..iterations {
        if iterations > 1 {
            println!("  Iteration {}/{}", i + 1, iterations);
        }

        // Measure duration query time
        let duration_start = Instant::now();
        let video_duration = get_video_duration_gstreamer(video_path, &config)?;
        let duration_query_time = duration_start.elapsed();

        let expected_frames = (video_duration * sample_rate) as usize;

        // Measure extraction time
        let extraction_start = Instant::now();

        // Use optimized V2 extractor (automatically uses VA-API if available)
        let frames = extract_frames_gstreamer_v2(
            video_path,
            sample_rate,
            |_current, _total| {}, // No progress callback for benchmarking
            &config,
        )?;

        let extraction_time = extraction_start.elapsed();
        let actual_frames = frames.len();
        let frames_per_second = if extraction_time.as_secs_f64() > 0.0 {
            actual_frames as f64 / extraction_time.as_secs_f64()
        } else {
            0.0
        };

        results.push(BenchmarkResult {
            duration_query_time,
            video_duration,
            expected_frames,
            extraction_time,
            actual_frames,
            frames_per_second,
        });

        // Print iteration results
        println!("    Duration query: {:?}", duration_query_time);
        println!("    Video duration: {:.2}s", video_duration);
        println!("    Expected frames: {}", expected_frames);
        println!("    Extraction time: {:?}", extraction_time);
        println!("    Actual frames extracted: {}", actual_frames);
        println!("    Frames per second: {:.2}", frames_per_second);
        println!(
            "    Time per frame: {:.2}ms",
            extraction_time.as_millis() as f64 / actual_frames as f64
        );
    }

    Ok(results)
}

fn print_summary(results: &[(PathBuf, Vec<BenchmarkResult>)], iterations: usize) {
    println!("=== Benchmark Summary ===");
    println!();

    for (video_path, video_results) in results {
        println!("File: {}", video_path.display());

        if iterations == 1 {
            let result = &video_results[0];
            println!("  Duration query: {:?}", result.duration_query_time);
            println!("  Video duration: {:.2}s", result.video_duration);
            println!("  Expected frames: {}", result.expected_frames);
            println!("  Extraction time: {:?}", result.extraction_time);
            println!("  Actual frames: {}", result.actual_frames);
            println!("  Frames/sec: {:.2}", result.frames_per_second);
            println!(
                "  Time/frame: {:.2}ms",
                result.extraction_time.as_millis() as f64 / result.actual_frames as f64
            );
        } else {
            // Calculate averages
            let avg_duration_query = video_results
                .iter()
                .map(|r| r.duration_query_time.as_secs_f64())
                .sum::<f64>()
                / iterations as f64;
            let avg_extraction = video_results
                .iter()
                .map(|r| r.extraction_time.as_secs_f64())
                .sum::<f64>()
                / iterations as f64;
            let avg_fps = video_results
                .iter()
                .map(|r| r.frames_per_second)
                .sum::<f64>()
                / iterations as f64;
            let avg_frames =
                video_results.iter().map(|r| r.actual_frames).sum::<usize>() / iterations;

            println!("  Average over {} iterations:", iterations);
            println!("    Duration query: {:.3}s", avg_duration_query);
            println!("    Extraction time: {:.3}s", avg_extraction);
            println!("    Frames extracted: {}", avg_frames);
            println!("    Frames/sec: {:.2}", avg_fps);
            println!(
                "    Time/frame: {:.2}ms",
                avg_extraction * 1000.0 / avg_frames as f64
            );
        }
        println!();
    }
}

fn find_video_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut video_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
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

    // Sort for consistent ordering
    video_files.sort();

    Ok(video_files)
}
