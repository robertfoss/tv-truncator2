//! End-to-end performance / soak tests (GStreamer + `tvt` binary).
//!
//! **CI:** These are **`#[ignore]`** so the default/fast tier (`cargo test`) stays bounded.
//! Run locally or in the full tier: `cargo test --test performance_tests -- --include-ignored`.
//!
//! **Guardrails:** Time bounds are conservative ceilings for downscaled fixtures — failures
//! usually mean a real regression or a pathological CI machine, not acceptable variance.
//!
//! Synthetic dry-runs iterate every `tests/samples/synthetic/<case>` with `segments.json`
//! ([MEMA-15](/MEMA/issues/MEMA-15)).

mod helpers;

use assert_cmd::Command;
use helpers::sample_fixtures;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

/// Wall-clock ceiling for a full dry-run over `tests/samples/downscaled` (when present).
const MAX_DRY_RUN_DOWNSCALED_SECS: u64 = 180;

/// Test performance of GStreamer frame extraction
#[test]
#[ignore = "slow: GStreamer dry-run over samples; cargo test --test performance_tests -- --include-ignored"]
fn test_gstreamer_frame_extraction_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // Test GStreamer extraction performance
    let start = Instant::now();
    let result = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/downscaled")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .arg("--verbose")
        .output();

    let gstreamer_duration = start.elapsed();

    assert!(result.is_ok());
    println!("GStreamer processing time: {:?}", gstreamer_duration);

    assert!(
        gstreamer_duration.as_secs() < MAX_DRY_RUN_DOWNSCALED_SECS,
        "GStreamer processing took too long: {:?}",
        gstreamer_duration
    );
}

/// Test memory usage during GStreamer processing
#[test]
#[ignore = "slow: GStreamer dry-run; cargo test --test performance_tests -- --include-ignored"]
fn test_gstreamer_memory_usage() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    let result = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/downscaled")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .output();

    assert!(result.is_ok());

    let temp_dir = tempdir().unwrap();
    let temp_files: Vec<_> = fs::read_dir(temp_dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .collect();

    assert_eq!(
        temp_files.len(),
        0,
        "GStreamer should not create temporary files"
    );
}

/// Test parallel processing performance
#[test]
#[ignore = "slow: multiple GStreamer dry-runs; cargo test --test performance_tests -- --include-ignored"]
fn test_parallel_processing_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    let worker_counts = vec![1, 2, 4];
    let mut results = Vec::new();

    for workers in worker_counts {
        let start = Instant::now();
        let result = Command::cargo_bin("tvt")
            .unwrap()
            .arg("--input")
            .arg("tests/samples/downscaled")
            .arg("--threshold")
            .arg("2")
            .arg("--min-duration")
            .arg("1.0")
            .arg("--parallel")
            .arg(workers.to_string())
            .arg("--dry-run")
            .output();

        let duration = start.elapsed();
        assert!(result.is_ok());

        results.push((workers, duration));
        println!("Workers: {}, Time: {:?}", workers, duration);
    }

    assert!(results.len() >= 2);
}

/// Test quick mode performance
#[test]
#[ignore = "slow: paired normal vs quick dry-runs; cargo test --test performance_tests -- --include-ignored"]
fn test_quick_mode_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    let start_normal = Instant::now();
    let result_normal = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/downscaled")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .output();

    let normal_duration = start_normal.elapsed();
    assert!(result_normal.is_ok());

    let start_quick = Instant::now();
    let result_quick = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/downscaled")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--quick")
        .arg("--dry-run")
        .output();

    let quick_duration = start_quick.elapsed();
    assert!(result_quick.is_ok());

    println!("Normal mode: {:?}", normal_duration);
    println!("Quick mode: {:?}", quick_duration);

    println!(
        "Quick mode improvement: {:.1}%",
        (normal_duration.as_secs_f64() - quick_duration.as_secs_f64())
            / normal_duration.as_secs_f64()
            * 100.0
    );
}

/// Test algorithm performance comparison
#[test]
#[ignore = "slow: algorithm sweep dry-runs; cargo test --test performance_tests -- --include-ignored"]
fn test_algorithm_performance_comparison() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // Match clap `SimilarityAlgorithm` value names (kebab-case).
    let algorithms = vec!["current", "multi-hash", "ssim-features"];
    let mut results = Vec::new();

    for algorithm in algorithms {
        let start = Instant::now();
        let result = Command::cargo_bin("tvt")
            .unwrap()
            .arg("--input")
            .arg("tests/samples/downscaled")
            .arg("--threshold")
            .arg("2")
            .arg("--min-duration")
            .arg("1.0")
            .arg("--parallel")
            .arg("1")
            .arg("--algorithm")
            .arg(algorithm)
            .arg("--dry-run")
            .output();

        let duration = start.elapsed();
        assert!(result.is_ok());

        results.push((algorithm, duration));
        println!("Algorithm {}: {:?}", algorithm, duration);
    }

    assert_eq!(results.len(), 3);
}

/// Test synthetic samples performance
#[test]
#[ignore = "slow: synthetic fixture sweep; cargo test --test performance_tests -- --include-ignored"]
fn test_synthetic_samples_performance() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dirs = sample_fixtures::synthetic_subdirs_with_segments(&repo_root);
    assert!(
        !dirs.is_empty(),
        "expected synthetic fixtures under tests/samples/synthetic with segments.json"
    );

    for dir_path in dirs {
        let rel = dir_path
            .strip_prefix(&repo_root)
            .unwrap_or(&dir_path)
            .to_string_lossy()
            .replace('\\', "/");
        if !dir_path.exists() {
            println!("Skipping {} - directory not found", rel);
            continue;
        }
        if !sample_fixtures::dir_has_mkv_video(&dir_path) {
            println!(
                "Skipping {} - no .mkv in fixture dir (optional media / LFS)",
                rel
            );
            continue;
        }

        let start = Instant::now();
        let result = Command::cargo_bin("tvt")
            .unwrap()
            .arg("--input")
            .arg(&dir_path)
            .arg("--threshold")
            .arg("2")
            .arg("--min-duration")
            .arg("1.0")
            .arg("--parallel")
            .arg("1")
            .arg("--dry-run")
            .output();

        let duration = start.elapsed();
        assert!(result.is_ok());

        println!("Synthetic sample {}: {:?}", rel, duration);

        assert!(
            duration.as_secs() < 120,
            "Synthetic sample {} took too long: {:?}",
            rel,
            duration
        );
    }
}

/// Test error handling performance
#[test]
fn test_error_handling_performance() {
    let start = Instant::now();
    let _ = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("nonexistent_directory")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--dry-run")
        .output();

    let duration = start.elapsed();

    assert!(
        duration.as_secs() < 5,
        "Error handling took too long: {:?}",
        duration
    );
}

/// Test large file handling performance
#[test]
#[ignore = "slow: verbose dry-run; cargo test --test performance_tests -- --include-ignored"]
fn test_large_file_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    let start = Instant::now();
    let result = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/downscaled")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .arg("--verbose")
        .output();

    let duration = start.elapsed();
    assert!(result.is_ok());

    println!("Large file processing time: {:?}", duration);

    assert!(
        duration.as_secs() < 600,
        "Large file processing took too long: {:?}",
        duration
    );
}
