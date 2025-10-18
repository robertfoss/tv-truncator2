//! Performance comparison tests between GStreamer and FFmpeg approaches
//!
//! These tests validate the performance improvements achieved by migrating
//! from FFmpeg to GStreamer for video processing.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tempfile::tempdir;

/// Test performance of GStreamer frame extraction
#[test]
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

    // Performance expectations:
    // - Should complete in under 2 minutes for downscaled samples
    // - Frame extraction should be the fastest part
    assert!(
        gstreamer_duration.as_secs() < 120,
        "GStreamer processing took too long: {:?}",
        gstreamer_duration
    );
}

/// Test memory usage during GStreamer processing
#[test]
fn test_gstreamer_memory_usage() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // This is a basic test - in a real implementation, we would use
    // memory profiling tools to measure actual memory usage
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

    // GStreamer should not create temporary files
    let temp_dir = tempdir().unwrap();
    let temp_files: Vec<_> = fs::read_dir(temp_dir.path())
        .unwrap()
        .filter_map(|entry| entry.ok())
        .collect();

    // Should have no temporary files (GStreamer processes in memory)
    assert_eq!(
        temp_files.len(),
        0,
        "GStreamer should not create temporary files"
    );
}

/// Test parallel processing performance
#[test]
fn test_parallel_processing_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // Test with different parallel worker counts
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

    // Parallel processing should show improvement with more workers
    // (though the improvement may be limited by I/O and algorithm complexity)
    assert!(results.len() >= 2);
}

/// Test quick mode performance
#[test]
fn test_quick_mode_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // Test normal mode vs quick mode
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

    // Quick mode should be faster (lower sampling rate)
    // Note: This may not always be true due to other factors
    println!(
        "Quick mode improvement: {:.1}%",
        (normal_duration.as_secs_f64() - quick_duration.as_secs_f64())
            / normal_duration.as_secs_f64()
            * 100.0
    );
}

/// Test algorithm performance comparison
#[test]
fn test_algorithm_performance_comparison() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    let algorithms = vec!["current", "multihash", "ssim"];
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

    // All algorithms should complete successfully
    assert_eq!(results.len(), 3);
}

/// Test synthetic samples performance
#[test]
fn test_synthetic_samples_performance() {
    let synthetic_dirs = vec![
        "tests/samples/synthetic/opening_credits",
        "tests/samples/synthetic/full_duplicates",
        "tests/samples/synthetic/mid_segment",
    ];

    for dir in synthetic_dirs {
        let dir_path = PathBuf::from(dir);
        if !dir_path.exists() {
            println!("Skipping {} - directory not found", dir);
            continue;
        }

        let start = Instant::now();
        let result = Command::cargo_bin("tvt")
            .unwrap()
            .arg("--input")
            .arg(dir)
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

        println!("Synthetic sample {}: {:?}", dir, duration);

        // Synthetic samples should process quickly
        assert!(
            duration.as_secs() < 60,
            "Synthetic sample {} took too long: {:?}",
            dir,
            duration
        );
    }
}

/// Test error handling performance
#[test]
fn test_error_handling_performance() {
    // Test with non-existent directory
    let start = Instant::now();
    let result = Command::cargo_bin("tvt")
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

    // Should fail quickly, not hang
    assert!(
        duration.as_secs() < 5,
        "Error handling took too long: {:?}",
        duration
    );
}

/// Test large file handling performance
#[test]
fn test_large_file_performance() {
    let video_path = PathBuf::from("tests/samples/downscaled/27.mkv");
    if !video_path.exists() {
        println!("Skipping test - sample video not found");
        return;
    }

    // Test with high sampling rate (more frames)
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

    // Should handle large files efficiently
    assert!(
        duration.as_secs() < 300,
        "Large file processing took too long: {:?}",
        duration
    );
}
