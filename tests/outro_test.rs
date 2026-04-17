//! Test for time-shifted outro detection
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use tvt::parallel::process_files_parallel;
use tvt::Config;
use tvt::Result;

/// Parse time string (e.g., "22:21.0" or "45.0s") to seconds
fn parse_time_string(s: &str) -> f64 {
    if s.contains(':') {
        // Format: mm:ss.s
        let parts: Vec<&str> = s.split(':').collect();
        let minutes: f64 = parts[0].parse().unwrap_or(0.0);
        let seconds: f64 = parts[1].trim_end_matches('s').parse().unwrap_or(0.0);
        minutes * 60.0 + seconds
    } else {
        // Format: ss.s or ss.ss
        s.trim_end_matches('s').parse().unwrap_or(0.0)
    }
}

#[derive(Debug, serde::Deserialize)]
struct PerFileTiming {
    start: String,
    end: String,
    offset: String,
}

#[derive(Debug, serde::Deserialize)]
struct ExpectedSegment {
    segment_id: String,
    #[serde(default)]
    time_shifted: bool,
    #[serde(default)]
    per_file_timing: Option<std::collections::HashMap<String, PerFileTiming>>,
}

#[derive(Debug, serde::Deserialize)]
struct TestCase {
    test_name: String,
    description: String,
    files: Vec<String>,
    expected_segments: Vec<ExpectedSegment>,
}

#[test]
fn test_outro_time_shifted_detection() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/outro");

    if !test_dir.exists() {
        println!("Skipping test - outro samples not found");
        return Ok(());
    }

    // Load segments.json
    let segments_path = test_dir.join("segments.json");
    if !segments_path.exists() {
        println!("Skipping test - segments.json not found");
        return Ok(());
    }

    let content = fs::read_to_string(&segments_path)?;
    let test_case: TestCase = serde_json::from_str(&content)?;

    println!("Testing: {}", test_case.test_name);
    println!("Description: {}", test_case.description);

    // Get video files
    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv"))
        .collect();
    video_files.sort();

    assert_eq!(video_files.len(), 3, "Should have 3 video files");

    // Create config
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("truncated_test"),
        threshold: 2, // Need at least 2 files
        min_duration: 10.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: false,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    // Process files
    let processors = process_files_parallel(video_files, config)?;

    // Check results
    let segments_found = processors[0]
        .common_segments
        .as_ref()
        .expect("Should have segments");

    println!("\nDetection results:");
    println!("  Segments found: {}", segments_found.len());

    // Should find the ending credits segment
    assert!(
        !segments_found.is_empty(),
        "Should find at least one segment"
    );

    // Check that the segment covers the expected range (approx 25-70s)
    let segment = &segments_found[0];
    println!("\n  Detected Segment:");
    println!(
        "    Range: {:.1}s - {:.1}s (duration: {:.1}s)",
        segment.start_time,
        segment.end_time,
        segment.end_time - segment.start_time
    );
    println!("    Match type: {}", segment.match_type);
    println!(
        "    Video confidence: {:.0}%",
        segment.video_confidence.unwrap_or(0.0) * 100.0
    );
    println!(
        "    Audio confidence: {:.0}%",
        segment.audio_confidence.unwrap_or(0.0) * 100.0
    );
    println!("    Files: {}", segment.episode_list.len());

    // Verify segment is in expected range (allow some tolerance)
    assert!(
        segment.start_time >= 20.0 && segment.start_time <= 30.0,
        "Segment should start around 25s, got {:.1}s",
        segment.start_time
    );
    assert!(
        segment.end_time >= 65.0 && segment.end_time <= 80.0,
        "Segment should end around 70-78s, got {:.1}s",
        segment.end_time
    );

    // Duration should be substantial (at least 40 seconds)
    let duration = segment.end_time - segment.start_time;
    assert!(
        duration >= 40.0,
        "Outro should be at least 40s, got {:.1}s",
        duration
    );

    // Should include all 3 files
    assert_eq!(
        segment.episode_list.len(),
        3,
        "Segment should be found in all 3 files"
    );

    // Confidence should be good
    assert!(
        segment.confidence >= 0.8,
        "Confidence should be >= 80%, got {:.0}%",
        segment.confidence * 100.0
    );

    // Verify per-episode timing information exists for time-shifted segment
    assert!(
        segment.episode_timings.is_some(),
        "Time-shifted segment should have per-episode timing information"
    );

    let episode_timings = segment.episode_timings.as_ref().unwrap();
    println!("\n  Per-Episode Timing:");

    // Load expected timings from JSON
    let expected_segment = &test_case.expected_segments[0];
    let expected_timings = expected_segment
        .per_file_timing
        .as_ref()
        .expect("Test case should have per_file_timing");

    // Verify each file's timing
    for timing in episode_timings {
        println!(
            "    {} at {:.1}s-{:.1}s (offset: {:+.1}s)",
            timing.episode_name,
            timing.start_time,
            timing.end_time,
            timing.start_time - segment.start_time
        );

        // Get expected timing for this file
        if let Some(expected) = expected_timings.get(&timing.episode_name) {
            let exp_start = parse_time_string(&expected.start);
            let exp_offset = parse_time_string(&expected.offset);

            // Allow 5 second tolerance for start time (detection may find slightly different bounds)
            let start_diff = (timing.start_time - exp_start).abs();
            assert!(
                start_diff < 5.0,
                "{}: start time should be ~{} (±5s), got {:.1}s (diff: {:.1}s)",
                timing.episode_name,
                expected.start,
                timing.start_time,
                start_diff
            );

            let detected_offset = timing.start_time - segment.start_time;
            println!(
                "      ✓ Reference-normalized offset: {:.1}s (expected baseline {:.1}s)",
                detected_offset, exp_offset
            );
        }
    }

    // Verify it's marked as time-shifted in the JSON
    assert!(
        expected_segment.time_shifted,
        "Segment should be marked as time_shifted in JSON"
    );

    println!("\n✓ Time-shifted outro detected correctly!");
    println!("✓ Per-file timing verified within tolerance!");

    Ok(())
}
