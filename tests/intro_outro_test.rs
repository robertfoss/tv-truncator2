//! Test for intro+outro detection with different length middle sections
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};
use tvt::parallel::process_files_parallel;
use tvt::segment_detector::MatchType;
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
        // Format: ss.ss or ss.ss
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
    match_type: String,
    start_time: String,
    end_time: String,
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
fn test_intro_outro_detection() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/intro_outro");

    if !test_dir.exists() {
        println!("Skipping test - intro_outro samples not found");
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
    println!("Expected {} segments", test_case.expected_segments.len());

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
        threshold: 2,
        min_duration: 5.0, // Lower threshold to detect shorter segments
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

    // Should find both intro and outro segments
    assert!(
        segments_found.len() >= 2,
        "Should find at least 2 segments (intro and outro), got {}",
        segments_found.len()
    );

    // Verify each expected segment
    for (i, expected) in test_case.expected_segments.iter().enumerate() {
        println!(
            "\n  Validating expected segment {}: {}",
            i + 1,
            expected.segment_id
        );

        // Parse expected time (handle both numeric and string formats)
        let expected_start = parse_time_string(&expected.start_time);

        // Find this segment in detected segments
        let detected = segments_found.iter().find(|s| {
            // Match by approximate time range
            (s.start_time - expected_start).abs() < 10.0
        });

        assert!(
            detected.is_some(),
            "Expected segment '{}' at {}-{} not found in detected segments",
            expected.segment_id,
            expected.start_time,
            expected.end_time
        );

        let segment = detected.unwrap();
        println!(
            "    Detected at: {:.1}s - {:.1}s (duration: {:.1}s)",
            segment.start_time,
            segment.end_time,
            segment.end_time - segment.start_time
        );
        println!("    Match type: {}", segment.match_type);

        // Verify match type
        let expected_match_type = match expected.match_type.as_str() {
            "audio" => MatchType::Audio,
            "video" => MatchType::Video,
            "audio+video" => MatchType::AudioAndVideo,
            _ => panic!("Invalid match type in JSON: {}", expected.match_type),
        };

        assert_eq!(
            segment.match_type, expected_match_type,
            "Segment '{}' should have match type {:?}, got {:?}",
            expected.segment_id, expected_match_type, segment.match_type
        );

        // Verify all 3 files are included
        assert_eq!(
            segment.episode_list.len(),
            3,
            "Segment '{}' should include all 3 files",
            expected.segment_id
        );

        // Verify per-file timing if segment is time-shifted and has expected timing data
        if expected.time_shifted && expected.per_file_timing.is_some() {
            assert!(
                segment.episode_timings.is_some(),
                "Time-shifted segment '{}' should have per-episode timing",
                expected.segment_id
            );

            let episode_timings = segment.episode_timings.as_ref().unwrap();
            println!("    Per-file timing:");

            if let Some(ref expected_timings) = expected.per_file_timing {
                for timing in episode_timings {
                    println!(
                        "      {} at {:.1}s-{:.1}s (offset: {:+.1}s)",
                        timing.episode_name,
                        timing.start_time,
                        timing.end_time,
                        timing.start_time - segment.start_time
                    );

                    // Validate timing
                    if let Some(exp_timing) = expected_timings.get(&timing.episode_name) {
                        let exp_start = parse_time_string(&exp_timing.start);
                        let exp_offset = parse_time_string(&exp_timing.offset);

                        let start_diff = (timing.start_time - exp_start).abs();
                        assert!(
                            start_diff < 10.0,
                            "{}: start time should be ~{} (±10s), got {:.1}s",
                            timing.episode_name,
                            exp_timing.start,
                            timing.start_time
                        );

                        let detected_offset = timing.start_time - segment.start_time;
                        let offset_diff = (detected_offset - exp_offset).abs();
                        assert!(
                            offset_diff < 5.0,
                            "{}: offset should be ~{} (±5s), got {:.1}s",
                            timing.episode_name,
                            exp_timing.offset,
                            detected_offset
                        );

                        println!("        ✓ Timing validated");
                    }
                }
            }
        } else {
            println!("    Not time-shifted (same position in all files)");
        }
    }

    println!("\n✓ intro_outro sample processed correctly!");
    println!("✓ Both intro and outro segments detected!");
    println!("✓ Match types correct!");
    println!("✓ Time shifts validated!");

    Ok(())
}
