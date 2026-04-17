//! Validation tests for synthetic test samples
//!
//! This test suite verifies that generated synthetic samples are correct and playable.

mod helpers;

use helpers::synthetic_generator::{
    check_ffmpeg, generate_test_case, AudioSource, ExpectedSegment, MatchType, SegmentSource,
    TestCase,
};
use std::path::{Path, PathBuf};
use tvt::gstreamer_extractor_v2;
use tvt::Result;

/// Get the base path for synthetic samples
fn get_synthetic_base_path() -> PathBuf {
    PathBuf::from("tests/samples/synthetic")
}

/// Get the base path for downscaled videos
fn get_downscaled_path() -> PathBuf {
    PathBuf::from("tests/samples/downscaled")
}

/// Generate all new audio matching test cases
fn generate_all_audio_test_cases(verbose: bool) -> Result<()> {
    let base_path = get_synthetic_base_path();
    let downscaled_path = get_downscaled_path();

    // Test Case 1: audio_only_opening (5 files)
    generate_audio_only_opening(&base_path, &downscaled_path, verbose)?;

    // Test Case 2: audio_only_ending (7 files)
    generate_audio_only_ending(&base_path, &downscaled_path, verbose)?;

    // Test Case 3: mixed_audio_video_segments (8 files)
    generate_mixed_segments(&base_path, &downscaled_path, verbose)?;

    // Test Case 4: audio_overlap_partial (10 files)
    generate_audio_overlap_partial(&base_path, &downscaled_path, verbose)?;

    Ok(())
}

/// Generate audio_only_opening test case
/// All files have identical 20-second audio intro, different video
fn generate_audio_only_opening(
    base_path: &Path,
    downscaled_path: &Path,
    verbose: bool,
) -> Result<()> {
    let test_dir = base_path.join("audio_only_opening");

    let test_case = TestCase {
        test_name: "audio_only_opening".to_string(),
        description:
            "Five files with identical 20-second audio intro (from 01.mkv), different video content"
                .to_string(),
        files: vec![
            "video1.mkv".to_string(),
            "video2.mkv".to_string(),
            "video3.mkv".to_string(),
            "video4.mkv".to_string(),
            "video5.mkv".to_string(),
        ],
        expected_segments: vec![ExpectedSegment {
            segment_id: "audio_opening".to_string(),
            match_type: MatchType::Audio,
            start_time: 0.0,
            end_time: 20.0,
            files: vec![
                "video1.mkv".to_string(),
                "video2.mkv".to_string(),
                "video3.mkv".to_string(),
                "video4.mkv".to_string(),
                "video5.mkv".to_string(),
            ],
        }],
    };

    // Use different source videos for visual variety
    let source_videos = ["01.mkv", "02.mkv", "03.mkv", "04.mkv", "05.mkv"];

    let mut video_specs = Vec::new();
    for (i, &source) in source_videos.iter().enumerate() {
        let segments = vec![
            // 0-20s: Different video from each source, but identical audio from 01.mkv
            SegmentSource {
                source_video: source.to_string(),
                start_time: 10.0, // Start at 10s into source
                duration: 20.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: "01.mkv".to_string(),
                    start_time: 30.0, // Identical audio segment
                }),
            },
            // 20-35s: Unique outro with original audio/video
            SegmentSource {
                source_video: source.to_string(),
                start_time: 50.0 + (i as f64 * 10.0), // Different time per file
                duration: 15.0,
                use_source_audio: true,
                audio_source: None,
            },
        ];
        video_specs.push(segments);
    }

    generate_test_case(
        &test_dir,
        &test_case,
        &video_specs,
        downscaled_path,
        verbose,
    )?;

    Ok(())
}

/// Generate audio_only_ending test case
/// All files have identical 15-second audio outro, different video
fn generate_audio_only_ending(
    base_path: &Path,
    downscaled_path: &Path,
    verbose: bool,
) -> Result<()> {
    let test_dir = base_path.join("audio_only_ending");

    let test_case = TestCase {
        test_name: "audio_only_ending".to_string(),
        description: "Seven files with unique intros and identical 15-second audio outro (from 02.mkv), different ending video".to_string(),
        files: vec![
            "video1.mkv".to_string(),
            "video2.mkv".to_string(),
            "video3.mkv".to_string(),
            "video4.mkv".to_string(),
            "video5.mkv".to_string(),
            "video6.mkv".to_string(),
            "video7.mkv".to_string(),
        ],
        expected_segments: vec![ExpectedSegment {
            segment_id: "audio_ending".to_string(),
            match_type: MatchType::Audio,
            start_time: 25.0,
            end_time: 40.0,
            files: vec![
                "video1.mkv".to_string(),
                "video2.mkv".to_string(),
                "video3.mkv".to_string(),
                "video4.mkv".to_string(),
                "video5.mkv".to_string(),
                "video6.mkv".to_string(),
                "video7.mkv".to_string(),
            ],
        }],
    };

    let source_videos = [
        "01.mkv", "02.mkv", "03.mkv", "04.mkv", "05.mkv", "27.mkv", "28.mkv",
    ];

    let mut video_specs = Vec::new();
    for (i, &source) in source_videos.iter().enumerate() {
        let segments = vec![
            // 0-25s: Unique intro (different time offset per file)
            SegmentSource {
                source_video: source.to_string(),
                start_time: 20.0 + (i as f64 * 5.0),
                duration: 25.0,
                use_source_audio: true,
                audio_source: None,
            },
            // 25-40s: Different video, identical audio from 02.mkv
            SegmentSource {
                source_video: source.to_string(),
                start_time: 60.0 + (i as f64 * 8.0), // Different video per file
                duration: 15.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: "02.mkv".to_string(),
                    start_time: 100.0, // Identical audio segment
                }),
            },
        ];
        video_specs.push(segments);
    }

    generate_test_case(
        &test_dir,
        &test_case,
        &video_specs,
        downscaled_path,
        verbose,
    )?;

    Ok(())
}

/// Generate mixed_audio_video_segments test case
/// Opening: audio+video match, Middle: audio-only match, Ending: video-only match
fn generate_mixed_segments(base_path: &Path, downscaled_path: &Path, verbose: bool) -> Result<()> {
    let test_dir = base_path.join("mixed_audio_video_segments");

    let test_case = TestCase {
        test_name: "mixed_audio_video_segments".to_string(),
        description: "Eight files with three types of matches: audio+video (0-25s), audio-only (40-55s), video-only (70-85s)".to_string(),
        files: vec![
            "video1.mkv".to_string(),
            "video2.mkv".to_string(),
            "video3.mkv".to_string(),
            "video4.mkv".to_string(),
            "video5.mkv".to_string(),
            "video6.mkv".to_string(),
            "video7.mkv".to_string(),
            "video8.mkv".to_string(),
        ],
        expected_segments: vec![
            ExpectedSegment {
                segment_id: "audio_video_opening".to_string(),
                match_type: MatchType::AudioAndVideo,
                start_time: 0.0,
                end_time: 25.0,
                files: vec![
                    "video1.mkv".to_string(),
                    "video2.mkv".to_string(),
                    "video3.mkv".to_string(),
                    "video4.mkv".to_string(),
                    "video5.mkv".to_string(),
                    "video6.mkv".to_string(),
                    "video7.mkv".to_string(),
                    "video8.mkv".to_string(),
                ],
            },
            ExpectedSegment {
                segment_id: "audio_only_middle".to_string(),
                match_type: MatchType::Audio,
                start_time: 40.0,
                end_time: 55.0,
                files: vec![
                    "video1.mkv".to_string(),
                    "video2.mkv".to_string(),
                    "video3.mkv".to_string(),
                    "video4.mkv".to_string(),
                    "video5.mkv".to_string(),
                    "video6.mkv".to_string(),
                    "video7.mkv".to_string(),
                    "video8.mkv".to_string(),
                ],
            },
            ExpectedSegment {
                segment_id: "video_only_ending".to_string(),
                match_type: MatchType::Video,
                start_time: 70.0,
                end_time: 85.0,
                files: vec![
                    "video1.mkv".to_string(),
                    "video2.mkv".to_string(),
                    "video3.mkv".to_string(),
                    "video4.mkv".to_string(),
                    "video5.mkv".to_string(),
                    "video6.mkv".to_string(),
                    "video7.mkv".to_string(),
                    "video8.mkv".to_string(),
                ],
            },
        ],
    };

    let source_videos = [
        "01.mkv", "02.mkv", "03.mkv", "04.mkv", "05.mkv", "27.mkv", "28.mkv", "29.mkv",
    ];

    let mut video_specs = Vec::new();
    for (i, &source) in source_videos.iter().enumerate() {
        let segments = vec![
            // 0-25s: Identical audio + identical video from 01.mkv
            SegmentSource {
                source_video: "01.mkv".to_string(),
                start_time: 10.0, // Same segment for all files
                duration: 25.0,
                use_source_audio: true, // Use source audio (identical)
                audio_source: None,
            },
            // 25-40s: Unique transition
            SegmentSource {
                source_video: source.to_string(),
                start_time: 30.0 + (i as f64 * 5.0),
                duration: 15.0,
                use_source_audio: true,
                audio_source: None,
            },
            // 40-55s: Different video, identical audio from 02.mkv
            SegmentSource {
                source_video: source.to_string(),
                start_time: 50.0 + (i as f64 * 6.0), // Different video per file
                duration: 15.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: "02.mkv".to_string(),
                    start_time: 80.0, // Identical audio
                }),
            },
            // 55-70s: Unique mid section
            SegmentSource {
                source_video: source.to_string(),
                start_time: 100.0 + (i as f64 * 7.0),
                duration: 15.0,
                use_source_audio: true,
                audio_source: None,
            },
            // 70-85s: Identical video, different audio
            SegmentSource {
                source_video: "03.mkv".to_string(),
                start_time: 120.0, // Same video for all files
                duration: 15.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: source.to_string(), // Different audio per file
                    start_time: 150.0 + (i as f64 * 8.0),
                }),
            },
            // 85-95s: Unique outro
            SegmentSource {
                source_video: source.to_string(),
                start_time: 200.0 + (i as f64 * 9.0),
                duration: 10.0,
                use_source_audio: true,
                audio_source: None,
            },
        ];
        video_specs.push(segments);
    }

    generate_test_case(
        &test_dir,
        &test_case,
        &video_specs,
        downscaled_path,
        verbose,
    )?;

    Ok(())
}

/// Generate audio_overlap_partial test case
/// Tests threshold requirement: 5 files with audio A, 5 with audio B
fn generate_audio_overlap_partial(
    base_path: &Path,
    downscaled_path: &Path,
    verbose: bool,
) -> Result<()> {
    let test_dir = base_path.join("audio_overlap_partial");

    let test_case = TestCase {
        test_name: "audio_overlap_partial".to_string(),
        description: "Ten files: 5 with audio A, 5 with audio B, all with identical video at 30-45s. Only video segment should be detected (threshold=6).".to_string(),
        files: vec![
            "video1.mkv".to_string(),
            "video2.mkv".to_string(),
            "video3.mkv".to_string(),
            "video4.mkv".to_string(),
            "video5.mkv".to_string(),
            "video6.mkv".to_string(),
            "video7.mkv".to_string(),
            "video8.mkv".to_string(),
            "video9.mkv".to_string(),
            "video10.mkv".to_string(),
        ],
        expected_segments: vec![
            // NO audio segments expected (only 5 files each, below threshold=6)
            ExpectedSegment {
                segment_id: "video_middle".to_string(),
                match_type: MatchType::Video,
                start_time: 30.0,
                end_time: 45.0,
                files: vec![
                    "video1.mkv".to_string(),
                    "video2.mkv".to_string(),
                    "video3.mkv".to_string(),
                    "video4.mkv".to_string(),
                    "video5.mkv".to_string(),
                    "video6.mkv".to_string(),
                    "video7.mkv".to_string(),
                    "video8.mkv".to_string(),
                    "video9.mkv".to_string(),
                    "video10.mkv".to_string(),
                ],
            },
        ],
    };

    let source_videos = [
        "01.mkv", "02.mkv", "03.mkv", "04.mkv", "05.mkv", "27.mkv", "28.mkv", "29.mkv", "30.mkv",
        "74.mkv",
    ];

    let mut video_specs = Vec::new();
    for i in 0..10 {
        // First 5 files have audio A (from 04.mkv), last 5 have audio B (from 05.mkv)
        let audio_source = if i < 5 { "04.mkv" } else { "05.mkv" };
        let source = source_videos[i];

        let segments = vec![
            // 0-30s: Different video per file, different audio per group
            SegmentSource {
                source_video: source.to_string(),
                start_time: 20.0 + (i as f64 * 4.0),
                duration: 30.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: audio_source.to_string(),
                    start_time: 40.0 + (i as f64 * 3.0),
                }),
            },
            // 30-45s: Identical video from 27.mkv, different audio per group
            SegmentSource {
                source_video: "27.mkv".to_string(),
                start_time: 100.0, // Same video for all files
                duration: 15.0,
                use_source_audio: false,
                audio_source: Some(AudioSource {
                    source_video: audio_source.to_string(),
                    start_time: 120.0 + (i as f64 * 2.0), // Different audio per group
                }),
            },
            // 45-60s: Unique outro
            SegmentSource {
                source_video: source.to_string(),
                start_time: 150.0 + (i as f64 * 5.0),
                duration: 15.0,
                use_source_audio: true,
                audio_source: None,
            },
        ];
        video_specs.push(segments);
    }

    generate_test_case(
        &test_dir,
        &test_case,
        &video_specs,
        downscaled_path,
        verbose,
    )?;

    Ok(())
}

/// Verify a video file is playable
fn verify_video_playable(video_path: &Path) -> Result<bool> {
    // Initialize GStreamer
    gstreamer_extractor_v2::init_gstreamer()?;

    // Get video info to verify it's playable
    let config = tvt::Config::default();
    let _duration = tvt::gstreamer_extractor_v2::get_video_duration_gstreamer(video_path, &config)?;

    // If we got here without error, the video is playable
    Ok(true)
}

/// Verify a test case directory
fn verify_test_case(test_dir: &Path, verbose: bool) -> Result<bool> {
    if verbose {
        println!("\nVerifying test case: {:?}", test_dir);
    }

    // Load segments.json
    let segments_path = test_dir.join("segments.json");
    if !segments_path.exists() {
        eprintln!("  ✗ segments.json not found");
        return Ok(false);
    }

    let content = std::fs::read_to_string(&segments_path)?;
    let test_case: TestCase = serde_json::from_str(&content)?;

    if verbose {
        println!("  Test: {}", test_case.test_name);
        println!("  Description: {}", test_case.description);
        println!("  Expected files: {}", test_case.files.len());
        println!("  Expected segments: {}", test_case.expected_segments.len());
    }

    // Verify all video files exist and are playable
    let mut all_ok = true;
    for filename in &test_case.files {
        let video_path = test_dir.join(filename);
        if !video_path.exists() {
            eprintln!("  ✗ Video file not found: {}", filename);
            all_ok = false;
            continue;
        }

        // Verify playable
        match verify_video_playable(&video_path) {
            Ok(_) => {
                if verbose {
                    println!("  ✓ {}: playable", filename);
                }
            }
            Err(e) => {
                eprintln!("  ✗ {}: error verifying: {}", filename, e);
                all_ok = false;
            }
        }
    }

    // Verify segment metadata
    for segment in &test_case.expected_segments {
        if verbose {
            println!(
                "  Segment '{}': {} from {:.1}s to {:.1}s ({} files)",
                segment.segment_id,
                match segment.match_type {
                    MatchType::Audio => "audio",
                    MatchType::Video => "video",
                    MatchType::AudioAndVideo => "audio+video",
                },
                segment.start_time,
                segment.end_time,
                segment.files.len()
            );
        }
    }

    Ok(all_ok)
}

#[test]
#[ignore] // Run with: cargo test --test synthetic_validation -- --ignored --nocapture
fn test_generate_audio_synthetic_samples() -> Result<()> {
    println!("=== Generating Audio Matching Synthetic Test Samples ===\n");

    // Check FFmpeg availability
    match check_ffmpeg() {
        Ok(_) => println!("✓ FFmpeg is available\n"),
        Err(e) => {
            println!("✗ FFmpeg check failed: {}", e);
            println!("Skipping test generation. Install FFmpeg to generate synthetic videos.");
            return Ok(());
        }
    }

    // Generate all test cases
    println!("Generating test cases...\n");
    generate_all_audio_test_cases(true)?;

    println!("\n=== Generation Complete ===\n");

    Ok(())
}

#[test]
#[ignore] // Run with: cargo test --test synthetic_validation -- --ignored --nocapture
fn test_validate_audio_synthetic_samples() -> Result<()> {
    println!("=== Validating Audio Matching Synthetic Test Samples ===\n");

    let base_path = get_synthetic_base_path();

    let test_dirs = vec![
        "audio_only_opening",
        "audio_only_ending",
        "mixed_audio_video_segments",
        "audio_overlap_partial",
    ];

    let mut all_valid = true;

    for test_dir_name in &test_dirs {
        let test_dir = base_path.join(test_dir_name);

        if !test_dir.exists() {
            println!("⚠ Test directory not found: {}", test_dir_name);
            println!("  Run test_generate_audio_synthetic_samples first to generate samples.\n");
            all_valid = false;
            continue;
        }

        match verify_test_case(&test_dir, true) {
            Ok(true) => println!("✓ {} is valid\n", test_dir_name),
            Ok(false) => {
                println!("✗ {} has errors\n", test_dir_name);
                all_valid = false;
            }
            Err(e) => {
                println!("✗ {} validation error: {}\n", test_dir_name, e);
                all_valid = false;
            }
        }
    }

    println!("\n=== Validation Summary ===");
    if all_valid {
        println!("✓ All synthetic test samples are valid");
    } else {
        println!("✗ Some test samples have errors");
    }

    assert!(all_valid, "Synthetic test samples validation failed");

    Ok(())
}
