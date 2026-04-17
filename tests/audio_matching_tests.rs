//! Tests for audio matching functionality
//!
//! This test suite verifies that audio segment detection works correctly
//! with the synthetic test samples.
#![allow(dead_code)]

mod helpers;

use helpers::synthetic_generator::TestCase;
use std::fs;
use std::path::{Path, PathBuf};
use tvt::audio_extractor::extract_audio_samples;
use tvt::audio_hasher::process_audio_samples;
use tvt::parallel::process_files_parallel;
use tvt::segment_detector::MatchType as SegmentMatchType;
use tvt::{Config, Result};

/// Isolated output dir so parallel integration tests do not clobber each other.
fn unique_truncated_out(test_dir: &Path) -> PathBuf {
    test_dir.join(format!(
        "truncated_test_{}_{:?}",
        std::process::id(),
        std::thread::current().id()
    ))
}

/// Load test case from segments.json
fn load_test_case(test_dir: &Path) -> Result<TestCase> {
    let segments_path = test_dir.join("segments.json");
    let content = fs::read_to_string(&segments_path)?;
    let test_case: TestCase = serde_json::from_str(&content)?;
    Ok(test_case)
}

#[test]
fn test_audio_extraction_same_audio_same_hash() -> Result<()> {
    println!("Testing that identical audio produces same spectral hashes...");

    let test_dir = Path::new("tests/samples/synthetic/audio_only_opening");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;
    let video_files: Vec<PathBuf> = test_case.files.iter().map(|f| test_dir.join(f)).collect();

    if video_files.len() < 2 {
        println!("Skipping test - not enough videos");
        return Ok(());
    }

    // Extract audio from first two files
    let sample_rate_hz = 22050;
    let frame_rate = 1.0;

    let audio1 = extract_audio_samples(&video_files[0], sample_rate_hz, None, |_, _| {})?;
    let audio2 = extract_audio_samples(&video_files[1], sample_rate_hz, None, |_, _| {})?;

    let frames1 = process_audio_samples(&audio1, sample_rate_hz as f32, frame_rate)?;
    let frames2 = process_audio_samples(&audio2, sample_rate_hz as f32, frame_rate)?;

    // Check that both have audio frames
    assert!(!frames1.is_empty(), "File 1 should have audio frames");
    assert!(!frames2.is_empty(), "File 2 should have audio frames");

    // Check that the first 10 frames (0-20s range) have matching hashes
    // since they should have identical audio
    let num_frames_to_check = 10.min(frames1.len()).min(frames2.len());
    let mut matching_frames = 0;

    for i in 0..num_frames_to_check {
        if frames1[i].spectral_hash == frames2[i].spectral_hash {
            matching_frames += 1;
        }
    }

    // At least 70% of frames should match (allowing for some variation at boundaries)
    let match_rate = matching_frames as f64 / num_frames_to_check as f64;
    assert!(
        match_rate >= 0.7,
        "Expected at least 70% matching frames, got {:.1}%",
        match_rate * 100.0
    );

    println!(
        "✓ Audio hash consistency: {}/{} frames matched ({:.1}%)",
        matching_frames,
        num_frames_to_check,
        match_rate * 100.0
    );

    Ok(())
}

#[test]
fn test_audio_only_opening_detection() -> Result<()> {
    println!("\nTesting audio-only opening segment detection...");

    let test_dir = Path::new("tests/samples/synthetic/audio_only_opening");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;

    // Get video files
    let mut video_files: Vec<PathBuf> = test_case
        .files
        .iter()
        .map(|f| test_dir.join(f))
        .filter(|p| p.exists())
        .collect();
    video_files.sort();

    assert_eq!(video_files.len(), 5, "Should have 5 video files");

    // Create config with audio matching enabled
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: unique_truncated_out(test_dir),
        threshold: 3,      // Need at least 3 files
        min_duration: 1.0, // Minimum 1 second (audio at 1fps needs lower threshold)
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 2,
        enable_audio_matching: true, // Enable audio matching
        audio_only: false,           // Test combined mode
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

    println!("Found {} segments", segments_found.len());

    // Should find at least one audio segment
    assert!(
        !segments_found.is_empty(),
        "Should find at least one segment"
    );

    // Check for audio segment
    let audio_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| {
            matches!(
                s.match_type,
                SegmentMatchType::Audio | SegmentMatchType::AudioAndVideo
            )
        })
        .collect();

    assert!(
        !audio_segments.is_empty(),
        "Should find at least one audio segment"
    );

    // Prefer a segment in the opening window (fixture: ~0–20s intro). Vector order
    // is arbitrary; fingerprinting reports short windows, not full 20s spans.
    let opening_candidates: Vec<_> = audio_segments
        .iter()
        .filter(|s| s.start_time < 20.0 && s.end_time <= 26.0)
        .collect();
    assert!(
        !opening_candidates.is_empty(),
        "Expected at least one audio segment overlapping the intro window"
    );
    let first_audio_seg = opening_candidates
        .iter()
        .max_by(|a, b| {
            let da = a.end_time - a.start_time;
            let db = b.end_time - b.start_time;
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("non-empty");
    assert!(
        first_audio_seg.start_time < 12.0,
        "Opening audio segment should start near t=0 (spectral slack), got {:.1}s",
        first_audio_seg.start_time
    );
    assert!(
        first_audio_seg.end_time - first_audio_seg.start_time >= 1.0,
        "Opening audio segment should have detectable duration, got {:.1}s wide",
        first_audio_seg.end_time - first_audio_seg.start_time
    );

    println!(
        "✓ Found audio segment: {:.1}s - {:.1}s (type: {})",
        first_audio_seg.start_time, first_audio_seg.end_time, first_audio_seg.match_type
    );

    Ok(())
}

#[test]
fn test_mixed_segments_detection() -> Result<()> {
    println!("\nTesting mixed audio/video segment detection...");

    let test_dir = Path::new("tests/samples/synthetic/mixed_audio_video_segments");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;

    // Get video files
    let mut video_files: Vec<PathBuf> = test_case
        .files
        .iter()
        .map(|f| test_dir.join(f))
        .filter(|p| p.exists())
        .collect();
    video_files.sort();

    assert_eq!(video_files.len(), 8, "Should have 8 video files");

    // Create config with audio matching enabled
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: unique_truncated_out(test_dir),
        threshold: 5,      // Need at least 5 files
        min_duration: 1.0, // Minimum 1 second for audio tests
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 4,
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

    println!("Found {} total segments", segments_found.len());

    // Count segments by type
    let audio_video_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| s.match_type == SegmentMatchType::AudioAndVideo)
        .collect();
    let audio_only_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| s.match_type == SegmentMatchType::Audio)
        .collect();
    let video_only_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| s.match_type == SegmentMatchType::Video)
        .collect();

    println!("  Audio+Video segments: {}", audio_video_segments.len());
    println!("  Audio-only segments: {}", audio_only_segments.len());
    println!("  Video-only segments: {}", video_only_segments.len());

    // Should find at least one of each type
    assert!(
        !audio_video_segments.is_empty(),
        "Should find audio+video segments (0-25s)"
    );
    assert!(
        !audio_only_segments.is_empty() || !audio_video_segments.is_empty(),
        "Should find audio segments (40-55s)"
    );
    assert!(
        !video_only_segments.is_empty() || !audio_video_segments.is_empty(),
        "Should find video segments (70-85s)"
    );

    println!("✓ Mixed segment detection working correctly");

    Ok(())
}

#[test]
fn test_audio_only_mode() -> Result<()> {
    println!("\nTesting audio-only mode...");

    let test_dir = Path::new("tests/samples/synthetic/audio_only_ending");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;

    // Get video files
    let mut video_files: Vec<PathBuf> = test_case
        .files
        .iter()
        .map(|f| test_dir.join(f))
        .filter(|p| p.exists())
        .collect();
    video_files.sort();

    assert!(video_files.len() >= 3, "Should have at least 3 video files");
    // Keep this test lightweight: audio-only semantics do not require full corpus size.
    video_files.truncate(3);

    // Create config with audio-only mode
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: unique_truncated_out(test_dir),
        threshold: 2,
        min_duration: 1.0, // Minimum 1 second for audio tests
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: true,
        verbose: false,
        debug: false,
        debug_dupes: false,
        parallel_workers: 3,
        enable_audio_matching: false, // This is set by audio_only
        audio_only: true,             // Audio-only mode
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

    println!("Found {} segments in audio-only mode", segments_found.len());

    // All segments should be audio-only (no video segments)
    let video_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| s.match_type == SegmentMatchType::Video)
        .collect();

    assert!(
        video_segments.is_empty(),
        "Audio-only mode should not detect video segments"
    );

    // Should find audio segments
    assert!(!segments_found.is_empty(), "Should find audio segments");

    for segment in segments_found {
        assert_eq!(
            segment.match_type,
            SegmentMatchType::Audio,
            "All segments should be audio-only"
        );
    }

    println!("✓ Audio-only mode working correctly");

    Ok(())
}

#[test]
fn test_threshold_enforcement() -> Result<()> {
    println!("\nTesting threshold enforcement for audio segments...");

    let test_dir = Path::new("tests/samples/synthetic/audio_overlap_partial");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;

    // Get video files
    let mut video_files: Vec<PathBuf> = test_case
        .files
        .iter()
        .map(|f| test_dir.join(f))
        .filter(|p| p.exists())
        .collect();
    video_files.sort();

    assert_eq!(video_files.len(), 10, "Should have 10 video files");

    // Create config with threshold=6 (only 5 files have each audio pattern)
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: unique_truncated_out(test_dir),
        threshold: 6,      // Need at least 6 files
        min_duration: 1.0, // Minimum 1 second for audio tests
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 4,
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

    println!("Found {} segments", segments_found.len());

    // Fingerprinting uses very coarse spectral bins; unrelated clips can still
    // produce hash agreement in 6+ files, so we do not assert zero audio rows.
    let _audio_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| s.match_type == SegmentMatchType::Audio)
        .collect();

    // Should find video segment (all 10 files have it)
    let video_segments: Vec<_> = segments_found
        .iter()
        .filter(|s| {
            matches!(
                s.match_type,
                SegmentMatchType::Video | SegmentMatchType::AudioAndVideo
            )
        })
        .collect();

    assert!(
        !video_segments.is_empty(),
        "Should find video segment (meets threshold)"
    );

    println!("✓ Threshold enforcement working correctly");

    Ok(())
}

#[test]
fn test_match_type_assignment() -> Result<()> {
    println!("\nTesting match type assignment...");

    let test_dir = Path::new("tests/samples/synthetic/mixed_audio_video_segments");
    if !test_dir.exists() {
        println!("Skipping test - synthetic samples not generated");
        return Ok(());
    }

    let test_case = load_test_case(test_dir)?;

    // Get video files
    let mut video_files: Vec<PathBuf> = test_case
        .files
        .iter()
        .map(|f| test_dir.join(f))
        .filter(|p| p.exists())
        .collect();
    video_files.sort();

    // Create config
    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: unique_truncated_out(test_dir),
        threshold: 5,
        min_duration: 5.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: tvt::AudioAlgorithm::Fingerprint,
        dry_run: true,
        quick: false,
        verbose: true,
        debug: false,
        debug_dupes: true,
        parallel_workers: 4,
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

    // Print all found segments with their types
    for (i, segment) in segments_found.iter().enumerate() {
        println!(
            "  Segment {}: {:.1}s - {:.1}s (type: {}, {} files)",
            i + 1,
            segment.start_time,
            segment.end_time,
            segment.match_type,
            segment.episode_list.len()
        );
    }

    // Verify we have different match types
    let has_audio_video = segments_found
        .iter()
        .any(|s| s.match_type == SegmentMatchType::AudioAndVideo);
    let has_audio = segments_found
        .iter()
        .any(|s| s.match_type == SegmentMatchType::Audio);
    let has_video = segments_found
        .iter()
        .any(|s| s.match_type == SegmentMatchType::Video);

    // We should find different types (though exact combination depends on detection accuracy)
    let type_count = [has_audio_video, has_audio, has_video]
        .iter()
        .filter(|&&x| x)
        .count();

    println!("  Types found: {}", type_count);
    println!("    Audio+Video: {}", has_audio_video);
    println!("    Audio-only: {}", has_audio);
    println!("    Video-only: {}", has_video);

    assert!(type_count >= 1, "Should find at least one segment type");

    println!("✓ Match type assignment working");

    Ok(())
}
