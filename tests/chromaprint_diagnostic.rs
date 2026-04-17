//! Diagnostic test for Chromaprint to understand why it's not finding segments

use std::path::Path;
use tvt::audio_chromaprint::detect_audio_segments_chromaprint;
use tvt::audio_extractor::{extract_audio_samples, EpisodeAudio};
use tvt::audio_features::extract_chromaprint_landmarks;
use tvt::audio_hasher::process_audio_samples;
use tvt::Result;
use tvt::{AudioAlgorithm, Config};

#[test]
fn test_chromaprint_landmark_extraction() -> Result<()> {
    let test_file = Path::new("tests/samples/synthetic/intro/intro_1.mkv");

    if !test_file.exists() {
        println!("Skipping - test file not found");
        return Ok(());
    }

    println!("\n=== Chromaprint Diagnostic: Landmark Extraction ===");

    // Extract audio
    let audio_samples = extract_audio_samples(test_file, 22050, None, |_, _| {})?;
    println!("Extracted {} audio samples", audio_samples.len());

    // Extract landmarks
    let landmarks = extract_chromaprint_landmarks(&audio_samples, 22050.0)?;
    println!("Extracted {} landmarks", landmarks.len());

    // Show first 10 landmarks
    println!("\nFirst 10 landmarks:");
    for (i, landmark) in landmarks.iter().take(10).enumerate() {
        println!(
            "  {}: hash={:08x}, timestamp={:.2}s",
            i, landmark.hash, landmark.timestamp
        );
    }

    // Count unique hashes
    let mut unique_hashes = std::collections::HashSet::new();
    for landmark in &landmarks {
        unique_hashes.insert(landmark.hash);
    }
    println!(
        "\nUnique hashes: {} / {}",
        unique_hashes.len(),
        landmarks.len()
    );

    assert!(!landmarks.is_empty(), "Should extract some landmarks");

    Ok(())
}

#[test]
fn test_chromaprint_simple_match() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/intro");

    if !test_dir.exists() {
        println!("Skipping - test directory not found");
        return Ok(());
    }

    println!("\n=== Chromaprint Diagnostic: Simple Matching Test ===");

    let video_files = vec![test_dir.join("intro_1.mkv"), test_dir.join("intro_2.mkv")];

    let mut episode_audio = Vec::new();
    for video_path in &video_files {
        if !video_path.exists() {
            println!("Skipping - {} not found", video_path.display());
            continue;
        }

        println!("Extracting audio from: {}", video_path.display());
        let audio_samples = extract_audio_samples(video_path, 22050, None, |_, _| {})?;
        let audio_frames = process_audio_samples(&audio_samples, 22050.0, 1.0)?;

        episode_audio.push(EpisodeAudio {
            episode_path: video_path.clone(),
            audio_frames,
            raw_samples: audio_samples,
            sample_rate: 22050.0,
        });
    }

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("test_out"),
        threshold: 2,
        min_duration: 5.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: AudioAlgorithm::Chromaprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true, // Enable debug
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let segments = detect_audio_segments_chromaprint(&episode_audio, &config, true)?;

    println!("\n=== Results ===");
    println!("Found {} segments", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Segment {}: {:.1}s-{:.1}s ({} episodes, conf={:.2})",
            i,
            seg.start_time,
            seg.end_time,
            seg.episode_list.len(),
            seg.confidence
        );
    }

    Ok(())
}
