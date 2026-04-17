//! Test individual algorithms on downscaled_2file

use std::fs;
use std::path::{Path, PathBuf};
use tvt::audio_chromaprint::detect_audio_segments_chromaprint;
use tvt::audio_energy_bands::detect_audio_segments_energy_bands;
use tvt::audio_extractor::{extract_audio_samples, EpisodeAudio};
use tvt::audio_hasher::process_audio_samples;
use tvt::audio_spectral_v2::detect_audio_segments_spectral_v2;
use tvt::Result;
use tvt::{AudioAlgorithm, Config};

fn extract_episode_audio(video_files: &[PathBuf]) -> Result<Vec<EpisodeAudio>> {
    let mut episode_audio = Vec::new();
    let sample_rate = 22050;

    for video_path in video_files {
        println!("  Extracting audio from: {}", video_path.display());
        let audio_samples = extract_audio_samples(video_path, sample_rate, None, |_, _| {})?;
        let audio_frames = process_audio_samples(&audio_samples, sample_rate as f32, 1.0)?;

        episode_audio.push(EpisodeAudio {
            episode_path: video_path.clone(),
            audio_frames,
            raw_samples: audio_samples,
            sample_rate: sample_rate as f32,
        });
    }

    Ok(episode_audio)
}

#[test]
#[ignore]
fn test_chromaprint_downscaled() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/downscaled_2file");

    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n=== Chromaprint on downscaled_2file ===");

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("mkv")
                && !path.to_string_lossy().contains("truncated")
        })
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("test_out"),
        threshold: 2,
        min_duration: 30.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: AudioAlgorithm::Chromaprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let segments = detect_audio_segments_chromaprint(&episode_audio, &config, true)?;

    println!("\n=== Results ===");
    println!(
        "Found {} segments (expected 2: opening 0-114s, ending 1341-1411s)",
        segments.len()
    );
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Segment {}: {:.1}s-{:.1}s ({} episodes, conf={:.2})",
            i,
            seg.start_time,
            seg.end_time,
            seg.episode_list.len(),
            seg.audio_confidence.unwrap_or(seg.confidence)
        );
    }

    // Validate: should find exactly 2 segments
    assert_eq!(
        segments.len(),
        2,
        "Should find exactly 2 segments (opening and ending)"
    );

    // First should be opening (starts near 0, ends before 400s)
    assert!(
        segments[0].start_time < 10.0,
        "Opening should start near 0, got {:.1}s",
        segments[0].start_time
    );
    assert!(
        segments[0].end_time > 100.0 && segments[0].end_time < 400.0,
        "Opening should end around 114s (allowing margin), got {:.1}s",
        segments[0].end_time
    );

    // Second should be ending (starts after 1000s, ends after 1400s)
    assert!(
        segments[1].start_time > 1000.0 && segments[1].start_time < 1350.0,
        "Ending should start around 1341s (allowing margin), got {:.1}s",
        segments[1].start_time
    );
    assert!(
        segments[1].end_time > 1400.0,
        "Ending should end after 1400s, got {:.1}s",
        segments[1].end_time
    );

    println!("✅ Chromaprint PASSED: Found both opening and ending segments correctly");
    Ok(())
}

#[test]
#[ignore]
fn test_spectralv2_downscaled() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/downscaled_2file");

    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n=== SpectralV2 on downscaled_2file ===");

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("mkv")
                && !path.to_string_lossy().contains("truncated")
        })
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("test_out"),
        threshold: 2,
        min_duration: 30.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: AudioAlgorithm::SpectralV2,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let segments = detect_audio_segments_spectral_v2(&episode_audio, &config, true)?;

    println!("\n=== Results ===");
    println!("Found {} segments (expected 2)", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Segment {}: {:.1}s-{:.1}s ({} episodes, conf={:.2})",
            i,
            seg.start_time,
            seg.end_time,
            seg.episode_list.len(),
            seg.audio_confidence.unwrap_or(seg.confidence)
        );
    }

    assert_eq!(segments.len(), 2, "Should find exactly 2 segments");

    // Validate segments are in reasonable ranges
    assert!(segments[0].start_time < 10.0, "Opening should start near 0");
    assert!(
        segments[0].end_time > 100.0 && segments[0].end_time < 400.0,
        "Opening should be in reasonable range"
    );
    assert!(
        segments[1].start_time > 1000.0,
        "Ending should start late in episode"
    );
    assert!(
        segments[1].end_time > 1400.0,
        "Ending should end after 1400s"
    );

    println!("✅ SpectralV2 PASSED");
    Ok(())
}

#[test]
#[ignore]
fn test_energybands_downscaled() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/downscaled_2file");

    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n=== EnergyBands on downscaled_2file ===");

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("mkv")
                && !path.to_string_lossy().contains("truncated")
        })
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("test_out"),
        threshold: 2,
        min_duration: 30.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: AudioAlgorithm::EnergyBands,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let segments = detect_audio_segments_energy_bands(&episode_audio, &config, true)?;

    println!("\n=== Results ===");
    println!("Found {} segments (expected 2)", segments.len());
    for (i, seg) in segments.iter().enumerate() {
        println!(
            "  Segment {}: {:.1}s-{:.1}s ({} episodes, conf={:.2})",
            i,
            seg.start_time,
            seg.end_time,
            seg.episode_list.len(),
            seg.audio_confidence.unwrap_or(seg.confidence)
        );
    }

    assert_eq!(segments.len(), 2, "Should find exactly 2 segments");

    // Validate segments are in reasonable ranges (allow for partial detection)
    assert!(segments[0].start_time < 10.0, "Opening should start near 0");
    assert!(
        segments[0].end_time > 30.0 && segments[0].end_time < 400.0,
        "Opening should be in reasonable range (may be partial)"
    );
    assert!(
        segments[1].start_time > 1000.0,
        "Ending should start late in episode"
    );
    assert!(
        segments[1].end_time > 1350.0,
        "Ending should be near end of episode"
    );

    println!("✅ EnergyBands PASSED (partial segment detection is acceptable)");
    Ok(())
}
