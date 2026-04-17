//! Baseline test to capture current audio algorithm performance
//! This captures metrics before improvements are made

use std::fs;
use std::path::{Path, PathBuf};
use tvt::audio_comparison::{compare_all_algorithms, print_comparison_report, ExpectedSegment};
use tvt::audio_extractor::{extract_audio_samples, EpisodeAudio};
use tvt::audio_hasher::process_audio_samples;
use tvt::Result;
use tvt::{AudioAlgorithm, Config};

/// Helper function to extract audio from video files
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
fn baseline_intro() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/intro");
    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    println!("BASELINE: intro sample (32s intro at 0.0s, 3 files)");
    println!("{}", "=".repeat(80));

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv"))
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("truncated_test"),
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
        debug_dupes: false,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let expected = vec![ExpectedSegment {
        start_time: 0.0,
        end_time: 32.0,
        min_episodes: 3,
    }];

    let metrics = compare_all_algorithms(&episode_audio, &config, &expected, true)?;
    print_comparison_report(&metrics);

    Ok(())
}

#[test]
fn baseline_outro() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/outro");
    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    println!("BASELINE: outro sample (time-shifted outro, 3 files)");
    println!("{}", "=".repeat(80));

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv"))
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("truncated_test"),
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
        debug_dupes: false,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    // Outro is time-shifted: 25s-70s, 25s-75s, 28.8s-78.6s
    let expected = vec![ExpectedSegment {
        start_time: 25.0,
        end_time: 70.0,
        min_episodes: 3,
    }];

    let metrics = compare_all_algorithms(&episode_audio, &config, &expected, true)?;
    print_comparison_report(&metrics);

    Ok(())
}

#[test]
fn baseline_intro_outro() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/intro_outro");
    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    println!("BASELINE: intro_outro sample (intro 0-35s + outro 55-109s, 3 files)");
    println!("{}", "=".repeat(80));

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv"))
        .collect();
    video_files.sort();

    let episode_audio = extract_episode_audio(&video_files)?;

    let config = Config {
        input_dir: test_dir.to_path_buf(),
        output_dir: test_dir.join("truncated_test"),
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
        debug_dupes: false,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let expected = vec![
        ExpectedSegment {
            start_time: 0.0,
            end_time: 35.0,
            min_episodes: 3,
        },
        ExpectedSegment {
            start_time: 55.0,
            end_time: 109.0,
            min_episodes: 3,
        },
    ];

    let metrics = compare_all_algorithms(&episode_audio, &config, &expected, true)?;
    print_comparison_report(&metrics);

    Ok(())
}

#[test]
#[ignore] // Run explicitly due to large files
fn baseline_downscaled_2file() -> Result<()> {
    let test_dir = Path::new("tests/samples/synthetic/downscaled_2file");
    if !test_dir.exists() {
        println!("Skipping - samples not found");
        return Ok(());
    }

    println!("\n{}", "=".repeat(80));
    println!("BASELINE: downscaled_2file (full episodes, 2 files)");
    println!("{}", "=".repeat(80));

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
        output_dir: test_dir.join("truncated_test"),
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
        debug_dupes: false,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    };

    let expected = vec![
        ExpectedSegment {
            start_time: 0.0,
            end_time: 114.0,
            min_episodes: 2,
        },
        ExpectedSegment {
            start_time: 1341.0,
            end_time: 1411.0,
            min_episodes: 2,
        },
    ];

    let metrics = compare_all_algorithms(&episode_audio, &config, &expected, true)?;
    print_comparison_report(&metrics);

    Ok(())
}
