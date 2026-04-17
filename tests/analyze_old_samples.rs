//! Analyze existing synthetic samples for audio patterns
//!
//! This tool scans the existing synthetic test samples to identify
//! whether they have identical audio segments.

use std::fs;
use std::path::{Path, PathBuf};
use tvt::audio_extractor::extract_audio_samples;
use tvt::audio_hasher::process_audio_samples;
use tvt::Result;

/// Analyze a test directory for audio patterns
fn analyze_test_directory(test_dir: &Path, verbose: bool) -> Result<()> {
    if verbose {
        println!(
            "\n=== Analyzing: {} ===",
            test_dir.file_name().unwrap().to_string_lossy()
        );
    }

    // Get all .mkv files
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

    if video_files.len() < 2 {
        if verbose {
            println!("  Skipping - less than 2 video files");
        }
        return Ok(());
    }

    if verbose {
        println!("  Files: {}", video_files.len());
    }

    // Extract audio from all files
    let sample_rate_hz = 22050;
    let frame_rate = 1.0;
    let mut all_audio_frames = Vec::new();

    for (i, video_file) in video_files.iter().enumerate() {
        if verbose {
            println!("  Extracting audio from file {}...", i + 1);
        }

        let audio_samples = match extract_audio_samples(video_file, sample_rate_hz, None, |_, _| {})
        {
            Ok(samples) => samples,
            Err(e) => {
                println!("  ✗ Failed to extract audio: {}", e);
                continue;
            }
        };

        let frames = process_audio_samples(&audio_samples, sample_rate_hz as f32, frame_rate)?;
        all_audio_frames.push((video_file.clone(), frames));
    }

    if all_audio_frames.len() < 2 {
        if verbose {
            println!("  Not enough audio extracted for comparison");
        }
        return Ok(());
    }

    // Compare audio patterns between files
    let num_files = all_audio_frames.len();
    let min_frames = all_audio_frames
        .iter()
        .map(|(_, frames)| frames.len())
        .min()
        .unwrap_or(0);

    if min_frames == 0 {
        if verbose {
            println!("  No audio frames to compare");
        }
        return Ok(());
    }

    // Check if first N frames match across all files
    let frames_to_check = 20.min(min_frames);
    let mut intro_matches = true;
    let mut outro_matches = true;

    // Check intro (first N frames)
    for frame_idx in 0..frames_to_check {
        let first_hash = all_audio_frames[0].1[frame_idx].spectral_hash;
        for (_, frames) in &all_audio_frames[1..] {
            if frames[frame_idx].spectral_hash != first_hash {
                intro_matches = false;
                break;
            }
        }
        if !intro_matches {
            break;
        }
    }

    // Check outro (last N frames)
    for i in 0..frames_to_check {
        let frame_idx = min_frames - frames_to_check + i;
        let first_hash = all_audio_frames[0].1[frame_idx].spectral_hash;
        for (_, frames) in &all_audio_frames[1..] {
            if frames.len() <= frame_idx || frames[frame_idx].spectral_hash != first_hash {
                outro_matches = false;
                break;
            }
        }
        if !outro_matches {
            break;
        }
    }

    // Report findings
    if intro_matches {
        println!(
            "  ✓ Identical audio INTRO detected (first ~{}s across all {} files)",
            frames_to_check, num_files
        );
    } else {
        println!("  ✗ Audio intros are different");
    }

    if outro_matches {
        println!(
            "  ✓ Identical audio OUTRO detected (last ~{}s across all {} files)",
            frames_to_check, num_files
        );
    } else {
        println!("  ✗ Audio outros are different");
    }

    // Check for mid-section matches
    if min_frames > frames_to_check * 2 + 10 {
        let mid_start = frames_to_check;
        let mid_end = min_frames - frames_to_check;
        let mid_frames_to_check = 10.min(mid_end - mid_start);

        let mut mid_matches = true;
        for i in 0..mid_frames_to_check {
            let frame_idx = mid_start + i;
            let first_hash = all_audio_frames[0].1[frame_idx].spectral_hash;
            for (_, frames) in &all_audio_frames[1..] {
                if frames[frame_idx].spectral_hash != first_hash {
                    mid_matches = false;
                    break;
                }
            }
            if !mid_matches {
                break;
            }
        }

        if mid_matches {
            println!("  ✓ Identical audio MID-SECTION detected");
        } else {
            println!("  ✗ Mid-section audio is different");
        }
    }

    println!(
        "  Summary: Audio matching would {} change detection results",
        if intro_matches || outro_matches {
            "potentially"
        } else {
            "NOT"
        }
    );

    Ok(())
}

#[test]
#[ignore] // Run with: cargo test --test analyze_old_samples -- --ignored --nocapture
fn test_analyze_all_old_samples() -> Result<()> {
    println!("=== Analyzing Existing Synthetic Samples for Audio Patterns ===\n");

    let synthetic_dir = Path::new("tests/samples/synthetic");
    let old_sample_dirs = [
        "intro",
        "outro",
        "mid_segment",
        "multiple_segments",
        "full_duplicates",
        "intro_outro",
    ];

    for dir_name in &old_sample_dirs {
        let test_dir = synthetic_dir.join(dir_name);
        if test_dir.exists() {
            analyze_test_directory(&test_dir, true)?;
        } else {
            println!("\n=== Skipping: {} (not found) ===", dir_name);
        }
    }

    println!("\n=== Analysis Complete ===");

    Ok(())
}
