//! Tests to verify video cutting works correctly

use std::path::{Path, PathBuf};
use std::process::Command;
use tvt::{Config, Result};
use tvt::parallel::process_files_parallel;

/// Get video duration using ffprobe
fn get_video_duration(video_path: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("quiet")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("csv=p=0")
        .arg(video_path)
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    let duration_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(duration_str.parse::<f64>()?)
}

#[test]
fn test_cutting_removes_correct_duration_downscaled_2file() -> Result<()> {
    use std::fs;
    
    // Use downscaled_2file test samples
    let input_dir = PathBuf::from("tests/samples/downscaled_2file");
    
    if !input_dir.exists() {
        println!("Skipping test - downscaled_2file samples not found");
        return Ok(());
    }
    
    let output_dir = input_dir.join("truncated_test");
    
    // Clean up any previous test output
    let _ = fs::remove_dir_all(&output_dir);
    
    // Create config for processing
    let config = Config {
        input_dir: input_dir.clone(),
        output_dir: output_dir.clone(),
        threshold: 2, // Find segments in at least 2 files
        min_duration: 10.0, // Minimum 10 seconds
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        dry_run: false, // Actually cut the videos
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: false,
        parallel_workers: 1,
    };
    
    // Find video files (should be 2: episodes 01 and 02)
    let mut video_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv")
        })
        .collect();
    video_files.sort();
    
    // Should have at least 2 files
    assert!(video_files.len() >= 2, "Need at least 2 video files for testing");
    
    // Get only first 2 files for faster testing
    video_files.truncate(2);
    
    println!("Testing with {} files", video_files.len());
    
    // Get original durations
    let original_duration_01 = get_video_duration(&video_files[0])?;
    let original_duration_02 = get_video_duration(&video_files[1])?;
    
    println!("Original durations: {:.2}s, {:.2}s", original_duration_01, original_duration_02);
    
    // Process files
    let processors = process_files_parallel(video_files.clone(), config)?;
    
    // Check that files were processed successfully
    let completed_count = processors.iter().filter(|p| {
        matches!(p.state, tvt::state_machine::ProcessingState::Done { .. })
    }).count();
    
    assert_eq!(completed_count, 2, "Both files should be processed successfully");
    
    // Get truncated file paths
    let truncated_01 = output_dir.join(video_files[0].file_name().unwrap());
    let truncated_02 = output_dir.join(video_files[1].file_name().unwrap());
    
    // Verify truncated files exist
    assert!(truncated_01.exists(), "Truncated file 01 should exist");
    assert!(truncated_02.exists(), "Truncated file 02 should exist");
    
    // Get truncated durations
    let truncated_duration_01 = get_video_duration(&truncated_01)?;
    let truncated_duration_02 = get_video_duration(&truncated_02)?;
    
    println!("Truncated durations: {:.2}s, {:.2}s", truncated_duration_01, truncated_duration_02);
    
    // Calculate removed durations
    let removed_01 = original_duration_01 - truncated_duration_01;
    let removed_02 = original_duration_02 - truncated_duration_02;
    
    println!("Removed durations: {:.2}s, {:.2}s", removed_01, removed_02);
    
    // Verify that approximately 30-32 seconds were removed (allowing for keyframe alignment)
    // The opening credits are about 31.8 seconds
    assert!(removed_01 >= 28.0 && removed_01 <= 35.0, 
        "Should remove ~31.8s from file 01, but removed {:.2}s", removed_01);
    assert!(removed_02 >= 28.0 && removed_02 <= 35.0, 
        "Should remove ~31.8s from file 02, but removed {:.2}s", removed_02);
    
    // Verify truncated files are shorter than originals
    assert!(truncated_duration_01 < original_duration_01, 
        "Truncated file 01 should be shorter than original");
    assert!(truncated_duration_02 < original_duration_02, 
        "Truncated file 02 should be shorter than original");
    
    // Cleanup
    let _ = fs::remove_dir_all(&output_dir);
    
    Ok(())
}

#[test]
fn test_truncated_files_are_valid_videos() -> Result<()> {
    use std::fs;
    
    let input_dir = PathBuf::from("tests/samples/downscaled_2file");
    
    if !input_dir.exists() {
        println!("Skipping test - downscaled_2file samples not found");
        return Ok(());
    }
    
    let output_dir = input_dir.join("truncated_test_validity");
    let _ = fs::remove_dir_all(&output_dir);
    
    let config = Config {
        input_dir: input_dir.clone(),
        output_dir: output_dir.clone(),
        threshold: 2,
        min_duration: 10.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        dry_run: false,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: false,
        parallel_workers: 1,
    };
    
    let mut video_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv")
        })
        .collect();
    video_files.sort();
    video_files.truncate(2);
    
    // Process files
    let _ = process_files_parallel(video_files.clone(), config)?;
    
    // Check that truncated files can be probed (indicates they're valid videos)
    for video_file in &video_files {
        let truncated_path = output_dir.join(video_file.file_name().unwrap());
        
        if truncated_path.exists() {
            // Try to get video info - this will fail if the file is corrupted
            let duration = get_video_duration(&truncated_path)?;
            assert!(duration > 0.0, "Truncated file should have positive duration");
            
            println!("Truncated file is valid: {} ({:.2}s)", 
                truncated_path.file_name().unwrap().to_string_lossy(), duration);
        }
    }
    
    // Cleanup
    let _ = fs::remove_dir_all(&output_dir);
    
    Ok(())
}

#[test]
fn test_single_segment_cutting() -> Result<()> {
    use std::fs;
    
    let input_dir = PathBuf::from("tests/samples/downscaled_2file");
    
    if !input_dir.exists() {
        println!("Skipping test - downscaled_2file samples not found");
        return Ok(());
    }
    
    let output_dir = input_dir.join("truncated_test_single");
    let _ = fs::remove_dir_all(&output_dir);
    
    let config = Config {
        input_dir: input_dir.clone(),
        output_dir: output_dir.clone(),
        threshold: 2,
        min_duration: 10.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        dry_run: false,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: false,
        parallel_workers: 1,
    };
    
    let mut video_files: Vec<PathBuf> = fs::read_dir(&input_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("mkv")
        })
        .collect();
    video_files.sort();
    video_files.truncate(2);
    
    // Process files
    let processors = process_files_parallel(video_files, config)?;
    
    // Verify at least one segment was detected
    let total_segments: usize = processors.iter()
        .filter_map(|p| p.common_segments.as_ref())
        .map(|segs| segs.len())
        .sum();
    
    assert!(total_segments > 0, "Should detect at least one common segment");
    
    println!("Detected {} common segment(s)", total_segments);
    
    // Cleanup
    let _ = fs::remove_dir_all(&output_dir);
    
    Ok(())
}

