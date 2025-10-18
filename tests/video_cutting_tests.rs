//! Integration tests for video cutting functionality

use std::path::Path;
use std::process::Command;
use tempfile::TempDir;
use tvt::video_processor::cut_video_segments;

/// Create a simple test video using FFmpeg
fn create_test_video(output_path: &Path, duration: f64) -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new("ffmpeg")
        .arg("-f")
        .arg("lavfi")
        .arg("-i")
        .arg(&format!(
            "testsrc=duration={}:size=320x240:rate=1",
            duration
        ))
        .arg("-c:v")
        .arg("libx264")
        .arg("-preset")
        .arg("ultrafast")
        .arg("-y")
        .arg(output_path)
        .output()?;

    if !output.status.success() {
        return Err(format!("FFmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }

    Ok(())
}

#[test]
fn test_cut_video_segments_no_removal() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.mp4");
    let output_path = temp_dir.path().join("output.mp4");

    // Create a test video
    create_test_video(&input_path, 5.0).unwrap();

    // Test with no segments to remove (should copy the file)
    let result = cut_video_segments(&input_path, &output_path, &[]);
    assert!(result.is_ok());

    // Verify output file exists
    assert!(output_path.exists());

    // Verify file was copied (not processed)
    let input_size = std::fs::metadata(&input_path).unwrap().len();
    let output_size = std::fs::metadata(&output_path).unwrap().len();
    assert_eq!(input_size, output_size);
}

#[test]
fn test_cut_video_segments_single_removal() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.mp4");
    let output_path = temp_dir.path().join("output.mp4");

    // Create a test video (10 seconds)
    create_test_video(&input_path, 10.0).unwrap();

    // Remove segment from 2-4 seconds (should result in 8 second video)
    let segments_to_remove = vec![(2.0, 4.0)];
    let result = cut_video_segments(&input_path, &output_path, &segments_to_remove);
    if let Err(e) = &result {
        println!("Error: {}", e);
    }
    assert!(result.is_ok());

    // Verify output file exists
    assert!(output_path.exists());

    // Verify output is smaller than input (segment was removed)
    let input_size = std::fs::metadata(&input_path).unwrap().len();
    let output_size = std::fs::metadata(&output_path).unwrap().len();
    assert!(output_size < input_size);
}

#[test]
fn test_cut_video_segments_multiple_removals() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.mp4");
    let output_path = temp_dir.path().join("output.mp4");

    // Create a test video (15 seconds)
    create_test_video(&input_path, 15.0).unwrap();

    // Remove multiple segments: 2-4 seconds and 8-10 seconds
    let segments_to_remove = vec![(2.0, 4.0), (8.0, 10.0)];
    let result = cut_video_segments(&input_path, &output_path, &segments_to_remove);
    assert!(result.is_ok());

    // Verify output file exists
    assert!(output_path.exists());

    // Verify output is smaller than input (segments were removed)
    let input_size = std::fs::metadata(&input_path).unwrap().len();
    let output_size = std::fs::metadata(&output_path).unwrap().len();
    assert!(output_size < input_size);
}

#[test]
fn test_cut_video_segments_remove_all() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.mp4");
    let output_path = temp_dir.path().join("output.mp4");

    // Create a test video (5 seconds)
    create_test_video(&input_path, 5.0).unwrap();

    // Remove entire video (0-5 seconds)
    let segments_to_remove = vec![(0.0, 5.0)];
    let result = cut_video_segments(&input_path, &output_path, &segments_to_remove);
    assert!(result.is_ok());

    // Verify output file exists (should be empty video)
    assert!(output_path.exists());

    // Verify output is much smaller than input (almost everything removed)
    let input_size = std::fs::metadata(&input_path).unwrap().len();
    let output_size = std::fs::metadata(&output_path).unwrap().len();
    assert!(output_size < input_size);
}

#[test]
fn test_cut_video_segments_edge_cases() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }

    let temp_dir = TempDir::new().unwrap();
    let input_path = temp_dir.path().join("input.mp4");
    let output_path = temp_dir.path().join("output.mp4");

    // Create a test video (10 seconds)
    create_test_video(&input_path, 10.0).unwrap();

    // Test removing from the beginning
    let segments_to_remove = vec![(0.0, 2.0)];
    let result = cut_video_segments(&input_path, &output_path, &segments_to_remove);
    assert!(result.is_ok());
    assert!(output_path.exists());

    // Test removing from the end
    let output_path2 = temp_dir.path().join("output2.mp4");
    let segments_to_remove = vec![(8.0, 10.0)];
    let result = cut_video_segments(&input_path, &output_path2, &segments_to_remove);
    assert!(result.is_ok());
    assert!(output_path2.exists());

    // Test removing from the middle
    let output_path3 = temp_dir.path().join("output3.mp4");
    let segments_to_remove = vec![(4.0, 6.0)];
    let result = cut_video_segments(&input_path, &output_path3, &segments_to_remove);
    assert!(result.is_ok());
    assert!(output_path3.exists());
}
