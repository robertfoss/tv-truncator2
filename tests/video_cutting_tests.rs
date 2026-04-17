//! Integration tests for video cutting functionality

use std::path::Path;
use std::process::Command;
use tempfile::TempDir;
use tvt::video_processor::cut_video_segments;

fn ffprobe_duration_secs(path: &Path) -> Option<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

fn assert_duration_approx(path: &Path, expected: f64, tol: f64) {
    let Some(d) = ffprobe_duration_secs(path) else {
        panic!(
            "ffprobe failed or returned no duration for {}",
            path.display()
        );
    };
    assert!(
        (d - expected).abs() <= tol,
        "duration for {}: got {d} expected {expected} (±{tol})",
        path.display()
    );
}

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
        .arg("-pix_fmt")
        .arg("yuv420p")
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
    if Command::new("ffprobe").arg("-version").output().is_err() {
        println!("Skipping test: ffprobe not available");
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

    // Stream-copy + concat can mux slightly larger than the source; duration is authoritative.
    assert_duration_approx(&output_path, 8.0, 0.25);
}

#[test]
fn test_cut_video_segments_multiple_removals() {
    // Skip this test if FFmpeg is not available
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        println!("Skipping test: FFmpeg not available");
        return;
    }
    if Command::new("ffprobe").arg("-version").output().is_err() {
        println!("Skipping test: ffprobe not available");
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

    assert_duration_approx(&output_path, 11.0, 0.25);
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
    if Command::new("ffprobe").arg("-version").output().is_err() {
        println!("Skipping test: ffprobe not available");
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
    assert_duration_approx(&output_path, 8.0, 0.25);

    // Test removing from the end
    let output_path2 = temp_dir.path().join("output2.mp4");
    let segments_to_remove = vec![(8.0, 10.0)];
    let result = cut_video_segments(&input_path, &output_path2, &segments_to_remove);
    assert!(result.is_ok());
    assert!(output_path2.exists());
    assert_duration_approx(&output_path2, 8.0, 0.25);

    // Test removing from the middle
    let output_path3 = temp_dir.path().join("output3.mp4");
    let segments_to_remove = vec![(4.0, 6.0)];
    let result = cut_video_segments(&input_path, &output_path3, &segments_to_remove);
    assert!(result.is_ok());
    assert!(output_path3.exists());
    assert_duration_approx(&output_path3, 8.0, 0.25);
}
