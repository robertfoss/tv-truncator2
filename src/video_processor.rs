//! Video processing and cutting operations

use crate::Result;
use std::path::Path;
use std::process::Command;
use std::fs;
use anyhow::Context;
use crate::gstreamer_cutter;

/// Cut segments from a video file (removes the specified segments)
pub fn cut_video_segments(
    input_path: &Path,
    output_path: &Path,
    segments_to_remove: &[(f64, f64)],
) -> Result<()> {
    if segments_to_remove.is_empty() {
        // No segments to remove, just copy the file
        fs::copy(input_path, output_path)
            .with_context(|| format!("Failed to copy file from {} to {}", input_path.display(), output_path.display()))?;
        return Ok(());
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    // Get video duration first
    let duration = get_video_duration(input_path)?;
    
    // Sort segments by start time
    let mut sorted_segments = segments_to_remove.to_vec();
    sorted_segments.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Build segments to keep (inverse of segments to remove)
    let segments_to_keep = build_segments_to_keep(&sorted_segments, duration);
    
    if segments_to_keep.is_empty() {
        // All content is being removed, create empty output
        return create_empty_video(output_path);
    }

    // Use a simpler approach: create multiple input files and concatenate
    cut_video_with_segments(input_path, output_path, &segments_to_keep)
}

/// Get video duration using ffprobe
fn get_video_duration(input_path: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("quiet")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("csv=p=0")
        .arg(input_path)
        .output()
        .with_context(|| "Failed to execute ffprobe")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe failed: {}", String::from_utf8_lossy(&output.stderr));
    }

    let duration_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
    duration_str.parse::<f64>()
        .with_context(|| format!("Failed to parse duration: {}", duration_str))
}

/// Build segments to keep (inverse of segments to remove)
fn build_segments_to_keep(segments_to_remove: &[(f64, f64)], duration: f64) -> Vec<(f64, f64)> {
    let mut segments_to_keep = Vec::new();
    let mut last_end = 0.0;
    
    for (start, end) in segments_to_remove {
        if last_end < *start {
            // Add segment from last_end to start
            segments_to_keep.push((last_end, *start));
        }
        last_end = *end;
    }
    
    // Add final segment if there's content after the last removal
    if last_end < duration {
        segments_to_keep.push((last_end, duration));
    }
    
    segments_to_keep
}

/// Cut video using GStreamer
fn cut_video_with_segments(
    input_path: &Path,
    output_path: &Path,
    segments_to_keep: &[(f64, f64)],
) -> Result<()> {
    // Use GStreamer for video cutting
    gstreamer_cutter::cut_video_segments_gstreamer(input_path, output_path, segments_to_keep)
}

/// Create an empty video file using GStreamer
fn create_empty_video(output_path: &Path) -> Result<()> {
    // For now, create a minimal empty file
    // In a full implementation, we would use GStreamer to create a proper empty video
    fs::write(output_path, b"")?;
    Ok(())
}


/// Verify that all streams are properly synchronized after cutting
pub fn verify_stream_synchronization(_video_path: &Path) -> Result<bool> {
    // TODO: Implement stream synchronization verification
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use std::fs;

    #[test]
    fn test_cut_video_segments_empty() {
        let temp_dir = tempdir().unwrap();
        let input_path = temp_dir.path().join("input.mkv");
        let output_path = temp_dir.path().join("output.mkv");
        
        // Create a dummy input file
        fs::write(&input_path, "dummy video content").unwrap();
        
        let result = cut_video_segments(&input_path, &output_path, &[]);
        assert!(result.is_ok());
        
        // Should have copied the file
        assert!(output_path.exists());
    }

    #[test]
    fn test_build_segments_to_keep() {
        let segments_to_remove = vec![(10.0, 20.0), (30.0, 40.0)];
        let duration = 60.0;
        
        let segments_to_keep = build_segments_to_keep(&segments_to_remove, duration);
        
        assert_eq!(segments_to_keep.len(), 3);
        assert_eq!(segments_to_keep[0], (0.0, 10.0));
        assert_eq!(segments_to_keep[1], (20.0, 30.0));
        assert_eq!(segments_to_keep[2], (40.0, 60.0));
    }

    #[test]
    fn test_build_segments_to_keep_no_gaps() {
        let segments_to_remove = vec![(0.0, 10.0), (10.0, 20.0)];
        let duration = 30.0;
        
        let segments_to_keep = build_segments_to_keep(&segments_to_remove, duration);
        
        assert_eq!(segments_to_keep.len(), 1);
        assert_eq!(segments_to_keep[0], (20.0, 30.0));
    }

}
