//! Video cutting via **ffmpeg** (stream copy / concat).
//!
//! Duration for planning cuts uses GStreamer (`analyzer::get_video_duration`) so timestamps stay
//! consistent with decode/extract paths elsewhere in TVT.

use crate::Result;
use anyhow::Context;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static CUT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Cut segments from a video file (removes the specified segments)
pub fn cut_video_segments(
    input_path: &Path,
    output_path: &Path,
    segments_to_remove: &[(f64, f64)],
) -> Result<()> {
    if segments_to_remove.is_empty() {
        // No segments to remove, just copy the file
        fs::copy(input_path, output_path).with_context(|| {
            format!(
                "Failed to copy file from {} to {}",
                input_path.display(),
                output_path.display()
            )
        })?;
        return Ok(());
    }

    // Create output directory if it doesn't exist
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    // Get video duration first (GStreamer — same clock as extraction)
    let duration = crate::analyzer::get_video_duration(input_path)?;

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

/// Convert non-overlapping “segments to keep” into “segments to remove” for [`cut_video_segments`].
pub fn segments_to_remove_from_keep(duration: f64, keep: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if duration <= 0.0 {
        return Vec::new();
    }
    if keep.is_empty() {
        return vec![(0.0, duration)];
    }

    let mut sorted: Vec<(f64, f64)> = keep
        .iter()
        .copied()
        .map(|(s, e)| (s.clamp(0.0, duration), e.clamp(0.0, duration)))
        .filter(|(s, e)| e > s)
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for seg in sorted {
        if let Some(last) = merged.last_mut() {
            if seg.0 <= last.1 {
                last.1 = last.1.max(seg.1);
            } else {
                merged.push(seg);
            }
        } else {
            merged.push(seg);
        }
    }

    let mut remove = Vec::new();
    let mut cursor = 0.0f64;
    for (s, e) in merged {
        if cursor < s {
            remove.push((cursor, s));
        }
        cursor = cursor.max(e);
    }
    if cursor < duration {
        remove.push((cursor, duration));
    }
    remove
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

/// Cut video using ffmpeg (more reliable than GStreamer for cutting)
fn cut_video_with_segments(
    input_path: &Path,
    output_path: &Path,
    segments_to_keep: &[(f64, f64)],
) -> Result<()> {
    // For multiple segments, we need to extract each segment and concatenate them
    // For now, implement single segment cutting (most common case)

    if segments_to_keep.len() == 1 {
        // Simple case: one continuous segment
        let (start, end) = segments_to_keep[0];

        // Use ffmpeg to cut the segment
        let status = Command::new("ffmpeg")
            .arg("-y") // Overwrite output
            .arg("-ss")
            .arg(format!("{:.3}", start))
            // `-t` after input-seeking (`-ss` before `-i`) can mis-trim some MP4 streams; `-to` is an
            // absolute stop time and matches the intended keep-range end.
            .arg("-to")
            .arg(format!("{:.3}", end))
            .arg("-i")
            .arg(input_path)
            .arg("-c")
            .arg("copy") // Stream copy (no re-encoding)
            .arg(output_path)
            .output()
            .with_context(|| "Failed to execute ffmpeg")?;

        if !status.status.success() {
            anyhow::bail!("ffmpeg failed: {}", String::from_utf8_lossy(&status.stderr));
        }

        Ok(())
    } else {
        // Multiple segments - need to concatenate
        cut_and_concatenate_segments(input_path, output_path, segments_to_keep)
    }
}

/// Cut and concatenate multiple segments
fn cut_and_concatenate_segments(
    input_path: &Path,
    output_path: &Path,
    segments_to_keep: &[(f64, f64)],
) -> Result<()> {
    use std::io::Write;

    // Create temporary directory for segment files (must be unique per call: parallel tests share
    // the same PID and would otherwise clobber each other's segment files).
    let temp_dir_path = std::env::temp_dir().join(format!(
        "tvt_cutting_{}_{}",
        std::process::id(),
        CUT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&temp_dir_path)?;
    let mut segment_files = Vec::new();

    // Extract each segment
    for (i, (start, end)) in segments_to_keep.iter().enumerate() {
        // Use `.mp4` plus `-reset_timestamps 1` so each excerpt has a correct container duration
        // under stream copy; `.mkv` excerpts often kept the source duration metadata and broke
        // concat duration for downstream consumers (and integration tests).
        let segment_path = temp_dir_path.join(format!("segment_{:03}.mp4", i));

        let status = Command::new("ffmpeg")
            .arg("-y")
            .arg("-ss")
            .arg(format!("{:.3}", start))
            .arg("-to")
            .arg(format!("{:.3}", end))
            .arg("-i")
            .arg(input_path)
            .arg("-c")
            .arg("copy")
            .arg("-reset_timestamps")
            .arg("1")
            .arg(&segment_path)
            .output()
            .with_context(|| "Failed to execute ffmpeg")?;

        if !status.status.success() {
            anyhow::bail!(
                "ffmpeg failed to extract segment: {}",
                String::from_utf8_lossy(&status.stderr)
            );
        }

        segment_files.push(segment_path);
    }

    // Create concat file list
    let concat_file = temp_dir_path.join("concat.txt");
    let mut file = fs::File::create(&concat_file)?;
    for segment_path in &segment_files {
        writeln!(file, "file '{}'", segment_path.display())?;
    }
    drop(file);

    // Concatenate segments
    let status = Command::new("ffmpeg")
        .arg("-y")
        .arg("-f")
        .arg("concat")
        .arg("-safe")
        .arg("0")
        .arg("-i")
        .arg(&concat_file)
        .arg("-c")
        .arg("copy")
        .arg(output_path)
        .output()
        .with_context(|| "Failed to execute ffmpeg for concatenation")?;

    if !status.status.success() {
        anyhow::bail!(
            "ffmpeg concatenation failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );
    }

    // Clean up temporary directory
    let _ = fs::remove_dir_all(&temp_dir_path);

    Ok(())
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
    use std::fs;
    use tempfile::tempdir;

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

    #[test]
    fn test_segments_to_remove_from_keep_round_trip() {
        let duration = 100.0;
        let remove = vec![(10.0, 20.0), (50.0, 60.0)];
        let keep = build_segments_to_keep(&remove, duration);
        let back = segments_to_remove_from_keep(duration, &keep);
        assert_eq!(back, remove);
    }
}
