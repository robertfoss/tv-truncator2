//! Historical name: cutting used to be prototyped with a GStreamer encode/mux pipeline.
//!
//! The stable implementation is **ffmpeg** stream copy in [`crate::video_processor`]. This module
//! keeps the old entry point as a thin wrapper so callers get deterministic cuts without the
//! abandoned sleep/seek experiment.

use crate::Result;
use std::path::Path;

/// Cut video by keeping the listed `(start, end)` segments (seconds).
///
/// Despite the name, this uses the same ffmpeg-backed path as [`crate::video_processor::cut_video_segments`]
/// after converting “keep” ranges into “remove” ranges using GStreamer-reported duration.
pub fn cut_video_segments_gstreamer(
    input_path: &Path,
    output_path: &Path,
    segments_to_keep: &[(f64, f64)],
) -> Result<()> {
    if segments_to_keep.is_empty() {
        return Err(anyhow::anyhow!("No segments to keep"));
    }

    let duration = crate::analyzer::get_video_duration(input_path)?;
    let segments_to_remove =
        crate::video_processor::segments_to_remove_from_keep(duration, segments_to_keep);
    crate::video_processor::cut_video_segments(input_path, output_path, &segments_to_remove)
}

/// Build segments to keep from segments to remove
///
/// This function takes a list of segments to remove and returns the segments to keep.
/// It handles the logic of inverting the removal list.
///
/// # Arguments
/// * `duration` - Total duration of the video in seconds
/// * `segments_to_remove` - Vector of (start_time, end_time) tuples for segments to remove
///
/// # Returns
/// * `Vec<(f64, f64)>` - Vector of segments to keep
pub fn build_segments_to_keep(duration: f64, segments_to_remove: &[(f64, f64)]) -> Vec<(f64, f64)> {
    if segments_to_remove.is_empty() {
        return vec![(0.0, duration)];
    }

    // Sort segments by start time
    let mut sorted_remove = segments_to_remove.to_vec();
    sorted_remove.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut segments_to_keep = Vec::new();
    let mut current_time = 0.0;

    for (start, end) in sorted_remove {
        // Add segment before this removal if there's a gap
        if current_time < start {
            segments_to_keep.push((current_time, start));
        }
        current_time = end;
    }

    // Add final segment if there's content after the last removal
    if current_time < duration {
        segments_to_keep.push((current_time, duration));
    }

    segments_to_keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_segments_to_keep_empty() {
        let segments = build_segments_to_keep(100.0, &[]);
        assert_eq!(segments, vec![(0.0, 100.0)]);
    }

    #[test]
    fn test_build_segments_to_keep_single_removal() {
        let segments = build_segments_to_keep(100.0, &[(20.0, 30.0)]);
        assert_eq!(segments, vec![(0.0, 20.0), (30.0, 100.0)]);
    }

    #[test]
    fn test_build_segments_to_keep_multiple_removals() {
        let segments = build_segments_to_keep(100.0, &[(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]);
        assert_eq!(
            segments,
            vec![(0.0, 10.0), (20.0, 30.0), (40.0, 50.0), (60.0, 100.0)]
        );
    }

    #[test]
    fn test_build_segments_to_keep_adjacent_removals() {
        let segments = build_segments_to_keep(100.0, &[(10.0, 20.0), (20.0, 30.0)]);
        assert_eq!(segments, vec![(0.0, 10.0), (30.0, 100.0)]);
    }

    #[test]
    fn test_build_segments_to_keep_beginning_removal() {
        let segments = build_segments_to_keep(100.0, &[(0.0, 20.0)]);
        assert_eq!(segments, vec![(20.0, 100.0)]);
    }

    #[test]
    fn test_build_segments_to_keep_ending_removal() {
        let segments = build_segments_to_keep(100.0, &[(80.0, 100.0)]);
        assert_eq!(segments, vec![(0.0, 80.0)]);
    }
}
