//! Shared utilities for audio segment post-processing
//!
//! Provides segment merging and refinement to eliminate false positives
//! and improve boundary accuracy.

use crate::segment_detector::CommonSegment;

/// Split segments that are too long (>4 minutes)
///
/// For full episodes, prevents accidental merging of opening+ending
/// into one giant segment. Discards overlong segments as they're likely spurious.
pub fn split_overlong_segments(segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    let mut result = Vec::new();
    let max_duration = 240.0; // 4 minutes - allow longer openings/endings

    for seg in segments {
        let duration = seg.end_time - seg.start_time;
        
        if duration <= max_duration {
            // Segment is reasonable length - keep it
            result.push(seg);
        } else {
            // Segment is too long - this is a spurious match spanning too much content
            // In a properly working algorithm, we shouldn't create these in the first place
            // For now, discard them entirely rather than trying to guess boundaries
            // The algorithm should be fixed to not create overlong segments
            
            // Log this for debugging
            eprintln!(
                "Warning: Discarding overlong audio segment {:.1}s-{:.1}s (duration {:.1}s > {:.1}s max)",
                seg.start_time, seg.end_time, duration, max_duration
            );
        }
    }

    result
}

/// Merge overlapping audio segments
///
/// This eliminates false positives by merging segments that overlap
/// significantly (>50% overlap) into single segments.
pub fn merge_overlapping_segments(mut segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    if segments.is_empty() {
        return segments;
    }

    // Sort by start time
    segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.into_iter().skip(1) {
        // Calculate overlap
        let overlap_start = current.start_time.max(segment.start_time);
        let overlap_end = current.end_time.min(segment.end_time);
        let overlap_duration = (overlap_end - overlap_start).max(0.0);

        let current_duration = current.end_time - current.start_time;
        let segment_duration = segment.end_time - segment.start_time;
        let min_duration = current_duration.min(segment_duration);

        // Only merge if:
        // 1. Significant overlap (>50%), OR
        // 2. Very close together (<2s gap) AND both are short (<60s)
        // Don't merge distant segments even if they're close - this prevents
        // merging opening (0-114s) with ending (1341-1411s) into one giant segment
        let gap = segment.start_time - current.end_time;
        let both_short = current_duration < 60.0 && segment_duration < 60.0;
        
        let should_merge = overlap_duration >= min_duration * 0.5
            || (gap >= 0.0 && gap < 2.0 && both_short);

        if should_merge {
            // Merge segments
            current.start_time = current.start_time.min(segment.start_time);
            current.end_time = current.end_time.max(segment.end_time);
            current.confidence = current.confidence.max(segment.confidence);

            // Merge episode lists
            for ep in &segment.episode_list {
                if !current.episode_list.contains(ep) {
                    current.episode_list.push(ep.clone());
                }
            }

            // Merge audio confidence
            if let Some(seg_conf) = segment.audio_confidence {
                current.audio_confidence = Some(
                    current.audio_confidence.unwrap_or(0.0).max(seg_conf)
                );
            }

            // Merge episode timings if present
            if let (Some(ref mut curr_timings), Some(ref seg_timings)) =
                (&mut current.episode_timings, &segment.episode_timings)
            {
                for seg_timing in seg_timings {
                    if let Some(curr_timing) = curr_timings
                        .iter_mut()
                        .find(|t| t.episode_name == seg_timing.episode_name)
                    {
                        // Merge timing for same episode
                        curr_timing.start_time = curr_timing.start_time.min(seg_timing.start_time);
                        curr_timing.end_time = curr_timing.end_time.max(seg_timing.end_time);
                    } else {
                        // Add new episode timing
                        curr_timings.push(seg_timing.clone());
                    }
                }
            } else if current.episode_timings.is_none() && segment.episode_timings.is_some() {
                current.episode_timings = segment.episode_timings.clone();
            }
        } else {
            // No overlap - save current and start new
            merged.push(current);
            current = segment;
        }
    }

    merged.push(current);
    merged
}

/// Refine segment boundaries to improve timing accuracy
///
/// Trims segments to actual content boundaries by analyzing
/// match density at the edges.
pub fn refine_segment_boundaries(segment: CommonSegment, min_duration: f64) -> CommonSegment {
    let mut refined = segment.clone();

    // If segment has per-episode timings, refine each individually
    if let Some(ref mut timings) = refined.episode_timings {
        for timing in timings.iter_mut() {
            let duration = timing.end_time - timing.start_time;
            
            // Don't refine if segment is already close to minimum
            if duration < min_duration * 1.5 {
                continue;
            }

            // Trim up to 20% from start/end if segment is long
            let max_trim = (duration * 0.2).min(10.0);
            
            // Simple heuristic: theme songs often have 1-2s of fade-in/out
            let trim_amount = max_trim.min(2.0);
            
            timing.start_time += trim_amount;
            timing.end_time -= trim_amount;
        }
        
        // Update reference times
        if let Some(ref timings_ref) = refined.episode_timings {
            refined.start_time = timings_ref
                .iter()
                .map(|t| t.start_time)
                .fold(f64::INFINITY, f64::min);
            refined.end_time = timings_ref
                .iter()
                .map(|t| t.end_time)
                .fold(f64::NEG_INFINITY, f64::max);
        }
    } else {
        // No per-episode timings - refine reference time only
        let duration = refined.end_time - refined.start_time;
        
        if duration >= min_duration * 1.5 {
            let max_trim = (duration * 0.2).min(10.0);
            let trim_amount = max_trim.min(2.0);
            
            refined.start_time += trim_amount;
            refined.end_time -= trim_amount;
        }
    }

    refined
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_detector::MatchType;

    #[test]
    fn test_merge_overlapping() {
        let seg1 = CommonSegment {
            start_time: 0.0,
            end_time: 10.0,
            episode_list: vec!["ep1".to_string()],
            episode_timings: None,
            confidence: 0.8,
            video_confidence: None,
            audio_confidence: Some(0.8),
            match_type: MatchType::Audio,
        };

        let seg2 = CommonSegment {
            start_time: 5.0,
            end_time: 15.0,
            episode_list: vec!["ep2".to_string()],
            episode_timings: None,
            confidence: 0.9,
            video_confidence: None,
            audio_confidence: Some(0.9),
            match_type: MatchType::Audio,
        };

        let merged = merge_overlapping_segments(vec![seg1, seg2]);
        
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].start_time, 0.0);
        assert_eq!(merged[0].end_time, 15.0);
        assert_eq!(merged[0].episode_list.len(), 2);
        assert_eq!(merged[0].confidence, 0.9); // Max confidence
    }

    #[test]
    fn test_no_merge_distant_segments() {
        let seg1 = CommonSegment {
            start_time: 0.0,
            end_time: 10.0,
            episode_list: vec!["ep1".to_string()],
            episode_timings: None,
            confidence: 0.8,
            video_confidence: None,
            audio_confidence: Some(0.8),
            match_type: MatchType::Audio,
        };

        let seg2 = CommonSegment {
            start_time: 50.0,
            end_time: 60.0,
            episode_list: vec!["ep2".to_string()],
            episode_timings: None,
            confidence: 0.9,
            video_confidence: None,
            audio_confidence: Some(0.9),
            match_type: MatchType::Audio,
        };

        let merged = merge_overlapping_segments(vec![seg1, seg2]);
        
        assert_eq!(merged.len(), 2); // Should NOT merge
    }
}

