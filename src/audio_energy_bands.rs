//! Energy band pattern matching for audio segment detection
//!
//! Simple but effective approach using mel-frequency energy bands
//! to detect recurring patterns like theme songs and credits.

use crate::audio_extractor::EpisodeAudio;
use crate::audio_features::{extract_energy_bands, EnergyBands};
use crate::audio_segment_utils::{merge_overlapping_segments, split_overlong_segments};
use crate::segment_detector::{CommonSegment, EpisodeSegmentTiming, MatchType};
use crate::Result;
use std::collections::HashMap;

/// Window size for pattern matching (in frames)
/// Larger window for more robust theme song matching
const PATTERN_WINDOW_SIZE: usize = 60;

/// Step size for sliding window
const PATTERN_STEP_SIZE: usize = 5;

/// Correlation threshold for pattern matching
const CORRELATION_THRESHOLD: f64 = 0.65;

/// Detect audio segments using energy band patterns
///
/// This algorithm:
/// 1. Extracts energy in mel-frequency bands over time
/// 2. Uses pattern correlation to find matching regions
/// 3. Handles time shifts via exhaustive window comparison
/// 4. Simple and robust for theme songs/credits
pub fn detect_audio_segments_energy_bands(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [EnergyBands] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    // Step 1: Extract energy band features from all episodes
    let mut episode_features: Vec<Vec<EnergyBands>> = Vec::new();

    for (ep_id, episode) in episode_audio.iter().enumerate() {
        // Use raw audio samples from episode
        let features = extract_energy_bands(&episode.raw_samples, episode.sample_rate)?;

        if debug_dupes {
            println!(
                "  Episode {}: {} energy band frames",
                ep_id,
                features.len()
            );
        }

        episode_features.push(features);
    }

    // Step 2: Find matching patterns between all episode pairs
    let mut potential_matches: Vec<PatternMatch> = Vec::new();

    for i in 0..episode_features.len() {
        for j in (i + 1)..episode_features.len() {
            let matches = find_matching_patterns(&episode_features[i], &episode_features[j], i, j);
            potential_matches.extend(matches);
        }
    }

    if debug_dupes {
        println!(
            "  Found {} potential pattern matches",
            potential_matches.len()
        );
    }

    // Filter out over-long segments
    let max_segment_duration = 240.0; // Allow up to 4 minutes
    let filtered_matches: Vec<PatternMatch> = potential_matches
        .into_iter()
        .filter(|m| {
            let dur1 = m.end1 - m.start1;
            let dur2 = m.end2 - m.start2;
            dur1 <= max_segment_duration && dur2 <= max_segment_duration
        })
        .collect();

    if debug_dupes {
        println!(
            "  After duration filtering: {} matches",
            filtered_matches.len()
        );
    }

    // Step 3: Group matches into segments
    let segments = group_matches_into_segments(
        &filtered_matches,
        episode_audio,
        config.threshold,
        config.min_duration,
        debug_dupes,
    )?;

    if debug_dupes {
        println!(
            "🎵 [EnergyBands] Detected {} audio segments",
            segments.len()
        );
    }

    Ok(segments)
}

/// Represents a matching pattern between two episodes
#[derive(Debug, Clone)]
struct PatternMatch {
    episode1: usize,
    episode2: usize,
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    correlation: f64,
}

/// Find matching energy band patterns between two episodes
fn find_matching_patterns(
    features1: &[EnergyBands],
    features2: &[EnergyBands],
    ep1_id: usize,
    ep2_id: usize,
) -> Vec<PatternMatch> {
    let mut matches = Vec::new();

    if features1.len() < PATTERN_WINDOW_SIZE || features2.len() < PATTERN_WINDOW_SIZE {
        return matches;
    }

    // For long videos, only search beginning and end
    let max_time1 = features1.last().unwrap().timestamp;
    let search_regions = if max_time1 > 600.0 {
        // Energy bands: ~5 frames/second, so 2 min = ~600 frames
        let frames_per_2min = 600;
        vec![
            (0, features1.len().min(frames_per_2min)),
            (features1.len().saturating_sub(frames_per_2min), features1.len()),
        ]
    } else {
        vec![(0, features1.len())]
    };

    // Slide window across search regions
    for (region_start, region_end) in search_regions {
        for pos1 in (region_start..region_end.saturating_sub(PATTERN_WINDOW_SIZE)).step_by(PATTERN_STEP_SIZE) {
            let window1 = &features1[pos1..pos1 + PATTERN_WINDOW_SIZE];

            // Find best matching window in second episode
            let mut best_pos2 = 0;
            let mut best_correlation = 0.0;

            for pos2 in (0..features2.len() - PATTERN_WINDOW_SIZE).step_by(PATTERN_STEP_SIZE) {
                let window2 = &features2[pos2..pos2 + PATTERN_WINDOW_SIZE];

                let correlation = compute_pattern_correlation(window1, window2);

                if correlation > best_correlation {
                    best_correlation = correlation;
                    best_pos2 = pos2;
                }
            }

            // If correlation is good enough, add match
            if best_correlation >= CORRELATION_THRESHOLD {
                let start1 = window1.first().unwrap().timestamp;
                let end1 = window1.last().unwrap().timestamp;
                let window2 = &features2[best_pos2..best_pos2 + PATTERN_WINDOW_SIZE];
                let start2 = window2.first().unwrap().timestamp;
                let end2 = window2.last().unwrap().timestamp;

                matches.push(PatternMatch {
                    episode1: ep1_id,
                    episode2: ep2_id,
                    start1,
                    end1,
                    start2,
                    end2,
                    correlation: best_correlation,
                });
            }
        }
    }

    matches
}

/// Compute correlation between two energy band patterns
fn compute_pattern_correlation(pattern1: &[EnergyBands], pattern2: &[EnergyBands]) -> f64 {
    let mut total_correlation = 0.0;
    let num_frames = pattern1.len().min(pattern2.len());

    for i in 0..num_frames {
        let correlation = compute_frame_correlation(&pattern1[i], &pattern2[i]);
        total_correlation += correlation;
    }

    total_correlation / num_frames as f64
}

/// Compute correlation between two energy band frames
fn compute_frame_correlation(frame1: &EnergyBands, frame2: &EnergyBands) -> f64 {
    let num_bands = frame1.bands.len().min(frame2.bands.len());

    if num_bands == 0 {
        return 0.0;
    }

    // Normalize bands
    let norm1: f32 = frame1.bands.iter().map(|b| b * b).sum::<f32>().sqrt();
    let norm2: f32 = frame2.bands.iter().map(|b| b * b).sum::<f32>().sqrt();

    if norm1 < 1e-6 || norm2 < 1e-6 {
        return 0.0;
    }

    // Compute normalized dot product (cosine similarity)
    let mut dot_product = 0.0f32;
    for i in 0..num_bands {
        dot_product += frame1.bands[i] * frame2.bands[i];
    }

    let correlation = (dot_product / (norm1 * norm2)) as f64;

    // Also factor in RMS energy similarity
    let energy_ratio = if frame1.rms_energy > frame2.rms_energy {
        frame2.rms_energy / frame1.rms_energy.max(1e-6)
    } else {
        frame1.rms_energy / frame2.rms_energy.max(1e-6)
    };

    // Combine correlation and energy similarity
    correlation * 0.8 + energy_ratio as f64 * 0.2
}

/// Group matches into common segments
fn group_matches_into_segments(
    matches: &[PatternMatch],
    episode_audio: &[EpisodeAudio],
    threshold: usize,
    min_duration: f64,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if matches.is_empty() {
        return Ok(Vec::new());
    }

    // Group matches by approximate time
    // Use strict grouping: both start AND end must be close
    let mut segment_groups: Vec<Vec<&PatternMatch>> = Vec::new();

    for m in matches {
        let mut found_group = false;

        for group in &mut segment_groups {
            if let Some(first) = group.first() {
                // Must overlap significantly - check both start and end
                let start_close = (m.start1 - first.start1).abs() < 30.0;
                let end_close = (m.end1 - first.end1).abs() < 30.0;

                if start_close && end_close {
                    group.push(m);
                    found_group = true;
                    break;
                }
            }
        }

        if !found_group {
            segment_groups.push(vec![m]);
        }
    }

    if debug_dupes {
        println!("  Grouped into {} segment candidates", segment_groups.len());
    }

    // Convert groups to CommonSegments
    let mut common_segments = Vec::new();

    for group in segment_groups {
        // Build per-episode timing
        let mut episode_timings_map: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

        for m in &group {
            episode_timings_map
                .entry(m.episode1)
                .or_insert_with(Vec::new)
                .push((m.start1, m.end1));

            episode_timings_map
                .entry(m.episode2)
                .or_insert_with(Vec::new)
                .push((m.start2, m.end2));
        }

        if episode_timings_map.len() < threshold {
            continue;
        }

        // Merge overlapping ranges for each episode
        let mut episode_timings = Vec::new();
        for (ep_id, mut ranges) in episode_timings_map {
            ranges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Merge overlapping
            let mut merged = vec![ranges[0]];
            for &(start, end) in &ranges[1..] {
                let last = merged.last_mut().unwrap();
                if start <= last.1 + 2.0 {
                    last.1 = last.1.max(end);
                } else {
                    merged.push((start, end));
                }
            }

            // Use largest range
            let (start, end) = merged
                .iter()
                .max_by_key(|(s, e)| ((e - s) * 1000.0) as i64)
                .unwrap();

            if end - start >= min_duration {
                let ep_name = episode_audio[ep_id]
                    .episode_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                episode_timings.push(EpisodeSegmentTiming {
                    episode_name: ep_name,
                    start_time: *start,
                    end_time: *end,
                });
            }
        }

        if episode_timings.len() < threshold {
            continue;
        }

        // Calculate confidence
        let avg_correlation =
            group.iter().map(|m| m.correlation).sum::<f64>() / group.len() as f64;

        let time_shifted = is_time_shifted(&episode_timings);

        let ref_start = episode_timings
            .iter()
            .map(|t| t.start_time)
            .fold(f64::INFINITY, f64::min);
        let ref_end = episode_timings
            .iter()
            .map(|t| t.end_time)
            .fold(f64::NEG_INFINITY, f64::max);

        let episode_names = episode_timings
            .iter()
            .map(|t| t.episode_name.clone())
            .collect();

        common_segments.push(CommonSegment {
            start_time: ref_start,
            end_time: ref_end,
            episode_list: episode_names,
            episode_timings: if time_shifted {
                Some(episode_timings)
            } else {
                None
            },
            confidence: avg_correlation,
            video_confidence: None,
            audio_confidence: Some(avg_correlation),
            match_type: MatchType::Audio,
        });
    }

    // Split overlong segments first
    let split = split_overlong_segments(common_segments);
    
    // Then merge overlapping segments to eliminate false positives
    let merged = merge_overlapping_segments(split);

    Ok(merged)
}

/// Check if segment is time-shifted
fn is_time_shifted(timings: &[EpisodeSegmentTiming]) -> bool {
    if timings.len() < 2 {
        return false;
    }

    let first_start = timings[0].start_time;
    for timing in &timings[1..] {
        if (timing.start_time - first_start).abs() > 2.0 {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_correlation() {
        let frame1 = EnergyBands {
            bands: vec![1.0, 0.5, 0.3, 0.1],
            rms_energy: 1.0,
            flux: 0.1,
            timestamp: 0.0,
        };

        let frame2 = EnergyBands {
            bands: vec![1.0, 0.5, 0.3, 0.1],
            rms_energy: 1.0,
            flux: 0.1,
            timestamp: 0.0,
        };

        let correlation = compute_frame_correlation(&frame1, &frame2);
        assert!(correlation > 0.95);
    }
}

