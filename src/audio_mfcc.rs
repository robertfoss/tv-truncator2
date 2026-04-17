//! MFCC-based audio matching with time-shift detection
//!
//! Uses Mel-Frequency Cepstral Coefficients for robust audio characterization
//! and sliding window comparison for time-shift capable segment detection.

use crate::audio_extractor::EpisodeAudio;
use crate::audio_features::{extract_mfcc_features, MfccFeatures};
use crate::audio_segment_utils::{merge_overlapping_segments, split_overlong_segments};
use crate::segment_detector::{CommonSegment, EpisodeSegmentTiming, MatchType};
use crate::Result;
use std::collections::HashMap;

/// Window size for MFCC comparison (in frames)
/// Larger window = more robust matching for theme songs
const MFCC_WINDOW_SIZE: usize = 60;

/// Step size for sliding window (in frames)
/// Use larger steps for efficiency - don't need to check every position
const MFCC_STEP_SIZE: usize = 15;

/// Distance threshold for MFCCs to be considered matching
const MFCC_DISTANCE_THRESHOLD: f32 = 0.35;

/// Minimum matching window duration (in seconds)
#[allow(dead_code)]
const MIN_MATCH_DURATION: f64 = 5.0;

/// Detect audio segments using MFCC features
///
/// This algorithm:
/// 1. Extracts MFCC features from all episodes
/// 2. Uses sliding window comparison to find matching regions
/// 3. Handles time-shifted segments by comparing all window positions
/// 4. Clusters matches into contiguous segments
pub fn detect_audio_segments_mfcc(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [MFCC] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    // Step 1: Extract MFCC features from all episodes
    // For efficiency, only extract from regions we'll search (first/last 2 min for long videos)
    // We use a two-pass approach: process beginning and ending regions separately
    let mut episode_start_mfccs: Vec<Vec<MfccFeatures>> = Vec::new();
    let mut episode_end_mfccs: Vec<Vec<MfccFeatures>> = Vec::new();

    for (ep_id, episode) in episode_audio.iter().enumerate() {
        let duration = episode.raw_samples.len() as f64 / episode.sample_rate as f64;

        // For long videos (>10 min), extract MFCC from first 2 min and last 2 min separately
        if duration > 600.0 {
            let samples_per_2min = (episode.sample_rate * 120.0) as usize;

            // Extract from beginning (first 2 min)
            let start_samples = episode.raw_samples.len().min(samples_per_2min);
            let start_mfccs =
                extract_mfcc_features(&episode.raw_samples[..start_samples], episode.sample_rate)?;

            // Extract from end (last 2 min)
            let end_offset = episode.raw_samples.len().saturating_sub(samples_per_2min);
            let mut end_mfccs =
                extract_mfcc_features(&episode.raw_samples[end_offset..], episode.sample_rate)?;

            // Adjust timestamps for end segment to reflect actual position in video
            let time_offset = end_offset as f64 / episode.sample_rate as f64;
            for mfcc in &mut end_mfccs {
                mfcc.timestamp += time_offset;
            }

            if debug_dupes {
                println!(
                    "  Episode {}: {} start frames, {} end frames (duration: {:.1}s)",
                    ep_id,
                    start_mfccs.len(),
                    end_mfccs.len(),
                    duration
                );
            }

            episode_start_mfccs.push(start_mfccs);
            episode_end_mfccs.push(end_mfccs);
        } else {
            // Short video - extract from entire file, use for both start and end
            let mfccs = extract_mfcc_features(&episode.raw_samples, episode.sample_rate)?;

            if debug_dupes {
                println!(
                    "  Episode {}: {} MFCC frames (duration: {:.1}s)",
                    ep_id,
                    mfccs.len(),
                    duration
                );
            }

            episode_start_mfccs.push(mfccs.clone());
            episode_end_mfccs.push(mfccs);
        }
    }

    // Step 2: Compare all episode pairs using sliding windows
    // Process beginning and ending regions separately to avoid cross-boundary windows
    let mut potential_matches: Vec<SegmentMatch> = Vec::new();

    // Pass 1: Find matches in beginning regions (intros)
    for i in 0..episode_start_mfccs.len() {
        for j in (i + 1)..episode_start_mfccs.len() {
            let matches =
                find_matching_windows(&episode_start_mfccs[i], &episode_start_mfccs[j], i, j);
            potential_matches.extend(matches);
        }
    }

    // Pass 2: Find matches in ending regions (outros)
    for i in 0..episode_end_mfccs.len() {
        for j in (i + 1)..episode_end_mfccs.len() {
            let matches = find_matching_windows(&episode_end_mfccs[i], &episode_end_mfccs[j], i, j);
            potential_matches.extend(matches);
        }
    }

    if debug_dupes {
        println!(
            "  Found {} potential window matches",
            potential_matches.len()
        );
    }

    // Filter out over-long segments
    let max_segment_duration = 240.0; // Allow up to 4 minutes
    let filtered_matches: Vec<SegmentMatch> = potential_matches
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

    // Step 3: Group matches by episodes and time regions
    // Note: We don't need the MFCC data for grouping, only for extraction
    let segments = group_matches_into_segments(
        &filtered_matches,
        episode_audio,
        config.threshold,
        config.min_duration,
        debug_dupes,
    )?;

    if debug_dupes {
        println!("🎵 [MFCC] Detected {} audio segments", segments.len());
    }

    Ok(segments)
}

/// Represents a matching window between two episodes
#[derive(Debug, Clone)]
struct SegmentMatch {
    episode1: usize,
    episode2: usize,
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    distance: f32,
}

/// Find matching windows between two episode MFCC sequences
fn find_matching_windows(
    mfccs1: &[MfccFeatures],
    mfccs2: &[MfccFeatures],
    ep1_id: usize,
    ep2_id: usize,
) -> Vec<SegmentMatch> {
    let mut matches = Vec::new();

    if mfccs1.len() < MFCC_WINDOW_SIZE || mfccs2.len() < MFCC_WINDOW_SIZE {
        return matches;
    }

    // Note: For long videos, we've already extracted only beginning/end during feature extraction,
    // so we search the entire extracted region here.

    // Slide window across first episode
    for pos1 in (0..mfccs1.len().saturating_sub(MFCC_WINDOW_SIZE)).step_by(MFCC_STEP_SIZE) {
        let window1 = &mfccs1[pos1..pos1 + MFCC_WINDOW_SIZE];

        // Find best match in second episode (search whole extracted region)
        let mut best_pos2 = 0;
        let mut best_distance = f32::MAX;

        for pos2 in (0..mfccs2.len() - MFCC_WINDOW_SIZE).step_by(MFCC_STEP_SIZE) {
            let window2 = &mfccs2[pos2..pos2 + MFCC_WINDOW_SIZE];

            let distance = compute_window_distance(window1, window2);

            if distance < best_distance {
                best_distance = distance;
                best_pos2 = pos2;
            }
        }

        // If match is good enough, add it
        if best_distance < MFCC_DISTANCE_THRESHOLD {
            let start1 = window1.first().unwrap().timestamp;
            let end1 = window1.last().unwrap().timestamp;
            let window2 = &mfccs2[best_pos2..best_pos2 + MFCC_WINDOW_SIZE];
            let start2 = window2.first().unwrap().timestamp;
            let end2 = window2.last().unwrap().timestamp;

            matches.push(SegmentMatch {
                episode1: ep1_id,
                episode2: ep2_id,
                start1,
                end1,
                start2,
                end2,
                distance: best_distance,
            });
        }
    }

    matches
}

/// Compute distance between two MFCC windows
fn compute_window_distance(window1: &[MfccFeatures], window2: &[MfccFeatures]) -> f32 {
    let mut total_distance = 0.0f32;
    let num_frames = window1.len().min(window2.len());

    for i in 0..num_frames {
        let dist = compute_mfcc_distance(&window1[i], &window2[i]);
        total_distance += dist;
    }

    total_distance / num_frames as f32
}

/// Compute Euclidean distance between two MFCC feature vectors
fn compute_mfcc_distance(mfcc1: &MfccFeatures, mfcc2: &MfccFeatures) -> f32 {
    let mut sum_sq = 0.0f32;

    // Distance based on coefficients
    for i in 0..mfcc1.coefficients.len().min(mfcc2.coefficients.len()) {
        let diff = mfcc1.coefficients[i] - mfcc2.coefficients[i];
        sum_sq += diff * diff;
    }

    // Add delta contribution (weighted lower)
    for i in 0..mfcc1.deltas.len().min(mfcc2.deltas.len()) {
        let diff = mfcc1.deltas[i] - mfcc2.deltas[i];
        sum_sq += diff * diff * 0.5;
    }

    // Add delta-delta contribution (weighted even lower)
    for i in 0..mfcc1.delta_deltas.len().min(mfcc2.delta_deltas.len()) {
        let diff = mfcc1.delta_deltas[i] - mfcc2.delta_deltas[i];
        sum_sq += diff * diff * 0.25;
    }

    sum_sq.sqrt()
}

/// Group matches into common segments
fn group_matches_into_segments(
    matches: &[SegmentMatch],
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
    let mut segment_groups: Vec<Vec<&SegmentMatch>> = Vec::new();

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

        // Check if enough episodes participate
        if episode_timings_map.len() < threshold {
            continue;
        }

        // Merge overlapping ranges for each episode
        let mut episode_timings = Vec::new();
        for (ep_id, mut ranges) in episode_timings_map {
            ranges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Merge overlapping ranges
            let mut merged = vec![ranges[0]];
            for &(start, end) in &ranges[1..] {
                let last = merged.last_mut().unwrap();
                if start <= last.1 + 2.0 {
                    last.1 = last.1.max(end);
                } else {
                    merged.push((start, end));
                }
            }

            // Use the largest merged range
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

        // Calculate confidence from average distance
        let avg_distance = group.iter().map(|m| m.distance).sum::<f32>() / group.len() as f32;
        let confidence = (1.0 - avg_distance.min(1.0)) as f64;

        // Check if time-shifted
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
            confidence,
            video_confidence: None,
            audio_confidence: Some(confidence),
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
    fn test_mfcc_distance() {
        let mfcc1 = MfccFeatures {
            coefficients: vec![1.0, 2.0, 3.0],
            deltas: vec![0.1, 0.2, 0.3],
            delta_deltas: vec![0.01, 0.02, 0.03],
            timestamp: 0.0,
        };

        let mfcc2 = MfccFeatures {
            coefficients: vec![1.0, 2.0, 3.0],
            deltas: vec![0.1, 0.2, 0.3],
            delta_deltas: vec![0.01, 0.02, 0.03],
            timestamp: 0.0,
        };

        let distance = compute_mfcc_distance(&mfcc1, &mfcc2);
        assert!(distance < 0.001);
    }
}
