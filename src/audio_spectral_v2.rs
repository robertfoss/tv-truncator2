//! Improved spectral hash matching with proper quantization
//!
//! Fixes the ultra-coarse hashing in the original implementation by
//! using meaningful 64-bit hashes with multiple frequency bands.

use crate::audio_extractor::EpisodeAudio;
use crate::audio_features::{extract_spectral_hash_v2, SpectralHashV2};
use crate::audio_segment_utils::{merge_overlapping_segments, split_overlong_segments};
use crate::hasher::{hamming_distance, RollingHash};
use crate::segment_detector::{CommonSegment, EpisodeSegmentTiming, MatchType};
use crate::Result;
use std::collections::HashMap;

/// Window size for rolling hash (in frames)
/// Larger window = more robust matching for theme songs/credits
const ROLLING_WINDOW_SIZE: usize = 60;

/// Similarity threshold (as fraction, 0.0-1.0)
const SIMILARITY_THRESHOLD: f64 = 0.70;

/// Minimum segment duration (seconds)
#[allow(dead_code)]
const MIN_SEGMENT_DURATION: f64 = 5.0;

/// Detect audio segments using improved spectral hashing
///
/// This algorithm:
/// 1. Extracts 64-bit spectral hashes with proper quantization
/// 2. Uses rolling hash over windows for robustness
/// 3. Compares all window pairs to handle time shifts
/// 4. Groups similar sequences into segments
pub fn detect_audio_segments_spectral_v2(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [SpectralV2] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    // Step 1: Extract spectral hashes from all episodes
    let mut episode_hashes: Vec<Vec<SpectralHashV2>> = Vec::new();

    for (ep_id, episode) in episode_audio.iter().enumerate() {
        // Use raw audio samples from episode
        let hashes = extract_spectral_hash_v2(&episode.raw_samples, episode.sample_rate)?;

        if debug_dupes {
            println!("  Episode {}: {} hash frames", ep_id, hashes.len());
        }

        episode_hashes.push(hashes);
    }

    // Step 2: Generate rolling hashes for each episode
    let mut episode_rolling_hashes: Vec<Vec<(u64, f64, f64)>> = Vec::new();

    for hashes in &episode_hashes {
        let mut rolling = RollingHash::new(ROLLING_WINDOW_SIZE);
        let mut rolling_hashes = Vec::new();

        for (i, hash_frame) in hashes.iter().enumerate() {
            if let Some(roll_hash) = rolling.add(hash_frame.hash) {
                let start_idx = i.saturating_sub(ROLLING_WINDOW_SIZE - 1);
                let start_time = hashes[start_idx].timestamp;
                let end_time = hash_frame.timestamp;

                rolling_hashes.push((roll_hash, start_time, end_time));
            }
        }

        episode_rolling_hashes.push(rolling_hashes);
    }

    // Step 3: Find matching sequences using exhaustive comparison
    let mut potential_matches: Vec<SegmentMatch> = Vec::new();

    for i in 0..episode_rolling_hashes.len() {
        for j in (i + 1)..episode_rolling_hashes.len() {
            let matches = find_matching_sequences(
                &episode_rolling_hashes[i],
                &episode_rolling_hashes[j],
                i,
                j,
            );
            potential_matches.extend(matches);
        }
    }

    if debug_dupes {
        println!(
            "  Found {} potential sequence matches",
            potential_matches.len()
        );
    }

    // Filter out over-long segments before grouping
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
            "  After duration filtering (<{}s): {} matches",
            max_segment_duration,
            filtered_matches.len()
        );
    }

    // Step 4: Group matches into segments
    let segments = group_matches_into_segments(
        &filtered_matches,
        episode_audio,
        config.threshold,
        config.min_duration,
        debug_dupes,
    )?;

    if debug_dupes {
        println!("🎵 [SpectralV2] Detected {} audio segments", segments.len());
    }

    Ok(segments)
}

/// Represents a matching sequence between two episodes
#[derive(Debug, Clone)]
struct SegmentMatch {
    episode1: usize,
    episode2: usize,
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    similarity: f64,
}

/// Find matching sequences between two episodes
fn find_matching_sequences(
    hashes1: &[(u64, f64, f64)],
    hashes2: &[(u64, f64, f64)],
    ep1_id: usize,
    ep2_id: usize,
) -> Vec<SegmentMatch> {
    let mut matches = Vec::new();

    // For long videos, only search beginning and end regions
    let max_time1 = hashes1.last().map(|(_, _, t)| *t).unwrap_or(0.0);
    let search_indices: Vec<usize> = if max_time1 > 600.0 {
        // Long video - search first 2 min and last 2 min
        // At ~4 fps hash rate, this is ~480 frames per 2 min
        let frames_per_2min = 480;
        let beginning: Vec<usize> = (0..hashes1.len().min(frames_per_2min)).collect();
        let ending: Vec<usize> = (hashes1.len().saturating_sub(frames_per_2min)..hashes1.len()).collect();
        beginning.into_iter().chain(ending).collect()
    } else {
        // Short video - use all positions
        (0..hashes1.len()).collect()
    };

    // For each sampled position in episode 1, find best match in episode 2
    for &i in &search_indices {
        let (hash1, start1, end1) = &hashes1[i];
        let mut best_similarity = 0.0;
        let mut best_match = None;

        for (hash2, start2, end2) in hashes2 {
            let hamming_dist = hamming_distance(*hash1, *hash2);
            let similarity = 1.0 - (hamming_dist as f64 / 64.0);

            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some((*start2, *end2));
            }
        }

        if best_similarity >= SIMILARITY_THRESHOLD {
            if let Some((start2, end2)) = best_match {
                matches.push(SegmentMatch {
                    episode1: ep1_id,
                    episode2: ep2_id,
                    start1: *start1,
                    end1: *end1,
                    start2,
                    end2,
                    similarity: best_similarity,
                });
            }
        }
    }

    matches
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
                // Must overlap significantly - check both start and end times
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

            let ep_name = episode_audio[ep_id]
                .episode_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            if end - start >= min_duration {
                episode_timings.push(EpisodeSegmentTiming {
                    episode_name: ep_name,
                    start_time: *start,
                    end_time: *end,
                });
            } else if debug_dupes {
                println!(
                    "  Rejected {}: duration {:.1}s < {:.1}s",
                    ep_name,
                    end - start,
                    min_duration
                );
            }
        }

        if episode_timings.len() < threshold {
            if debug_dupes {
                println!(
                    "  Rejected group: only {} episodes (need {})",
                    episode_timings.len(), threshold
                );
            }
            continue;
        }

        // Calculate confidence
        let avg_similarity =
            group.iter().map(|m| m.similarity).sum::<f64>() / group.len() as f64;

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
            confidence: avg_similarity,
            video_confidence: None,
            audio_confidence: Some(avg_similarity),
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
    fn test_is_time_shifted() {
        let timings = vec![
            EpisodeSegmentTiming {
                episode_name: "ep1".to_string(),
                start_time: 10.0,
                end_time: 20.0,
            },
            EpisodeSegmentTiming {
                episode_name: "ep2".to_string(),
                start_time: 10.5,
                end_time: 20.5,
            },
        ];
        assert!(!is_time_shifted(&timings));

        let timings_shifted = vec![
            EpisodeSegmentTiming {
                episode_name: "ep1".to_string(),
                start_time: 10.0,
                end_time: 20.0,
            },
            EpisodeSegmentTiming {
                episode_name: "ep2".to_string(),
                start_time: 15.0,
                end_time: 25.0,
            },
        ];
        assert!(is_time_shifted(&timings_shifted));
    }
}

