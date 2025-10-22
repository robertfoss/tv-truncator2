//! Chromaprint-style audio fingerprinting for segment matching
//!
//! Uses landmark-based fingerprinting with sequence matching for robust
//! detection of recurring segments at any position (time-shift capable).

use crate::audio_extractor::EpisodeAudio;
use crate::audio_features::{extract_chromaprint_landmarks, Landmark};
use crate::audio_segment_utils::{merge_overlapping_segments as merge_segments_util, split_overlong_segments};
use crate::segment_detector::{CommonSegment, EpisodeSegmentTiming, MatchType};
use crate::Result;
use std::collections::HashMap;

/// Minimum cluster size (number of matching landmarks in sequence)
const MIN_SEQUENCE_LENGTH: usize = 3;

/// Minimum segment duration (in seconds)
const MIN_SEGMENT_DURATION: f64 = 5.0;

/// Maximum segment duration (in seconds) - prevents matching entire episodes
/// Allow up to 4 minutes for longer openings/endings
const MAX_SEGMENT_DURATION: f64 = 240.0;

/// Maximum time gap between consecutive landmarks in a sequence (seconds)
const MAX_LANDMARK_GAP: f64 = 3.0;

/// Detect audio segments using chromaprint-style fingerprinting
///
/// This algorithm:
/// 1. Extracts landmark fingerprints from all episodes
/// 2. Finds matching sequences of landmarks between episode pairs
/// 3. Groups pairwise matches into multi-episode segments
/// 4. Handles time-shifted segments naturally
pub fn detect_audio_segments_chromaprint(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [Chromaprint] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    // Step 1: Extract landmarks from all episodes
    let mut episode_landmarks: Vec<Vec<Landmark>> = Vec::new();

    for (ep_id, episode) in episode_audio.iter().enumerate() {
        // For long videos (>10 min), only extract landmarks from beginning and end
        // This focuses on openings/endings and avoids matching random middle content
        let duration = if !episode.raw_samples.is_empty() {
            episode.raw_samples.len() as f64 / episode.sample_rate as f64
        } else {
            0.0
        };

        let landmarks = if duration > 600.0 {
            // Long video - extract from first 2 min and last 2 min
            // Provides enough context to find 90s openings/endings
            let samples_per_2min = (episode.sample_rate * 120.0) as usize;
            
            let beginning_samples = &episode.raw_samples[..samples_per_2min.min(episode.raw_samples.len())];
            let ending_start = episode.raw_samples.len().saturating_sub(samples_per_2min);
            let ending_samples = &episode.raw_samples[ending_start..];
            
            // Extract from beginning
            let mut landmarks = extract_chromaprint_landmarks(beginning_samples, episode.sample_rate)?;
            
            // Extract from ending and adjust timestamps
            // IMPORTANT: Keep these separate by adding them with large timestamp offset
            let mut ending_landmarks = extract_chromaprint_landmarks(ending_samples, episode.sample_rate)?;
            let time_offset = ending_start as f64 / episode.sample_rate as f64;
            for landmark in &mut ending_landmarks {
                landmark.timestamp += time_offset;
            }
            landmarks.extend(ending_landmarks);
            
            landmarks
        } else {
            // Short video - extract from entire video
            extract_chromaprint_landmarks(&episode.raw_samples, episode.sample_rate)?
        };

        if debug_dupes {
            println!("  Episode {}: {} landmarks extracted (duration={:.1}s)", 
                     ep_id, landmarks.len(), duration);
        }

        episode_landmarks.push(landmarks);
    }

    // Step 2: Find matching sequences between all episode pairs
    let mut all_matches: Vec<SequenceMatch> = Vec::new();

    for i in 0..episode_landmarks.len() {
        for j in (i + 1)..episode_landmarks.len() {
            let matches = find_matching_sequences(
                &episode_landmarks[i],
                &episode_landmarks[j],
                i,
                j,
            );
            
            if debug_dupes && !matches.is_empty() {
                println!(
                    "  Episodes {} vs {}: {} matching sequences",
                    i, j, matches.len()
                );
            }
            
            all_matches.extend(matches);
        }
    }

    if debug_dupes {
        println!("  Total sequence matches found: {}", all_matches.len());
    }

    // Filter out spurious long sequences before grouping
    let filtered_matches: Vec<SequenceMatch> = all_matches
        .into_iter()
        .filter(|m| {
            let dur1 = m.end1 - m.start1;
            let dur2 = m.end2 - m.start2;
            // Keep only reasonable theme song/credits durations
            let duration_ok = dur1 >= MIN_SEGMENT_DURATION 
                && dur2 >= MIN_SEGMENT_DURATION
                && dur1 <= MAX_SEGMENT_DURATION 
                && dur2 <= MAX_SEGMENT_DURATION;
            let similar_duration = (dur1 - dur2).abs() < 30.0;
            duration_ok && similar_duration
        })
        .collect();

    if debug_dupes {
        println!("  After filtering: {} valid sequences", filtered_matches.len());
    }

    // Step 3: Group matches into multi-episode segments
    let segments = group_matches_into_segments(&filtered_matches, episode_audio, config, debug_dupes);

    if debug_dupes {
        println!("🎵 [Chromaprint] Detected {} segments", segments.len());
    }

    Ok(segments)
}

/// Represents a matching sequence of landmarks between two episodes
#[derive(Debug, Clone)]
struct SequenceMatch {
    episode1: usize,
    episode2: usize,
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    match_count: usize,
}

/// Find matching sequences between two episode landmark sets
fn find_matching_sequences(
    landmarks1: &[Landmark],
    landmarks2: &[Landmark],
    ep1_id: usize,
    ep2_id: usize,
) -> Vec<SequenceMatch> {
    let mut matches = Vec::new();

    if landmarks1.is_empty() || landmarks2.is_empty() {
        return matches;
    }

    // Simple approach: find all hash matches and cluster them by time proximity
    let mut hash_matches: Vec<(f64, f64)> = Vec::new();

    // Build hash set for episode 2 for O(1) lookup
    let mut hash_map2: HashMap<u32, Vec<f64>> = HashMap::new();
    for landmark in landmarks2 {
        hash_map2
            .entry(landmark.hash)
            .or_insert_with(Vec::new)
            .push(landmark.timestamp);
    }

    // Find all hash matches
    for landmark1 in landmarks1 {
        if let Some(times2) = hash_map2.get(&landmark1.hash) {
            for &time2 in times2 {
                hash_matches.push((landmark1.timestamp, time2));
            }
        }
    }

    // DEBUG: Log number of hash matches found
    // eprintln!("DEBUG: Episodes {} vs {}: {} hash matches from {} and {} landmarks",
    //     ep1_id, ep2_id, hash_matches.len(), landmarks1.len(), landmarks2.len());

    if hash_matches.is_empty() {
        return matches;
    }

    // Group hash matches by time offset bins
    // This finds regions where the same audio content appears
    let mut offset_bins: HashMap<i32, Vec<(f64, f64)>> = HashMap::new();

    for &(t1, t2) in &hash_matches {
        // Round offset to nearest 5 seconds to group similar offsets
        let offset = (t2 - t1) / 5.0;
        let offset_bin = offset.round() as i32;
        
        offset_bins
            .entry(offset_bin)
            .or_insert_with(Vec::new)
            .push((t1, t2));
    }

    // eprintln!("DEBUG: Episodes {} vs {}: {} offset bins from {} hash matches",
    //     ep1_id, ep2_id, offset_bins.len(), hash_matches.len());

    // For each offset bin, find contiguous matching regions
    for (_offset_bin, mut bin_matches) in offset_bins {
        // Sort by episode 1 time
        bin_matches.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find contiguous regions with STRICT density requirements
        let mut start_idx = 0;
        while start_idx < bin_matches.len() {
            let region_start1 = bin_matches[start_idx].0;
            let region_start2 = bin_matches[start_idx].1;
            
            // Extend region as long as matches are VERY DENSE
            let mut end_idx = start_idx;
            let mut match_count = 1;
            
            for i in (start_idx + 1)..bin_matches.len() {
                let time_since_start = bin_matches[i].0 - region_start1;
                let time_since_last = bin_matches[i].0 - bin_matches[end_idx].0;
                
                // Calculate current density
                let current_density = (match_count + 1) as f64 / time_since_start.max(1.0);
                
                // Stop if:
                // 1. Gap too large (>3s), OR
                // 2. Total duration reaching max limit (4 min), OR  
                // 3. Density dropping too low (< 0.05 matches/sec for quality)
                // 4. Huge time jump (>300s) indicating beginning->ending span
                if time_since_last >= MAX_LANDMARK_GAP 
                    || time_since_start >= MAX_SEGMENT_DURATION  // Stop at 4 minutes
                    || current_density < 0.05  // Reasonable density requirement
                    || time_since_last > 300.0
                {
                    break;
                }
                
                end_idx = i;
                match_count += 1;
            }

            let region_end1 = bin_matches[end_idx].0;
            let region_end2 = bin_matches[end_idx].1;
            let duration1 = region_end1 - region_start1;
            let duration2 = region_end2 - region_start2;
            let final_density = match_count as f64 / duration1.max(1.0);

            // Accept only if:
            // 1. Good density (>= 0.05 matches/sec for quality)
            // 2. Enough total matches (>= 3)
            // 3. Reasonable duration (5-240s for openings/endings)
            if match_count >= MIN_SEQUENCE_LENGTH
                && final_density >= 0.05  // Reasonable density requirement
                && duration1 >= MIN_SEGMENT_DURATION
                && duration2 >= MIN_SEGMENT_DURATION
                && duration1 <= MAX_SEGMENT_DURATION
                && duration2 <= MAX_SEGMENT_DURATION
            {
                matches.push(SequenceMatch {
                    episode1: ep1_id,
                    episode2: ep2_id,
                    start1: region_start1,
                    end1: region_end1,
                    start2: region_start2,
                    end2: region_end2,
                    match_count,
                });
            }

            start_idx = end_idx + 1;
        }
    }

    // eprintln!("DEBUG: Episodes {} vs {}: {} sequences found",
    //     ep1_id, ep2_id, matches.len());

    matches
}

/// Group pairwise matches into multi-episode segments
fn group_matches_into_segments(
    matches: &[SequenceMatch],
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Vec<CommonSegment> {
    if matches.is_empty() {
        return Vec::new();
    }

    // Group matches by approximate time range
    // Use very strict grouping: both start AND end must be close
    let mut segment_groups: Vec<Vec<&SequenceMatch>> = Vec::new();

    for m in matches {
        let mut found_group = false;

        for group in &mut segment_groups {
            if let Some(first) = group.first() {
                // Check if this match overlaps significantly with the group
                // Use BOTH start and end times for matching
                let start_close = (m.start1 - first.start1).abs() < 30.0;
                let end_close = (m.end1 - first.end1).abs() < 30.0;
                
                // Must match on both start AND end to be same segment
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

    let mut common_segments = Vec::new();

    // Convert each group to a CommonSegment
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
        if episode_timings_map.len() < config.threshold {
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

            // Use the largest merged range for this episode
            let (start, end) = merged
                .iter()
                .max_by_key(|(s, e)| ((e - s) * 1000.0) as i64)
                .unwrap();

            if end - start >= config.min_duration {
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

        if episode_timings.len() < config.threshold {
            continue;
        }

        // Calculate confidence from match density
        let avg_match_count =
            group.iter().map(|m| m.match_count).sum::<usize>() as f64 / group.len() as f64;
        let confidence = (avg_match_count / 50.0).min(1.0) * 0.8 + 0.2;

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

        let episode_names: Vec<String> = episode_timings
            .iter()
            .map(|t| t.episode_name.clone())
            .collect();

        if debug_dupes {
            println!(
                "  ✓ Chromaprint segment: {:.1}s-{:.1}s across {} episodes (conf={:.2}, time-shifted={})",
                ref_start, ref_end, episode_names.len(), confidence, time_shifted
            );
        }

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

    // Split any overlong segments first (prevents merging opening+ending)
    let split = split_overlong_segments(common_segments);
    
    // Then merge overlapping segments to eliminate false positives
    let merged = merge_segments_util(split);

    merged
}

/// Merge overlapping audio segments (DEPRECATED - use audio_segment_utils::merge_overlapping_segments)
#[allow(dead_code)]
fn merge_overlapping_audio_segments(mut segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    if segments.is_empty() {
        return segments;
    }

    segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.into_iter().skip(1) {
        // Check for overlap
        let overlap_start = current.start_time.max(segment.start_time);
        let overlap_end = current.end_time.min(segment.end_time);
        let overlap_duration = (overlap_end - overlap_start).max(0.0);

        let current_duration = current.end_time - current.start_time;
        let segment_duration = segment.end_time - segment.start_time;
        let min_duration = current_duration.min(segment_duration);

        // Merge if significant overlap (>50%) or adjacent (within 2s)
        if overlap_duration > min_duration * 0.5 || segment.start_time <= current.end_time + 2.0 {
            // Merge
            current.end_time = current.end_time.max(segment.end_time);
            current.confidence = current.confidence.max(segment.confidence);

            // Merge episode lists
            for ep in &segment.episode_list {
                if !current.episode_list.contains(ep) {
                    current.episode_list.push(ep.clone());
                }
            }

            // Merge episode timings
            if let (Some(ref mut curr_timings), Some(ref seg_timings)) =
                (&mut current.episode_timings, &segment.episode_timings)
            {
                for seg_timing in seg_timings {
                    if let Some(curr_timing) = curr_timings
                        .iter_mut()
                        .find(|t| t.episode_name == seg_timing.episode_name)
                    {
                        curr_timing.start_time = curr_timing.start_time.min(seg_timing.start_time);
                        curr_timing.end_time = curr_timing.end_time.max(seg_timing.end_time);
                    } else {
                        curr_timings.push(seg_timing.clone());
                    }
                }
            }
        } else {
            // No overlap, save current and start new
            merged.push(current);
            current = segment;
        }
    }

    merged.push(current);
    merged
}

/// Check if a segment is time-shifted across episodes
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
