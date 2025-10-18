//! Common segment detection logic

use crate::analyzer::EpisodeFrames;
use crate::hasher::{hamming_distance, RollingHash};
use crate::similarity::{
    calculate_adaptive_threshold, calculate_similarity_score, compute_ssim_from_features,
    FrameFeatures, KeyPoint, MultiScaleHash, SimilarityAlgorithm,
};
use crate::Result;
use std::collections::HashMap;

/// Represents a common segment found across multiple episodes
#[derive(Debug, Clone)]
pub struct CommonSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub episode_list: Vec<String>,
    pub confidence: f64,
}

/// Represents a sequence hash with episode information
#[derive(Debug, Clone)]
pub struct SequenceHash {
    hash: u64,
    episode_id: usize,
    start_time: f64,
    end_time: f64,
}

/// Detect common segments across episodes using rolling hash
pub fn detect_common_segments(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    match config.similarity_algorithm {
        SimilarityAlgorithm::Current => {
            detect_with_current_algorithm(episode_frames, config, debug_dupes)
        }
        SimilarityAlgorithm::MultiHash => {
            detect_with_multi_hash(episode_frames, config, debug_dupes)
        }
        SimilarityAlgorithm::SsimFeatures => {
            detect_with_ssim_features(episode_frames, config, debug_dupes)
        }
        SimilarityAlgorithm::Both => {
            let results1 = detect_with_multi_hash(episode_frames, config, debug_dupes)?;
            let _results2 = detect_with_ssim_features(episode_frames, config, debug_dupes)?;
            // For now, return multi-hash results (comparison will be implemented later)
            Ok(results1)
        }
    }
}

/// Detect using current algorithm (existing implementation)
fn detect_with_current_algorithm(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    // First, check for fully identical videos
    if let Some(full_duplicate_segments) =
        detect_full_video_duplicates(episode_frames, config, debug_dupes)?
    {
        return Ok(full_duplicate_segments);
    }

    // Check for identical opening segments
    if let Some(opening_segments) = detect_opening_segments(episode_frames, config, debug_dupes)? {
        return Ok(opening_segments);
    }

    let mut sequence_hashes = Vec::new();
    // Use smaller window size for shorter videos to detect opening segments
    let min_frames = episode_frames
        .iter()
        .map(|ep| ep.frames.len())
        .min()
        .unwrap_or(0);
    let window_size = if min_frames < 20 {
        (config.min_duration as usize).max(3) // Smaller window for short videos
    } else {
        (config.min_duration as usize).max(10) // Standard window for longer videos
    };

    // For opening segment detection, also try with a very small window size
    let opening_window_size = 3;

    // Generate rolling hashes for each episode
    for (episode_id, episode) in episode_frames.iter().enumerate() {
        let mut rolling_hash = RollingHash::new(window_size);

        for (i, frame) in episode.frames.iter().enumerate() {
            if let Some(_hash) = rolling_hash.add(frame.perceptual_hash) {
                let start_time = if i + 1 >= window_size {
                    episode.frames[i + 1 - window_size].timestamp
                } else {
                    episode.frames[0].timestamp
                };

                sequence_hashes.push(SequenceHash {
                    hash: _hash,
                    episode_id,
                    start_time,
                    end_time: frame.timestamp,
                });
            }
        }

        // Also generate sequences with smaller window size for opening segment detection
        if opening_window_size < window_size {
            let mut opening_hash = RollingHash::new(opening_window_size);

            for (i, frame) in episode.frames.iter().enumerate() {
                if let Some(_hash) = opening_hash.add(frame.perceptual_hash) {
                    let start_time = if i + 1 >= opening_window_size {
                        episode.frames[i + 1 - opening_window_size].timestamp
                    } else {
                        episode.frames[0].timestamp
                    };

                    sequence_hashes.push(SequenceHash {
                        hash: _hash,
                        episode_id,
                        start_time,
                        end_time: frame.timestamp,
                    });
                }
            }
        }
    }

    // Group sequences by similar hashes (using Hamming distance)
    let mut hash_groups: HashMap<u64, Vec<SequenceHash>> = HashMap::new();
    let similarity_threshold = config.similarity_threshold;

    for seq in sequence_hashes {
        // Try to find an existing group with similar hash
        let mut target_group_hash = None;

        for (group_hash, group_sequences) in hash_groups.iter() {
            if let Some(first_seq) = group_sequences.first() {
                // Calculate Hamming distance between hashes
                let hamming_dist = hamming_distance(seq.hash, first_seq.hash);
                let max_distance = (64.0 * (1.0 - similarity_threshold)) as u32;

                if hamming_dist <= max_distance {
                    target_group_hash = Some(*group_hash);
                    break;
                }
            }
        }

        // Add to existing group or create new one
        if let Some(group_hash) = target_group_hash {
            hash_groups.get_mut(&group_hash).unwrap().push(seq);
        } else {
            hash_groups.insert(seq.hash, vec![seq]);
        }
    }

    if debug_dupes {
        println!("\n=== DEBUG: Duplicate Detection Analysis ===");
        println!("Total unique hash sequences: {}", hash_groups.len());
        println!("Window size: {} frames", window_size);
        println!("Threshold: {} episodes", config.threshold);
        println!("Min duration: {} seconds", config.min_duration);

        // Show segments in the specific time range 21:10-22:00 (1270-1320 seconds)
        println!("\nSegments in time range 21:10-22:00 (1270-1320 seconds):");
        let mut target_time_segments = Vec::new();
        for (hash, sequences) in &hash_groups {
            for seq in sequences {
                if seq.start_time >= 1270.0 && seq.start_time <= 1320.0 {
                    target_time_segments.push((hash, seq));
                }
            }
        }
        target_time_segments.sort_by(|a, b| a.1.start_time.partial_cmp(&b.1.start_time).unwrap());

        for (i, (hash, seq)) in target_time_segments.iter().enumerate() {
            let episode_name = episode_frames[seq.episode_id]
                .episode_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown");
            println!(
                "  {}: Hash 0x{:x} in {} at {:.1}s - {:.1}s (duration: {:.1}s)",
                i + 1,
                hash,
                episode_name,
                seq.start_time,
                seq.end_time,
                seq.end_time - seq.start_time
            );
        }

        // Show Hamming distances between 27.mkv and 28.mkv segments in this time range
        println!("\nHamming distances between 27.mkv and 28.mkv segments in this time range:");
        let segments_27: Vec<_> = target_time_segments
            .iter()
            .filter(|(_, seq)| {
                episode_frames[seq.episode_id]
                    .episode_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    == "27.mkv"
            })
            .collect();
        let segments_28: Vec<_> = target_time_segments
            .iter()
            .filter(|(_, seq)| {
                episode_frames[seq.episode_id]
                    .episode_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    == "28.mkv"
            })
            .collect();

        for (_, seq_27) in &segments_27 {
            for (_, seq_28) in &segments_28 {
                let hamming_dist = hamming_distance(seq_27.hash, seq_28.hash);
                let similarity = (1.0 - (hamming_dist as f64 / 64.0)) * 100.0;
                println!("  27.mkv {:.1}s-{:.1}s vs 28.mkv {:.1}s-{:.1}s: Hamming distance = {}, Similarity = {:.1}%",
                         seq_27.start_time, seq_27.end_time, seq_28.start_time, seq_28.end_time,
                         hamming_dist, similarity);
            }
        }

        // Show top hash groups by frequency
        let mut sorted_groups: Vec<_> = hash_groups.iter().collect();
        sorted_groups.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        println!("\nTop 10 most frequent hash sequences:");
        for (i, (hash, sequences)) in sorted_groups.iter().take(10).enumerate() {
            let episode_count = sequences
                .iter()
                .map(|s| s.episode_id)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let time_range =
                if let (Some(first), Some(last)) = (sequences.first(), sequences.last()) {
                    format!("{:.1}s - {:.1}s", first.start_time, last.end_time)
                } else {
                    "N/A".to_string()
                };
            println!(
                "  {}: Hash 0x{:x} appears {} times across {} episodes, time: {}",
                i + 1,
                hash,
                sequences.len(),
                episode_count,
                time_range
            );
        }
    }

    // Find sequences that appear in enough episodes
    let mut common_segments = Vec::new();

    for (_hash, sequences) in hash_groups {
        if sequences.len() >= config.threshold {
            // Group by episode to avoid duplicates
            let mut episode_segments: HashMap<usize, Vec<&SequenceHash>> = HashMap::new();
            for seq in &sequences {
                episode_segments
                    .entry(seq.episode_id)
                    .or_insert_with(Vec::new)
                    .push(seq);
            }

            if episode_segments.len() >= config.threshold {
                // Calculate average start and end times
                let total_duration: f64 = sequences.iter().map(|s| s.end_time - s.start_time).sum();
                let avg_duration = total_duration / sequences.len() as f64;

                let avg_start: f64 =
                    sequences.iter().map(|s| s.start_time).sum::<f64>() / sequences.len() as f64;
                let avg_end = avg_start + avg_duration;

                // Get episode names
                let episode_names: Vec<String> = episode_segments
                    .keys()
                    .map(|&id| {
                        episode_frames[id]
                            .episode_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    })
                    .collect();

                // Calculate confidence based on consistency
                let confidence = calculate_confidence(&sequences, avg_duration);

                let segment = CommonSegment {
                    start_time: avg_start,
                    end_time: avg_end,
                    episode_list: episode_names,
                    confidence,
                };

                // Validate that segment meets min_duration requirement
                let segment_duration = segment.end_time - segment.start_time;
                if segment_duration >= config.min_duration {
                    if debug_dupes {
                        println!(
                            "\nFound segment: {:.1}s - {:.1}s (duration: {:.1}s)",
                            segment.start_time, segment.end_time, segment_duration
                        );
                        println!("  Confidence: {:.1}%", segment.confidence * 100.0);
                        println!("  Episodes: {}", segment.episode_list.join(", "));
                        println!("  Raw sequences: {}", sequences.len());
                    }

                    common_segments.push(segment);
                } else if debug_dupes {
                    println!("\nSkipping segment: {:.1}s - {:.1}s (duration: {:.1}s) - below min_duration {:.1}s", 
                             segment.start_time, segment.end_time, segment_duration, config.min_duration);
                }
            }
        }
    }

    // Apply deduplication and merge overlapping segments
    let deduplicated_segments = deduplicate_similar_segments(common_segments);
    let merged_segments = merge_overlapping_segments(deduplicated_segments);

    // Sort by confidence (highest first)
    let mut final_segments = merged_segments;
    final_segments.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(final_segments)
}

/// Detect identical opening segments by comparing the first few frames
fn detect_opening_segments(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Option<Vec<CommonSegment>>> {
    if episode_frames.len() < config.threshold {
        return Ok(None);
    }

    // Check if the first few frames are identical across episodes
    let min_frames = episode_frames
        .iter()
        .map(|ep| ep.frames.len())
        .min()
        .unwrap_or(0);
    if min_frames < 5 {
        return Ok(None);
    }

    // Compare first 160 frames of each episode (32 seconds at 5 fps)
    let check_frames = 160.min(min_frames);
    let mut identical_episodes = Vec::new();

    for (episode_id, episode) in episode_frames.iter().enumerate() {
        let mut is_identical = true;

        // Compare with the first episode
        for i in 0..check_frames {
            if i >= episode.frames.len() || i >= episode_frames[0].frames.len() {
                is_identical = false;
                break;
            }

            let hash1 = episode.frames[i].perceptual_hash;
            let hash2 = episode_frames[0].frames[i].perceptual_hash;

            // Check if hashes are similar (allowing for some variation)
            let hamming_dist = hamming_distance(hash1, hash2);
            let max_distance = (64.0 * (1.0 - config.similarity_threshold)) as u32;

            if hamming_dist > max_distance {
                is_identical = false;
                break;
            }
        }

        if is_identical {
            identical_episodes.push(episode_id);
        }
    }

    if identical_episodes.len() >= config.threshold {
        // Calculate the duration of the identical opening segment
        let first_episode = &episode_frames[0];
        let start_time = first_episode.frames[0].timestamp;
        let end_time = first_episode.frames[check_frames - 1].timestamp;
        let duration = end_time - start_time;

        if duration >= config.min_duration {
            let episode_names: Vec<String> = identical_episodes
                .iter()
                .map(|&id| {
                    episode_frames[id]
                        .episode_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string()
                })
                .collect();

            let segment = CommonSegment {
                start_time,
                end_time,
                episode_list: episode_names,
                confidence: 1.0, // Opening segments are highly confident
            };

            if debug_dupes {
                println!(
                    "Found opening segment: {:.1}s - {:.1}s (duration: {:.1}s)",
                    segment.start_time, segment.end_time, duration
                );
                println!("  Episodes: {}", segment.episode_list.join(", "));
            }

            return Ok(Some(vec![segment]));
        }
    }

    Ok(None)
}

/// Detect fully identical videos by comparing frame counts and key frame hashes
fn detect_full_video_duplicates(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Option<Vec<CommonSegment>>> {
    if episode_frames.len() < config.threshold {
        return Ok(None);
    }

    // Group episodes by frame count
    let mut frame_count_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for (episode_id, episode) in episode_frames.iter().enumerate() {
        frame_count_groups
            .entry(episode.frames.len())
            .or_insert_with(Vec::new)
            .push(episode_id);
    }

    // Find groups with enough episodes
    for (frame_count, episode_ids) in frame_count_groups {
        if episode_ids.len() >= config.threshold {
            // Check if these episodes are fully identical
            if let Some(identical_segments) =
                check_episodes_identical(&episode_ids, episode_frames, config, debug_dupes)?
            {
                if debug_dupes {
                    println!("\n=== DEBUG: Full Video Detection ===");
                    println!(
                        "Found {} fully identical episodes with {} frames each",
                        episode_ids.len(),
                        frame_count
                    );
                }
                return Ok(Some(identical_segments));
            }
        }
    }

    Ok(None)
}

/// Check if a group of episodes are fully identical
fn check_episodes_identical(
    episode_ids: &[usize],
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Option<Vec<CommonSegment>>> {
    if episode_ids.len() < config.threshold {
        return Ok(None);
    }

    let first_episode_id = episode_ids[0];
    let first_episode = &episode_frames[first_episode_id];
    let frame_count = first_episode.frames.len();

    // Sample key frames: first 10, middle 10, last 10
    let sample_indices = if frame_count <= 30 {
        (0..frame_count).collect::<Vec<_>>()
    } else {
        let mut indices = Vec::new();
        // First 10 frames
        indices.extend(0..10.min(frame_count));
        // Middle 10 frames
        let middle_start = frame_count / 2 - 5;
        indices.extend(middle_start..(middle_start + 10).min(frame_count));
        // Last 10 frames
        let last_start = frame_count.saturating_sub(10);
        indices.extend(last_start..frame_count);
        indices.sort();
        indices.dedup();
        indices
    };

    if debug_dupes {
        println!(
            "Checking {} episodes for full identity using {} sample frames",
            episode_ids.len(),
            sample_indices.len()
        );
    }

    // Compare sample frames across all episodes
    let mut identical_episodes = vec![first_episode_id];

    for &episode_id in &episode_ids[1..] {
        let episode = &episode_frames[episode_id];
        let mut matches = 0;

        for &frame_idx in &sample_indices {
            if frame_idx >= episode.frames.len() {
                break;
            }

            let first_hash = first_episode.frames[frame_idx].perceptual_hash;
            let current_hash = episode.frames[frame_idx].perceptual_hash;

            if first_hash == current_hash {
                matches += 1;
            }
        }

        let match_ratio = matches as f64 / sample_indices.len() as f64;
        if match_ratio >= 0.95 {
            // 95% match threshold for full video identity
            identical_episodes.push(episode_id);
        } else if debug_dupes {
            println!(
                "Episode {} only matches {:.1}% of sample frames",
                episode_id,
                match_ratio * 100.0
            );
        }
    }

    if identical_episodes.len() >= config.threshold {
        // Create segment for full video duration
        let episode_names: Vec<String> = identical_episodes
            .iter()
            .map(|&id| {
                episode_frames[id]
                    .episode_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            })
            .collect();

        let first_frame_time = first_episode.frames[0].timestamp;
        let last_frame_time = first_episode.frames.last().unwrap().timestamp;
        let duration = last_frame_time - first_frame_time;

        if duration >= config.min_duration {
            let segment = CommonSegment {
                start_time: first_frame_time,
                end_time: last_frame_time,
                episode_list: episode_names,
                confidence: 1.0, // Full identity = 100% confidence
            };

            if debug_dupes {
                println!(
                    "Found full video duplicate: {:.1}s - {:.1}s (duration: {:.1}s)",
                    segment.start_time, segment.end_time, duration
                );
                println!("  Episodes: {}", segment.episode_list.join(", "));
            }

            return Ok(Some(vec![segment]));
        }
    }

    Ok(None)
}

/// Detect using multi-scale perceptual hashing
fn detect_with_multi_hash(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    // First, check for fully identical videos
    if let Some(full_duplicate_segments) =
        detect_full_video_duplicates(episode_frames, config, debug_dupes)?
    {
        return Ok(full_duplicate_segments);
    }

    // Check for identical opening segments
    if let Some(opening_segments) = detect_opening_segments(episode_frames, config, debug_dupes)? {
        return Ok(opening_segments);
    }

    let mut multi_hashes = Vec::new();
    let window_size = (config.min_duration as usize).max(10);

    // Generate multi-scale hashes for each episode
    for (episode_id, episode) in episode_frames.iter().enumerate() {
        let mut rolling_hash = RollingHash::new(window_size);

        for (i, frame) in episode.frames.iter().enumerate() {
            if let Some(_hash) = rolling_hash.add(frame.perceptual_hash) {
                if debug_dupes && episode_id == 0 && i < 5 {
                    println!(
                        "  Generated hash for episode {} frame {}: 0x{:x}",
                        episode_id, i, _hash
                    );
                }
                let start_time = if i + 1 >= window_size {
                    episode.frames[i + 1 - window_size].timestamp
                } else {
                    episode.frames[0].timestamp
                };

                // Generate multi-scale hash for this frame
                // For now, we'll use the existing perceptual hash as a base and create variations
                // In a full implementation, we'd need the actual image data to call generate_multi_scale_hash
                let base_hash = frame.perceptual_hash;

                // Create different hash variations based on the perceptual hash
                // This simulates different hash algorithms working on the same image
                // Use more realistic variations that would occur with different algorithms
                let multi_hash = MultiScaleHash {
                    dhash: base_hash.rotate_left(1) ^ 0xAAAAAAAAAAAAAAAA, // Simulate dHash with rotation
                    phash: base_hash,                                     // Use existing pHash
                    ahash: base_hash.rotate_right(2) ^ 0x5555555555555555, // Simulate aHash with rotation
                    color_hash: base_hash.rotate_left(3) ^ 0x3333333333333333, // Simulate color hash with rotation
                };

                multi_hashes.push((multi_hash, episode_id, start_time, frame.timestamp));
            }
        }
    }

    // Calculate adaptive threshold - be more permissive for real-world content
    let hashes: Vec<_> = multi_hashes
        .iter()
        .map(|(hash, _, _, _)| hash.clone())
        .collect();
    let adaptive_threshold = calculate_adaptive_threshold(&hashes, config.similarity_threshold);

    if debug_dupes {
        println!("\n=== DEBUG: Multi-Hash Detection ===");
        println!("Adaptive threshold: {:.3}", adaptive_threshold);
        println!("Base threshold: {:.3}", config.similarity_threshold);
        println!("Total multi-hashes: {}", multi_hashes.len());
        println!("Window size: {}", window_size);
    }

    // Group by similarity using multi-scale hashing
    let mut hash_groups: HashMap<u64, Vec<(MultiScaleHash, usize, f64, f64)>> = HashMap::new();

    for (multi_hash, episode_id, start_time, end_time) in multi_hashes {
        // Try to find an existing group with similar hash
        let mut target_group_hash = None;

        for (group_hash, group_hashes) in hash_groups.iter() {
            if let Some((first_hash, _, _, _)) = group_hashes.first() {
                let similarity = calculate_similarity_score(&multi_hash, first_hash);

                if debug_dupes && hash_groups.len() < 5 {
                    println!(
                        "  Similarity: {:.3} (threshold: {:.3})",
                        similarity, adaptive_threshold
                    );
                }

                if similarity >= adaptive_threshold {
                    target_group_hash = Some(*group_hash);
                    break;
                }
            }
        }

        // Add to existing group or create new one
        if let Some(group_hash) = target_group_hash {
            hash_groups
                .get_mut(&group_hash)
                .unwrap()
                .push((multi_hash, episode_id, start_time, end_time));
        } else {
            // Use a simple hash of the multi-scale hash for grouping
            let group_key =
                multi_hash.dhash ^ multi_hash.phash ^ multi_hash.ahash ^ multi_hash.color_hash;
            hash_groups.insert(
                group_key,
                vec![(multi_hash, episode_id, start_time, end_time)],
            );
        }
    }

    // Find sequences that appear in enough episodes
    let mut common_segments = Vec::new();

    for (_hash, sequences) in hash_groups {
        if sequences.len() >= config.threshold {
            // Group by episode to avoid duplicates
            let mut episode_segments: HashMap<usize, Vec<&(MultiScaleHash, usize, f64, f64)>> =
                HashMap::new();
            for seq in &sequences {
                episode_segments
                    .entry(seq.1)
                    .or_insert_with(Vec::new)
                    .push(seq);
            }

            if episode_segments.len() >= config.threshold {
                // For identical files, we want to find the full video duration
                // Check if this group covers most of the video duration
                let mut episode_ranges: Vec<(f64, f64)> = Vec::new();
                let mut episode_names: Vec<String> = Vec::new();

                for (&episode_id, episode_sequences) in &episode_segments {
                    let episode_name = episode_frames[episode_id]
                        .episode_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string();

                    // Find the range covered by this episode's sequences
                    let mut min_start = f64::INFINITY;
                    let mut max_end = f64::NEG_INFINITY;

                    for seq in episode_sequences {
                        min_start = min_start.min(seq.2);
                        max_end = max_end.max(seq.3);
                    }

                    episode_ranges.push((min_start, max_end));
                    episode_names.push(episode_name);
                }

                // Calculate the overall segment range
                let segment_start = episode_ranges
                    .iter()
                    .map(|(start, _)| *start)
                    .fold(f64::INFINITY, f64::min);
                let segment_end = episode_ranges
                    .iter()
                    .map(|(_, end)| *end)
                    .fold(f64::NEG_INFINITY, f64::max);

                // Calculate confidence based on similarity scores
                let mut total_similarity = 0.0;
                let mut similarity_count = 0;

                for i in 0..sequences.len() {
                    for j in i + 1..sequences.len() {
                        let sim = calculate_similarity_score(&sequences[i].0, &sequences[j].0);
                        total_similarity += sim;
                        similarity_count += 1;
                    }
                }

                let confidence = if similarity_count > 0 {
                    total_similarity / similarity_count as f64
                } else {
                    0.0
                };

                let segment = CommonSegment {
                    start_time: segment_start,
                    end_time: segment_end,
                    episode_list: episode_names,
                    confidence,
                };

                // Validate that segment meets min_duration requirement
                let segment_duration = segment.end_time - segment.start_time;
                if segment_duration >= config.min_duration {
                    if debug_dupes {
                        println!(
                            "\nFound multi-hash segment: {:.1}s - {:.1}s (duration: {:.1}s)",
                            segment.start_time, segment.end_time, segment_duration
                        );
                        println!("  Confidence: {:.1}%", segment.confidence * 100.0);
                        println!("  Episodes: {}", segment.episode_list.join(", "));
                        println!("  Raw sequences: {}", sequences.len());
                        println!("  Episode ranges: {:?}", episode_ranges);
                    }

                    common_segments.push(segment);
                } else if debug_dupes {
                    println!("\nSkipping multi-hash segment: {:.1}s - {:.1}s (duration: {:.1}s) - below min_duration {:.1}s",
                                 segment.start_time, segment.end_time, segment_duration, config.min_duration);
                }
            }
        }
    }

    // Apply deduplication and merge overlapping segments
    let deduplicated_segments = deduplicate_similar_segments(common_segments);
    let merged_segments = merge_overlapping_segments(deduplicated_segments);

    // Sort by confidence (highest first)
    let mut final_segments = merged_segments;
    final_segments.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(final_segments)
}

/// Detect using SSIM + Feature matching
fn detect_with_ssim_features(
    episode_frames: &[EpisodeFrames],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    // First, check for fully identical videos
    if let Some(full_duplicate_segments) =
        detect_full_video_duplicates(episode_frames, config, debug_dupes)?
    {
        return Ok(full_duplicate_segments);
    }

    // Check for identical opening segments
    if let Some(opening_segments) = detect_opening_segments(episode_frames, config, debug_dupes)? {
        return Ok(opening_segments);
    }

    use crate::similarity::match_features;

    let mut frame_features = Vec::new();
    let window_size = (config.min_duration as usize).max(10);

    // Extract features for each episode
    for (_episode_id, episode) in episode_frames.iter().enumerate() {
        let mut episode_features = Vec::new();

        for frame in &episode.frames {
            // Create more realistic features based on the perceptual hash
            // This simulates SSIM and feature extraction working on the same image
            let hash_normalized = frame.perceptual_hash as f64 / (u64::MAX as f64);

            // Simulate SSIM signature with multiple features
            let ssim_signature = vec![
                hash_normalized,
                (frame.perceptual_hash >> 32) as f64 / (u32::MAX as f64),
                (frame.perceptual_hash & 0xFFFFFFFF) as f64 / (u32::MAX as f64),
                ((frame.perceptual_hash >> 16) & 0xFFFF) as f64 / (u16::MAX as f64),
            ];

            // Simulate keypoints based on hash bits
            let mut keypoints = Vec::new();
            let mut descriptors = Vec::new();

            // Generate keypoints based on hash bits
            for i in 0..8 {
                if (frame.perceptual_hash >> i) & 1 == 1 {
                    keypoints.push(KeyPoint {
                        x: (i as f32 * 10.0) % 100.0,
                        y: (i as f32 * 15.0) % 100.0,
                        response: hash_normalized as f32,
                    });

                    // Add descriptor for this keypoint
                    for j in 0..25 {
                        descriptors.push((hash_normalized * (j as f64 + 1.0)) as f32);
                    }
                }
            }

            let features = FrameFeatures {
                ssim_signature,
                keypoints,
                descriptors,
                timestamp: frame.timestamp,
            };
            episode_features.push(features);
        }

        frame_features.push(episode_features);
    }

    if debug_dupes {
        println!("\n=== DEBUG: SSIM+Features Detection ===");
        println!("Window size: {} frames", window_size);
        println!("Threshold: {} episodes", config.threshold);
        println!("Min duration: {} seconds", config.min_duration);
    }

    // Find similar segments using SSIM and feature matching
    let mut common_segments = Vec::new();

    // Find similar segments using SSIM and feature matching
    // Use a more efficient approach: find segments that appear in multiple episodes

    // Create a map to store segments by their content signature
    let mut segment_groups: HashMap<String, Vec<(usize, f64, f64, f64)>> = HashMap::new();

    // Compare each episode with every other episode
    for i in 0..episode_frames.len() {
        for j in i + 1..episode_frames.len() {
            let features1 = &frame_features[i];
            let features2 = &frame_features[j];

            // Use larger step size to avoid too many overlapping matches
            let window_size_frames = window_size;
            let step_size = window_size_frames; // No overlap to reduce duplicates

            for start_idx in
                (0..features1.len().saturating_sub(window_size_frames)).step_by(step_size)
            {
                let end_idx = (start_idx + window_size_frames).min(features1.len());
                if end_idx - start_idx < window_size_frames {
                    continue;
                }

                let window1 = &features1[start_idx..end_idx];

                // Try different alignments in the second video
                for start2_idx in
                    (0..features2.len().saturating_sub(window_size_frames)).step_by(step_size)
                {
                    let end2_idx = (start2_idx + window_size_frames).min(features2.len());
                    if end2_idx - start2_idx < window_size_frames {
                        continue;
                    }

                    let window2 = &features2[start2_idx..end2_idx];

                    // Calculate similarity between windows
                    let mut total_similarity = 0.0;
                    let mut valid_pairs = 0;

                    for (feat1, feat2) in window1.iter().zip(window2.iter()) {
                        let ssim_sim = compute_ssim_from_features(feat1, feat2);
                        let feature_sim = match_features(feat1, feat2);
                        let combined_sim = ssim_sim * 0.6 + feature_sim * 0.4;

                        total_similarity += combined_sim;
                        valid_pairs += 1;
                    }

                    if valid_pairs > 0 {
                        let avg_similarity = total_similarity / valid_pairs as f64;

                        if avg_similarity >= config.similarity_threshold {
                            let start_time1 = window1[0].timestamp;
                            let end_time1 = window1.last().unwrap().timestamp;
                            let start_time2 = window2[0].timestamp;
                            let end_time2 = window2.last().unwrap().timestamp;

                            let duration = end_time1 - start_time1;
                            if duration >= config.min_duration {
                                // Create a content signature for this segment including episode indices and content hash
                                let content_hash = (window1[0].ssim_signature[0] * 1000.0) as u64;
                                let signature = format!(
                                    "{}-{}-{:.1}-{:.1}-{:x}",
                                    i, j, start_time1, end_time1, content_hash
                                );

                                // Add both episodes to this segment group
                                segment_groups
                                    .entry(signature.clone())
                                    .or_insert_with(Vec::new)
                                    .push((i, start_time1, end_time1, avg_similarity));
                                segment_groups
                                    .entry(signature)
                                    .or_insert_with(Vec::new)
                                    .push((j, start_time2, end_time2, avg_similarity));
                            }
                        }
                    }
                }
            }
        }
    }

    // Process segment groups to find segments that appear in enough episodes
    for (_signature, episodes) in segment_groups {
        let episode_count = episodes.len();
        if episode_count >= config.threshold {
            // Group by episode to avoid duplicates
            let mut episode_segments: HashMap<usize, Vec<(f64, f64, f64)>> = HashMap::new();
            for (episode_id, start_time, end_time, confidence) in &episodes {
                episode_segments
                    .entry(*episode_id)
                    .or_insert_with(Vec::new)
                    .push((*start_time, *end_time, *confidence));
            }

            if episode_segments.len() >= config.threshold {
                // Calculate average timing and confidence
                let mut total_confidence = 0.0;
                let mut confidence_count = 0;
                let mut episode_names = Vec::new();

                for (&episode_id, episode_times) in &episode_segments {
                    let episode_name = episode_frames[episode_id]
                        .episode_path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    episode_names.push(episode_name);

                    for (_, _, conf) in episode_times {
                        total_confidence += conf;
                        confidence_count += 1;
                    }
                }

                let avg_confidence = if confidence_count > 0 {
                    total_confidence / confidence_count as f64
                } else {
                    0.0
                };

                // Use the first episode's timing as reference
                let (start_time, end_time) =
                    if let Some(first_episode) = episode_segments.values().next() {
                        if let Some((start, end, _)) = first_episode.first() {
                            (*start, *end)
                        } else {
                            continue;
                        }
                    } else {
                        continue;
                    };

                let segment = CommonSegment {
                    start_time,
                    end_time,
                    episode_list: episode_names,
                    confidence: avg_confidence,
                };

                // Validate that segment meets min_duration requirement
                let segment_duration = segment.end_time - segment.start_time;
                if segment_duration >= config.min_duration {
                    if debug_dupes {
                        println!(
                            "\nFound SSIM+Features segment: {:.1}s - {:.1}s (duration: {:.1}s)",
                            segment.start_time, segment.end_time, segment_duration
                        );
                        println!("  Confidence: {:.1}%", segment.confidence * 100.0);
                        println!("  Episodes: {}", segment.episode_list.join(", "));
                        println!("  Raw episodes: {}", episode_count);
                    }

                    common_segments.push(segment);
                } else if debug_dupes {
                    println!("\nSkipping SSIM+Features segment: {:.1}s - {:.1}s (duration: {:.1}s) - below min_duration {:.1}s",
                                 segment.start_time, segment.end_time, segment_duration, config.min_duration);
                }
            }
        }
    }

    // Apply advanced deduplication and merge overlapping segments
    let deduplicated_segments = deduplicate_similar_segments(common_segments);
    let merged_segments = merge_overlapping_segments(deduplicated_segments);
    let filtered_segments: Vec<CommonSegment> = merged_segments
        .into_iter()
        .filter(|segment| segment.episode_list.len() >= config.threshold)
        .collect();

    // Sort by confidence (highest first)
    let mut final_segments = filtered_segments;
    final_segments.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    Ok(final_segments)
}

/// Calculate confidence score for a segment based on consistency
fn calculate_confidence(sequences: &[SequenceHash], avg_duration: f64) -> f64 {
    if sequences.is_empty() {
        return 0.0;
    }

    // Calculate duration consistency
    let duration_variance: f64 = sequences
        .iter()
        .map(|s| {
            let duration = s.end_time - s.start_time;
            (duration - avg_duration).powi(2)
        })
        .sum::<f64>()
        / sequences.len() as f64;

    let duration_consistency = 1.0 / (1.0 + duration_variance);

    // Calculate episode count factor
    let episode_count_factor = (sequences.len() as f64 / 10.0).min(1.0);

    // Combine factors
    (duration_consistency * 0.7 + episode_count_factor * 0.3).min(1.0)
}

/// Find similar sequences using Hamming distance
pub fn find_similar_sequences(
    target_hash: u64,
    all_sequences: &[SequenceHash],
    max_distance: u32,
) -> Vec<&SequenceHash> {
    all_sequences
        .iter()
        .filter(|seq| hamming_distance(target_hash, seq.hash) <= max_distance)
        .collect()
}

/// Merge overlapping segments and deduplicate similar content
pub fn merge_overlapping_segments(mut segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    if segments.is_empty() {
        return segments;
    }

    // First, sort segments by start time
    segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.into_iter().skip(1) {
        // Check if segments overlap or are very close (within 2 seconds)
        let time_gap = segment.start_time - current.end_time;
        let overlap = current.end_time > segment.start_time;

        if overlap || time_gap <= 2.0 {
            // Merge segments
            current.end_time = current.end_time.max(segment.end_time);

            // Merge episode lists (remove duplicates)
            let mut all_episodes = current.episode_list.clone();
            for episode in &segment.episode_list {
                if !all_episodes.contains(episode) {
                    all_episodes.push(episode.clone());
                }
            }
            current.episode_list = all_episodes;

            // Use higher confidence
            current.confidence = current.confidence.max(segment.confidence);
        } else {
            // No overlap, add current segment and start new one
            merged.push(current);
            current = segment;
        }
    }

    merged.push(current);
    merged
}

/// Advanced deduplication for content-based similarity
pub fn deduplicate_similar_segments(segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    if segments.is_empty() {
        return segments;
    }

    let mut deduplicated = Vec::new();

    for segment in segments {
        let mut is_duplicate = false;

        // Check against already processed segments
        for existing in &mut deduplicated {
            // Check if segments are similar in timing and content
            let time_overlap = calculate_time_overlap(&segment, existing);
            let episode_overlap = calculate_episode_overlap(&segment, existing);

            // If significant overlap in both time and episodes, merge
            if time_overlap > 0.5 && episode_overlap > 0.7 {
                // Merge the segments
                existing.start_time = existing.start_time.min(segment.start_time);
                existing.end_time = existing.end_time.max(segment.end_time);
                existing.confidence = existing.confidence.max(segment.confidence);

                // Merge episode lists
                for episode in &segment.episode_list {
                    if !existing.episode_list.contains(episode) {
                        existing.episode_list.push(episode.clone());
                    }
                }

                is_duplicate = true;
                break;
            }
        }

        if !is_duplicate {
            deduplicated.push(segment);
        }
    }

    deduplicated
}

/// Calculate time overlap between two segments (0.0 to 1.0)
fn calculate_time_overlap(seg1: &CommonSegment, seg2: &CommonSegment) -> f64 {
    let overlap_start = seg1.start_time.max(seg2.start_time);
    let overlap_end = seg1.end_time.min(seg2.end_time);

    if overlap_end <= overlap_start {
        return 0.0;
    }

    let overlap_duration = overlap_end - overlap_start;
    let total_duration =
        (seg1.end_time - seg1.start_time) + (seg2.end_time - seg2.start_time) - overlap_duration;

    if total_duration <= 0.0 {
        return 0.0;
    }

    overlap_duration / total_duration
}

/// Calculate episode overlap between two segments (0.0 to 1.0)
fn calculate_episode_overlap(seg1: &CommonSegment, seg2: &CommonSegment) -> f64 {
    let common_episodes = seg1
        .episode_list
        .iter()
        .filter(|ep| seg2.episode_list.contains(ep))
        .count();

    let min_episodes = seg1.episode_list.len().min(seg2.episode_list.len());

    if min_episodes == 0 {
        return 0.0;
    }

    common_episodes as f64 / min_episodes as f64
}

/// Legacy function for backward compatibility
pub fn merge_overlapping_segments_legacy(mut segments: Vec<CommonSegment>) -> Vec<CommonSegment> {
    if segments.is_empty() {
        return segments;
    }

    // Sort by start time
    segments.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    let mut merged = Vec::new();
    let mut current = segments[0].clone();

    for segment in segments.into_iter().skip(1) {
        // Check if segments overlap or are very close
        if segment.start_time <= current.end_time + 5.0 {
            // 5 second tolerance
            // Merge segments
            current.end_time = current.end_time.max(segment.end_time);
            current.episode_list.extend(segment.episode_list);
            current.episode_list.sort();
            current.episode_list.dedup();
            current.confidence = (current.confidence + segment.confidence) / 2.0;
        } else {
            merged.push(current);
            current = segment;
        }
    }

    merged.push(current);
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyzer::{EpisodeFrames, Frame};
    use std::path::PathBuf;

    fn create_test_episode(name: &str, frames: Vec<(f64, u64)>) -> EpisodeFrames {
        EpisodeFrames {
            episode_path: PathBuf::from(name),
            frames: frames
                .into_iter()
                .map(|(timestamp, hash)| Frame {
                    timestamp,
                    perceptual_hash: hash,
                })
                .collect(),
        }
    }

    #[test]
    fn test_detect_common_segments_empty() {
        let episodes = vec![];
        let config = crate::Config::default();
        let result = detect_common_segments(&episodes, &config, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_detect_common_segments_single_episode() {
        let episodes = vec![create_test_episode(
            "ep1.mkv",
            vec![
                (0.0, 1),
                (1.0, 2),
                (2.0, 3),
                (3.0, 4),
                (4.0, 5),
                (5.0, 6),
                (6.0, 7),
                (7.0, 8),
                (8.0, 9),
                (9.0, 10),
                (10.0, 11),
                (11.0, 12),
                (12.0, 13),
                (13.0, 14),
                (14.0, 15),
            ],
        )];

        let config = crate::Config::default();
        let result = detect_common_segments(&episodes, &config, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty()); // Not enough episodes
    }

    #[test]
    fn test_detect_common_segments_identical_sequences() {
        // Create episodes with identical sequences
        let common_sequence = vec![
            (10.0, 100),
            (11.0, 101),
            (12.0, 102),
            (13.0, 103),
            (14.0, 104),
            (15.0, 105),
            (16.0, 106),
            (17.0, 107),
            (18.0, 108),
            (19.0, 109),
        ];

        let episodes = vec![
            create_test_episode("ep1.mkv", common_sequence.clone()),
            create_test_episode("ep2.mkv", common_sequence.clone()),
            create_test_episode("ep3.mkv", common_sequence.clone()),
        ];

        let config = crate::Config {
            threshold: 2,      // Lower threshold for test
            min_duration: 5.0, // Lower min duration for test
            ..crate::Config::default()
        };
        let result = detect_common_segments(&episodes, &config, false);
        assert!(result.is_ok());
        let segments = result.unwrap();
        assert!(!segments.is_empty());
        assert_eq!(segments[0].episode_list.len(), 3);
    }

    #[test]
    fn test_merge_overlapping_segments() {
        let segments = vec![
            CommonSegment {
                start_time: 0.0,
                end_time: 10.0,
                episode_list: vec!["ep1.mkv".to_string()],
                confidence: 0.8,
            },
            CommonSegment {
                start_time: 8.0,
                end_time: 18.0,
                episode_list: vec!["ep2.mkv".to_string()],
                confidence: 0.9,
            },
            CommonSegment {
                start_time: 25.0,
                end_time: 35.0,
                episode_list: vec!["ep3.mkv".to_string()],
                confidence: 0.7,
            },
        ];

        let merged = merge_overlapping_segments(segments);
        assert_eq!(merged.len(), 2); // First two should be merged
        assert_eq!(merged[0].start_time, 0.0);
        assert_eq!(merged[0].end_time, 18.0);
        assert_eq!(merged[0].episode_list.len(), 2);
        assert_eq!(merged[1].start_time, 25.0);
    }

    #[test]
    fn test_calculate_confidence() {
        let sequences = vec![
            SequenceHash {
                hash: 1,
                episode_id: 0,
                start_time: 0.0,
                end_time: 10.0,
            },
            SequenceHash {
                hash: 1,
                episode_id: 1,
                start_time: 0.0,
                end_time: 10.0,
            },
            SequenceHash {
                hash: 1,
                episode_id: 2,
                start_time: 0.0,
                end_time: 10.0,
            },
        ];

        let confidence = calculate_confidence(&sequences, 10.0);
        assert!(confidence > 0.0);
        assert!(confidence <= 1.0);
    }

    #[test]
    fn test_find_similar_sequences() {
        let sequences = vec![
            SequenceHash {
                hash: 0b1010,
                episode_id: 0,
                start_time: 0.0,
                end_time: 10.0,
            },
            SequenceHash {
                hash: 0b1000,
                episode_id: 1,
                start_time: 0.0,
                end_time: 10.0,
            },
            SequenceHash {
                hash: 0b0101,
                episode_id: 2,
                start_time: 0.0,
                end_time: 10.0,
            },
        ];

        let similar = find_similar_sequences(0b1010, &sequences, 1);
        assert_eq!(similar.len(), 2); // Should find exact match and 1-bit difference
    }
}
