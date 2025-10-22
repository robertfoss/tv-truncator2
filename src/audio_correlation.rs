//! Cross-correlation based audio matching
//!
//! This module implements audio segment matching using cross-correlation,
//! which is robust to phase shifts and encoding differences.

use crate::audio_extractor::EpisodeAudio;
use crate::segment_detector::{CommonSegment, MatchType};
use crate::Result;

/// Chunk size for cross-correlation analysis (in audio frames)
/// Larger chunks are more robust but slower
#[allow(dead_code)]
const CHUNK_SIZE: usize = 30; // 30 seconds at 1 fps

/// Correlation threshold for considering segments as matching
/// Lowered to handle encoding differences in real-world videos
const CORRELATION_THRESHOLD: f64 = 0.50; // 50% correlation (real-world encoded audio varies significantly)

/// Find matching audio segments using cross-correlation
///
/// This approach is more robust than spectral hashing for encoded audio
/// because it handles phase shifts and encoding artifacts better.
///
/// # Arguments
/// * `episode_audio` - Audio frames from all episodes
/// * `config` - Configuration settings
/// * `debug_dupes` - Whether to print debug information
///
/// # Returns
/// * `Result<Vec<CommonSegment>>` - Detected audio segments
pub fn detect_audio_segments_correlation(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [Cross-Correlation] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    let mut potential_matches = Vec::new();

    // Compare each pair of episodes
    for i in 0..episode_audio.len() {
        for j in (i + 1)..episode_audio.len() {
            let ep1 = &episode_audio[i];
            let ep2 = &episode_audio[j];

            if debug_dupes {
                println!(
                    "  Comparing {} ({} frames) vs {} ({} frames)",
                    ep1.episode_path.file_name().unwrap().to_string_lossy(),
                    ep1.audio_frames.len(),
                    ep2.episode_path.file_name().unwrap().to_string_lossy(),
                    ep2.audio_frames.len()
                );
            }

            // Find matching segments between this pair
            let matches = find_matching_segments_between_episodes(ep1, ep2, config, debug_dupes)?;

            for (start1, end1, start2, end2, correlation) in matches {
                potential_matches.push(PotentialMatch {
                    episode1_id: i,
                    episode2_id: j,
                    start1,
                    end1,
                    start2,
                    end2,
                    correlation,
                });
            }
        }
    }

    if debug_dupes {
        println!("  Found {} potential matches", potential_matches.len());
    }

    // Group matches into common segments
    let common_segments = group_matches_into_segments(
        &potential_matches,
        episode_audio,
        config,
        debug_dupes,
    )?;

    if debug_dupes {
        println!(
            "🎵 [Cross-Correlation] Detected {} audio segments",
            common_segments.len()
        );
    }

    Ok(common_segments)
}

/// Potential match between two episodes
#[derive(Debug, Clone)]
struct PotentialMatch {
    episode1_id: usize,
    episode2_id: usize,
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    correlation: f64,
}

/// Find matching segments between two episodes using cross-correlation
fn find_matching_segments_between_episodes(
    ep1: &EpisodeAudio,
    ep2: &EpisodeAudio,
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<(f64, f64, f64, f64, f64)>> {
    let mut matches = Vec::new();

    // Slide a window across ep1 and correlate with ep2
    // Use adaptive chunk size based on min_duration (not hardcoded CHUNK_SIZE)
    let min_chunk_frames = (config.min_duration as usize).max(10);
    let chunk_size = min_chunk_frames; // Use min_duration instead of CHUNK_SIZE for flexibility

    let mut pos1 = 0;
    let mut iteration = 0;
    while pos1 + chunk_size <= ep1.audio_frames.len() {
        iteration += 1;
        let chunk1_end = (pos1 + chunk_size).min(ep1.audio_frames.len());
        let chunk1_hashes: Vec<u64> = ep1.audio_frames[pos1..chunk1_end]
            .iter()
            .map(|f| f.spectral_hash)
            .collect();
        
        // Debug: print every 50 iterations to see progress
        if debug_dupes && iteration % 50 == 0 {
            let t = ep1.audio_frames[pos1].timestamp;
            println!("    Iteration {}: scanning ep1 at {:.1}s ({:.1} min)", iteration, t, t / 60.0);
        }

        // Find best match in ep2 using cross-correlation
        let best_match = find_best_correlation_match(&chunk1_hashes, ep2, chunk_size, debug_dupes);

        // Debug: print correlation in target range (22:21 area)
        if debug_dupes && pos1 < ep1.audio_frames.len() {
            let t = ep1.audio_frames[pos1].timestamp;
            if t >= 1330.0 && t <= 1350.0 {
                let corr_val = if let Some((_, corr)) = best_match {
                    format!("{:.3}", corr)
                } else {
                    "none".to_string()
                };
                println!("    >>> ep1 {:.1}s: best_corr={}", t, corr_val);
            }
        }

        if let Some((pos2, correlation)) = best_match {
            if correlation >= CORRELATION_THRESHOLD {
                let start1 = ep1.audio_frames[pos1].timestamp;
                let end1 = ep1.audio_frames[chunk1_end - 1].timestamp;
                let start2 = ep2.audio_frames[pos2].timestamp;
                let chunk2_end = (pos2 + chunk_size).min(ep2.audio_frames.len());
                let end2 = ep2.audio_frames[chunk2_end - 1].timestamp;

                if debug_dupes {
                    println!(
                        "    Match: ep1 [{:.1}s-{:.1}s] ↔ ep2 [{:.1}s-{:.1}s], corr={:.3}",
                        start1, end1, start2, end2, correlation
                    );
                }

                matches.push((start1, end1, start2, end2, correlation));

                // Skip ahead to avoid overlapping matches
                pos1 = chunk1_end;
                continue;
            } else if debug_dupes && correlation >= 0.3 {
                let start1 = ep1.audio_frames[pos1].timestamp;
                // Print correlations in key areas (opening and ending)
                if start1 < 120.0 || (start1 > 1300.0 && start1 < 1420.0) {
                    println!(
                        "    Below threshold (0.50): ep1 {:.1}s, corr={:.3}",
                        start1, correlation
                    );
                }
            }
        }

        // Move forward by half chunk for overlap
        pos1 += chunk_size / 2;
    }

    if debug_dupes {
        let final_time = if !ep1.audio_frames.is_empty() {
            ep1.audio_frames.last().unwrap().timestamp
        } else {
            0.0
        };
        println!("    Completed {} iterations, scanned up to {:.1}s ({:.1} min)",
            iteration, final_time, final_time / 60.0);
        println!("    Found {} matches", matches.len());
    }

    Ok(matches)
}

/// Find best correlation match for a chunk in another episode
fn find_best_correlation_match(
    chunk_hashes: &[u64],
    ep2: &EpisodeAudio,
    chunk_size: usize,
    _debug_dupes: bool,
) -> Option<(usize, f64)> {
    let mut best_pos = 0;
    let mut best_correlation = 0.0;

    let mut pos2 = 0;
    while pos2 + chunk_size <= ep2.audio_frames.len() {
        let chunk2_end = (pos2 + chunk_size).min(ep2.audio_frames.len());
        let chunk2_hashes: Vec<u64> = ep2.audio_frames[pos2..chunk2_end]
            .iter()
            .map(|f| f.spectral_hash)
            .collect();

        // Compute correlation between chunks
        let correlation = compute_hash_correlation(chunk_hashes, &chunk2_hashes);

        if correlation > best_correlation {
            best_correlation = correlation;
            best_pos = pos2;
        }

        pos2 += chunk_size / 4; // Slide by 1/4 chunk for better coverage
    }

    if best_correlation > 0.5 {
        Some((best_pos, best_correlation))
    } else {
        None
    }
}

/// Compute correlation between two hash sequences
///
/// Uses a simplified correlation: fraction of matching hashes
fn compute_hash_correlation(hashes1: &[u64], hashes2: &[u64]) -> f64 {
    let len = hashes1.len().min(hashes2.len());
    if len == 0 {
        return 0.0;
    }

    let mut matches = 0;
    for i in 0..len {
        if hashes1[i] == hashes2[i] {
            matches += 1;
        }
    }

    matches as f64 / len as f64
}

/// Group matches into common segments
fn group_matches_into_segments(
    matches: &[PotentialMatch],
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if matches.is_empty() {
        return Ok(Vec::new());
    }

    // Group matches by approximate time range
    let mut segment_groups: Vec<Vec<&PotentialMatch>> = Vec::new();

    for m in matches {
        // Find existing group that overlaps with this match
        let mut found_group = false;
        for group in &mut segment_groups {
            if let Some(first) = group.first() {
                // Check if this match overlaps with the group
                let time_diff = (m.start1 - first.start1).abs();
                if time_diff < 5.0 {
                    // Within 5 seconds - same segment
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
        // Count unique episodes in this group
        let mut episode_ids = std::collections::HashSet::new();
        for m in &group {
            episode_ids.insert(m.episode1_id);
            episode_ids.insert(m.episode2_id);
        }

        if episode_ids.len() >= config.threshold {
            // Calculate average timing
            let avg_start = group.iter().map(|m| m.start1.min(m.start2)).sum::<f64>()
                / group.len() as f64;
            let avg_end =
                group.iter().map(|m| m.end1.max(m.end2)).sum::<f64>() / group.len() as f64;
            let avg_correlation =
                group.iter().map(|m| m.correlation).sum::<f64>() / group.len() as f64;

            let duration = avg_end - avg_start;
            if duration >= config.min_duration {
                // Get episode names
                let episode_names: Vec<String> = episode_ids
                    .iter()
                    .map(|&id| {
                        episode_audio[id]
                            .episode_path
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("unknown")
                            .to_string()
                    })
                    .collect();

                if debug_dupes {
                    println!(
                        "  Segment: {:.1}s - {:.1}s ({:.1}s), {} episodes, corr={:.3}",
                        avg_start,
                        avg_end,
                        duration,
                        episode_names.len(),
                        avg_correlation
                    );
                }

                common_segments.push(CommonSegment {
                    start_time: avg_start,
                    end_time: avg_end,
                    episode_list: episode_names,
                    episode_timings: None,
                    confidence: avg_correlation,
                    video_confidence: None,
                    audio_confidence: Some(avg_correlation),
                    match_type: MatchType::Audio,
                });
            }
        }
    }

    Ok(common_segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_correlation() {
        let hashes1 = vec![1, 2, 3, 4, 5];
        let hashes2 = vec![1, 2, 3, 4, 5];

        let corr = compute_hash_correlation(&hashes1, &hashes2);
        assert_eq!(corr, 1.0); // Perfect match

        let hashes3 = vec![1, 2, 9, 4, 5];
        let corr2 = compute_hash_correlation(&hashes1, &hashes3);
        assert_eq!(corr2, 0.8); // 80% match
    }
}

