//! Algorithm comparison framework for audio matching
//!
//! Runs all audio algorithms on the same test data and generates
//! detailed comparison reports with precision, recall, F1, and performance metrics.

use crate::audio_chromaprint::detect_audio_segments_chromaprint;
use crate::audio_energy_bands::detect_audio_segments_energy_bands;
use crate::audio_extractor::EpisodeAudio;
use crate::audio_mfcc::detect_audio_segments_mfcc;
use crate::audio_spectral_v2::detect_audio_segments_spectral_v2;
use crate::segment_detector::CommonSegment;
use crate::{AudioAlgorithm, Config, Result};
use std::time::Instant;

/// Results from running a single algorithm
#[derive(Debug, Clone)]
pub struct AlgorithmResult {
    pub algorithm: AudioAlgorithm,
    pub segments: Vec<CommonSegment>,
    pub execution_time_ms: u128,
    pub error: Option<String>,
}

/// Comparison metrics for an algorithm
#[derive(Debug, Clone)]
pub struct ComparisonMetrics {
    pub algorithm: AudioAlgorithm,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub timing_accuracy_ms: f64,
    pub execution_time_ms: u128,
    pub segments_found: usize,
}

/// Expected segment for validation
#[derive(Debug, Clone)]
pub struct ExpectedSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub min_episodes: usize,
}

/// Run all algorithms and compare results
pub fn compare_all_algorithms(
    episode_audio: &[EpisodeAudio],
    config: &Config,
    expected_segments: &[ExpectedSegment],
    debug: bool,
) -> Result<Vec<ComparisonMetrics>> {
    let algorithms = vec![
        AudioAlgorithm::Chromaprint,
        AudioAlgorithm::Mfcc,
        AudioAlgorithm::SpectralV2,
        AudioAlgorithm::EnergyBands,
    ];

    let mut results = Vec::new();

    for algorithm in algorithms {
        if debug {
            println!("\n=== Testing {:?} ===", algorithm);
        }

        let result = run_algorithm(algorithm, episode_audio, config, debug)?;
        
        let metrics = if let Some(ref error) = result.error {
            if debug {
                println!("  ❌ Error: {}", error);
            }
            ComparisonMetrics {
                algorithm,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                timing_accuracy_ms: 0.0,
                execution_time_ms: result.execution_time_ms,
                segments_found: 0,
            }
        } else {
            calculate_metrics(&result, expected_segments, debug)
        };

        results.push(metrics);
    }

    Ok(results)
}

/// Run a single algorithm and measure performance
fn run_algorithm(
    algorithm: AudioAlgorithm,
    episode_audio: &[EpisodeAudio],
    config: &Config,
    debug: bool,
) -> Result<AlgorithmResult> {
    let start = Instant::now();

    let result = match algorithm {
        AudioAlgorithm::Chromaprint => {
            detect_audio_segments_chromaprint(episode_audio, config, debug)
        }
        AudioAlgorithm::Mfcc => detect_audio_segments_mfcc(episode_audio, config, debug),
        AudioAlgorithm::SpectralV2 => {
            detect_audio_segments_spectral_v2(episode_audio, config, debug)
        }
        AudioAlgorithm::EnergyBands => {
            detect_audio_segments_energy_bands(episode_audio, config, debug)
        }
        _ => {
            return Ok(AlgorithmResult {
                algorithm,
                segments: Vec::new(),
                execution_time_ms: 0,
                error: Some("Legacy algorithm not supported in comparison".to_string()),
            });
        }
    };

    let execution_time_ms = start.elapsed().as_millis();

    match result {
        Ok(segments) => Ok(AlgorithmResult {
            algorithm,
            segments,
            execution_time_ms,
            error: None,
        }),
        Err(e) => Ok(AlgorithmResult {
            algorithm,
            segments: Vec::new(),
            execution_time_ms,
            error: Some(format!("{}", e)),
        }),
    }
}

/// Calculate comparison metrics for a result
fn calculate_metrics(
    result: &AlgorithmResult,
    expected_segments: &[ExpectedSegment],
    debug: bool,
) -> ComparisonMetrics {
    if debug {
        println!(
            "  Found {} segments in {:.2}s",
            result.segments.len(),
            result.execution_time_ms as f64 / 1000.0
        );
        
        // Show segment details with confidence
        for (i, seg) in result.segments.iter().enumerate() {
            println!(
                "    Segment {}: {:.1}s-{:.1}s ({} episodes, confidence={:.2})",
                i, seg.start_time, seg.end_time, seg.episode_list.len(),
                seg.audio_confidence.unwrap_or(seg.confidence)
            );
        }
    }

    let (precision, recall, f1_score) =
        calculate_precision_recall(&result.segments, expected_segments, debug);

    let timing_accuracy_ms = calculate_timing_accuracy(&result.segments, expected_segments);

    ComparisonMetrics {
        algorithm: result.algorithm,
        precision,
        recall,
        f1_score,
        timing_accuracy_ms,
        execution_time_ms: result.execution_time_ms,
        segments_found: result.segments.len(),
    }
}

/// Calculate precision, recall, and F1 score
fn calculate_precision_recall(
    detected: &[CommonSegment],
    expected: &[ExpectedSegment],
    debug: bool,
) -> (f64, f64, f64) {
    if expected.is_empty() {
        return (1.0, 1.0, 1.0);
    }

    // Count true positives, false positives, false negatives
    let mut true_positives = 0;
    let mut matched_expected = vec![false; expected.len()];

    for detected_seg in detected {
        let mut matched = false;

        for (i, expected_seg) in expected.iter().enumerate() {
            if matched_expected[i] {
                continue;
            }

            // Check if detected segment overlaps with expected
            let overlap_start = detected_seg.start_time.max(expected_seg.start_time);
            let overlap_end = detected_seg.end_time.min(expected_seg.end_time);
            let overlap_duration = (overlap_end - overlap_start).max(0.0);

            let expected_duration = expected_seg.end_time - expected_seg.start_time;
            let detected_duration = detected_seg.end_time - detected_seg.start_time;
            let min_duration = expected_duration.min(detected_duration);

            // Consider it a match if overlap is > 50% of the smaller segment
            if overlap_duration > min_duration * 0.5 {
                // Also check episode count
                if detected_seg.episode_list.len() >= expected_seg.min_episodes {
                    true_positives += 1;
                    matched_expected[i] = true;
                    matched = true;
                    if debug {
                        println!(
                            "    ✓ Match: {:.1}s-{:.1}s ({} episodes)",
                            detected_seg.start_time,
                            detected_seg.end_time,
                            detected_seg.episode_list.len()
                        );
                    }
                    break;
                }
            }
        }

        if !matched && debug {
            println!(
                "    ✗ False positive: {:.1}s-{:.1}s ({} episodes)",
                detected_seg.start_time,
                detected_seg.end_time,
                detected_seg.episode_list.len()
            );
        }
    }

    let false_positives = detected.len().saturating_sub(true_positives);
    let false_negatives = matched_expected.iter().filter(|&&m| !m).count();

    let precision = if detected.is_empty() {
        if expected.is_empty() {
            1.0
        } else {
            0.0
        }
    } else {
        true_positives as f64 / detected.len() as f64
    };

    let recall = if expected.is_empty() {
        1.0
    } else {
        true_positives as f64 / expected.len() as f64
    };

    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    if debug {
        println!(
            "    Metrics: P={:.2} R={:.2} F1={:.2} (TP={} FP={} FN={})",
            precision, recall, f1_score, true_positives, false_positives, false_negatives
        );
    }

    (precision, recall, f1_score)
}

/// Calculate average timing accuracy (how close detected times are to expected)
fn calculate_timing_accuracy(
    detected: &[CommonSegment],
    expected: &[ExpectedSegment],
) -> f64 {
    let mut total_error = 0.0;
    let mut count = 0;

    for detected_seg in detected {
        // Find best matching expected segment
        let mut best_error = f64::INFINITY;

        for expected_seg in expected {
            let start_error = (detected_seg.start_time - expected_seg.start_time).abs();
            let end_error = (detected_seg.end_time - expected_seg.end_time).abs();
            let avg_error = (start_error + end_error) / 2.0;

            if avg_error < best_error {
                best_error = avg_error;
            }
        }

        if best_error < f64::INFINITY {
            total_error += best_error;
            count += 1;
        }
    }

    if count > 0 {
        (total_error / count as f64) * 1000.0 // Convert to milliseconds
    } else {
        0.0
    }
}

/// Print comparison report with segment details
pub fn print_comparison_report(metrics: &[ComparisonMetrics]) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║           Audio Algorithm Comparison Report                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!("\n{:<15} {:<10} {:<10} {:<10} {:<12} {:<12} {:<10}",
             "Algorithm", "Precision", "Recall", "F1 Score", "Time (ms)", "Timing Err", "Segments");
    println!("{}", "-".repeat(85));

    for metric in metrics {
        println!(
            "{:<15} {:<10.2} {:<10.2} {:<10.2} {:<12} {:<12.1} {:<10}",
            format!("{:?}", metric.algorithm),
            metric.precision,
            metric.recall,
            metric.f1_score,
            metric.execution_time_ms,
            metric.timing_accuracy_ms,
            metric.segments_found
        );
    }

    // Find best performers
    if let Some(best_f1) = metrics.iter().max_by(|a, b| {
        a.f1_score
            .partial_cmp(&b.f1_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!("\n🏆 Best F1 Score: {:?} ({:.2})", best_f1.algorithm, best_f1.f1_score);
    }

    if let Some(fastest) = metrics.iter().min_by_key(|m| m.execution_time_ms) {
        println!(
            "⚡ Fastest: {:?} ({}ms)",
            fastest.algorithm, fastest.execution_time_ms
        );
    }

    if let Some(best_timing) = metrics.iter().min_by(|a, b| {
        a.timing_accuracy_ms
            .partial_cmp(&b.timing_accuracy_ms)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        println!(
            "🎯 Most Accurate Timing: {:?} ({:.1}ms error)",
            best_timing.algorithm, best_timing.timing_accuracy_ms
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_recall_perfect() {
        let detected = vec![];
        let expected = vec![];

        let (precision, recall, f1) = calculate_precision_recall(&detected, &expected, false);
        assert_eq!(precision, 1.0);
        assert_eq!(recall, 1.0);
        assert_eq!(f1, 1.0);
    }

    #[test]
    fn test_timing_accuracy_empty() {
        let detected = vec![];
        let expected = vec![];

        let accuracy = calculate_timing_accuracy(&detected, &expected);
        assert_eq!(accuracy, 0.0);
    }
}

