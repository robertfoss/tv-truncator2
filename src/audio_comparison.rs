//! Algorithm comparison framework for audio matching
//!
//! Runs all audio algorithms on the same test data and generates
//! detailed comparison reports with precision, recall, F1, and performance metrics.

use crate::accuracy::evaluate_detection_accuracy;
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

/// Expected segment for validation (shared with [`crate::accuracy::ExpectedFixtureSegment`]).
pub type ExpectedSegment = crate::accuracy::ExpectedFixtureSegment;

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
                i,
                seg.start_time,
                seg.end_time,
                seg.episode_list.len(),
                seg.audio_confidence.unwrap_or(seg.confidence)
            );
        }
    }

    let m = evaluate_detection_accuracy(&result.segments, expected_segments, debug);

    ComparisonMetrics {
        algorithm: result.algorithm,
        precision: m.precision,
        recall: m.recall,
        f1_score: m.f1_score,
        timing_accuracy_ms: m.timing_mean_abs_error_ms,
        execution_time_ms: result.execution_time_ms,
        segments_found: result.segments.len(),
    }
}

/// Print comparison report with segment details
pub fn print_comparison_report(metrics: &[ComparisonMetrics]) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║           Audio Algorithm Comparison Report                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    println!(
        "\n{:<15} {:<10} {:<10} {:<10} {:<12} {:<12} {:<10}",
        "Algorithm", "Precision", "Recall", "F1 Score", "Time (ms)", "Timing Err", "Segments"
    );
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
        println!(
            "\n🏆 Best F1 Score: {:?} ({:.2})",
            best_f1.algorithm, best_f1.f1_score
        );
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
