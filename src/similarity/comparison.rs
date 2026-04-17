//! Algorithm comparison framework

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use super::common::{AlgorithmComparison, CommonSegment, ComparisonMetrics};

/// Compare two algorithm results and generate comparison report
pub fn compare_algorithms(
    multi_hash_results: Vec<CommonSegment>,
    ssim_features_results: Vec<CommonSegment>,
    multi_hash_time: Duration,
    ssim_features_time: Duration,
) -> AlgorithmComparison {
    let multi_hash_count = multi_hash_results.len();
    let ssim_features_count = ssim_features_results.len();

    // Find segments that appear in both results (agreement)
    let mut agreement = Vec::new();
    let mut multi_hash_only = Vec::new();
    let mut ssim_features_only = Vec::new();

    // Create a map of ssim results for quick lookup
    let mut ssim_map = HashMap::new();
    for segment in &ssim_features_results {
        let key = format!("{:.1}-{:.1}", segment.start_time, segment.end_time);
        ssim_map.insert(key, segment.clone());
    }

    // Check multi-hash results against ssim results
    for segment in &multi_hash_results {
        let key = format!("{:.1}-{:.1}", segment.start_time, segment.end_time);
        if let Some(_ssim_segment) = ssim_map.get(&key) {
            // Found in both - add to agreement
            agreement.push(segment.clone());
        } else {
            // Only in multi-hash
            multi_hash_only.push(segment.clone());
        }
    }

    // Check ssim results that weren't found in multi-hash
    let multi_keys: HashSet<String> = multi_hash_results
        .iter()
        .map(|segment| format!("{:.1}-{:.1}", segment.start_time, segment.end_time))
        .collect();
    for segment in &ssim_features_results {
        let key = format!("{:.1}-{:.1}", segment.start_time, segment.end_time);
        if !multi_keys.contains(&key) {
            ssim_features_only.push(segment.clone());
        }
    }

    // Calculate metrics
    let multi_hash_avg_confidence = if multi_hash_count > 0 {
        multi_hash_results.iter().map(|s| s.confidence).sum::<f64>() / multi_hash_count as f64
    } else {
        0.0
    };

    let ssim_features_avg_confidence = if ssim_features_count > 0 {
        ssim_features_results
            .iter()
            .map(|s| s.confidence)
            .sum::<f64>()
            / ssim_features_count as f64
    } else {
        0.0
    };

    let metrics = ComparisonMetrics {
        multi_hash_count,
        ssim_features_count,
        agreement_count: agreement.len(),
        multi_hash_avg_confidence,
        ssim_features_avg_confidence,
        processing_time_multi_hash: multi_hash_time,
        processing_time_ssim_features: ssim_features_time,
    };

    AlgorithmComparison {
        multi_hash_results,
        ssim_features_results,
        agreement,
        multi_hash_only,
        ssim_features_only,
        metrics,
    }
}

/// Print detailed algorithm comparison report
pub fn print_algorithm_comparison(comparison: &AlgorithmComparison) {
    println!("\n=== Algorithm Comparison ===");
    println!("Multi-Hash Algorithm:");
    println!("  Segments found: {}", comparison.metrics.multi_hash_count);
    println!(
        "  Average confidence: {:.1}%",
        comparison.metrics.multi_hash_avg_confidence * 100.0
    );
    println!(
        "  Processing time: {:?}",
        comparison.metrics.processing_time_multi_hash
    );

    println!("\nSSIM+Features Algorithm:");
    println!(
        "  Segments found: {}",
        comparison.metrics.ssim_features_count
    );
    println!(
        "  Average confidence: {:.1}%",
        comparison.metrics.ssim_features_avg_confidence * 100.0
    );
    println!(
        "  Processing time: {:?}",
        comparison.metrics.processing_time_ssim_features
    );

    println!("\nAgreement:");
    println!(
        "  Segments found by both: {}",
        comparison.metrics.agreement_count
    );
    println!("  Multi-Hash only: {}", comparison.multi_hash_only.len());
    println!(
        "  SSIM+Features only: {}",
        comparison.ssim_features_only.len()
    );

    // Show segments found by each algorithm
    if !comparison.agreement.is_empty() {
        println!("\nSegments found by both algorithms:");
        for (i, segment) in comparison.agreement.iter().enumerate() {
            let duration = segment.end_time - segment.start_time;
            println!(
                "  {}. {:.1}s - {:.1}s (duration: {:.1}s, confidence: {:.1}%)",
                i + 1,
                segment.start_time,
                segment.end_time,
                duration,
                segment.confidence * 100.0
            );
        }
    }

    if !comparison.multi_hash_only.is_empty() {
        println!("\nSegments found only by Multi-Hash:");
        for (i, segment) in comparison.multi_hash_only.iter().enumerate() {
            let duration = segment.end_time - segment.start_time;
            println!(
                "  {}. {:.1}s - {:.1}s (duration: {:.1}s, confidence: {:.1}%)",
                i + 1,
                segment.start_time,
                segment.end_time,
                duration,
                segment.confidence * 100.0
            );
        }
    }

    if !comparison.ssim_features_only.is_empty() {
        println!("\nSegments found only by SSIM+Features:");
        for (i, segment) in comparison.ssim_features_only.iter().enumerate() {
            let duration = segment.end_time - segment.start_time;
            println!(
                "  {}. {:.1}s - {:.1}s (duration: {:.1}s, confidence: {:.1}%)",
                i + 1,
                segment.start_time,
                segment.end_time,
                duration,
                segment.confidence * 100.0
            );
        }
    }

    // Performance comparison
    let speed_ratio = if comparison
        .metrics
        .processing_time_ssim_features
        .as_secs_f64()
        > 0.0
    {
        comparison.metrics.processing_time_multi_hash.as_secs_f64()
            / comparison
                .metrics
                .processing_time_ssim_features
                .as_secs_f64()
    } else {
        1.0
    };

    println!("\nPerformance:");
    if speed_ratio > 1.0 {
        println!(
            "  Multi-Hash is {:.1}x faster than SSIM+Features",
            speed_ratio
        );
    } else {
        println!(
            "  SSIM+Features is {:.1}x faster than Multi-Hash",
            1.0 / speed_ratio
        );
    }

    // Accuracy comparison
    let detection_ratio = if comparison.metrics.ssim_features_count > 0 {
        comparison.metrics.multi_hash_count as f64 / comparison.metrics.ssim_features_count as f64
    } else {
        1.0
    };

    println!("\nDetection:");
    if detection_ratio > 1.0 {
        println!(
            "  Multi-Hash found {:.1}x more segments than SSIM+Features",
            detection_ratio
        );
    } else {
        println!(
            "  SSIM+Features found {:.1}x more segments than Multi-Hash",
            1.0 / detection_ratio
        );
    }

    let agreement_rate =
        if comparison.metrics.multi_hash_count + comparison.metrics.ssim_features_count > 0 {
            (comparison.metrics.agreement_count as f64 * 2.0)
                / (comparison.metrics.multi_hash_count + comparison.metrics.ssim_features_count)
                    as f64
        } else {
            0.0
        };

    println!("  Agreement rate: {:.1}%", agreement_rate * 100.0);
}
