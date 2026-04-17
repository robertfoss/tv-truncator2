//! Detection accuracy metrics for comparing [`CommonSegment`](crate::segment_detector::CommonSegment)
//! outputs against fixture expectations (used for algorithm tuning, dedup experiments, and regression tracking).
//!
//! See [`crate::accuracy_store`] for persisting runs in SQLite.

use crate::segment_detector::CommonSegment;

/// One expected segment from a test fixture (for example `segments.json`), aligned with
/// [`crate::audio_comparison::ExpectedSegment`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedFixtureSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub min_episodes: usize,
}

/// Precision / recall / F1 plus timing error for a single fixture run.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionAccuracyMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    /// Mean absolute start/end error in milliseconds, averaged over detected segments
    /// (each segment paired with its closest expected segment by mean boundary error).
    pub timing_mean_abs_error_ms: f64,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub segments_detected: usize,
}

/// Evaluate overlap-based precision/recall and mean timing error.
///
/// Matching rules match the historical audio comparison harness: a detected segment matches an
/// expected segment when overlap exceeds 50% of the smaller duration and
/// `detected.episode_list.len() >= expected.min_episodes`. Each expected segment is matched at most once.
///
/// When `expected` is empty, precision/recall/F1 are defined as `1.0` (legacy behaviour from the
/// audio comparison tool).
pub fn evaluate_detection_accuracy(
    detected: &[CommonSegment],
    expected: &[ExpectedFixtureSegment],
    debug: bool,
) -> DetectionAccuracyMetrics {
    let (precision, recall, f1_score, tp, fp, fn_) = precision_recall_f1(detected, expected, debug);
    let timing_mean_abs_error_ms = mean_abs_timing_error_ms(detected, expected);

    DetectionAccuracyMetrics {
        precision,
        recall,
        f1_score,
        timing_mean_abs_error_ms,
        true_positives: tp,
        false_positives: fp,
        false_negatives: fn_,
        segments_detected: detected.len(),
    }
}

fn precision_recall_f1(
    detected: &[CommonSegment],
    expected: &[ExpectedFixtureSegment],
    debug: bool,
) -> (f64, f64, f64, usize, usize, usize) {
    if expected.is_empty() {
        return (1.0, 1.0, 1.0, 0, detected.len(), 0);
    }

    let mut true_positives = 0usize;
    let mut matched_expected = vec![false; expected.len()];

    for detected_seg in detected {
        let mut matched = false;

        for (i, expected_seg) in expected.iter().enumerate() {
            if matched_expected[i] {
                continue;
            }

            let overlap_start = detected_seg.start_time.max(expected_seg.start_time);
            let overlap_end = detected_seg.end_time.min(expected_seg.end_time);
            let overlap_duration = (overlap_end - overlap_start).max(0.0);

            let expected_duration = expected_seg.end_time - expected_seg.start_time;
            let detected_duration = detected_seg.end_time - detected_seg.start_time;
            let min_duration = expected_duration.min(detected_duration);

            if overlap_duration > min_duration * 0.5
                && detected_seg.episode_list.len() >= expected_seg.min_episodes
            {
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
        0.0
    } else {
        true_positives as f64 / detected.len() as f64
    };

    let recall = true_positives as f64 / expected.len() as f64;

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

    (
        precision,
        recall,
        f1_score,
        true_positives,
        false_positives,
        false_negatives,
    )
}

fn mean_abs_timing_error_ms(
    detected: &[CommonSegment],
    expected: &[ExpectedFixtureSegment],
) -> f64 {
    if expected.is_empty() {
        return 0.0;
    }

    let mut total_error = 0.0;
    let mut count = 0usize;

    for detected_seg in detected {
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
        (total_error / count as f64) * 1000.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segment_detector::MatchType;

    fn seg(start: f64, end: f64, episodes: usize) -> CommonSegment {
        CommonSegment {
            start_time: start,
            end_time: end,
            episode_list: vec!["e".to_string(); episodes],
            episode_timings: None,
            confidence: 1.0,
            video_confidence: None,
            audio_confidence: None,
            match_type: MatchType::Audio,
        }
    }

    #[test]
    fn empty_expected_and_detected_is_perfect() {
        let m = evaluate_detection_accuracy(&[], &[], false);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1_score, 1.0);
        assert_eq!(m.timing_mean_abs_error_ms, 0.0);
    }

    #[test]
    fn empty_expected_legacy_perfect_scores() {
        let m = evaluate_detection_accuracy(&[seg(0.0, 1.0, 2)], &[], false);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
    }

    #[test]
    fn single_overlap_match() {
        let detected = [seg(0.0, 10.0, 2)];
        let expected = [ExpectedFixtureSegment {
            start_time: 0.0,
            end_time: 10.0,
            min_episodes: 2,
        }];
        let m = evaluate_detection_accuracy(&detected, &expected, false);
        assert_eq!(m.true_positives, 1);
        assert_eq!(m.false_positives, 0);
        assert_eq!(m.false_negatives, 0);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1_score, 1.0);
    }
}
