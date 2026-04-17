//! Regression checks for [`tvt::accuracy::evaluate_detection_accuracy`] against
//! `tests/fixtures/accuracy_metric_baseline.json` (MEMA-26).

use serde::Deserialize;
use std::fs;
use tvt::accuracy::{evaluate_detection_accuracy, ExpectedFixtureSegment};
use tvt::segment_detector::{CommonSegment, MatchType};

#[derive(Debug, Deserialize)]
struct BaselineFile {
    version: u32,
    #[allow(dead_code)]
    description: String,
    cases: Vec<BaselineCase>,
}

#[derive(Debug, Deserialize)]
struct BaselineCase {
    id: String,
    detected: Vec<SegIn>,
    expected: Vec<ExpIn>,
    want_precision: f64,
    want_recall: f64,
    want_f1: f64,
    want_timing_ms: f64,
}

#[derive(Debug, Deserialize)]
struct SegIn {
    start: f64,
    end: f64,
    episodes: usize,
}

#[derive(Debug, Deserialize)]
struct ExpIn {
    start: f64,
    end: f64,
    min_episodes: usize,
}

fn to_common(s: &[SegIn]) -> Vec<CommonSegment> {
    s.iter()
        .map(|x| CommonSegment {
            start_time: x.start,
            end_time: x.end,
            episode_list: vec!["e".to_string(); x.episodes],
            episode_timings: None,
            confidence: 1.0,
            video_confidence: None,
            audio_confidence: None,
            match_type: MatchType::Audio,
        })
        .collect()
}

fn to_expected(e: &[ExpIn]) -> Vec<ExpectedFixtureSegment> {
    e.iter()
        .map(|x| ExpectedFixtureSegment {
            start_time: x.start,
            end_time: x.end,
            min_episodes: x.min_episodes,
        })
        .collect()
}

#[test]
fn baseline_json_matches_evaluator() {
    let root = env!("CARGO_MANIFEST_DIR");
    let path = format!("{root}/tests/fixtures/accuracy_metric_baseline.json");
    let raw = fs::read_to_string(&path).expect("baseline json");
    let file: BaselineFile = serde_json::from_str(&raw).expect("parse baseline");
    assert_eq!(file.version, 1);

    const EPS: f64 = 1e-9;

    for case in &file.cases {
        let det = to_common(&case.detected);
        let exp = to_expected(&case.expected);
        let m = evaluate_detection_accuracy(&det, &exp, false);

        assert!(
            (m.precision - case.want_precision).abs() < EPS,
            "case {} precision: got {} want {}",
            case.id,
            m.precision,
            case.want_precision
        );
        assert!(
            (m.recall - case.want_recall).abs() < EPS,
            "case {} recall: got {} want {}",
            case.id,
            m.recall,
            case.want_recall
        );
        assert!(
            (m.f1_score - case.want_f1).abs() < EPS,
            "case {} f1: got {} want {}",
            case.id,
            m.f1_score,
            case.want_f1
        );
        assert!(
            (m.timing_mean_abs_error_ms - case.want_timing_ms).abs() < EPS,
            "case {} timing ms: got {} want {}",
            case.id,
            m.timing_mean_abs_error_ms,
            case.want_timing_ms
        );
    }
}
