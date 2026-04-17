//! Unit tests for pure detection helpers: similarity scoring, adaptive thresholds,
//! algorithm comparison, and hash utilities (MEM-21 thin slice).

use std::collections::HashMap;
use std::time::Duration;

use tvt::hasher::{hamming_distance, is_similar, RollingHash};
use tvt::similarity::{
    calculate_adaptive_threshold, calculate_similarity_score, compare_algorithms, CommonSegment,
    MultiScaleHash,
};

fn sample_multi_hash(dhash: u64, phash: u64, ahash: u64, color_hash: u64) -> MultiScaleHash {
    MultiScaleHash {
        dhash,
        phash,
        ahash,
        color_hash,
    }
}

fn common_segment(start: f64, end: f64) -> CommonSegment {
    CommonSegment {
        start_time: start,
        end_time: end,
        episode_segments: vec![],
        confidence: 0.9,
        algorithm: "test".to_string(),
        similarity_scores: HashMap::new(),
    }
}

#[test]
fn identical_multi_scale_hashes_score_one() {
    let h = sample_multi_hash(
        0xAAAA_AAAA_AAAA_AAAA,
        0x5555_5555_5555_5555,
        0xFFFF_FFFF_FFFF_FFFF,
        0x0000_0000_0000_0001,
    );
    let score = calculate_similarity_score(&h, &h);
    assert!((score - 1.0).abs() < f64::EPSILON);
}

#[test]
fn orthogonal_hashes_score_zero_similarity_components() {
    let a = sample_multi_hash(0, 0, 0, 0);
    let b = sample_multi_hash(!0u64, !0u64, !0u64, !0u64);
    let score = calculate_similarity_score(&a, &b);
    assert!((score - 0.0).abs() < 1e-9);
}

#[test]
fn adaptive_threshold_empty_returns_base() {
    let base = 0.82;
    assert_eq!(calculate_adaptive_threshold(&[], base), base);
}

#[test]
fn adaptive_threshold_clamped_to_bounds() {
    let h = sample_multi_hash(1, 2, 3, 4);
    let out = calculate_adaptive_threshold(&[h.clone(), h], 0.99);
    assert!(out >= 0.5 && out <= 0.95);
}

#[test]
fn compare_algorithms_agreement_and_sides() {
    let overlap = common_segment(0.0, 10.0);
    let multi_extra = common_segment(20.0, 30.0);
    let ssim_extra = common_segment(40.0, 50.0);

    let multi = vec![overlap.clone(), multi_extra];
    let ssim = vec![overlap, ssim_extra.clone()];

    let cmp = compare_algorithms(
        multi,
        ssim,
        Duration::from_millis(12),
        Duration::from_millis(34),
    );

    assert_eq!(cmp.agreement.len(), 1);
    assert_eq!(cmp.multi_hash_only.len(), 1);
    assert_eq!(cmp.ssim_features_only.len(), 1);
    assert_eq!(cmp.metrics.agreement_count, 1);
    assert_eq!(cmp.metrics.multi_hash_count, 2);
    assert_eq!(cmp.metrics.ssim_features_count, 2);
}

#[test]
fn rolling_hash_window_emits_stable_hash_for_repeated_pattern() {
    let mut rh = RollingHash::new(3);
    assert_eq!(rh.add(5), None);
    assert_eq!(rh.add(5), None);
    let h1 = rh.add(5).expect("full window");
    let h2 = rh.add(5).expect("rolled");
    assert_eq!(h1, h2, "sliding identical windows should hash identically");
}

#[test]
fn hamming_and_is_similar_match_hasher_contract() {
    let a: u64 = 0b1010;
    let b: u64 = 0b1011;
    assert_eq!(hamming_distance(a, b), 1);
    assert!(is_similar(a, b, 1));
    assert!(!is_similar(a, b, 0));
}
