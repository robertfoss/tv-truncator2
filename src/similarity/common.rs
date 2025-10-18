//! Common types and utilities for similarity detection

use std::collections::HashMap;
use std::time::Duration;

/// Algorithm used for similarity detection
#[derive(Debug, Clone, PartialEq, clap::ValueEnum)]
pub enum SimilarityAlgorithm {
    /// Current algorithm (existing implementation)
    Current,
    /// Multi-scale perceptual hashing
    MultiHash,
    /// SSIM + Feature matching
    SsimFeatures,
    /// Run both algorithms and compare
    Both,
}

/// Enhanced segment information with per-episode timing
#[derive(Debug, Clone)]
pub struct EpisodeSegment {
    pub episode_name: String,
    pub start_time: f64,  // Can differ per episode (temporal shift)
    pub end_time: f64,
}

/// Enhanced common segment with algorithm metadata
#[derive(Debug, Clone)]
pub struct CommonSegment {
    pub start_time: f64,
    pub end_time: f64,
    pub episode_segments: Vec<EpisodeSegment>,  // New: per-episode timing
    pub confidence: f64,
    pub algorithm: String,  // Which algorithm detected it
    pub similarity_scores: HashMap<String, f64>,  // Detailed metrics
}

/// Scene segment for pre-processing
#[derive(Debug, Clone)]
pub struct SceneSegment {
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_time: f64,
    pub end_time: f64,
    pub complexity: f64,  // Scene complexity score
}

/// Keypoint for feature matching
#[derive(Debug, Clone)]
pub struct KeyPoint {
    pub x: f32,
    pub y: f32,
    pub response: f32,
}

/// DTW alignment result
#[derive(Debug, Clone)]
pub struct DtwAlignment {
    pub video1_range: (f64, f64),
    pub video2_range: (f64, f64),
    pub similarity: f64,
}

/// Algorithm comparison metrics
#[derive(Debug, Clone)]
pub struct ComparisonMetrics {
    pub multi_hash_count: usize,
    pub ssim_features_count: usize,
    pub agreement_count: usize,
    pub multi_hash_avg_confidence: f64,
    pub ssim_features_avg_confidence: f64,
    pub processing_time_multi_hash: Duration,
    pub processing_time_ssim_features: Duration,
}

/// Algorithm comparison results
#[derive(Debug, Clone)]
pub struct AlgorithmComparison {
    pub multi_hash_results: Vec<CommonSegment>,
    pub ssim_features_results: Vec<CommonSegment>,
    pub agreement: Vec<CommonSegment>,  // Found by both
    pub multi_hash_only: Vec<CommonSegment>,
    pub ssim_features_only: Vec<CommonSegment>,
    pub metrics: ComparisonMetrics,
}
