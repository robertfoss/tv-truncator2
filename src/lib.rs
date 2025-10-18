//! TVT (TV Truncator) - Library for analyzing and removing repetitive segments from TV episodes
//!
//! This library provides functionality to:
//! - Extract frames from video files using FFmpeg
//! - Generate perceptual hashes for frame comparison
//! - Detect common segments across multiple episodes
//! - Cut video segments while preserving synchronization

pub mod analyzer;
pub mod gstreamer_extractor;
pub mod gstreamer_extractor_optimized;
pub mod gstreamer_cutter;
pub mod hasher;
pub mod parallel;
pub mod progress_display;
pub mod segment_detector;
pub mod similarity;
pub mod state_machine;
pub mod synchronization;
pub mod video_processor;

/// Common error types used throughout the application
pub type Result<T> = anyhow::Result<T>;


/// Configuration for the TVT application
#[derive(Debug, Clone)]
pub struct Config {
    pub input_dir: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub threshold: usize,
    pub min_duration: f64,
    pub similarity: u8,
    pub similarity_threshold: f64,  // Make configurable (was hardcoded 0.9)
    pub similarity_algorithm: crate::similarity::SimilarityAlgorithm,
    pub dry_run: bool,
    pub quick: bool,
    pub verbose: bool,
    pub debug: bool,
    pub debug_dupes: bool,
    pub parallel_workers: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_dir: std::path::PathBuf::new(),
            output_dir: std::path::PathBuf::new(),
            threshold: 3,
            min_duration: 10.0,
            similarity: 90,
            similarity_threshold: 0.75,  // Default 75% similarity
            similarity_algorithm: crate::similarity::SimilarityAlgorithm::Current,
            dry_run: false,
            quick: false,
            verbose: false,
            debug: false,
            debug_dupes: false,
            parallel_workers: num_cpus::get().saturating_sub(1).max(1),
        }
    }
}
