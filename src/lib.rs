//! TVT (TV Truncator) - Library for analyzing and removing repetitive segments from TV episodes
//!
//! This library provides functionality to:
//! - Extract frames from video files using FFmpeg
//! - Generate perceptual hashes for frame comparison
//! - Detect common segments across multiple episodes
//! - Cut video segments while preserving synchronization

pub mod analyzer;
pub mod audio_correlation;
pub mod audio_extractor;
pub mod audio_hasher;
pub mod gstreamer_cutter;
pub mod gstreamer_extractor_v2;
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

/// Format time in seconds to mm:ss.s format
/// Minutes only shown if >= 60s
/// One decimal place for seconds
pub fn format_time(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.1}s", seconds)
    } else {
        let minutes = (seconds / 60.0).floor() as u32;
        let remaining_secs = seconds - (minutes as f64 * 60.0);
        format!("{}:{:04.1}", minutes, remaining_secs)
    }
}

/// Audio matching algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AudioAlgorithm {
    /// Spectral hash matching (fast, exact matches)
    SpectralHash,
    /// Cross-correlation matching (robust to phase shifts and encoding)
    CrossCorrelation,
}

/// Configuration for the TVT application
#[derive(Debug, Clone)]
pub struct Config {
    pub input_dir: std::path::PathBuf,
    pub output_dir: std::path::PathBuf,
    pub threshold: usize,
    pub min_duration: f64,
    pub similarity: u8,
    pub similarity_threshold: f64, // Make configurable (was hardcoded 0.9)
    pub similarity_algorithm: crate::similarity::SimilarityAlgorithm,
    pub audio_algorithm: AudioAlgorithm, // Audio matching algorithm
    pub dry_run: bool,
    pub quick: bool,
    pub verbose: bool,
    pub debug: bool,
    pub debug_dupes: bool,
    pub parallel_workers: usize,
    pub enable_audio_matching: bool, // Enable audio segment detection
    pub audio_only: bool,             // Only detect audio segments (skip video)
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_dir: std::path::PathBuf::new(),
            output_dir: std::path::PathBuf::new(),
            threshold: 3,
            min_duration: 10.0,
            similarity: 90,
            similarity_threshold: 0.75, // Default 75% similarity
            similarity_algorithm: crate::similarity::SimilarityAlgorithm::Current,
            audio_algorithm: AudioAlgorithm::CrossCorrelation, // More robust for encoded audio
            dry_run: false,
            quick: false,
            verbose: false,
            debug: false,
            debug_dupes: false,
            parallel_workers: num_cpus::get().saturating_sub(1).max(1),
            enable_audio_matching: true, // Audio matching always enabled
            audio_only: false,            // Video matching enabled by default
        }
    }
}
