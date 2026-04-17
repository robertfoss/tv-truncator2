//! TVT (TV Truncator) - Library for analyzing and removing repetitive segments from TV episodes
//!
//! This library provides functionality to:
//! - Extract frames and audio via GStreamer (`gstreamer_extractor_v2`, `audio_extractor`)
//! - Generate perceptual hashes for frame comparison
//! - Detect common segments across multiple episodes
//! - Cut video segments while preserving synchronization

pub mod analyzer;
pub mod audio_chromaprint;
pub mod audio_comparison;
pub mod audio_correlation;
pub mod audio_energy_bands;
pub mod audio_extractor;
pub mod audio_features;
pub mod audio_fingerprint;
pub mod audio_hasher;
pub mod audio_mfcc;
pub mod audio_segment_utils;
pub mod audio_spectral_v2;
pub mod gstreamer_cutter;
pub mod gstreamer_extractor_v2;
pub(crate) mod hamming_bk_tree;
pub mod hasher;
pub mod input_discovery;
pub mod parallel;
pub mod progress_display;
pub mod segment_detector;
pub mod similarity;
pub mod state_machine;
pub mod synchronization;
pub mod video_processor;

/// Common error types used throughout the application
pub type Result<T> = anyhow::Result<T>;

pub use input_discovery::discover_video_files;

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
    /// Chromaprint-style landmark fingerprinting (robust, time-shift capable)
    Chromaprint,
    /// MFCC-based matching with DTW (robust to encoding, good for speech/music)
    Mfcc,
    /// Improved spectral hash v2 (balanced speed/accuracy)
    SpectralV2,
    /// Energy band pattern matching (simple, effective for theme songs)
    EnergyBands,

    // Legacy algorithms (deprecated, kept for compatibility)
    /// Legacy spectral hash matching (deprecated, use SpectralV2)
    #[value(name = "spectral-hash")]
    SpectralHash,
    /// Legacy cross-correlation (deprecated, use Chromaprint)
    #[value(name = "cross-correlation")]
    CrossCorrelation,
    /// Legacy fingerprint (deprecated, use Chromaprint)
    #[value(name = "fingerprint")]
    Fingerprint,
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
    pub audio_only: bool,            // Only detect audio segments (skip video)
    /// Suppress decorative stdout and progress bars (for scripting).
    pub quiet: bool,
    /// Emit a single JSON object with run summary on stdout at end (implies quiet output).
    pub json_summary: bool,
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
            audio_algorithm: AudioAlgorithm::Chromaprint, // Most robust for encoded audio
            dry_run: false,
            quick: false,
            verbose: false,
            debug: false,
            debug_dupes: false,
            parallel_workers: num_cpus::get().saturating_sub(1).max(1),
            enable_audio_matching: true, // Audio matching always enabled
            audio_only: false,           // Video matching enabled by default
            quiet: false,
            json_summary: false,
        }
    }
}

/// Minimum episodes that must share a segment for it to be reported ([`Config::threshold`]),
/// capped so it never exceeds discovered input file count (otherwise detection is impossible).
pub fn effective_episode_threshold(requested: usize, input_file_count: usize) -> usize {
    if input_file_count == 0 {
        return requested;
    }
    requested.min(input_file_count).max(1)
}

#[cfg(test)]
mod effective_threshold_tests {
    use super::effective_episode_threshold;

    #[test]
    fn caps_when_impossible_otherwise() {
        assert_eq!(effective_episode_threshold(3, 2), 2);
        assert_eq!(effective_episode_threshold(10, 3), 3);
    }

    #[test]
    fn preserves_when_request_is_satisfiable() {
        assert_eq!(effective_episode_threshold(3, 5), 3);
        assert_eq!(effective_episode_threshold(2, 5), 2);
    }

    #[test]
    fn never_below_one_when_inputs_exist() {
        assert_eq!(effective_episode_threshold(0, 3), 1);
    }

    #[test]
    fn zero_inputs_leaves_requested_unchanged() {
        assert_eq!(effective_episode_threshold(3, 0), 3);
    }
}
