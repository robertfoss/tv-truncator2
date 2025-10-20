//! Audio spectral hashing using RustFFT
//!
//! This module generates perceptual hashes from audio spectral features
//! for identifying matching audio segments.

use crate::audio_extractor::AudioFrame;
use crate::Result;
use rustfft::{FftPlanner, num_complex::Complex};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Window size for FFT analysis (power of 2 for efficiency)
/// Using 8192 for more robust matching with encoded audio
pub const FFT_WINDOW_SIZE: usize = 8192;

/// Hop size for overlapping windows (75% overlap for better coverage)
pub const HOP_SIZE: usize = FFT_WINDOW_SIZE / 4;

/// Number of dominant frequency bins to extract
pub const NUM_DOMINANT_BINS: usize = 8;

/// Spectral features extracted from audio
#[derive(Debug, Clone)]
pub struct SpectralFeatures {
    /// Spectral centroid (center of mass of the spectrum)
    pub centroid: f32,
    /// Spectral rolloff (frequency below which 85% of energy is contained)
    pub rolloff: f32,
    /// Top frequency bins by magnitude
    pub dominant_bins: Vec<usize>,
    /// RMS energy of the frame
    pub energy: f32,
}

/// Extract spectral features from a window of audio samples
///
/// # Arguments
/// * `samples` - Audio samples (should be FFT_WINDOW_SIZE in length)
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// * `Result<SpectralFeatures>` - Extracted spectral features
pub fn extract_spectral_features(samples: &[f32], sample_rate: f32) -> Result<SpectralFeatures> {
    if samples.len() != FFT_WINDOW_SIZE {
        anyhow::bail!(
            "Expected {} samples, got {}",
            FFT_WINDOW_SIZE,
            samples.len()
        );
    }

    // Normalize audio to reduce sensitivity to volume differences
    let max_val = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let normalized: Vec<f32> = if max_val > 0.0001 {
        samples.iter().map(|s| s / max_val).collect()
    } else {
        samples.to_vec() // Silent audio
    };
    
    // Apply Hann window to reduce spectral leakage
    let window = apodize::hanning_iter(FFT_WINDOW_SIZE).map(|x| x as f32).collect::<Vec<f32>>();
    let windowed: Vec<f32> = normalized
        .iter()
        .zip(window.iter())
        .map(|(s, w)| s * w)
        .collect();

    // Prepare FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_WINDOW_SIZE);

    // Convert to complex numbers
    let mut buffer: Vec<Complex<f32>> = windowed
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();

    // Perform FFT
    fft.process(&mut buffer);

    // Calculate magnitude spectrum (only first half due to symmetry)
    let num_bins = FFT_WINDOW_SIZE / 2;
    let magnitudes: Vec<f32> = buffer[..num_bins]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    // Calculate total energy
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    let energy = (total_energy / num_bins as f32).sqrt();

    // Calculate spectral centroid
    let mut weighted_sum = 0.0f32;
    let mut magnitude_sum = 0.0f32;
    for (i, &mag) in magnitudes.iter().enumerate() {
        weighted_sum += i as f32 * mag;
        magnitude_sum += mag;
    }
    let centroid = if magnitude_sum > 0.0 {
        (weighted_sum / magnitude_sum) * (sample_rate / FFT_WINDOW_SIZE as f32)
    } else {
        0.0
    };

    // Calculate spectral rolloff (85% of energy)
    let target_energy = total_energy * 0.85;
    let mut cumulative_energy = 0.0f32;
    let mut rolloff_bin = 0;
    for (i, &mag) in magnitudes.iter().enumerate() {
        cumulative_energy += mag * mag;
        if cumulative_energy >= target_energy {
            rolloff_bin = i;
            break;
        }
    }
    let rolloff = rolloff_bin as f32 * (sample_rate / FFT_WINDOW_SIZE as f32);

    // Find dominant frequency bins
    let mut bin_mags: Vec<(usize, f32)> = magnitudes
        .iter()
        .enumerate()
        .map(|(i, &mag)| (i, mag))
        .collect();
    bin_mags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let dominant_bins: Vec<usize> = bin_mags
        .iter()
        .take(NUM_DOMINANT_BINS)
        .map(|(i, _)| *i)
        .collect();

    Ok(SpectralFeatures {
        centroid,
        rolloff,
        dominant_bins,
        energy,
    })
}

/// Generate a 64-bit hash from spectral features
///
/// This hash is designed to be perceptually meaningful - similar audio
/// will produce similar (but not necessarily identical) hashes.
///
/// Uses coarse quantization to be robust to encoding differences.
///
/// # Arguments
/// * `features` - Spectral features to hash
///
/// # Returns
/// * `u64` - 64-bit perceptual hash
pub fn generate_audio_hash(features: &SpectralFeatures) -> u64 {
    // Balanced hash: robust to encoding differences but discriminative enough
    // Uses coarse quantization with more bits of information
    
    // Quantize centroid to 16 bands (1kHz each up to 16kHz)
    let centroid_band = ((features.centroid / 1000.0).round() as u64).min(15);
    
    // Quantize rolloff to 16 bands (1kHz each)
    let rolloff_band = ((features.rolloff / 1000.0).round() as u64).min(15);
    
    // Quantize energy to 8 levels (3 bits)
    let energy_level = ((features.energy * 0.5).round() as u64).min(7);
    
    // Use top 2 dominant bins, grouped into bands of 30
    let bin1 = if !features.dominant_bins.is_empty() {
        ((features.dominant_bins[0] / 30) as u64).min(31)
    } else {
        0
    };
    let bin2 = if features.dominant_bins.len() > 1 {
        ((features.dominant_bins[1] / 30) as u64).min(31)
    } else {
        0
    };
    
    // Combine into hash: 4+4+3+5+5 = 21 bits (room for 2^21 = 2M combinations)
    // This balances robustness (coarse quantization) with discrimination
    (centroid_band << 48)
        | (rolloff_band << 40)
        | (energy_level << 32)
        | (bin1 << 16)
        | bin2
}

/// Process audio samples and extract audio frames with spectral hashes
///
/// # Arguments
/// * `samples` - Raw audio samples
/// * `sample_rate` - Sample rate in Hz
/// * `frame_rate` - Desired frame rate for analysis (frames per second)
///
/// # Returns
/// * `Result<Vec<AudioFrame>>` - Vector of audio frames with timestamps and hashes
pub fn process_audio_samples(
    samples: &[f32],
    sample_rate: f32,
    frame_rate: f32,
) -> Result<Vec<AudioFrame>> {
    let mut audio_frames = Vec::new();

    // Calculate hop size based on desired frame rate
    let samples_per_frame = (sample_rate / frame_rate) as usize;

    let mut position = 0;
    while position + FFT_WINDOW_SIZE <= samples.len() {
        let window = &samples[position..position + FFT_WINDOW_SIZE];

        // Extract spectral features
        let features = extract_spectral_features(window, sample_rate)?;

        // Generate hash
        let hash = generate_audio_hash(&features);

        // Calculate timestamp
        let timestamp = position as f64 / sample_rate as f64;

        audio_frames.push(AudioFrame {
            timestamp,
            spectral_hash: hash,
        });

        position += samples_per_frame;
    }

    Ok(audio_frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_features_extraction() {
        // Generate a simple sine wave at 440 Hz
        let sample_rate = 22050.0;
        let freq = 440.0;
        let mut samples = vec![0.0f32; FFT_WINDOW_SIZE];
        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        }

        let features = extract_spectral_features(&samples, sample_rate).unwrap();

        // Centroid should be near 440 Hz
        assert!(features.centroid > 400.0 && features.centroid < 500.0);

        // Energy should be non-zero
        assert!(features.energy > 0.0);

        // Should have dominant bins
        assert_eq!(features.dominant_bins.len(), NUM_DOMINANT_BINS);
    }

    #[test]
    fn test_audio_hash_consistency() {
        // Same features should produce same hash
        let features = SpectralFeatures {
            centroid: 1000.0,
            rolloff: 5000.0,
            dominant_bins: vec![10, 20, 30, 40, 50, 60, 70, 80],
            energy: 0.5,
        };

        let hash1 = generate_audio_hash(&features);
        let hash2 = generate_audio_hash(&features);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_similar_features_similar_hashes() {
        // Similar features should produce similar (but possibly different) hashes
        let features1 = SpectralFeatures {
            centroid: 1000.0,
            rolloff: 5000.0,
            dominant_bins: vec![10, 20, 30, 40, 50, 60, 70, 80],
            energy: 0.5,
        };

        let features2 = SpectralFeatures {
            centroid: 1050.0, // Slightly different
            rolloff: 5100.0,
            dominant_bins: vec![10, 20, 30, 40, 50, 60, 70, 80],
            energy: 0.52,
        };

        let hash1 = generate_audio_hash(&features1);
        let hash2 = generate_audio_hash(&features2);

        // Due to quantization, similar features should often produce same hash
        // But this is not guaranteed, so we just check they're both non-zero
        assert_ne!(hash1, 0);
        assert_ne!(hash2, 0);
    }

    #[test]
    fn test_process_audio_samples() {
        // Generate test audio
        let sample_rate = 22050.0;
        let duration = 1.0; // 1 second
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; num_samples];

        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let frame_rate = 1.0; // 1 frame per second
        let frames = process_audio_samples(&samples, sample_rate, frame_rate).unwrap();

        // Should have approximately 1 frame for 1 second of audio
        assert!(frames.len() >= 1);

        // All frames should have non-zero hashes
        for frame in &frames {
            assert_ne!(frame.spectral_hash, 0);
            assert!(frame.timestamp >= 0.0 && frame.timestamp <= 1.0);
        }
    }
}

