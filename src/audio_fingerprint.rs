//! Robust audio fingerprinting for encoded videos
//!
//! This module implements a fingerprinting approach based on energy patterns
//! in mel-frequency bands, which is robust to encoding differences.

use crate::audio_extractor::EpisodeAudio;
use crate::segment_detector::CommonSegment;
use crate::Result;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Number of mel-frequency bands to use
const MEL_BANDS: usize = 12;

/// FFT window size for fingerprinting
const FP_WINDOW_SIZE: usize = 4096;

/// Generate mel-frequency energy fingerprint from audio samples
///
/// Returns a vector of energy levels in mel-frequency bands
pub fn generate_mel_fingerprint(samples: &[f32], sample_rate: f32) -> Vec<u8> {
    if samples.len() < FP_WINDOW_SIZE {
        return vec![0; MEL_BANDS];
    }

    // Normalize samples
    let max_val = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let normalized: Vec<f32> = if max_val > 0.0001 {
        samples.iter().map(|s| s / max_val).collect()
    } else {
        vec![0.0; samples.len()]
    };

    // Apply window
    let window: Vec<f32> = apodize::hanning_iter(FP_WINDOW_SIZE)
        .map(|x| x as f32)
        .collect();
    let windowed: Vec<f32> = normalized[..FP_WINDOW_SIZE]
        .iter()
        .zip(window.iter())
        .map(|(s, w)| s * w)
        .collect();

    // FFT
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FP_WINDOW_SIZE);
    let mut buffer: Vec<Complex<f32>> = windowed
        .iter()
        .map(|&x| Complex { re: x, im: 0.0 })
        .collect();
    fft.process(&mut buffer);

    // Calculate magnitude spectrum
    let magnitudes: Vec<f32> = buffer[..FP_WINDOW_SIZE / 2]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();

    // Convert to mel scale and sum energy in each band
    let mel_energies = calculate_mel_band_energies(&magnitudes, sample_rate);

    // Quantize energies to 4 levels (2 bits each)
    mel_energies
        .iter()
        .map(|&energy| {
            if energy < 0.1 {
                0
            } else if energy < 0.3 {
                1
            } else if energy < 0.6 {
                2
            } else {
                3
            }
        })
        .collect()
}

/// Calculate energy in mel-frequency bands
fn calculate_mel_band_energies(magnitudes: &[f32], sample_rate: f32) -> Vec<f32> {
    let num_bins = magnitudes.len();
    let nyquist = sample_rate / 2.0;

    // Define mel band edges (in Hz)
    let mel_edges = get_mel_band_edges(MEL_BANDS, nyquist);

    let mut band_energies = vec![0.0f32; MEL_BANDS];

    for (band_idx, window) in mel_edges.windows(2).enumerate() {
        let low_freq = window[0];
        let high_freq = window[1];

        // Convert frequencies to bin indices
        let low_bin = ((low_freq / nyquist) * num_bins as f32) as usize;
        let high_bin = ((high_freq / nyquist) * num_bins as f32).min(num_bins as f32) as usize;

        // Sum energy in this band
        let mut energy = 0.0f32;
        for bin in low_bin..high_bin {
            energy += magnitudes[bin] * magnitudes[bin];
        }

        band_energies[band_idx] = energy.sqrt();
    }

    // Normalize
    let max_energy = band_energies.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_energy > 0.0 {
        for e in &mut band_energies {
            *e /= max_energy;
        }
    }

    band_energies
}

/// Get mel-frequency band edges
fn get_mel_band_edges(num_bands: usize, max_freq: f32) -> Vec<f32> {
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(max_freq);

    (0..=num_bands)
        .map(|i| {
            let mel = min_mel + (max_mel - min_mel) * i as f32 / num_bands as f32;
            mel_to_hz(mel)
        })
        .collect()
}

/// Convert Hz to mel scale
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale to Hz
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

/// Calculate fingerprint similarity (0.0 to 1.0)
pub fn fingerprint_similarity(fp1: &[u8], fp2: &[u8]) -> f64 {
    if fp1.len() != fp2.len() {
        return 0.0;
    }

    let mut matches = 0;
    for i in 0..fp1.len() {
        // Allow 1-level difference (adjacent energy levels still match)
        let diff = (fp1[i] as i32 - fp2[i] as i32).abs();
        if diff <= 1 {
            matches += 1;
        }
    }

    matches as f64 / fp1.len() as f64
}

/// Detect audio segments using mel-frequency fingerprinting
pub fn detect_audio_segments_fingerprint(
    episode_audio: &[EpisodeAudio],
    config: &crate::Config,
    debug_dupes: bool,
) -> Result<Vec<CommonSegment>> {
    if episode_audio.is_empty() {
        return Ok(Vec::new());
    }

    if debug_dupes {
        println!(
            "🎵 [Fingerprint] Detecting audio segments across {} episodes",
            episode_audio.len()
        );
    }

    // For now, use cross-correlation approach but with fingerprint similarity
    // This is a placeholder - full implementation would build a fingerprint database

    if debug_dupes {
        println!("🎵 [Fingerprint] Audio fingerprinting not yet implemented for long-form content");
        println!("  Falling back to spectral hash matching");
    }

    // Fall back to spectral hash for now
    crate::segment_detector::detect_audio_segments(episode_audio, config, debug_dupes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_conversion() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let hz_back = mel_to_hz(mel);
        assert!((hz - hz_back).abs() < 1.0);
    }

    #[test]
    fn test_fingerprint_similarity() {
        let fp1 = vec![0, 1, 2, 3, 2, 1, 0];
        let fp2 = vec![0, 1, 2, 3, 2, 1, 0];
        assert_eq!(fingerprint_similarity(&fp1, &fp2), 1.0);

        let fp3 = vec![0, 2, 2, 3, 2, 1, 0]; // One diff=1
        assert!(fingerprint_similarity(&fp1, &fp3) > 0.85);

        let fp4 = vec![3, 3, 3, 0, 0, 0, 3]; // All different
        assert!(fingerprint_similarity(&fp1, &fp4) < 0.3);
    }
}
