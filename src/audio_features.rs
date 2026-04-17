//! Core audio feature extraction for robust matching
//!
//! This module provides multiple feature extraction methods optimized for
//! detecting recurring audio segments (openings/endings) in encoded video.

use crate::Result;
use rustfft::{num_complex::Complex, FftPlanner};

/// Chromaprint-style landmark fingerprint
#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    /// Hash combining frequency pair and time delta
    pub hash: u32,
    /// Timestamp of the landmark in seconds
    pub timestamp: f64,
}

/// MFCC feature vector for a single frame
#[derive(Debug, Clone)]
pub struct MfccFeatures {
    /// 13 MFCC coefficients
    pub coefficients: Vec<f32>,
    /// Delta coefficients (first derivative)
    pub deltas: Vec<f32>,
    /// Delta-delta coefficients (second derivative)
    pub delta_deltas: Vec<f32>,
    /// Timestamp of this frame
    pub timestamp: f64,
}

/// Improved spectral hash with proper quantization
#[derive(Debug, Clone, Copy)]
pub struct SpectralHashV2 {
    /// 64-bit hash with meaningful quantization
    pub hash: u64,
    /// Timestamp of this frame
    pub timestamp: f64,
}

/// Energy band features for pattern matching
#[derive(Debug, Clone)]
pub struct EnergyBands {
    /// Energy levels in mel-frequency bands (8-12 bands)
    pub bands: Vec<f32>,
    /// Overall RMS energy
    pub rms_energy: f32,
    /// Spectral flux (change from previous frame)
    pub flux: f32,
    /// Timestamp of this frame
    pub timestamp: f64,
}

// Constants for chromaprint-style extraction
const CHROMAPRINT_WINDOW: usize = 4096;
const CHROMAPRINT_HOP: usize = CHROMAPRINT_WINDOW / 2;
const NUM_MEL_BANDS_CHROMAPRINT: usize = 12;

// Constants for MFCC extraction
const MFCC_WINDOW: usize = 2048;
const MFCC_HOP: usize = 512;
const NUM_MFCC_COEFFICIENTS: usize = 13;
const NUM_MEL_BANDS_MFCC: usize = 40;

// Constants for spectral hash v2
const SPECTRAL_V2_WINDOW: usize = 8192;
const SPECTRAL_V2_HOP: usize = SPECTRAL_V2_WINDOW / 4;
#[allow(dead_code)]
const NUM_FREQ_BANDS_V2: usize = 8;

// Constants for energy bands
const ENERGY_WINDOW: usize = 4096;
const ENERGY_HOP: usize = 1024;
const NUM_ENERGY_BANDS: usize = 12;

/// Extract chromaprint-style landmarks from audio samples
///
/// Uses spectral peak detection and landmark pairing to create
/// robust fingerprints resistant to encoding and volume changes.
pub fn extract_chromaprint_landmarks(samples: &[f32], sample_rate: f32) -> Result<Vec<Landmark>> {
    let mut landmarks = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(CHROMAPRINT_WINDOW);

    let mut position = 0;
    let mut prev_peaks: Vec<(usize, f32)> = Vec::new();

    while position + CHROMAPRINT_WINDOW <= samples.len() {
        let window_samples = &samples[position..position + CHROMAPRINT_WINDOW];
        let timestamp = position as f64 / sample_rate as f64;

        // Apply Hann window
        let window: Vec<f32> = apodize::hanning_iter(CHROMAPRINT_WINDOW)
            .map(|x| x as f32)
            .collect();
        let windowed: Vec<f32> = window_samples
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let mut buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft.process(&mut buffer);

        // Calculate magnitude spectrum
        let magnitudes: Vec<f32> = buffer[..CHROMAPRINT_WINDOW / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Convert to mel bands
        let mel_energies = compute_mel_bands(&magnitudes, sample_rate, NUM_MEL_BANDS_CHROMAPRINT);

        // Find peaks in mel bands
        let peaks = find_spectral_peaks(&mel_energies);

        // Create landmarks from peak pairs
        if !prev_peaks.is_empty() && !peaks.is_empty() {
            for (band1, energy1) in &prev_peaks {
                for (band2, energy2) in &peaks {
                    // Include energy information in hash for better distinctiveness
                    // Quantize energies to 4 levels (2 bits each)
                    let e1_quant = if *energy1 < 0.25 {
                        0
                    } else if *energy1 < 0.5 {
                        1
                    } else if *energy1 < 0.75 {
                        2
                    } else {
                        3
                    };
                    let e2_quant = if *energy2 < 0.25 {
                        0
                    } else if *energy2 < 0.5 {
                        1
                    } else if *energy2 < 0.75 {
                        2
                    } else {
                        3
                    };

                    // Hash: band1 (4 bits) | energy1 (2 bits) | band2 (4 bits) | energy2 (2 bits)
                    let hash = ((*band1 as u32) << 8)
                        | (e1_quant << 6)
                        | ((*band2 as u32) << 2)
                        | e2_quant;

                    landmarks.push(Landmark { hash, timestamp });
                }
            }
        }

        prev_peaks = peaks;
        position += CHROMAPRINT_HOP;
    }

    Ok(landmarks)
}

/// Extract MFCC features from audio samples
///
/// Returns 13 MFCCs plus delta and delta-delta coefficients
/// for robust audio characterization.
pub fn extract_mfcc_features(samples: &[f32], sample_rate: f32) -> Result<Vec<MfccFeatures>> {
    let mut features = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(MFCC_WINDOW);

    let mut position = 0;
    let mut mel_spectra = Vec::new();

    // First pass: extract mel spectra
    while position + MFCC_WINDOW <= samples.len() {
        let window_samples = &samples[position..position + MFCC_WINDOW];
        let timestamp = position as f64 / sample_rate as f64;

        // Apply Hann window
        let window: Vec<f32> = apodize::hanning_iter(MFCC_WINDOW)
            .map(|x| x as f32)
            .collect();
        let windowed: Vec<f32> = window_samples
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let mut buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft.process(&mut buffer);

        // Magnitude spectrum
        let magnitudes: Vec<f32> = buffer[..MFCC_WINDOW / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Mel filterbank
        let mel_energies = compute_mel_bands(&magnitudes, sample_rate, NUM_MEL_BANDS_MFCC);

        // DCT to get MFCCs
        let mfccs = compute_dct(&mel_energies, NUM_MFCC_COEFFICIENTS);

        mel_spectra.push((timestamp, mfccs));
        position += MFCC_HOP;
    }

    // Second pass: compute deltas
    for i in 0..mel_spectra.len() {
        let (timestamp, coeffs) = &mel_spectra[i];

        // Compute deltas using neighboring frames
        let deltas = compute_deltas(&mel_spectra, i);
        let delta_deltas = compute_delta_deltas(&mel_spectra, i);

        features.push(MfccFeatures {
            coefficients: coeffs.clone(),
            deltas,
            delta_deltas,
            timestamp: *timestamp,
        });
    }

    Ok(features)
}

/// Extract improved spectral hash v2
///
/// Uses proper 64-bit quantization with multiple frequency bands
/// to create distinctive yet robust hashes.
pub fn extract_spectral_hash_v2(samples: &[f32], sample_rate: f32) -> Result<Vec<SpectralHashV2>> {
    let mut hashes = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(SPECTRAL_V2_WINDOW);

    let mut position = 0;

    while position + SPECTRAL_V2_WINDOW <= samples.len() {
        let window_samples = &samples[position..position + SPECTRAL_V2_WINDOW];
        let timestamp = position as f64 / sample_rate as f64;

        // Normalize
        let max_val = window_samples
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, f32::max);
        let normalized: Vec<f32> = if max_val > 0.0001 {
            window_samples.iter().map(|s| s / max_val).collect()
        } else {
            window_samples.to_vec()
        };

        // Apply Hann window
        let window: Vec<f32> = apodize::hanning_iter(SPECTRAL_V2_WINDOW)
            .map(|x| x as f32)
            .collect();
        let windowed: Vec<f32> = normalized
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let mut buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft.process(&mut buffer);

        // Magnitude spectrum
        let magnitudes: Vec<f32> = buffer[..SPECTRAL_V2_WINDOW / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Generate hash from multiple frequency bands
        let hash = generate_spectral_hash_v2(&magnitudes, sample_rate);

        hashes.push(SpectralHashV2 { hash, timestamp });
        position += SPECTRAL_V2_HOP;
    }

    Ok(hashes)
}

/// Extract energy band features
///
/// Simple but effective for detecting theme songs and recurring patterns.
pub fn extract_energy_bands(samples: &[f32], sample_rate: f32) -> Result<Vec<EnergyBands>> {
    let mut features = Vec::new();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(ENERGY_WINDOW);

    let mut position = 0;
    let mut prev_bands: Option<Vec<f32>> = None;

    while position + ENERGY_WINDOW <= samples.len() {
        let window_samples = &samples[position..position + ENERGY_WINDOW];
        let timestamp = position as f64 / sample_rate as f64;

        // Apply Hann window
        let window: Vec<f32> = apodize::hanning_iter(ENERGY_WINDOW)
            .map(|x| x as f32)
            .collect();
        let windowed: Vec<f32> = window_samples
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // FFT
        let mut buffer: Vec<Complex<f32>> = windowed
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();
        fft.process(&mut buffer);

        // Magnitude spectrum
        let magnitudes: Vec<f32> = buffer[..ENERGY_WINDOW / 2]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        // Compute energy in mel bands
        let bands = compute_mel_bands(&magnitudes, sample_rate, NUM_ENERGY_BANDS);

        // Compute RMS energy
        let rms_energy = (bands.iter().map(|b| b * b).sum::<f32>() / bands.len() as f32).sqrt();

        // Compute spectral flux (change from previous frame)
        let flux = if let Some(ref prev) = prev_bands {
            bands
                .iter()
                .zip(prev.iter())
                .map(|(curr, prev)| (curr - prev).abs())
                .sum::<f32>()
                / bands.len() as f32
        } else {
            0.0
        };

        features.push(EnergyBands {
            bands: bands.clone(),
            rms_energy,
            flux,
            timestamp,
        });

        prev_bands = Some(bands);
        position += ENERGY_HOP;
    }

    Ok(features)
}

/// Compute mel-frequency bands from magnitude spectrum
fn compute_mel_bands(magnitudes: &[f32], sample_rate: f32, num_bands: usize) -> Vec<f32> {
    let nyquist = sample_rate / 2.0;
    let mel_edges = get_mel_band_edges(num_bands, nyquist);
    let num_bins = magnitudes.len();

    let mut bands = vec![0.0f32; num_bands];

    for (band_idx, window) in mel_edges.windows(2).enumerate() {
        let low_freq = window[0];
        let high_freq = window[1];

        let low_bin = ((low_freq / nyquist) * num_bins as f32) as usize;
        let high_bin = ((high_freq / nyquist) * num_bins as f32).min(num_bins as f32) as usize;

        let mut energy = 0.0f32;
        for bin in low_bin..high_bin {
            energy += magnitudes[bin] * magnitudes[bin];
        }

        bands[band_idx] = if energy > 0.0 {
            (energy / (high_bin - low_bin) as f32).sqrt()
        } else {
            0.0
        };
    }

    // Normalize
    let max_energy = bands.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_energy > 0.0 {
        for band in &mut bands {
            *band /= max_energy;
        }
    }

    bands
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

/// Find spectral peaks in mel bands
fn find_spectral_peaks(bands: &[f32]) -> Vec<(usize, f32)> {
    let mut peaks = Vec::new();

    // Find max energy to set adaptive threshold
    let max_energy = bands.iter().fold(0.0f32, |a, &b| a.max(b));
    let threshold = max_energy * 0.2; // 20% of max energy

    for (i, &energy) in bands.iter().enumerate() {
        if energy < threshold {
            continue;
        }

        // Check if this is a local maximum
        let is_peak =
            (i == 0 || energy > bands[i - 1]) && (i == bands.len() - 1 || energy > bands[i + 1]);

        if is_peak {
            peaks.push((i, energy));
        }
    }

    // Sort by energy and keep top peaks
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    peaks.truncate(8); // Keep top 8 peaks per frame for better coverage

    peaks
}

/// Compute DCT (Discrete Cosine Transform) for MFCC
fn compute_dct(input: &[f32], num_coeffs: usize) -> Vec<f32> {
    let n = input.len();
    let mut coeffs = vec![0.0f32; num_coeffs];

    for k in 0..num_coeffs {
        let mut sum = 0.0f32;
        for (i, &val) in input.iter().enumerate() {
            let log_val = if val > 1e-10 { val.ln() } else { -23.0 }; // log floor
            sum +=
                log_val * ((std::f32::consts::PI * k as f32 * (i as f32 + 0.5)) / n as f32).cos();
        }
        coeffs[k] = sum;
    }

    coeffs
}

/// Compute delta coefficients (first derivative)
fn compute_deltas(spectra: &[(f64, Vec<f32>)], index: usize) -> Vec<f32> {
    let num_coeffs = spectra[index].1.len();
    let mut deltas = vec![0.0f32; num_coeffs];

    let window = 2; // Use ±2 frames for delta computation

    for i in 0..num_coeffs {
        let mut sum = 0.0f32;
        let mut weight_sum = 0.0f32;

        for offset in -window..=window {
            let idx = (index as i32 + offset).max(0).min(spectra.len() as i32 - 1) as usize;
            let weight = offset as f32;
            sum += weight * spectra[idx].1[i];
            weight_sum += weight.abs();
        }

        deltas[i] = if weight_sum > 0.0 {
            sum / weight_sum
        } else {
            0.0
        };
    }

    deltas
}

/// Compute delta-delta coefficients (second derivative)
fn compute_delta_deltas(spectra: &[(f64, Vec<f32>)], index: usize) -> Vec<f32> {
    // Compute deltas for neighboring frames and then delta of deltas
    let num_coeffs = spectra[index].1.len();
    let mut delta_deltas = vec![0.0f32; num_coeffs];

    let window = 1;
    for i in 0..num_coeffs {
        let mut sum = 0.0f32;
        let mut count = 0;

        for offset in -window..=window {
            let idx = (index as i32 + offset).max(0).min(spectra.len() as i32 - 1) as usize;
            let deltas = compute_deltas(spectra, idx);
            sum += deltas[i];
            count += 1;
        }

        delta_deltas[i] = sum / count as f32;
    }

    delta_deltas
}

/// Generate improved 64-bit spectral hash
fn generate_spectral_hash_v2(magnitudes: &[f32], sample_rate: f32) -> u64 {
    // Divide spectrum into frequency bands and quantize energy in each
    let num_bins = magnitudes.len();
    let nyquist = sample_rate / 2.0;

    let mut hash = 0u64;

    // Define frequency bands (Hz)
    let band_edges = vec![
        0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, nyquist,
    ];

    for i in 0..band_edges.len() - 1 {
        let low_freq = band_edges[i];
        let high_freq = band_edges[i + 1];

        let low_bin = ((low_freq / nyquist) * num_bins as f32) as usize;
        let high_bin = ((high_freq / nyquist) * num_bins as f32).min(num_bins as f32) as usize;

        // Compute average energy in this band
        let mut energy = 0.0f32;
        for bin in low_bin..high_bin {
            energy += magnitudes[bin];
        }
        energy /= (high_bin - low_bin).max(1) as f32;

        // Quantize energy into 8 levels (3 bits per band)
        let quantized = if energy < 0.01 {
            0u64
        } else if energy < 0.05 {
            1u64
        } else if energy < 0.1 {
            2u64
        } else if energy < 0.2 {
            3u64
        } else if energy < 0.4 {
            4u64
        } else if energy < 0.6 {
            5u64
        } else if energy < 0.8 {
            6u64
        } else {
            7u64
        };

        // Pack into hash (3 bits per band, 8 bands = 24 bits)
        hash |= quantized << (i * 3);
    }

    // Add spectral shape information in remaining bits
    let centroid = compute_spectral_centroid(magnitudes, sample_rate);
    let rolloff = compute_spectral_rolloff(magnitudes);

    // Quantize centroid (0-8kHz) into 16 levels (4 bits)
    let centroid_quantized = ((centroid / 8000.0) * 15.0).min(15.0) as u64;
    hash |= centroid_quantized << 24;

    // Quantize rolloff (0-1) into 16 levels (4 bits)
    let rolloff_quantized = (rolloff * 15.0).min(15.0) as u64;
    hash |= rolloff_quantized << 28;

    hash
}

/// Compute spectral centroid
fn compute_spectral_centroid(magnitudes: &[f32], sample_rate: f32) -> f32 {
    let num_bins = magnitudes.len();
    let mut weighted_sum = 0.0f32;
    let mut magnitude_sum = 0.0f32;

    for (i, &mag) in magnitudes.iter().enumerate() {
        weighted_sum += i as f32 * mag;
        magnitude_sum += mag;
    }

    if magnitude_sum > 0.0 {
        (weighted_sum / magnitude_sum) * (sample_rate / (2.0 * num_bins as f32))
    } else {
        0.0
    }
}

/// Compute spectral rolloff (85% energy point)
fn compute_spectral_rolloff(magnitudes: &[f32]) -> f32 {
    let total_energy: f32 = magnitudes.iter().map(|m| m * m).sum();
    let target_energy = total_energy * 0.85;

    let mut cumulative_energy = 0.0f32;
    for (i, &mag) in magnitudes.iter().enumerate() {
        cumulative_energy += mag * mag;
        if cumulative_energy >= target_energy {
            return i as f32 / magnitudes.len() as f32;
        }
    }

    1.0
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
    fn test_chromaprint_extraction() {
        // Generate test audio (1 second sine wave)
        let sample_rate = 22050.0;
        let duration = 1.0;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; num_samples];

        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let landmarks = extract_chromaprint_landmarks(&samples, sample_rate).unwrap();
        assert!(!landmarks.is_empty());
    }

    #[test]
    fn test_mfcc_extraction() {
        let sample_rate = 22050.0;
        let duration = 1.0;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; num_samples];

        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let mfccs = extract_mfcc_features(&samples, sample_rate).unwrap();
        assert!(!mfccs.is_empty());
        assert_eq!(mfccs[0].coefficients.len(), NUM_MFCC_COEFFICIENTS);
    }

    #[test]
    fn test_spectral_hash_v2_extraction() {
        let sample_rate = 22050.0;
        let duration = 1.0;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; num_samples];

        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let hashes = extract_spectral_hash_v2(&samples, sample_rate).unwrap();
        assert!(!hashes.is_empty());
        assert_ne!(hashes[0].hash, 0);
    }

    #[test]
    fn test_energy_bands_extraction() {
        let sample_rate = 22050.0;
        let duration = 1.0;
        let num_samples = (sample_rate * duration) as usize;
        let mut samples = vec![0.0f32; num_samples];

        for (i, sample) in samples.iter_mut().enumerate() {
            let t = i as f32 / sample_rate;
            *sample = (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let features = extract_energy_bands(&samples, sample_rate).unwrap();
        assert!(!features.is_empty());
        assert_eq!(features[0].bands.len(), NUM_ENERGY_BANDS);
    }
}
