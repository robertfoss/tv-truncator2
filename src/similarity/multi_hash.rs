//! Multi-scale perceptual hashing implementation

use crate::Result;
use image::DynamicImage;

/// Multi-scale hash containing different hash types
#[derive(Debug, Clone)]
pub struct MultiScaleHash {
    pub dhash: u64,      // Difference hash
    pub phash: u64,      // Perceptual hash (existing)
    pub ahash: u64,      // Average hash
    pub color_hash: u64, // Color histogram hash
}

/// Generate multi-scale hash for an image
pub fn generate_multi_scale_hash(img: &DynamicImage) -> Result<MultiScaleHash> {
    let dhash = generate_dhash(img)?;
    let phash = generate_phash(img)?;
    let ahash = generate_ahash(img)?;
    let color_hash = generate_color_hash(img)?;

    Ok(MultiScaleHash {
        dhash,
        phash,
        ahash,
        color_hash,
    })
}

/// Calculate weighted similarity score between two multi-scale hashes
pub fn calculate_similarity_score(hash1: &MultiScaleHash, hash2: &MultiScaleHash) -> f64 {
    let dhash_sim = calculate_hamming_similarity(hash1.dhash, hash2.dhash);
    let phash_sim = calculate_hamming_similarity(hash1.phash, hash2.phash);
    let ahash_sim = calculate_hamming_similarity(hash1.ahash, hash2.ahash);
    let color_sim = calculate_hamming_similarity(hash1.color_hash, hash2.color_hash);

    // Weighted combination: dhash: 35%, phash: 30%, ahash: 20%, color: 15%
    dhash_sim * 0.35 + phash_sim * 0.30 + ahash_sim * 0.20 + color_sim * 0.15
}

/// Calculate Hamming similarity (1.0 - normalized_hamming_distance)
fn calculate_hamming_similarity(hash1: u64, hash2: u64) -> f64 {
    let hamming_dist = (hash1 ^ hash2).count_ones() as f64;
    1.0 - (hamming_dist / 64.0)
}

/// Generate difference hash (dHash)
fn generate_dhash(img: &DynamicImage) -> Result<u64> {
    // Resize to 9x8 for dHash
    let resized = img.resize_exact(9, 8, image::imageops::FilterType::Lanczos3);
    let gray = resized.to_luma8();

    let mut hash = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let left = gray.get_pixel(x, y)[0] as i32;
            let right = gray.get_pixel(x + 1, y)[0] as i32;

            if left > right {
                hash |= 1 << (y * 8 + x);
            }
        }
    }

    Ok(hash)
}

/// Generate perceptual hash (pHash) - using existing implementation
fn generate_phash(img: &DynamicImage) -> Result<u64> {
    use crate::analyzer::generate_perceptual_hash;
    generate_perceptual_hash(img)
}

/// Generate average hash (aHash)
fn generate_ahash(img: &DynamicImage) -> Result<u64> {
    // Resize to 8x8
    let resized = img.resize_exact(8, 8, image::imageops::FilterType::Lanczos3);
    let gray = resized.to_luma8();

    // Calculate average
    let mut sum = 0u32;
    for pixel in gray.pixels() {
        sum += pixel[0] as u32;
    }
    let avg = sum / 64;

    // Generate hash
    let mut hash = 0u64;
    for (i, pixel) in gray.pixels().enumerate() {
        if pixel[0] as u32 > avg {
            hash |= 1 << i;
        }
    }

    Ok(hash)
}

/// Generate color histogram hash
fn generate_color_hash(img: &DynamicImage) -> Result<u64> {
    // Resize to 8x8 for color histogram
    let resized = img.resize_exact(8, 8, image::imageops::FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Calculate color histogram (simplified)
    let mut r_sum = 0u32;
    let mut g_sum = 0u32;
    let mut b_sum = 0u32;

    for pixel in rgb.pixels() {
        r_sum += pixel[0] as u32;
        g_sum += pixel[1] as u32;
        b_sum += pixel[2] as u32;
    }

    let r_avg = r_sum / 64;
    let g_avg = g_sum / 64;
    let b_avg = b_sum / 64;

    // Create hash from dominant colors
    let mut hash = 0u64;
    for pixel in rgb.pixels() {
        hash = hash.wrapping_mul(31);
        if pixel[0] as u32 > r_avg {
            hash |= 1;
        }
        if pixel[1] as u32 > g_avg {
            hash |= 2;
        }
        if pixel[2] as u32 > b_avg {
            hash |= 4;
        }
    }

    Ok(hash)
}

/// Adaptive threshold calculation based on content
pub fn calculate_adaptive_threshold(hashes: &[MultiScaleHash], base_threshold: f64) -> f64 {
    if hashes.is_empty() {
        return base_threshold;
    }

    // Calculate content complexity metrics
    let mut complexity_scores = Vec::new();
    let mut motion_scores = Vec::new();
    let mut color_variance = Vec::new();

    for (i, hash) in hashes.iter().enumerate() {
        // Scene complexity (based on hash diversity)
        if i > 0 {
            let prev_hash = &hashes[i - 1];
            let complexity = calculate_similarity_score(hash, prev_hash);
            complexity_scores.push(1.0 - complexity);
        }

        // Motion level (temporal changes)
        if i > 0 {
            let motion = (hash.dhash ^ hashes[i - 1].dhash).count_ones() as f64 / 64.0;
            motion_scores.push(motion);
        }

        // Color variance (simplified)
        let color_variance_score = (hash.color_hash.count_ones() as f64) / 64.0;
        color_variance.push(color_variance_score);
    }

    // Calculate adaptive threshold
    let avg_complexity =
        complexity_scores.iter().sum::<f64>() / complexity_scores.len().max(1) as f64;
    let avg_motion = motion_scores.iter().sum::<f64>() / motion_scores.len().max(1) as f64;
    let avg_color_variance =
        color_variance.iter().sum::<f64>() / color_variance.len().max(1) as f64;

    // Adjust threshold based on content characteristics
    let adjustment = (avg_complexity * 0.3 + avg_motion * 0.4 + avg_color_variance * 0.3) * 0.1;
    let adaptive_threshold = base_threshold - adjustment;

    // Clamp between reasonable bounds
    adaptive_threshold.max(0.5).min(0.95)
}
