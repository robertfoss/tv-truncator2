//! SSIM + Feature matching implementation

use crate::Result;
use image::DynamicImage;
use super::common::{KeyPoint, DtwAlignment, SceneSegment};

/// Frame features for SSIM and feature matching
#[derive(Debug, Clone)]
pub struct FrameFeatures {
    pub ssim_signature: Vec<f64>,  // SSIM-based features
    pub keypoints: Vec<KeyPoint>,   // AKAZE keypoints
    pub descriptors: Vec<f32>,      // Feature descriptors
    pub timestamp: f64,
}


/// Extract features from a frame
pub fn extract_frame_features(img: &DynamicImage, timestamp: f64) -> Result<FrameFeatures> {
    let ssim_signature = extract_ssim_features(img)?;
    let (keypoints, descriptors) = extract_akaze_features(img)?;
    
    Ok(FrameFeatures {
        ssim_signature,
        keypoints,
        descriptors,
        timestamp,
    })
}

/// Extract SSIM-based features
fn extract_ssim_features(img: &DynamicImage) -> Result<Vec<f64>> {
    // Convert to grayscale and resize for SSIM calculation
    let gray = img.to_luma8();
    let resized = image::imageops::resize(&gray, 64, 64, image::imageops::FilterType::Lanczos3);
    
    // Calculate local SSIM features (simplified)
    let mut features = Vec::new();
    
    // Divide into 8x8 blocks and calculate local statistics
    for y in 0..8 {
        for x in 0..8 {
            let block_start_x = x * 8;
            let block_start_y = y * 8;
            
            let mut block_pixels = Vec::new();
            for by in 0..8 {
                for bx in 0..8 {
                    let pixel = resized.get_pixel(block_start_x + bx, block_start_y + by)[0] as f64;
                    block_pixels.push(pixel);
                }
            }
            
            // Calculate mean and variance for this block
            let mean = block_pixels.iter().sum::<f64>() / 64.0;
            let variance = block_pixels.iter()
                .map(|p| (p - mean).powi(2))
                .sum::<f64>() / 64.0;
            
            features.push(mean / 255.0);  // Normalize
            features.push(variance / (255.0 * 255.0));  // Normalize
        }
    }
    
    Ok(features)
}

/// Extract AKAZE features (simplified implementation)
fn extract_akaze_features(img: &DynamicImage) -> Result<(Vec<KeyPoint>, Vec<f32>)> {
    // For now, use a simplified feature extraction
    // In a full implementation, you would use the AKAZE crate
    
    let gray = img.to_luma8();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();
    
    // Simple corner detection (Harris-like)
    let width = gray.width() as usize;
    let height = gray.height() as usize;
    
    // Sample keypoints on a grid
    let step = 32; // Sample every 32 pixels
    for y in (step..height-step).step_by(step) {
        for x in (step..width-step).step_by(step) {
            // Calculate local gradient
            let gx = calculate_gradient_x(&gray, x, y);
            let gy = calculate_gradient_y(&gray, x, y);
            let response = (gx * gx + gy * gy).sqrt();
            
            if response > 10.0 {  // Threshold for corner strength
                keypoints.push(KeyPoint {
                    x: x as f32,
                    y: y as f32,
                    response: response as f32,
                });
                
                // Simple descriptor (local patch)
                let mut descriptor = Vec::new();
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        let px = (x as i32 + dx).max(0).min(width as i32 - 1) as usize;
                        let py = (y as i32 + dy).max(0).min(height as i32 - 1) as usize;
                        descriptor.push(gray.get_pixel(px as u32, py as u32)[0] as f32 / 255.0);
                    }
                }
                descriptors.extend(descriptor);
            }
        }
    }
    
    Ok((keypoints, descriptors))
}

/// Calculate horizontal gradient
fn calculate_gradient_x(gray: &image::GrayImage, x: usize, y: usize) -> f64 {
    let _width = gray.width() as usize;
    let _height = gray.height() as usize;
    
    if x == 0 || x >= _width - 1 {
        return 0.0;
    }
    
    let left = gray.get_pixel((x - 1) as u32, y as u32)[0] as f64;
    let right = gray.get_pixel((x + 1) as u32, y as u32)[0] as f64;
    
    right - left
}

/// Calculate vertical gradient
fn calculate_gradient_y(gray: &image::GrayImage, x: usize, y: usize) -> f64 {
    let _width = gray.width() as usize;
    let _height = gray.height() as usize;
    
    if y == 0 || y >= _height - 1 {
        return 0.0;
    }
    
    let top = gray.get_pixel(x as u32, (y - 1) as u32)[0] as f64;
    let bottom = gray.get_pixel(x as u32, (y + 1) as u32)[0] as f64;
    
    bottom - top
}

/// Compute SSIM between two images
pub fn compute_ssim(img1: &DynamicImage, img2: &DynamicImage) -> f64 {
    // Convert to grayscale and resize to same size
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();
    
    let (width, height) = gray1.dimensions();
    let (width2, height2) = gray2.dimensions();
    
    let target_width = width.min(width2);
    let target_height = height.min(height2);
    
    let resized1 = image::imageops::resize(&gray1, target_width, target_height, image::imageops::FilterType::Lanczos3);
    let resized2 = image::imageops::resize(&gray2, target_width, target_height, image::imageops::FilterType::Lanczos3);
    
    // Calculate SSIM (simplified version)
    let mut sum1 = 0.0;
    let mut sum2 = 0.0;
    let mut sum1_sq = 0.0;
    let mut sum2_sq = 0.0;
    let mut sum12 = 0.0;
    
    for y in 0..target_height {
        for x in 0..target_width {
            let p1 = resized1.get_pixel(x, y)[0] as f64;
            let p2 = resized2.get_pixel(x, y)[0] as f64;
            
            sum1 += p1;
            sum2 += p2;
            sum1_sq += p1 * p1;
            sum2_sq += p2 * p2;
            sum12 += p1 * p2;
        }
    }
    
    let n = (target_width * target_height) as f64;
    let mu1 = sum1 / n;
    let mu2 = sum2 / n;
    let sigma1_sq = (sum1_sq / n) - (mu1 * mu1);
    let sigma2_sq = (sum2_sq / n) - (mu2 * mu2);
    let sigma12 = (sum12 / n) - (mu1 * mu2);
    
    let c1 = 0.01 * 255.0 * 255.0;  // Constants for SSIM
    let c2 = 0.03 * 255.0 * 255.0;
    
    let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
    let denominator = (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1_sq + sigma2_sq + c2);
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Match features between two frame feature sets
pub fn match_features(feat1: &FrameFeatures, feat2: &FrameFeatures) -> f64 {
    if feat1.keypoints.is_empty() || feat2.keypoints.is_empty() {
        return 0.0;
    }
    
    // Simple feature matching based on descriptor similarity
    let mut matches = 0;
    let mut total_similarity = 0.0;
    
    for (i, _kp1) in feat1.keypoints.iter().enumerate() {
        let mut best_match = None;
        let mut best_similarity = 0.0;
        
        for (j, _kp2) in feat2.keypoints.iter().enumerate() {
            // Calculate descriptor similarity (simplified)
            let desc1_start = i * 25;  // 5x5 descriptor
            let desc2_start = j * 25;
            
            if desc1_start + 25 <= feat1.descriptors.len() && 
               desc2_start + 25 <= feat2.descriptors.len() {
                
                let similarity = calculate_descriptor_similarity(
                    &feat1.descriptors[desc1_start..desc1_start + 25],
                    &feat2.descriptors[desc2_start..desc2_start + 25],
                );
                
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_match = Some(j);
                }
            }
        }
        
        if let Some(_) = best_match {
            matches += 1;
            total_similarity += best_similarity;
        }
    }
    
    if matches == 0 {
        0.0
    } else {
        let match_ratio = matches as f64 / feat1.keypoints.len().max(feat2.keypoints.len()) as f64;
        let avg_similarity = total_similarity / matches as f64;
        
        // Be more selective: require both good match ratio and reasonable similarity
        if match_ratio < 0.2 || avg_similarity < 0.5 {
            0.0
        } else {
            match_ratio * avg_similarity * 0.9 // Scale down slightly to be more conservative
        }
    }
}

/// Calculate descriptor similarity (cosine similarity)
fn calculate_descriptor_similarity(desc1: &[f32], desc2: &[f32]) -> f64 {
    if desc1.len() != desc2.len() {
        return 0.0;
    }
    
    let mut dot_product = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for (a, b) in desc1.iter().zip(desc2.iter()) {
        dot_product += (*a as f64) * (*b as f64);
        norm1 += (*a as f64) * (*a as f64);
        norm2 += (*b as f64) * (*b as f64);
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1.sqrt() * norm2.sqrt())
    }
}

/// DTW alignment for handling temporal shifts
pub fn dtw_align_segments(
    features1: &[FrameFeatures],
    features2: &[FrameFeatures],
    max_shift: usize,
) -> Option<DtwAlignment> {
    if features1.is_empty() || features2.is_empty() {
        return None;
    }
    
    let _len1 = features1.len();
    let _len2 = features2.len();
    
    // Try different alignments within max_shift
    let mut best_alignment = None;
    let mut best_similarity = 0.0;
    
    // Try both directions of alignment
    for direction in 0..2 {
        let (feat_a, feat_b) = if direction == 0 {
            (features1, features2)
        } else {
            (features2, features1)
        };
        
        for shift in 0..=max_shift.min(feat_a.len().saturating_sub(1)) {
            let end_a = (feat_a.len() - shift).min(feat_b.len());
            if end_a < 10 { // Need at least 10 frames for a valid segment
                continue;
            }
            
            let mut total_similarity = 0.0;
            let mut valid_pairs = 0;
            
            // Sample frames for efficiency (every 3rd frame)
            for i in (0..end_a).step_by(3) {
                if shift + i < feat_a.len() && i < feat_b.len() {
                    let feat_a_frame = &feat_a[shift + i];
                    let feat_b_frame = &feat_b[i];
                    
                    // Calculate similarity between frames
                    let ssim_sim = compute_ssim_from_features(feat_a_frame, feat_b_frame);
                    let feature_sim = match_features(feat_a_frame, feat_b_frame);
                    let combined_sim = ssim_sim * 0.6 + feature_sim * 0.4;
                    
                    total_similarity += combined_sim;
                    valid_pairs += 1;
                }
            }
            
            if valid_pairs > 0 {
                let avg_similarity = total_similarity / valid_pairs as f64;
                if avg_similarity > best_similarity {
                    best_similarity = avg_similarity;
                    
                    let (video1_range, video2_range) = if direction == 0 {
                        (
                            (feat_a[shift].timestamp, feat_a[shift + end_a - 1].timestamp),
                            (feat_b[0].timestamp, feat_b[end_a - 1].timestamp)
                        )
                    } else {
                        (
                            (feat_b[0].timestamp, feat_b[end_a - 1].timestamp),
                            (feat_a[shift].timestamp, feat_a[shift + end_a - 1].timestamp)
                        )
                    };
                    
                    best_alignment = Some(DtwAlignment {
                        video1_range,
                        video2_range,
                        similarity: avg_similarity,
                    });
                }
            }
        }
    }
    
    best_alignment
}

/// Compute SSIM from frame features (simplified)
pub fn compute_ssim_from_features(feat1: &FrameFeatures, feat2: &FrameFeatures) -> f64 {
    if feat1.ssim_signature.len() != feat2.ssim_signature.len() {
        return 0.0;
    }
    
    // Calculate correlation between SSIM signatures
    let mut correlation = 0.0;
    let mut norm1 = 0.0;
    let mut norm2 = 0.0;
    
    for (a, b) in feat1.ssim_signature.iter().zip(feat2.ssim_signature.iter()) {
        correlation += a * b;
        norm1 += a * a;
        norm2 += b * b;
    }
    
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        correlation / (norm1.sqrt() * norm2.sqrt())
    }
}

/// Detect scene boundaries for efficiency
pub fn detect_scenes(frames: &[crate::analyzer::Frame]) -> Vec<SceneSegment> {
    if frames.is_empty() {
        return Vec::new();
    }
    
    let mut scenes = Vec::new();
    let mut current_scene_start = 0;
    let mut last_timestamp = frames[0].timestamp;
    
    for (i, frame) in frames.iter().enumerate() {
        // Simple scene detection based on time gaps
        let time_diff = frame.timestamp - last_timestamp;
        
        // If there's a significant time gap, start a new scene
        if time_diff > 2.0 {  // 2 second gap threshold
            if i > current_scene_start {
                scenes.push(SceneSegment {
                    start_frame: current_scene_start,
                    end_frame: i - 1,
                    start_time: frames[current_scene_start].timestamp,
                    end_time: last_timestamp,
                    complexity: calculate_scene_complexity(&frames[current_scene_start..i]),
                });
            }
            current_scene_start = i;
        }
        
        last_timestamp = frame.timestamp;
    }
    
    // Add the last scene
    if current_scene_start < frames.len() {
        scenes.push(SceneSegment {
            start_frame: current_scene_start,
            end_frame: frames.len() - 1,
            start_time: frames[current_scene_start].timestamp,
            end_time: last_timestamp,
            complexity: calculate_scene_complexity(&frames[current_scene_start..]),
        });
    }
    
    scenes
}

/// Calculate scene complexity (simplified)
fn calculate_scene_complexity(frames: &[crate::analyzer::Frame]) -> f64 {
    if frames.len() < 2 {
        return 0.0;
    }
    
    // Calculate average hash difference between consecutive frames
    let mut total_diff = 0.0;
    for i in 1..frames.len() {
        let diff = (frames[i].perceptual_hash ^ frames[i-1].perceptual_hash).count_ones() as f64;
        total_diff += diff;
    }
    
    total_diff / (frames.len() - 1) as f64 / 64.0  // Normalize
}
