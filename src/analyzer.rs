//! Video frame extraction and analysis module

use crate::Result;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Represents a single frame with its timestamp and perceptual hash
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub timestamp: f64,
    pub perceptual_hash: u64,
}

/// Represents frames extracted from an episode
#[derive(Debug, Clone)]
pub struct EpisodeFrames {
    pub episode_path: std::path::PathBuf,
    pub frames: Vec<Frame>,
}

/// Video metadata from ffprobe
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoInfo {
    pub duration: f64,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub bitrate: Option<u64>,
}

/// Get video duration in seconds using GStreamer
pub fn get_video_duration(video_path: &Path) -> Result<f64> {
    // Use GStreamer for duration extraction
    let config = crate::Config::default();
    crate::gstreamer_extractor::get_video_duration_gstreamer(video_path, &config)
}

/// Get video metadata using GStreamer discoverer
pub fn get_video_info(video_path: &Path) -> Result<VideoInfo> {
    // Use GStreamer for metadata extraction
    crate::gstreamer_extractor::get_video_info_gstreamer(video_path)
}

/// Generate perceptual hash for an image
/// TODO: Implement proper perceptual hashing using img_hash crate
pub fn generate_perceptual_hash(image: &DynamicImage) -> Result<u64> {
    // Simple hash based on image dimensions and first few pixels
    // This is a placeholder - will implement proper perceptual hashing later
    let rgb_image = image.to_rgb8();
    let mut hash = 0u64;
    hash = hash.wrapping_add(rgb_image.width() as u64);
    hash = hash.wrapping_mul(31);
    hash = hash.wrapping_add(rgb_image.height() as u64);
    hash = hash.wrapping_mul(31);

    // Add first few pixel values
    let pixels = rgb_image.as_raw();
    for (i, &pixel) in pixels.iter().take(16).enumerate() {
        hash = hash.wrapping_add((pixel as u64) << (i % 8));
    }

    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    use tempfile::tempdir;

    #[test]
    fn test_generate_perceptual_hash() {
        // Create a simple test image
        let img = RgbImage::new(100, 100);
        let dynamic_img = DynamicImage::ImageRgb8(img);

        let hash1 = generate_perceptual_hash(&dynamic_img).unwrap();
        let hash2 = generate_perceptual_hash(&dynamic_img).unwrap();

        // Same image should produce same hash
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_get_video_info_missing_file() {
        let temp_dir = tempdir().unwrap();
        let missing_file = temp_dir.path().join("missing.mkv");

        let result = get_video_info(&missing_file);
        assert!(result.is_err());
    }

    #[test]
    fn test_video_info_parsing() {
        // Test that VideoInfo can be serialized/deserialized
        let info = VideoInfo {
            duration: 120.5,
            width: 1920,
            height: 1080,
            fps: 23.976,
            bitrate: Some(5000000),
        };

        let json = serde_json::to_string(&info).unwrap();
        let parsed: VideoInfo = serde_json::from_str(&json).unwrap();

        assert_eq!(info.duration, parsed.duration);
        assert_eq!(info.width, parsed.width);
        assert_eq!(info.height, parsed.height);
        assert_eq!(info.fps, parsed.fps);
        assert_eq!(info.bitrate, parsed.bitrate);
    }
}
