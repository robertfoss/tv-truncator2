//! Tests for GStreamer-based frame extraction

use std::path::Path;
use tvt::gstreamer_extractor_v2;

#[test]
fn test_gstreamer_initialization() {
    let result = gstreamer_extractor_v2::init_gstreamer();
    assert!(result.is_ok());
}

#[test]
fn test_gstreamer_duration_extraction() {
    // Test with a sample video file
    let video_path =
        Path::new("tests/samples/identical_3files/[Yellow-Flash]_Hajime_no_Ippo_-_01.mkv");

    if video_path.exists() {
        let config = tvt::Config::default();
        let result = gstreamer_extractor_v2::get_video_duration_gstreamer(video_path, &config);
        assert!(result.is_ok());

        let duration = result.unwrap();
        assert!(duration > 0.0);
        println!("Video duration: {:.2} seconds", duration);
    } else {
        println!("Skipping test - sample video not found");
    }
}

#[test]
fn test_gstreamer_frame_extraction() {
    // Test with a sample video file
    let video_path =
        Path::new("tests/samples/identical_3files/[Yellow-Flash]_Hajime_no_Ippo_-_01.mkv");

    if video_path.exists() {
        let config = tvt::Config::default();
        let result = gstreamer_extractor_v2::extract_frames_gstreamer_v2(
            video_path,
            0.5, // 0.5 fps for quick testing
            |current, total| {
                println!("Progress: {}/{}", current, total);
            },
            &config,
        );

        assert!(result.is_ok());

        let frames_vec = result.unwrap();
        assert!(!frames_vec.is_empty());
        println!("Extracted {} frames", frames_vec.len());

        // Verify frame structure
        for (i, frame) in frames_vec.iter().enumerate() {
            assert!(frame.timestamp >= 0.0);
            assert!(frame.perceptual_hash != 0); // u64 hash should not be zero

            if i > 0 {
                assert!(frame.timestamp > frames_vec[i - 1].timestamp);
            }
        }
    } else {
        println!("Skipping test - sample video not found");
    }
}

#[test]
fn test_gstreamer_metadata_extraction() {
    // Test with a sample video file
    let video_path =
        Path::new("tests/samples/identical_3files/[Yellow-Flash]_Hajime_no_Ippo_-_01.mkv");

    if video_path.exists() {
        let result = gstreamer_extractor_v2::get_video_info_gstreamer(video_path);
        assert!(result.is_ok());

        let info = result.unwrap();
        assert!(info.duration > 0.0);
        assert!(info.width > 0);
        assert!(info.height > 0);
        assert!(info.fps > 0.0);

        println!(
            "Video info: {}x{} @ {:.2}fps, duration: {:.2}s",
            info.width, info.height, info.fps, info.duration
        );
    } else {
        println!("Skipping test - sample video not found");
    }
}
