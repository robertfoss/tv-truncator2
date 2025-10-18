//! Tests for progress tracking functionality

use indicatif::{ProgressBar, ProgressStyle};

#[test]
fn test_progress_bar_creation() {
    let pb = ProgressBar::new(100);
    assert_eq!(pb.length(), Some(100));

    // Test progress bar styling
    let style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("#>-");

    pb.set_style(style);
    pb.set_message("Test message".to_string());

    // Test progress updates
    pb.set_position(50);
    assert_eq!(pb.position(), 50);

    pb.inc(25);
    assert_eq!(pb.position(), 75);

    pb.finish_with_message("Completed".to_string());
    assert_eq!(pb.position(), 100);
}

// Commented out - old cache functionality no longer exists
// #[test]
// fn test_progress_bar_with_extract_frames() {
//     // Test that progress bar integration works with frame extraction
//     let temp_dir = TempDir::new().unwrap();
//     let cache_manager = CacheManager::new(temp_dir.path().to_path_buf());
//
//     let test_path = Path::new("nonexistent_video.mkv");
//     let sample_rate = 1.0;
//
//     // Create a progress bar
//     let pb = ProgressBar::new_spinner();
//     pb.set_style(
//         ProgressStyle::default_spinner()
//             .template("{spinner:.blue} {msg}")
//             .unwrap(),
//     );
//     pb.set_message("Testing progress...".to_string());
//
//     // Test that extract_frames_with_cache accepts progress bar
//     // This will fail because the video doesn't exist, but it should not panic
//     let result = extract_frames_with_cache(test_path, sample_rate, Some(&pb), Some(&cache_manager));
//     assert!(result.is_err());
//
//     // Verify progress bar was used (it should have been updated)
//     // The exact state depends on when the error occurred, but it should be in a valid state
//     // Since the file doesn't exist, the progress bar might not be updated, so we just check it's valid
//     assert!(pb.position() >= 0);
// }

#[test]
fn test_progress_bar_styles() {
    // Test different progress bar styles
    let pb = ProgressBar::new(100);

    // Test bar style
    let bar_style = ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames {msg}",
        )
        .unwrap()
        .progress_chars("#>-");

    pb.set_style(bar_style);
    pb.set_message("Extracting frames...".to_string());
    pb.set_position(50);

    // Test spinner style
    let spinner_pb = ProgressBar::new_spinner();
    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.blue} {msg}")
        .unwrap();

    spinner_pb.set_style(spinner_style);
    spinner_pb.set_message("Processing...".to_string());

    // Both should be valid
    assert!(pb.length().is_some());
    assert!(spinner_pb.length().is_none()); // Spinner doesn't have a length
}

#[test]
fn test_progress_bar_message_updates() {
    let pb = ProgressBar::new(100);

    // Test message updates
    pb.set_message("Initial message".to_string());
    assert_eq!(pb.message(), "Initial message");

    pb.set_message("Updated message".to_string());
    assert_eq!(pb.message(), "Updated message");

    // Test message with formatting
    pb.set_message(format!("Processing {} of {}", 25, 100));
    assert_eq!(pb.message(), "Processing 25 of 100");
}

#[test]
fn test_progress_bar_position_updates() {
    let pb = ProgressBar::new(100);

    // Test position updates
    pb.set_position(0);
    assert_eq!(pb.position(), 0);

    pb.set_position(50);
    assert_eq!(pb.position(), 50);

    pb.inc(25);
    assert_eq!(pb.position(), 75);

    pb.inc(25);
    assert_eq!(pb.position(), 100);

    // Test that position can exceed length (indicatif allows this)
    pb.inc(1);
    assert_eq!(pb.position(), 101); // indicatif allows exceeding length
}

#[test]
fn test_progress_bar_completion() {
    let pb = ProgressBar::new(100);

    // Test completion states
    assert!(!pb.is_finished());

    pb.finish_with_message("Completed".to_string());
    assert!(pb.is_finished());
    assert_eq!(pb.message(), "Completed");

    // Test that finished progress bar can still be updated (indicatif allows this)
    let initial_position = pb.position();
    pb.inc(1);
    assert_eq!(pb.position(), initial_position + 1);
}
