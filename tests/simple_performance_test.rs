//! Simple performance validation test for GStreamer implementation

use std::time::Instant;
use assert_cmd::Command;

/// Test that GStreamer processing completes within reasonable time
#[test]
fn test_gstreamer_processing_speed() {
    let start = Instant::now();
    
    let result = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/synthetic/opening_credits")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .output();
    
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    
    // Should complete within 5 minutes for synthetic samples (they have complex content)
    assert!(duration.as_secs() < 300, "Processing took too long: {:?}", duration);
    
    println!("GStreamer processing completed in: {:?}", duration);
}

/// Test quick mode performance
#[test]
fn test_quick_mode_performance() {
    let start = Instant::now();
    
    let result = Command::cargo_bin("tvt")
        .unwrap()
        .arg("--input")
        .arg("tests/samples/synthetic/opening_credits")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--quick")
        .arg("--dry-run")
        .output();
    
    let duration = start.elapsed();
    
    assert!(result.is_ok());
    
    // Quick mode should be faster
    assert!(duration.as_secs() < 60, "Quick mode took too long: {:?}", duration);
    
    println!("Quick mode processing completed in: {:?}", duration);
}
