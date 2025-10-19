//! Tests for identical segment detection

use assert_cmd::Command;
use std::path::Path;

#[test]
fn test_identical_3files_has_one_long_segment() {
    let mut cmd = Command::cargo_bin("tvt").unwrap();

    let result = cmd
        .arg("--input")
        .arg("tests/samples/identical_3files")
        .arg("--threshold")
        .arg("3")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--verbose")
        .assert();

    let output = String::from_utf8(result.get_output().stdout.clone()).unwrap();

    // Verify the command succeeded
    result.success();

    // Check that we found exactly 1 segment
    assert!(output.contains("Found 1 identical segment(s) across 3 file(s):"));

    // Check that the segment is long (should be around 92 seconds based on actual video content)
    // Note: This duration represents the actual identical content in the test files
    assert!(output.contains("Time: 0.00s - 92."));

    // Check that all 3 files are mentioned
    assert!(output.contains("[Yellow-Flash]_Hajime_no_Ippo_-_01 (Copy 2)_truncated.mkv"));
    assert!(output.contains("[Yellow-Flash]_Hajime_no_Ippo_-_01 (Copy)_truncated.mkv"));
    assert!(output.contains("[Yellow-Flash]_Hajime_no_Ippo_-_01_truncated.mkv"));

    // Check that the confidence is reasonable (should be around 100% for identical files)
    assert!(output.contains("Confidence: 100%"));

    // Check that total time is calculated correctly (actual duration is ~92s based on video content)
    assert!(output.contains("Total time that will be removed: 92."));
}

#[test]
fn test_identical_3files_segment_detection_with_different_thresholds() {
    // Test with threshold 2 (should still find the segment)
    let mut cmd = Command::cargo_bin("tvt").unwrap();

    let result = cmd
        .arg("--input")
        .arg("tests/samples/identical_3files")
        .arg("--threshold")
        .arg("2")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--verbose")
        .assert();

    let output = String::from_utf8(result.get_output().stdout.clone()).unwrap();

    result.success();

    // Should still find 1 segment with threshold 2
    assert!(output.contains("Found 1 identical segment(s) across 3 file(s):"));
}

#[test]
fn test_identical_3files_no_segments_with_high_threshold() {
    // Test with threshold 4 (should not find any segments since we only have 3 files)
    let mut cmd = Command::cargo_bin("tvt").unwrap();

    let result = cmd
        .arg("--input")
        .arg("tests/samples/identical_3files")
        .arg("--threshold")
        .arg("4")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--verbose")
        .assert();

    let output = String::from_utf8(result.get_output().stdout.clone()).unwrap();

    result.success();

    // Should not find any segments with threshold 4
    assert!(output.contains("No identical segments found that meet the threshold criteria."));
}

#[test]
fn test_identical_3files_with_dry_run() {
    // Test with dry run to ensure it still detects segments
    let mut cmd = Command::cargo_bin("tvt").unwrap();

    let result = cmd
        .arg("--input")
        .arg("tests/samples/identical_3files")
        .arg("--threshold")
        .arg("3")
        .arg("--min-duration")
        .arg("1.0")
        .arg("--parallel")
        .arg("1")
        .arg("--dry-run")
        .arg("--verbose")
        .assert();

    let output = String::from_utf8(result.get_output().stdout.clone()).unwrap();

    result.success();

    // Should still find the segment in dry run mode
    assert!(output.contains("Found 1 identical segment(s) across 3 file(s):"));
}

#[test]
fn test_identical_3files_verifies_files_exist() {
    // Verify that the test files actually exist
    let test_dir = Path::new("tests/samples/identical_3files");
    assert!(test_dir.exists(), "Test directory should exist");
    assert!(test_dir.is_dir(), "Test directory should be a directory");

    let files = std::fs::read_dir(test_dir).unwrap();
    let video_files: Vec<_> = files
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ["mkv", "mp4", "avi", "mov", "flv", "webm"].contains(&ext))
                    .unwrap_or(false)
        })
        .collect();
    assert_eq!(
        video_files.len(),
        3,
        "Should have exactly 3 video files in the test directory"
    );

    // Verify specific files exist
    let expected_files = [
        "[Yellow-Flash]_Hajime_no_Ippo_-_01_truncated.mkv",
        "[Yellow-Flash]_Hajime_no_Ippo_-_01 (Copy)_truncated.mkv",
        "[Yellow-Flash]_Hajime_no_Ippo_-_01 (Copy 2)_truncated.mkv",
    ];

    for expected_file in &expected_files {
        let file_path = test_dir.join(expected_file);
        assert!(file_path.exists(), "File {} should exist", expected_file);
    }
}
