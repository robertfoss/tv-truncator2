//! CLI UX: help grouping, quiet/json flags (no heavy pipeline).

use assert_cmd::Command;
use predicates::str::contains;

#[test]
fn help_shows_group_headings() {
    let assert = Command::cargo_bin("tvt").unwrap().arg("--help").assert();
    let out = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        out.contains("Input and output")
            && out.contains("Safety")
            && out.contains("Detection")
            && out.contains("Performance")
            && out.contains("Output and logging"),
        "grouped help headings missing:\n{}",
        out
    );
}

#[test]
fn help_mentions_dry_run_and_json_summary() {
    let assert = Command::cargo_bin("tvt").unwrap().arg("--help").assert();
    let out = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(out.contains("--dry-run"));
    assert!(out.contains("--json-summary"));
    assert!(out.contains("--quiet") || out.contains("-q"));
}

#[test]
fn help_mentions_hardware_video_decode_opt_out() {
    let assert = Command::cargo_bin("tvt").unwrap().arg("--help").assert();
    let out = String::from_utf8_lossy(&assert.get_output().stdout);
    assert!(
        out.contains("--no-hardware-video-decode"),
        "expected --no-hardware-video-decode in help:\n{}",
        out
    );
}

#[test]
fn json_summary_on_empty_dir_emits_valid_json() {
    let tmp = tempfile::tempdir().unwrap();
    let assert = Command::cargo_bin("tvt")
        .unwrap()
        .args(["-i", tmp.path().to_str().unwrap(), "--json-summary"])
        .assert()
        .success();
    let out = String::from_utf8_lossy(&assert.get_output().stdout);
    let v: serde_json::Value = serde_json::from_str(out.trim()).expect("stdout should be JSON");
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["video_files_found"], 0);
}

#[test]
fn missing_input_errors_cleanly() {
    Command::cargo_bin("tvt")
        .unwrap()
        .args(["-i", "/nonexistent/tvt-path-xyz"])
        .assert()
        .failure()
        .stderr(contains("does not exist"));
}
