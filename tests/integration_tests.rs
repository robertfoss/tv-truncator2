//! Integration tests for TVT

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("tvt").unwrap();
    cmd.arg("--help");
    cmd.assert().success().stdout(predicate::str::contains(
        "TV Truncator - Remove repetitive segments",
    ));
}

#[test]
fn test_cli_missing_input() {
    let mut cmd = Command::cargo_bin("tvt").unwrap();
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("required"));
}

#[test]
fn test_cli_invalid_input() {
    let temp_dir = tempdir().unwrap();
    let non_existent = temp_dir.path().join("nonexistent");

    let mut cmd = Command::cargo_bin("tvt").unwrap();
    cmd.arg("--input").arg(non_existent);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("does not exist"));
}

#[test]
fn test_cli_valid_input() {
    let temp_dir = tempdir().unwrap();

    let mut cmd = Command::cargo_bin("tvt").unwrap();
    cmd.arg("--input").arg(temp_dir.path());
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("TVT - TV Truncator"));
}
