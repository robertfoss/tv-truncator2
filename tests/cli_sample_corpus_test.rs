//! CLI smoke tests against in-repo sample corpora ([MEMA-7](/MEMA/issues/MEMA-7)).
//!
//! Skips when fixture media is absent (e.g. minimal CI checkout without samples).

use assert_cmd::Command;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn intro_fixture_dir() -> PathBuf {
    repo_root().join("tests/samples/synthetic/intro")
}

#[test]
fn cli_dry_run_quick_parallel_on_intro_fixture() {
    let intro = intro_fixture_dir();
    if !intro.join("intro_1.mkv").is_file() {
        eprintln!("skip: intro fixture media not present");
        return;
    }

    let mut cmd = Command::cargo_bin("tvt").unwrap();
    cmd.current_dir(&repo_root());
    cmd.args([
        "--input",
        "tests/samples/synthetic/intro",
        "--dry-run",
        "--quick",
        "--parallel",
        "2",
        "--quiet",
    ]);
    cmd.assert().success();
}

#[test]
fn cli_recursive_finds_same_video_count_as_flat_intro() {
    let intro = intro_fixture_dir();
    if !intro.join("intro_1.mkv").is_file() {
        eprintln!("skip: intro fixture media not present");
        return;
    }

    for (extra, label) in [
        (Vec::<&str>::new(), "flat"),
        (vec!["--recursive"], "recursive"),
    ] {
        let mut cmd = Command::cargo_bin("tvt").unwrap();
        cmd.current_dir(&repo_root());
        cmd.args([
            "--input",
            "tests/samples/synthetic/intro",
            "--dry-run",
            "--quick",
            "--parallel",
            "2",
            "--json-summary",
        ]);
        cmd.args(&extra);
        let out = cmd.output().expect("run tvt");
        assert!(
            out.status.success(),
            "{} stderr: {}",
            label,
            String::from_utf8_lossy(&out.stderr)
        );
        let v: serde_json::Value =
            serde_json::from_slice(&out.stdout).expect("json-summary stdout");
        assert_eq!(
            v["video_files_found"].as_u64(),
            Some(3),
            "{}: expected 3 inputs discovered",
            label
        );
    }
}
