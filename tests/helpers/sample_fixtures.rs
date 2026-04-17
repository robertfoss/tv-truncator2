//! Shared helpers for discovering committed sample fixtures (MEMA-15).

use std::fs;
use std::path::{Path, PathBuf};

/// Every `segments.json` under `dir` (recursive).
///
/// Only referenced from `sample_fixture_coverage`; other integration crates warn otherwise.
#[allow(dead_code)]
pub fn collect_segments_json_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|n| n == "segments.json") {
            out.push(path);
        } else if path.is_dir() {
            collect_segments_json_recursive(&path, out);
        }
    }
}

/// Immediate subdirectories of `tests/samples/synthetic` that contain `segments.json`.
///
/// Only referenced from `performance_tests`; other integration crates pull `helpers` and would
/// otherwise warn about dead code.
#[allow(dead_code)]
pub fn synthetic_subdirs_with_segments(repo_root: &Path) -> Vec<PathBuf> {
    let synthetic = repo_root.join("tests/samples/synthetic");
    let mut dirs = Vec::new();
    let Ok(rd) = fs::read_dir(&synthetic) else {
        return dirs;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() && path.join("segments.json").is_file() {
            dirs.push(path);
        }
    }
    dirs.sort();
    dirs
}

/// True if `path` has at least one `.mkv` file as a direct child (matches how CLI is invoked per fixture).
#[allow(dead_code)]
pub fn dir_has_mkv_video(dir: &Path) -> bool {
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    rd.flatten().any(|e| {
        e.path().is_file()
            && e.path()
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("mkv"))
    })
}
