//! Discover input video paths for the CLI and tests.

use crate::Result;
use std::fs;
use std::path::{Path, PathBuf};

/// Collect video files under `dir`.
///
/// When `recursive` is true, walks subdirectories but skips any directory named `truncated`
/// (typical tool output folder). Results are sorted by lowercase filename for stable ordering.
pub fn discover_video_files(dir: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut video_files = Vec::new();
    if recursive {
        find_video_files_recursive(dir, &mut video_files)?;
    } else {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && is_video_extension(path.extension().and_then(|e| e.to_str())) {
                video_files.push(path);
            }
        }
    }

    video_files.sort_by(|a, b| {
        a.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase()
            .cmp(
                &b.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_lowercase(),
            )
    });

    Ok(video_files)
}

fn find_video_files_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            let skip = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("truncated"));
            if skip {
                continue;
            }
            find_video_files_recursive(&path, out)?;
        } else if is_video_extension(path.extension().and_then(|e| e.to_str())) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_video_extension(ext: Option<&str>) -> bool {
    let Some(ext_str) = ext else {
        return false;
    };
    matches!(
        ext_str.to_lowercase().as_str(),
        "mp4" | "mkv" | "avi" | "mov" | "wmv" | "flv" | "webm"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn is_video_extension_case_insensitive() {
        assert!(is_video_extension(Some("MKV")));
        assert!(is_video_extension(Some("mkv")));
        assert!(!is_video_extension(Some("txt")));
        assert!(!is_video_extension(None));
    }

    #[test]
    fn discover_flat_reads_only_direct_children() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/samples/synthetic/intro");
        if !dir.join("intro_1.mkv").is_file() {
            eprintln!("skip: intro fixtures missing");
            return;
        }
        let paths = discover_video_files(&dir, false).expect("discover");
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn discover_recursive_skips_truncated_subdir() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/samples/synthetic/intro");
        if !dir.join("intro_1.mkv").is_file() {
            eprintln!("skip: intro fixtures missing");
            return;
        }
        let flat = discover_video_files(&dir, false).expect("flat");
        let rec = discover_video_files(&dir, true).expect("recursive");
        assert_eq!(
            flat.len(),
            rec.len(),
            "truncated/ must not add extra inputs"
        );
        assert_eq!(flat.len(), 3);
    }

    #[test]
    fn discover_downscaled_2file_finds_two_inputs() {
        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/samples/synthetic/downscaled_2file");
        let marker =
            dir.join("[Yellow-Flash]_Hajime_no_Ippo_-_01_[Blu-Ray][720p][10bit][C49D620A].mkv");
        if !marker.is_file() {
            eprintln!("skip: downscaled_2file media missing");
            return;
        }
        let paths = discover_video_files(&dir, false).expect("discover");
        assert_eq!(paths.len(), 2);
    }
}
