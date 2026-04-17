//! Golden fixtures + regression harness for critical paths ([MEM-24](/MEM/issues/MEM-24)).
//!
//! - Validates in-repo `segments.json` files and the golden manifest.
//! - Smoke-tests GStreamer init (same dependency chain as production).
//! - Locks a few pure numeric contracts (hasher / similarity) so refactors cannot drift silently.

use serde::Deserialize;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

use tvt::hasher::{hamming_distance, is_similar};
use tvt::similarity::calculate_similarity_score;
use tvt::similarity::MultiScaleHash;

#[derive(Debug, Deserialize)]
struct GoldenManifest {
    version: u32,
    paths: Vec<ManifestPath>,
}

#[derive(Debug, Deserialize)]
struct ManifestPath {
    id: String,
    #[allow(dead_code)]
    role: String,
    roots: Vec<String>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn golden_manifest_loads() {
    let p = repo_root().join("tests/fixtures/golden_manifest.json");
    let raw = fs::read_to_string(&p).expect("golden_manifest.json must exist");
    let m: GoldenManifest = serde_json::from_str(&raw).expect("valid manifest JSON");
    assert_eq!(m.version, 1);
    assert!(
        m.paths.iter().any(|x| x.id == "detection_synthetic"),
        "manifest must list detection_synthetic"
    );
}

/// Parse human-readable time tokens used in fixtures: seconds (`"32.0s"`, `"0.0"`), or `m:ss.d` / `mm:ss.d`.
fn parse_time_token(raw: &str) -> Option<f64> {
    let s = raw.trim();
    let s = s.strip_suffix('s').unwrap_or(s).trim();
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 2 {
        let m: f64 = parts[0].parse().ok()?;
        let sec: f64 = parts[1].parse().ok()?;
        return Some(m * 60.0 + sec);
    }
    if parts.len() == 3 {
        let h: f64 = parts[0].parse().ok()?;
        let m: f64 = parts[1].parse().ok()?;
        let sec: f64 = parts[2].parse().ok()?;
        return Some(h * 3600.0 + m * 60.0 + sec);
    }
    None
}

fn time_field_to_seconds(v: &Value, key: &str) -> Option<f64> {
    match v.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => parse_time_token(s),
        _ => None,
    }
}

fn validate_segments_json(path: &Path, root: &Path) {
    let raw = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let v: Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("JSON {}: {e}", path.display()));

    let test_name = v
        .get("test_name")
        .and_then(|x| x.as_str())
        .unwrap_or_else(|| panic!("{} missing test_name", path.display()));
    assert!(!test_name.is_empty(), "{}: empty test_name", path.display());

    let files = v
        .get("files")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("{} missing files array", path.display()));
    assert!(
        !files.is_empty(),
        "{}: files must be non-empty",
        path.display()
    );

    let segs = v
        .get("expected_segments")
        .and_then(|x| x.as_array())
        .unwrap_or_else(|| panic!("{} missing expected_segments", path.display()));

    for seg in segs {
        let sid = seg
            .get("segment_id")
            .and_then(|x| x.as_str())
            .unwrap_or_else(|| panic!("{} segment missing segment_id", path.display()));
        assert!(!sid.is_empty(), "{}: empty segment_id", path.display());

        let start = time_field_to_seconds(seg, "start_time")
            .unwrap_or_else(|| panic!("{} segment {sid}: bad start_time", path.display()));
        let end = time_field_to_seconds(seg, "end_time")
            .unwrap_or_else(|| panic!("{} segment {sid}: bad end_time", path.display()));
        assert!(
            end >= start,
            "{} segment {sid}: end_time must be >= start_time (got {start} .. {end})",
            path.display()
        );

        let listed = seg
            .get("files")
            .and_then(|x| x.as_array())
            .unwrap_or_else(|| panic!("{} segment {sid}: missing files", path.display()));
        for f in listed {
            let name = f.as_str().unwrap_or_else(|| {
                panic!("{} segment {sid}: files must be strings", path.display())
            });
            let media = root.join(name);
            if !media.exists() {
                eprintln!(
                    "note: optional media missing (ok for CI without LFS): {}",
                    media.display()
                );
            }
        }
    }
}

fn collect_segments_json_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let direct = dir.join("segments.json");
    if direct.is_file() {
        out.push(direct);
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let sj = p.join("segments.json");
            if sj.is_file() {
                out.push(sj);
            }
        }
    }
}

#[test]
fn all_golden_segments_json_parse_and_schema() {
    let root = repo_root();
    let manifest_path = root.join("tests/fixtures/golden_manifest.json");
    let raw = fs::read_to_string(&manifest_path).expect("manifest");
    let m: GoldenManifest = serde_json::from_str(&raw).expect("manifest JSON");

    for entry in &m.paths {
        for rel in &entry.roots {
            let abs = root.join(rel);
            assert!(abs.is_dir(), "manifest path {} must exist", abs.display());
            let mut paths = Vec::new();
            collect_segments_json_under(&abs, &mut paths);
            paths.sort();
            assert!(
                !paths.is_empty(),
                "no segments.json under {}",
                abs.display()
            );
            for p in paths {
                let fixture_root = p.parent().expect("segments.json parent");
                validate_segments_json(&p, fixture_root);
            }
        }
    }
}

#[test]
fn gstreamer_init_regression() {
    let r = tvt::gstreamer_extractor_v2::init_gstreamer();
    assert!(r.is_ok(), "GStreamer init is a critical path: {r:?}");
}

#[test]
fn hasher_hamming_golden_vectors() {
    assert_eq!(hamming_distance(0b1010, 0b1011), 1);
    assert!(is_similar(0b1010, 0b1011, 1));
    assert!(!is_similar(0b1010, 0b1011, 0));
}

#[test]
fn similarity_score_identical_multi_hash_is_one() {
    let h = MultiScaleHash {
        dhash: 0xAAAA_AAAA_AAAA_AAAA,
        phash: 0x5555_5555_5555_5555,
        ahash: 0xFFFF_FFFF_FFFF_FFFF,
        color_hash: 1,
    };
    let s = calculate_similarity_score(&h, &h);
    assert!(
        (s - 1.0).abs() < 1e-9,
        "identical MultiScaleHash must score 1.0, got {s}"
    );
}
