//! Ensures committed `segments.json` fixtures stay tied to the golden manifest ([MEMA-15](/MEMA/issues/MEMA-15)).
//!
//! Fast CI (`cargo test`) runs these; full tier additionally runs `#[ignore]` suites with
//! `cargo test -- --include-ignored` (see `.github/workflows/ci-full.yml`).

mod helpers;

use helpers::sample_fixtures;
use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn manifest_root_prefixes() -> Vec<String> {
    let raw = fs::read_to_string(repo_root().join("tests/fixtures/golden_manifest.json"))
        .expect("golden_manifest.json must exist");
    let m: serde_json::Value = serde_json::from_str(&raw).expect("valid golden_manifest.json");
    m["paths"]
        .as_array()
        .expect("paths array")
        .iter()
        .flat_map(|entry| entry["roots"].as_array().expect("roots").iter())
        .filter_map(|r| r.as_str().map(String::from))
        .collect()
}

fn is_under_manifest_root(rel: &str, roots: &[String]) -> bool {
    roots.iter().any(|root| {
        rel == root.as_str() || rel.starts_with(&format!("{}/", root.trim_end_matches('/')))
    })
}

#[test]
fn every_segments_json_under_tests_samples_is_under_golden_manifest_roots() {
    let root = repo_root();
    let samples = root.join("tests/samples");
    assert!(samples.is_dir(), "tests/samples must exist (fixture tree)");

    let mut segments_files = Vec::new();
    sample_fixtures::collect_segments_json_recursive(&samples, &mut segments_files);
    segments_files.sort();

    assert!(
        !segments_files.is_empty(),
        "expected at least one segments.json under tests/samples"
    );

    let roots = manifest_root_prefixes();
    assert!(
        !roots.is_empty(),
        "golden_manifest.json must list at least one root"
    );

    for path in &segments_files {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");

        assert!(
            is_under_manifest_root(&rel, &roots),
            "segments.json at {} is not under any golden manifest `roots` entry; add the parent to tests/fixtures/golden_manifest.json or move the fixture",
            rel
        );
    }
}
