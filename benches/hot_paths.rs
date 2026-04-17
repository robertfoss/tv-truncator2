//! Criterion micro-benchmarks for CPU-hot paths (pure Rust, no GStreamer I/O).
//!
//! Run: `cargo bench --bench hot_paths`
//! Quick smoke: `cargo bench --bench hot_paths -- --quick`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tvt::hasher::{
    hamming_distance, is_similar, rolling_hash_analysis_vector, rolling_hash_analysis_vector_par,
    RollingHash,
};

fn bench_rolling_hash_fill_window(c: &mut Criterion) {
    c.bench_function("rolling_hash_512_updates_window8", |b| {
        b.iter(|| {
            let mut rh = RollingHash::new(8);
            for i in 0..512u64 {
                black_box(rh.add(black_box(i)));
            }
        });
    });
}

/// Synthetic 50k perceptual hashes — same workload as Criterion “before” (sequential)
/// vs “after” (parallel window fingerprints).
fn bench_rolling_analysis_50k_sequential_vs_parallel(c: &mut Criterion) {
    const N: usize = 50_000;
    let data: Vec<u64> = (0..N as u64).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();

    c.bench_function("rolling_analysis_50k_sequential", |b| {
        b.iter(|| black_box(rolling_hash_analysis_vector(black_box(&data))));
    });

    c.bench_function("rolling_analysis_50k_parallel", |b| {
        b.iter(|| black_box(rolling_hash_analysis_vector_par(black_box(&data))));
    });
}

fn bench_hamming_and_similar(c: &mut Criterion) {
    c.bench_function("hamming_distance_u64", |b| {
        b.iter(|| {
            hamming_distance(
                black_box(0xdead_beef_cafe_u64),
                black_box(0xcafe_babe_face_u64),
            )
        });
    });

    c.bench_function("is_similar_threshold16", |b| {
        b.iter(|| {
            is_similar(
                black_box(0x0000_ffff_0000_u64),
                black_box(0x0000_fffe_0001_u64),
                black_box(16),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_rolling_hash_fill_window,
    bench_rolling_analysis_50k_sequential_vs_parallel,
    bench_hamming_and_similar
);
criterion_main!(benches);
