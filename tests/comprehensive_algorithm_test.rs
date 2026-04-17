//! Comprehensive test of all video and audio algorithms on downscaled samples
//!
//! This test evaluates all audio and video algorithms against a dataset of 12
//! TV episodes and generates a detailed report comparing their performance.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tvt::analyzer::EpisodeFrames;
use tvt::audio_chromaprint::detect_audio_segments_chromaprint;
use tvt::audio_correlation::detect_audio_segments_correlation;
use tvt::audio_energy_bands::detect_audio_segments_energy_bands;
use tvt::audio_extractor::{extract_audio_samples, EpisodeAudio};
use tvt::audio_fingerprint::detect_audio_segments_fingerprint;
use tvt::audio_hasher::process_audio_samples;
use tvt::audio_mfcc::detect_audio_segments_mfcc;
use tvt::audio_spectral_v2::detect_audio_segments_spectral_v2;
use tvt::gstreamer_extractor_v2::extract_frames_gstreamer_v2;
use tvt::segment_detector::{detect_audio_segments, detect_common_segments, CommonSegment};
use tvt::Result;
use tvt::{AudioAlgorithm, Config};

/// Expected intro segment timing (seconds)
const EXPECTED_INTRO_START: f64 = 0.0;
const EXPECTED_INTRO_END: f64 = 90.0;

/// Expected outro segment timing (seconds)
const EXPECTED_OUTRO_START: f64 = 1341.0;
const EXPECTED_OUTRO_END: f64 = 1411.0;

/// Margin for timing accuracy (seconds)
const TIMING_MARGIN: f64 = 20.0;

/// Result of testing a single algorithm
#[derive(Debug, Clone)]
struct AlgorithmResult {
    algorithm_name: String,
    success: bool,
    error_message: Option<String>,
    segments_found: usize,
    intro_segment: Option<SegmentInfo>,
    outro_segment: Option<SegmentInfo>,
    execution_time: Duration,
}

#[derive(Debug, Clone)]
struct SegmentInfo {
    start_time: f64,
    end_time: f64,
    confidence: f64,
    episode_count: usize,
}

impl AlgorithmResult {
    fn new(algorithm_name: String) -> Self {
        Self {
            algorithm_name,
            success: false,
            error_message: None,
            segments_found: 0,
            intro_segment: None,
            outro_segment: None,
            execution_time: Duration::default(),
        }
    }

    fn is_intro_accurate(&self) -> bool {
        self.intro_segment.as_ref().map_or(false, |seg| {
            (seg.start_time - EXPECTED_INTRO_START).abs() < TIMING_MARGIN
                && (seg.end_time - EXPECTED_INTRO_END).abs() < TIMING_MARGIN
        })
    }

    fn is_outro_accurate(&self) -> bool {
        self.outro_segment.as_ref().map_or(false, |seg| {
            (seg.start_time - EXPECTED_OUTRO_START).abs() < TIMING_MARGIN
                && (seg.end_time - EXPECTED_OUTRO_END).abs() < TIMING_MARGIN
        })
    }

    fn status_indicator(&self) -> &str {
        if !self.success {
            return "❌";
        }
        if self.intro_segment.is_some() && self.outro_segment.is_some() {
            "✅"
        } else if self.intro_segment.is_some() || self.outro_segment.is_some() {
            "⚠️"
        } else {
            "❌"
        }
    }
}

/// Load all video files from the downscaled directory
fn load_video_files() -> Result<Vec<PathBuf>> {
    let test_dir = Path::new("tests/samples/downscaled");

    if !test_dir.exists() {
        anyhow::bail!("Test directory not found: {}", test_dir.display());
    }

    let mut video_files: Vec<PathBuf> = fs::read_dir(test_dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path.extension().and_then(|s| s.to_str()) == Some("mkv")
                && !path.to_string_lossy().contains("truncated")
        })
        .collect();

    video_files.sort();

    if video_files.is_empty() {
        anyhow::bail!("No video files found in {}", test_dir.display());
    }

    Ok(video_files)
}

/// Extract audio from all video files
fn extract_episode_audio(video_files: &[PathBuf]) -> Result<Vec<EpisodeAudio>> {
    let mut episode_audio = Vec::new();
    let sample_rate = 22050;

    println!("Extracting audio from {} files...", video_files.len());
    for (i, video_path) in video_files.iter().enumerate() {
        println!(
            "  [{}/{}] {}",
            i + 1,
            video_files.len(),
            video_path.file_name().unwrap().to_string_lossy()
        );
        let audio_samples = extract_audio_samples(video_path, sample_rate, None, |_, _| {})?;
        let audio_frames = process_audio_samples(&audio_samples, sample_rate as f32, 1.0)?;

        episode_audio.push(EpisodeAudio {
            episode_path: video_path.clone(),
            audio_frames,
            raw_samples: audio_samples,
            sample_rate: sample_rate as f32,
        });
    }

    Ok(episode_audio)
}

/// Extract video frames from all video files
fn extract_episode_frames(video_files: &[PathBuf], config: &Config) -> Result<Vec<EpisodeFrames>> {
    let mut episode_frames = Vec::new();
    let sample_rate = if config.quick { 0.5 } else { 1.0 };

    println!(
        "Extracting video frames from {} files...",
        video_files.len()
    );
    for (i, video_path) in video_files.iter().enumerate() {
        println!(
            "  [{}/{}] {}",
            i + 1,
            video_files.len(),
            video_path.file_name().unwrap().to_string_lossy()
        );
        let frames = extract_frames_gstreamer_v2(video_path, sample_rate, |_, _| {}, config)?;

        episode_frames.push(EpisodeFrames {
            episode_path: video_path.clone(),
            frames,
        });
    }

    Ok(episode_frames)
}

/// Create base configuration
fn create_base_config() -> Config {
    Config {
        input_dir: PathBuf::from("tests/samples/downscaled"),
        output_dir: PathBuf::from("tests/samples/downscaled/test_out"),
        threshold: 2,
        min_duration: 30.0,
        similarity: 90,
        similarity_threshold: 0.75,
        similarity_algorithm: tvt::similarity::SimilarityAlgorithm::Current,
        audio_algorithm: AudioAlgorithm::Chromaprint,
        dry_run: true,
        quick: false,
        verbose: false,
        debug: false,
        debug_dupes: true,
        parallel_workers: 2,
        enable_audio_matching: true,
        audio_only: false,
        quiet: false,
        json_summary: false,
    }
}

/// Parse segments and categorize as intro/outro
fn parse_segments(segments: Vec<CommonSegment>) -> (Option<SegmentInfo>, Option<SegmentInfo>) {
    let mut intro: Option<SegmentInfo> = None;
    let mut outro: Option<SegmentInfo> = None;

    for seg in segments {
        let confidence = seg
            .audio_confidence
            .or(seg.video_confidence)
            .unwrap_or(seg.confidence);

        // Check if it's an intro (starts near beginning)
        if seg.start_time < 200.0 {
            intro = Some(SegmentInfo {
                start_time: seg.start_time,
                end_time: seg.end_time,
                confidence,
                episode_count: seg.episode_list.len(),
            });
        }
        // Check if it's an outro (starts after 1000s)
        else if seg.start_time > 1000.0 {
            outro = Some(SegmentInfo {
                start_time: seg.start_time,
                end_time: seg.end_time,
                confidence,
                episode_count: seg.episode_list.len(),
            });
        }
    }

    (intro, outro)
}

/// Test a single audio algorithm
fn test_audio_algorithm(
    name: &str,
    episode_audio: &[EpisodeAudio],
    config: &Config,
    detect_fn: impl Fn(&[EpisodeAudio], &Config, bool) -> Result<Vec<CommonSegment>>,
) -> AlgorithmResult {
    let mut result = AlgorithmResult::new(name.to_string());

    println!("\n=== Testing {} ===", name);
    let start = Instant::now();

    match detect_fn(episode_audio, config, true) {
        Ok(segments) => {
            result.execution_time = start.elapsed();
            result.success = true;
            result.segments_found = segments.len();

            let (intro, outro) = parse_segments(segments);
            result.intro_segment = intro;
            result.outro_segment = outro;

            println!(
                "  ✓ Found {} segments in {:.2}s",
                result.segments_found,
                result.execution_time.as_secs_f64()
            );
            if let Some(ref intro) = result.intro_segment {
                println!(
                    "    Intro: {:.1}s-{:.1}s (conf={:.2}, {} eps)",
                    intro.start_time, intro.end_time, intro.confidence, intro.episode_count
                );
            }
            if let Some(ref outro) = result.outro_segment {
                println!(
                    "    Outro: {:.1}s-{:.1}s (conf={:.2}, {} eps)",
                    outro.start_time, outro.end_time, outro.confidence, outro.episode_count
                );
            }
        }
        Err(e) => {
            result.execution_time = start.elapsed();
            result.success = false;
            result.error_message = Some(e.to_string());
            println!("  ✗ Failed: {}", e);
        }
    }

    result
}

/// Test a single video algorithm
fn test_video_algorithm(
    name: &str,
    episode_frames: &[EpisodeFrames],
    config: &Config,
) -> AlgorithmResult {
    let mut result = AlgorithmResult::new(name.to_string());

    println!("\n=== Testing {} ===", name);
    let start = Instant::now();

    match detect_common_segments(episode_frames, config, true, None) {
        Ok(segments) => {
            result.execution_time = start.elapsed();
            result.success = true;
            result.segments_found = segments.len();

            let (intro, outro) = parse_segments(segments);
            result.intro_segment = intro;
            result.outro_segment = outro;

            println!(
                "  ✓ Found {} segments in {:.2}s",
                result.segments_found,
                result.execution_time.as_secs_f64()
            );
            if let Some(ref intro) = result.intro_segment {
                println!(
                    "    Intro: {:.1}s-{:.1}s (conf={:.2}, {} eps)",
                    intro.start_time, intro.end_time, intro.confidence, intro.episode_count
                );
            }
            if let Some(ref outro) = result.outro_segment {
                println!(
                    "    Outro: {:.1}s-{:.1}s (conf={:.2}, {} eps)",
                    outro.start_time, outro.end_time, outro.confidence, outro.episode_count
                );
            }
        }
        Err(e) => {
            result.execution_time = start.elapsed();
            result.success = false;
            result.error_message = Some(e.to_string());
            println!("  ✗ Failed: {}", e);
        }
    }

    result
}

/// Generate markdown report
fn generate_report(
    audio_results: &[AlgorithmResult],
    video_results: &[AlgorithmResult],
) -> Result<()> {
    let mut report = String::new();

    report.push_str("# Comprehensive Algorithm Test Report\n\n");

    // Executive Summary
    report.push_str("## Executive Summary\n\n");
    let audio_success_count = audio_results
        .iter()
        .filter(|r| r.success && r.intro_segment.is_some() && r.outro_segment.is_some())
        .count();
    let video_success_count = video_results
        .iter()
        .filter(|r| r.success && r.intro_segment.is_some() && r.outro_segment.is_some())
        .count();

    report.push_str(&format!(
        "- **Audio Algorithms Tested**: {} ({} fully successful)\n",
        audio_results.len(),
        audio_success_count
    ));
    report.push_str(&format!(
        "- **Video Algorithms Tested**: {} ({} fully successful)\n",
        video_results.len(),
        video_success_count
    ));
    report.push_str(&format!(
        "- **Dataset**: 12 anime episodes (Hajime no Ippo)\n"
    ));
    report.push_str(&format!(
        "- **Expected Segments**: Opening (0-90s) and Ending (1341-1411s)\n"
    ));

    // Best performers
    if let Some(best_audio) = audio_results
        .iter()
        .filter(|r| r.success)
        .min_by_key(|r| r.execution_time)
    {
        report.push_str(&format!(
            "- **Fastest Audio Algorithm**: {} ({:.2}s)\n",
            best_audio.algorithm_name,
            best_audio.execution_time.as_secs_f64()
        ));
    }
    if let Some(best_video) = video_results
        .iter()
        .filter(|r| r.success)
        .min_by_key(|r| r.execution_time)
    {
        report.push_str(&format!(
            "- **Fastest Video Algorithm**: {} ({:.2}s)\n",
            best_video.algorithm_name,
            best_video.execution_time.as_secs_f64()
        ));
    }

    report.push_str("\n---\n\n");

    // Quick Results Table
    report.push_str("## Quick Results Overview\n\n");
    report.push_str("### Audio Algorithms\n\n");
    report.push_str("| Algorithm | Status | Segments | Intro | Outro | Time (s) |\n");
    report.push_str("|-----------|--------|----------|-------|-------|----------|\n");

    for result in audio_results {
        let intro_status = if result.is_intro_accurate() {
            "✅"
        } else if result.intro_segment.is_some() {
            "⚠️"
        } else {
            "❌"
        };
        let outro_status = if result.is_outro_accurate() {
            "✅"
        } else if result.outro_segment.is_some() {
            "⚠️"
        } else {
            "❌"
        };

        report.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.2} |\n",
            result.algorithm_name,
            result.status_indicator(),
            result.segments_found,
            intro_status,
            outro_status,
            result.execution_time.as_secs_f64()
        ));
    }

    report.push_str("\n### Video Algorithms\n\n");
    report.push_str("| Algorithm | Status | Segments | Intro | Outro | Time (s) |\n");
    report.push_str("|-----------|--------|----------|-------|-------|----------|\n");

    for result in video_results {
        let intro_status = if result.is_intro_accurate() {
            "✅"
        } else if result.intro_segment.is_some() {
            "⚠️"
        } else {
            "❌"
        };
        let outro_status = if result.is_outro_accurate() {
            "✅"
        } else if result.outro_segment.is_some() {
            "⚠️"
        } else {
            "❌"
        };

        report.push_str(&format!(
            "| {} | {} | {} | {} | {} | {:.2} |\n",
            result.algorithm_name,
            result.status_indicator(),
            result.segments_found,
            intro_status,
            outro_status,
            result.execution_time.as_secs_f64()
        ));
    }

    report.push_str("\n**Legend**: ✅ = Accurate, ⚠️ = Partial/Inaccurate, ❌ = Not Found\n\n");

    report.push_str("---\n\n");

    // Recommendations
    report.push_str("## Recommendations\n\n");
    report.push_str("### Top Audio Algorithms\n\n");

    // Find best audio algorithms by different criteria
    let best_accuracy: Vec<_> = audio_results
        .iter()
        .filter(|r| r.is_intro_accurate() && r.is_outro_accurate())
        .collect();

    if !best_accuracy.is_empty() {
        report.push_str("**Most Accurate** (found both segments correctly):\n");
        for r in &best_accuracy {
            let avg_conf = (r
                .intro_segment
                .as_ref()
                .map(|s| s.confidence)
                .unwrap_or(0.0)
                + r.outro_segment
                    .as_ref()
                    .map(|s| s.confidence)
                    .unwrap_or(0.0))
                / 2.0;
            report.push_str(&format!(
                "- **{}**: {:.1}% avg confidence, {:.2}s\n",
                r.algorithm_name,
                avg_conf * 100.0,
                r.execution_time.as_secs_f64()
            ));
        }
        report.push_str("\n");
    }

    if let Some(fastest) = audio_results
        .iter()
        .filter(|r| r.success)
        .min_by_key(|r| r.execution_time)
    {
        report.push_str(&format!(
            "**Fastest**: **{}** ({:.2}s)\n\n",
            fastest.algorithm_name,
            fastest.execution_time.as_secs_f64()
        ));
    }

    let balanced: Vec<_> = audio_results
        .iter()
        .filter(|r| {
            r.is_intro_accurate() && r.is_outro_accurate() && r.execution_time.as_secs() < 120
        })
        .collect();

    if !balanced.is_empty() {
        report.push_str("**Best Balanced** (accurate and reasonably fast):\n");
        for r in &balanced {
            report.push_str(&format!(
                "- **{}** ({:.2}s)\n",
                r.algorithm_name,
                r.execution_time.as_secs_f64()
            ));
        }
        report.push_str("\n");
    }

    report.push_str("### Top Video Algorithms\n\n");

    let best_video_accuracy: Vec<_> = video_results
        .iter()
        .filter(|r| r.is_intro_accurate() && r.is_outro_accurate())
        .collect();

    if !best_video_accuracy.is_empty() {
        report.push_str("**Most Accurate**:\n");
        for r in &best_video_accuracy {
            report.push_str(&format!(
                "- **{}** ({:.2}s)\n",
                r.algorithm_name,
                r.execution_time.as_secs_f64()
            ));
        }
        report.push_str("\n");
    }

    if let Some(fastest_video) = video_results
        .iter()
        .filter(|r| r.success)
        .min_by_key(|r| r.execution_time)
    {
        report.push_str(&format!(
            "**Fastest**: **{}** ({:.2}s)\n\n",
            fastest_video.algorithm_name,
            fastest_video.execution_time.as_secs_f64()
        ));
    }

    report.push_str("---\n\n");

    // Test Setup
    report.push_str("## Test Setup\n\n");
    report.push_str("### Dataset\n");
    report.push_str("- **Source**: Hajime no Ippo (Boxing anime)\n");
    report.push_str("- **Episodes**: 12 files (episodes 01-05, 27-30, 74-76)\n");
    report.push_str("- **Format**: MKV (downscaled for testing)\n");
    report.push_str("- **Episode Length**: ~23.5 minutes each\n\n");

    report.push_str("### Expected Segments\n");
    report.push_str("Based on manual verification:\n");
    report.push_str("- **Opening**: 0s to 90s (1:30 theme song)\n");
    report.push_str("- **Ending**: 1341s to 1411s (22:21 to 23:31, ending credits)\n\n");

    report.push_str("### Configuration\n");
    report.push_str("```\n");
    report.push_str("threshold: 2 (minimum 2 episodes required)\n");
    report.push_str("min_duration: 30s\n");
    report.push_str("similarity: 90%\n");
    report.push_str("similarity_threshold: 0.75\n");
    report.push_str("parallel_workers: 2\n");
    report.push_str("```\n\n");

    report.push_str("---\n\n");

    // Detailed Audio Results
    report.push_str("## Detailed Audio Algorithm Results\n\n");

    for result in audio_results {
        report.push_str(&format!("### {}\n\n", result.algorithm_name));

        if !result.success {
            report.push_str(&format!("**Status**: ❌ Failed\n\n"));
            if let Some(ref err) = result.error_message {
                report.push_str(&format!("**Error**: {}\n\n", err));
            }
            continue;
        }

        report.push_str(&format!(
            "**Status**: {} {}\n\n",
            result.status_indicator(),
            if result.intro_segment.is_some() && result.outro_segment.is_some() {
                "Success"
            } else {
                "Partial"
            }
        ));

        report.push_str(&format!(
            "**Execution Time**: {:.2}s\n\n",
            result.execution_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "**Segments Found**: {}\n\n",
            result.segments_found
        ));

        if let Some(ref intro) = result.intro_segment {
            report.push_str("**Opening Segment**:\n");
            report.push_str(&format!(
                "- Time: {:.1}s to {:.1}s (duration: {:.1}s)\n",
                intro.start_time,
                intro.end_time,
                intro.end_time - intro.start_time
            ));
            report.push_str(&format!(
                "- Expected: {:.1}s to {:.1}s\n",
                EXPECTED_INTRO_START, EXPECTED_INTRO_END
            ));
            report.push_str(&format!(
                "- Accuracy: {}\n",
                if result.is_intro_accurate() {
                    "✅ Accurate"
                } else {
                    "⚠️ Off by >20s"
                }
            ));
            report.push_str(&format!("- Confidence: {:.1}%\n", intro.confidence * 100.0));
            report.push_str(&format!("- Episodes: {}\n", intro.episode_count));
        } else {
            report.push_str("**Opening Segment**: ❌ Not found\n");
        }
        report.push_str("\n");

        if let Some(ref outro) = result.outro_segment {
            report.push_str("**Ending Segment**:\n");
            report.push_str(&format!(
                "- Time: {:.1}s to {:.1}s (duration: {:.1}s)\n",
                outro.start_time,
                outro.end_time,
                outro.end_time - outro.start_time
            ));
            report.push_str(&format!(
                "- Expected: {:.1}s to {:.1}s\n",
                EXPECTED_OUTRO_START, EXPECTED_OUTRO_END
            ));
            report.push_str(&format!(
                "- Accuracy: {}\n",
                if result.is_outro_accurate() {
                    "✅ Accurate"
                } else {
                    "⚠️ Off by >20s"
                }
            ));
            report.push_str(&format!("- Confidence: {:.1}%\n", outro.confidence * 100.0));
            report.push_str(&format!("- Episodes: {}\n", outro.episode_count));
        } else {
            report.push_str("**Ending Segment**: ❌ Not found\n");
        }
        report.push_str("\n");
    }

    report.push_str("---\n\n");

    // Detailed Video Results
    report.push_str("## Detailed Video Algorithm Results\n\n");

    for result in video_results {
        report.push_str(&format!("### {}\n\n", result.algorithm_name));

        if !result.success {
            report.push_str(&format!("**Status**: ❌ Failed\n\n"));
            if let Some(ref err) = result.error_message {
                report.push_str(&format!("**Error**: {}\n\n", err));
            }
            continue;
        }

        report.push_str(&format!(
            "**Status**: {} {}\n\n",
            result.status_indicator(),
            if result.intro_segment.is_some() && result.outro_segment.is_some() {
                "Success"
            } else {
                "Partial"
            }
        ));

        report.push_str(&format!(
            "**Execution Time**: {:.2}s\n\n",
            result.execution_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "**Segments Found**: {}\n\n",
            result.segments_found
        ));

        if let Some(ref intro) = result.intro_segment {
            report.push_str("**Opening Segment**:\n");
            report.push_str(&format!(
                "- Time: {:.1}s to {:.1}s (duration: {:.1}s)\n",
                intro.start_time,
                intro.end_time,
                intro.end_time - intro.start_time
            ));
            report.push_str(&format!(
                "- Expected: {:.1}s to {:.1}s\n",
                EXPECTED_INTRO_START, EXPECTED_INTRO_END
            ));
            report.push_str(&format!(
                "- Accuracy: {}\n",
                if result.is_intro_accurate() {
                    "✅ Accurate"
                } else {
                    "⚠️ Off by >20s"
                }
            ));
            report.push_str(&format!("- Confidence: {:.1}%\n", intro.confidence * 100.0));
            report.push_str(&format!("- Episodes: {}\n", intro.episode_count));
        } else {
            report.push_str("**Opening Segment**: ❌ Not found\n");
        }
        report.push_str("\n");

        if let Some(ref outro) = result.outro_segment {
            report.push_str("**Ending Segment**:\n");
            report.push_str(&format!(
                "- Time: {:.1}s to {:.1}s (duration: {:.1}s)\n",
                outro.start_time,
                outro.end_time,
                outro.end_time - outro.start_time
            ));
            report.push_str(&format!(
                "- Expected: {:.1}s to {:.1}s\n",
                EXPECTED_OUTRO_START, EXPECTED_OUTRO_END
            ));
            report.push_str(&format!(
                "- Accuracy: {}\n",
                if result.is_outro_accurate() {
                    "✅ Accurate"
                } else {
                    "⚠️ Off by >20s"
                }
            ));
            report.push_str(&format!("- Confidence: {:.1}%\n", outro.confidence * 100.0));
            report.push_str(&format!("- Episodes: {}\n", outro.episode_count));
        } else {
            report.push_str("**Ending Segment**: ❌ Not found\n");
        }
        report.push_str("\n");
    }

    report.push_str("---\n\n");

    // Performance Comparison
    report.push_str("## Performance Comparison\n\n");

    report.push_str("### Audio Algorithm Performance\n\n");
    let mut audio_by_time = audio_results
        .iter()
        .filter(|r| r.success)
        .collect::<Vec<_>>();
    audio_by_time.sort_by(|a, b| a.execution_time.cmp(&b.execution_time));

    if let Some(fastest) = audio_by_time.first() {
        let fastest_time = fastest.execution_time.as_secs_f64().max(0.01); // Avoid division by very small numbers

        for result in &audio_by_time {
            let time = result.execution_time.as_secs_f64();
            let relative = time / fastest_time;
            // Cap bar length at 40 characters max
            let bar_length = ((time / fastest_time) * 20.0).min(40.0) as usize;
            let bar = "█".repeat(bar_length);

            report.push_str(&format!(
                "**{}**: {:.2}s {}\n",
                result.algorithm_name, time, bar
            ));
            report.push_str(&format!("  {:.2}x relative to fastest\n\n", relative));
        }
    }

    report.push_str("### Video Algorithm Performance\n\n");
    let mut video_by_time = video_results
        .iter()
        .filter(|r| r.success)
        .collect::<Vec<_>>();
    video_by_time.sort_by(|a, b| a.execution_time.cmp(&b.execution_time));

    if !video_by_time.is_empty() {
        let fastest_time = video_by_time
            .first()
            .unwrap()
            .execution_time
            .as_secs_f64()
            .max(0.01);

        for result in &video_by_time {
            let time = result.execution_time.as_secs_f64();
            let relative = time / fastest_time;
            // Cap bar length at 40 characters max
            let bar_length = ((time / fastest_time) * 20.0).min(40.0) as usize;
            let bar = "█".repeat(bar_length);

            report.push_str(&format!(
                "**{}**: {:.2}s {}\n",
                result.algorithm_name, time, bar
            ));
            report.push_str(&format!("  {:.2}x relative to fastest\n\n", relative));
        }
    }

    report.push_str("---\n\n");

    // Final Recommendations
    report.push_str("## Final Recommendations\n\n");

    report.push_str("### When to Use Each Algorithm\n\n");

    report.push_str("**Audio Algorithms**:\n\n");
    report.push_str(
        "- **Chromaprint**: Best for time-shifted segments, robust to encoding differences\n",
    );
    report.push_str("- **MFCC**: Excellent for speech/music, handles encoding artifacts well\n");
    report.push_str("- **SpectralV2**: Balanced speed/accuracy, good general purpose\n");
    report.push_str("- **EnergyBands**: Fast and simple, best for clear theme songs\n");
    report.push_str("- **Fingerprint**: Legacy, may be less robust than newer algorithms\n");
    report.push_str("- **CrossCorrelation**: Legacy, slower but handles phase shifts\n");
    report.push_str("- **SpectralHash**: Legacy, basic spectral matching\n\n");

    report.push_str("**Video Algorithms**:\n\n");
    report.push_str("- **Current**: Proven rolling hash implementation\n");
    report.push_str("- **MultiHash**: Multi-scale perceptual hashing for better accuracy\n");
    report.push_str("- **SsimFeatures**: SSIM + feature matching for robust detection\n\n");

    report.push_str("### Best Overall Choice\n\n");

    // Find the best overall audio algorithm
    let best_audio_overall = audio_results
        .iter()
        .filter(|r| r.is_intro_accurate() && r.is_outro_accurate())
        .min_by_key(|r| r.execution_time);

    if let Some(best) = best_audio_overall {
        report.push_str(&format!(
            "**Audio**: **{}** - Found both segments accurately in {:.2}s\n\n",
            best.algorithm_name,
            best.execution_time.as_secs_f64()
        ));
    }

    let best_video_overall = video_results
        .iter()
        .filter(|r| r.is_intro_accurate() && r.is_outro_accurate())
        .min_by_key(|r| r.execution_time);

    if let Some(best) = best_video_overall {
        report.push_str(&format!(
            "**Video**: **{}** - Found both segments accurately in {:.2}s\n\n",
            best.algorithm_name,
            best.execution_time.as_secs_f64()
        ));
    }

    // Write report to file
    fs::write("COMPREHENSIVE_ALGORITHM_REPORT.md", report)?;
    println!("\n📊 Report generated: COMPREHENSIVE_ALGORITHM_REPORT.md");

    Ok(())
}

#[test]
#[ignore]
fn test_all_algorithms() -> Result<()> {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     COMPREHENSIVE ALGORITHM TEST - ALL AUDIO & VIDEO        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load video files
    let video_files = load_video_files()?;
    println!("✓ Loaded {} video files\n", video_files.len());

    // Create base config
    let config = create_base_config();

    // Extract audio once (reuse for all audio tests)
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    EXTRACTING AUDIO                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    let episode_audio = extract_episode_audio(&video_files)?;
    println!("\n✓ Audio extraction complete\n");

    // Test all audio algorithms
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  TESTING AUDIO ALGORITHMS                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut audio_results = Vec::new();

    audio_results.push(test_audio_algorithm(
        "Chromaprint",
        &episode_audio,
        &config,
        detect_audio_segments_chromaprint,
    ));

    audio_results.push(test_audio_algorithm(
        "MFCC",
        &episode_audio,
        &config,
        detect_audio_segments_mfcc,
    ));

    audio_results.push(test_audio_algorithm(
        "SpectralV2",
        &episode_audio,
        &config,
        detect_audio_segments_spectral_v2,
    ));

    audio_results.push(test_audio_algorithm(
        "EnergyBands",
        &episode_audio,
        &config,
        detect_audio_segments_energy_bands,
    ));

    audio_results.push(test_audio_algorithm(
        "Fingerprint (Legacy)",
        &episode_audio,
        &config,
        detect_audio_segments_fingerprint,
    ));

    audio_results.push(test_audio_algorithm(
        "CrossCorrelation (Legacy)",
        &episode_audio,
        &config,
        detect_audio_segments_correlation,
    ));

    audio_results.push(test_audio_algorithm(
        "SpectralHash (Legacy)",
        &episode_audio,
        &config,
        detect_audio_segments,
    ));

    println!("\n✓ Audio algorithm testing complete\n");

    // Test all video algorithms
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  TESTING VIDEO ALGORITHMS                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    let mut video_results = Vec::new();

    // Test Current algorithm
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    EXTRACTING FRAMES (Current)               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    let mut config_current = config.clone();
    config_current.similarity_algorithm = tvt::similarity::SimilarityAlgorithm::Current;
    let episode_frames_current = extract_episode_frames(&video_files, &config_current)?;
    video_results.push(test_video_algorithm(
        "Current",
        &episode_frames_current,
        &config_current,
    ));

    // Test MultiHash algorithm
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                   EXTRACTING FRAMES (MultiHash)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    let mut config_multihash = config.clone();
    config_multihash.similarity_algorithm = tvt::similarity::SimilarityAlgorithm::MultiHash;
    let episode_frames_multihash = extract_episode_frames(&video_files, &config_multihash)?;
    video_results.push(test_video_algorithm(
        "MultiHash",
        &episode_frames_multihash,
        &config_multihash,
    ));

    // Test SsimFeatures algorithm
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                 EXTRACTING FRAMES (SsimFeatures)             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    let mut config_ssim = config.clone();
    config_ssim.similarity_algorithm = tvt::similarity::SimilarityAlgorithm::SsimFeatures;
    let episode_frames_ssim = extract_episode_frames(&video_files, &config_ssim)?;
    video_results.push(test_video_algorithm(
        "SsimFeatures",
        &episode_frames_ssim,
        &config_ssim,
    ));

    println!("\n✓ Video algorithm testing complete\n");

    // Generate report
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    GENERATING REPORT                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    generate_report(&audio_results, &video_results)?;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                  ALL TESTS COMPLETE!                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    Ok(())
}
