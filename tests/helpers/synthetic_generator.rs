//! Synthetic video generator for testing audio/video matching algorithms
//!
//! This module generates test videos by extracting and recombining segments
//! from real videos to create controlled test cases.
#![allow(dead_code)]

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Type of segment matching
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum MatchType {
    Audio,
    Video,
    #[serde(rename = "audio+video")]
    AudioAndVideo,
}

/// Expected segment from segments.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedSegment {
    pub segment_id: String,
    pub match_type: MatchType,
    pub start_time: f64,
    pub end_time: f64,
    pub files: Vec<String>,
}

/// Test case metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub test_name: String,
    pub description: String,
    pub files: Vec<String>,
    pub expected_segments: Vec<ExpectedSegment>,
}

/// Segment source specification
#[derive(Debug, Clone)]
pub struct SegmentSource {
    /// Source video file (from downscaled directory)
    pub source_video: String,
    /// Start time in source video
    pub start_time: f64,
    /// Duration of segment
    pub duration: f64,
    /// Whether to use source audio (true) or replace it
    pub use_source_audio: bool,
    /// If use_source_audio is false, audio source
    pub audio_source: Option<AudioSource>,
}

/// Audio source for segment
#[derive(Debug, Clone)]
pub struct AudioSource {
    /// Source video file for audio
    pub source_video: String,
    /// Start time in audio source
    pub start_time: f64,
}

/// Extract a segment from a video file
fn extract_segment(
    source_path: &Path,
    output_path: &Path,
    start_time: f64,
    duration: f64,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!(
            "  Extracting {:.1}s from {:?} starting at {:.1}s",
            duration,
            source_path.file_name().unwrap(),
            start_time
        );
    }

    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{:.3}", start_time))
        .arg("-i")
        .arg(source_path)
        .arg("-t")
        .arg(format!("{:.3}", duration))
        .arg("-c")
        .arg("copy")
        .arg(output_path)
        .output()
        .context("Failed to run ffmpeg for segment extraction")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("FFmpeg extraction failed: {}", stderr);
    }

    Ok(())
}

/// Replace audio in a video segment
fn replace_audio(
    video_path: &Path,
    audio_source_path: &Path,
    audio_start: f64,
    output_path: &Path,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!(
            "  Replacing audio with audio from {:?} at {:.1}s",
            audio_source_path.file_name().unwrap(),
            audio_start
        );
    }

    // First extract audio segment
    let temp_audio = output_path.with_extension("temp_audio.m4a");
    let video_duration = get_video_duration(video_path)?;

    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-ss")
        .arg(format!("{:.3}", audio_start))
        .arg("-i")
        .arg(audio_source_path)
        .arg("-t")
        .arg(format!("{:.3}", video_duration))
        .arg("-vn")
        .arg("-c:a")
        .arg("copy")
        .arg(&temp_audio)
        .output()
        .context("Failed to extract audio")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let _ = fs::remove_file(&temp_audio);
        anyhow::bail!("Audio extraction failed: {}", stderr);
    }

    // Combine video with new audio
    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(video_path)
        .arg("-i")
        .arg(&temp_audio)
        .arg("-map")
        .arg("0:v")
        .arg("-map")
        .arg("1:a")
        .arg("-c")
        .arg("copy")
        .arg("-shortest")
        .arg(output_path)
        .output()
        .context("Failed to combine video and audio")?;

    // Clean up temp file
    let _ = fs::remove_file(&temp_audio);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Audio replacement failed: {}", stderr);
    }

    Ok(())
}

/// Get video duration using ffprobe
fn get_video_duration(video_path: &Path) -> Result<f64> {
    let output = Command::new("ffprobe")
        .arg("-v")
        .arg("error")
        .arg("-show_entries")
        .arg("format=duration")
        .arg("-of")
        .arg("default=noprint_wrappers=1:nokey=1")
        .arg(video_path)
        .output()
        .context("Failed to run ffprobe")?;

    if !output.status.success() {
        anyhow::bail!("ffprobe failed");
    }

    let duration_str = String::from_utf8_lossy(&output.stdout);
    let duration: f64 = duration_str
        .trim()
        .parse()
        .context("Failed to parse duration")?;

    Ok(duration)
}

/// Concatenate video segments
fn concatenate_segments(
    segment_paths: &[PathBuf],
    output_path: &Path,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("  Concatenating {} segments", segment_paths.len());
    }

    // Create concat file
    let concat_file = output_path.with_extension("concat.txt");
    let mut file = fs::File::create(&concat_file)?;
    for segment_path in segment_paths {
        // Convert to absolute path for concat file
        let abs_path = segment_path
            .canonicalize()
            .unwrap_or_else(|_| segment_path.to_path_buf());
        writeln!(file, "file '{}'", abs_path.display())?;
    }
    drop(file);

    // Concatenate
    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-f")
        .arg("concat")
        .arg("-safe")
        .arg("0")
        .arg("-i")
        .arg(&concat_file)
        .arg("-c")
        .arg("copy")
        .arg(output_path)
        .output()
        .context("Failed to concatenate segments")?;

    // Clean up concat file
    let _ = fs::remove_file(&concat_file);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Concatenation failed: {}", stderr);
    }

    Ok(())
}

/// Generate a single synthetic video file from segment specifications
pub fn generate_video_from_segments(
    output_path: &Path,
    segments: &[SegmentSource],
    downscaled_dir: &Path,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Generating video: {:?}", output_path.file_name().unwrap());
    }

    let temp_dir = output_path.parent().unwrap().join("temp_segments");
    fs::create_dir_all(&temp_dir)?;

    let mut segment_paths = Vec::new();

    for (i, segment) in segments.iter().enumerate() {
        let source_path = downscaled_dir.join(&segment.source_video);
        let temp_segment = temp_dir.join(format!("seg_{:03}.mkv", i));

        // Extract video segment
        extract_segment(
            &source_path,
            &temp_segment,
            segment.start_time,
            segment.duration,
            verbose,
        )?;

        // Replace audio if needed
        if !segment.use_source_audio {
            if let Some(ref audio_src) = segment.audio_source {
                let audio_source_path = downscaled_dir.join(&audio_src.source_video);
                let temp_output = temp_dir.join(format!("seg_{:03}_audio.mkv", i));

                replace_audio(
                    &temp_segment,
                    &audio_source_path,
                    audio_src.start_time,
                    &temp_output,
                    verbose,
                )?;

                // Replace temp segment with audio-replaced version
                fs::remove_file(&temp_segment)?;
                fs::rename(&temp_output, &temp_segment)?;
            }
        }

        segment_paths.push(temp_segment);
    }

    // Concatenate all segments
    if segment_paths.len() > 1 {
        concatenate_segments(&segment_paths, output_path, verbose)?;
    } else {
        // Just copy the single segment
        fs::copy(&segment_paths[0], output_path)?;
    }

    // Clean up temp directory
    for seg_path in &segment_paths {
        let _ = fs::remove_file(seg_path);
    }
    let _ = fs::remove_dir(&temp_dir);

    if verbose {
        println!("  ✓ Generated: {:?}", output_path.file_name().unwrap());
    }

    Ok(())
}

/// Generate a test case directory with multiple videos and segments.json
pub fn generate_test_case(
    output_dir: &Path,
    test_case: &TestCase,
    video_specs: &[Vec<SegmentSource>],
    downscaled_dir: &Path,
    verbose: bool,
) -> Result<()> {
    // Create output directory
    fs::create_dir_all(output_dir)?;

    if verbose {
        println!("\nGenerating test case: {}", test_case.test_name);
        println!("  Description: {}", test_case.description);
        println!("  Files: {}", test_case.files.len());
    }

    // Generate each video file
    for (i, filename) in test_case.files.iter().enumerate() {
        let video_path = output_dir.join(filename);
        generate_video_from_segments(&video_path, &video_specs[i], downscaled_dir, verbose)?;
    }

    // Write segments.json
    let segments_path = output_dir.join("segments.json");
    let json = serde_json::to_string_pretty(&test_case)?;
    fs::write(&segments_path, json)?;

    if verbose {
        println!("  ✓ Generated: segments.json\n");
    }

    Ok(())
}

/// Check if ffmpeg is available
pub fn check_ffmpeg() -> Result<()> {
    let output = Command::new("ffmpeg")
        .arg("-version")
        .output()
        .context("FFmpeg not found. Please install FFmpeg to generate synthetic videos.")?;

    if !output.status.success() {
        anyhow::bail!("FFmpeg is not working correctly");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_type_serialization() {
        let audio = MatchType::Audio;
        let json = serde_json::to_string(&audio).unwrap();
        assert_eq!(json, "\"audio\"");

        let video = MatchType::Video;
        let json = serde_json::to_string(&video).unwrap();
        assert_eq!(json, "\"video\"");

        let both = MatchType::AudioAndVideo;
        let json = serde_json::to_string(&both).unwrap();
        assert_eq!(json, "\"audio+video\"");
    }

    #[test]
    fn test_test_case_serialization() {
        let test_case = TestCase {
            test_name: "test".to_string(),
            description: "Test case".to_string(),
            files: vec!["video1.mkv".to_string()],
            expected_segments: vec![ExpectedSegment {
                segment_id: "seg1".to_string(),
                match_type: MatchType::Audio,
                start_time: 0.0,
                end_time: 10.0,
                files: vec!["video1.mkv".to_string()],
            }],
        };

        let json = serde_json::to_string_pretty(&test_case).unwrap();
        assert!(json.contains("\"audio\""));
    }
}
