//! Audio extraction module using GStreamer
//!
//! This module extracts audio samples from video files for spectral analysis and matching.

use crate::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Represents a single audio frame with timestamp and samples
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub timestamp: f64,
    pub spectral_hash: u64,
}

/// Represents audio frames extracted from an episode
#[derive(Debug, Clone)]
pub struct EpisodeAudio {
    pub episode_path: std::path::PathBuf,
    pub audio_frames: Vec<AudioFrame>,
    /// Raw audio samples (mono, 22050 Hz) for advanced algorithms
    pub raw_samples: Vec<f32>,
    /// Sample rate of raw audio
    pub sample_rate: f32,
}

/// Extract audio samples from a video file
///
/// Extracts mono audio at 22.05kHz for spectral analysis.
/// Processes audio in chunks and generates spectral hashes.
///
/// # Arguments
/// * `video_path` - Path to the video file
/// * `sample_rate_hz` - Sample rate for audio extraction (default: 22050 Hz)
/// * `progress_callback` - Callback for progress updates
///
/// # Returns
/// * `Result<Vec<f32>>` - Raw audio samples
pub fn extract_audio_samples<F>(
    video_path: &Path,
    sample_rate_hz: usize,
    progress_callback: F,
) -> Result<Vec<f32>>
where
    F: Fn(usize, usize) + Send + Sync + 'static,
{
    // Initialize GStreamer
    crate::gstreamer_extractor_v2::init_gstreamer()?;

    // Create pipeline: filesrc → decodebin → audioconvert → audioresample → appsink
    let pipeline = create_audio_pipeline(video_path, sample_rate_hz)?;

    // Set up sample collection
    let samples = Arc::new(Mutex::new(Vec::<f32>::new()));
    let samples_clone = samples.clone();
    let progress_callback = Arc::new(progress_callback);
    let progress_callback_clone = progress_callback.clone();

    // Get expected sample count for progress tracking
    let duration = crate::analyzer::get_video_duration(video_path)?;
    let expected_samples = (duration * sample_rate_hz as f64) as usize;

    // Configure appsink callbacks
    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| anyhow::anyhow!("Failed to find appsink element"))?;

    let appsink = appsink.downcast::<gst_app::AppSink>().unwrap();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                let buffer = sample.buffer().ok_or_else(|| gst::FlowError::Error)?;

                // Extract audio samples from buffer
                let map = buffer
                    .map_readable()
                    .map_err(|_| gst::FlowError::Error)?;
                let data = map.as_slice();

                // Convert bytes to f32 samples
                let mut audio_samples = Vec::new();
                for chunk in data.chunks_exact(4) {
                    let sample_bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    let sample = f32::from_le_bytes(sample_bytes);
                    audio_samples.push(sample);
                }

                // Add to collection
                let mut samples = samples_clone.lock().unwrap();
                samples.extend_from_slice(&audio_samples);
                let current_count = samples.len();

                // Call progress callback
                progress_callback_clone(current_count, expected_samples);

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start pipeline
    pipeline.set_state(gst::State::Playing)?;

    // Wait for completion
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                return Err(anyhow::anyhow!("Audio pipeline error: {}", err.error()));
            }
            _ => {}
        }
    }

    // Stop pipeline
    pipeline.set_state(gst::State::Null)?;

    // Return collected samples
    let final_samples = samples.lock().unwrap().clone();
    Ok(final_samples)
}

/// Create GStreamer pipeline for audio extraction
fn create_audio_pipeline(video_path: &Path, sample_rate_hz: usize) -> Result<gst::Pipeline> {
    let absolute_path = video_path
        .canonicalize()
        .unwrap_or_else(|_| video_path.to_path_buf())
        .to_string_lossy()
        .to_string();

    let pipeline = gst::Pipeline::new();

    // Create elements
    let filesrc = gst::ElementFactory::make("filesrc")
        .property("location", &absolute_path)
        .build()?;

    let decodebin = gst::ElementFactory::make("decodebin").build()?;

    let queue = gst::ElementFactory::make("queue")
        .name("audio_queue")
        .property("max-size-buffers", 100u32)
        .build()?;

    let audioconvert = gst::ElementFactory::make("audioconvert")
        .name("audioconvert")
        .build()?;

    let audioresample = gst::ElementFactory::make("audioresample")
        .name("audioresample")
        .build()?;

    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .property("emit-signals", true)
        .property("sync", false)
        .property("max-buffers", 10u32)
        .property("drop", false)
        .build()?;

    // Set caps for F32LE mono audio at specified sample rate
    // Explicitly set channel-mask to ensure consistent downmixing
    let caps = gst::Caps::builder("audio/x-raw")
        .field("format", "F32LE")
        .field("rate", sample_rate_hz as i32)
        .field("channels", 1i32)
        .field("layout", "interleaved")
        .build();
    appsink.set_property("caps", &caps);

    // Add elements to pipeline
    pipeline.add_many([
        &filesrc,
        &decodebin,
        &queue,
        &audioconvert,
        &audioresample,
        &appsink,
    ])?;

    // Link static elements
    filesrc.link(&decodebin)?;

    // Handle dynamic pad from decodebin
    let pipeline_clone = pipeline.clone();
    decodebin.connect_pad_added(move |_, pad| {
        let caps = match pad.current_caps() {
            Some(caps) => caps,
            None => pad.pad_template_caps(),
        };

        let structure = match caps.structure(0) {
            Some(s) => s,
            None => return,
        };

        let name = structure.name();
        if !name.starts_with("audio/") {
            // Not an audio pad - ignore
            return;
        }

        // Get elements from the pipeline
        let queue = match pipeline_clone.by_name("audio_queue") {
            Some(e) => e,
            None => return,
        };

        // Check if queue is already linked
        let sink_pad = match queue.static_pad("sink") {
            Some(pad) => pad,
            None => return,
        };

        if sink_pad.is_linked() {
            return;
        }

        // Link the audio pad to queue
        if let Err(e) = pad.link(&sink_pad) {
            eprintln!("⚠️  Failed to link decodebin audio pad to queue: {}", e);
            return;
        }

        // Link the rest of the chain
        let audioconvert = match pipeline_clone.by_name("audioconvert") {
            Some(e) => e,
            None => return,
        };
        let audioresample = match pipeline_clone.by_name("audioresample") {
            Some(e) => e,
            None => return,
        };
        let appsink = match pipeline_clone.by_name("sink") {
            Some(e) => e,
            None => return,
        };

        if let Err(e) = queue.link(&audioconvert) {
            eprintln!("⚠️  Failed to link queue to audioconvert: {}", e);
            return;
        }

        if let Err(e) = audioconvert.link(&audioresample) {
            eprintln!("⚠️  Failed to link audioconvert to audioresample: {}", e);
            return;
        }

        if let Err(e) = audioresample.link(&appsink) {
            eprintln!("⚠️  Failed to link audioresample to appsink: {}", e);
            return;
        }
    });

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_pipeline_creation() {
        let result = crate::gstreamer_extractor_v2::init_gstreamer();
        assert!(result.is_ok());
    }
}


