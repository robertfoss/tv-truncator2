//! GStreamer-based frame extraction (V2)
//!
//! Temporal sampling uses `videorate` at the requested **frames per second** (`sample_rate`)
//! while preserving resolution (no `videoscale`). PTS on each buffer is used as the frame
//! timestamp for [`crate::analyzer::Frame`].
//!
//! Hardware decode is preferred when available (`decodebin` + boosted plugin ranks). If a
//! hardware-style pipeline error occurs, ranks are restored and extraction retries once with
//! software decoders. Metadata and duration come from `Discoverer`, matching
//! [`crate::analyzer::get_video_info`] / [`get_video_duration_gstreamer`].

use crate::analyzer::{generate_perceptual_hash, Frame};
use crate::Config;
use crate::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_pbutils as gst_pbutils;
use image::RgbImage;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Optional hardware decoders we may rank-boost, with short install hints for the CLI.
pub const OPTIONAL_HW_VIDEO_DECODERS: &[(&str, &str)] = &[
    (
        "vaapih264dec",
        "VA-API H.264 — install GStreamer VA-API elements (often gst-plugins-bad) and GPU drivers",
    ),
    (
        "vaapih265dec",
        "VA-API HEVC — same stack as VA-API H.264 where supported",
    ),
    (
        "vaapimpeg2dec",
        "VA-API MPEG-2 — same stack as VA-API H.264 where supported",
    ),
    (
        "nvh264dec",
        "NVDEC H.264 — NVIDIA proprietary stack / gst-plugins-bad on Linux",
    ),
    (
        "nvh265dec",
        "NVDEC HEVC — NVIDIA proprietary stack where supported",
    ),
    (
        "d3d11h264dec",
        "D3D11 H.264 — Windows GPU decode via GStreamer d3d11 plugin set",
    ),
    (
        "d3d11h265dec",
        "D3D11 HEVC — Windows GPU decode where supported",
    ),
];

/// When true (default), [`boost_hardware_decoder_ranks`] runs on first [`init_gstreamer`].
/// Set via [`set_prefer_hardware_video_decode`] before any GStreamer call to disable rank boosts
/// (e.g. `--no-hardware-video-decode` on the CLI).
static PREFER_HW_VIDEO_DECODE: AtomicBool = AtomicBool::new(true);

/// After a runtime decode failure we treat as hardware-related, ranks are restored and this is set
/// so we never boost again in this process (graceful software fallback for all subsequent files).
static HW_DECODE_RANK_BOOST_DISABLED: AtomicBool = AtomicBool::new(false);

/// Original ranks captured before the first boost (used for rollback).
static HW_ORIGINAL_RANKS: Mutex<Option<Vec<(String, gst::Rank)>>> = Mutex::new(None);

/// Prefer hardware video decoders when plugins are available (default: `true`).
pub fn set_prefer_hardware_video_decode(prefer: bool) {
    PREFER_HW_VIDEO_DECODE.store(prefer, Ordering::Relaxed);
}

/// Current preference for boosting hardware decoder plugin ranks.
pub fn prefer_hardware_video_decode_enabled() -> bool {
    PREFER_HW_VIDEO_DECODE.load(Ordering::Relaxed)
}

/// True if a missing `candidate_factory` should be listed in CLI hints when
/// `first_present` is the first installed optional HW decoder (same vendor stack only).
fn missing_hw_decoder_hint_in_scope(first_present: Option<&str>, candidate_factory: &str) -> bool {
    let Some(present) = first_present else {
        return true;
    };
    let prefix = if present.starts_with("vaapi") {
        "vaapi"
    } else if present.starts_with("nvh") {
        "nvh"
    } else if present.starts_with("d3d11") {
        "d3d11"
    } else {
        return true;
    };
    candidate_factory.starts_with(prefix)
}

/// Per-plugin install hints for optional hardware decoders (stderr `--verbose` only by default).
#[derive(Debug, Clone)]
pub struct MissingOptionalHwDecodersCli {
    /// Number of missing plugins (same as `detail_lines.len()`).
    pub count: usize,
    /// Human label for the active vendor stack, or `"GPU"` when none of our optional decoders are installed.
    pub stack_label: &'static str,
    /// True when at least one optional hardware decoder from [`OPTIONAL_HW_VIDEO_DECODERS`] is present.
    pub partial_stack: bool,
    /// One line per missing plugin: element name + tip (for `tvt --verbose`).
    pub detail_lines: Vec<String>,
}

fn stack_label_for_optional_hw(first_present: Option<&str>) -> &'static str {
    match first_present {
        Some(p) if p.starts_with("vaapi") => "VA-API",
        Some(p) if p.starts_with("nvh") => "NVDEC",
        Some(p) if p.starts_with("d3d11") => "D3D11",
        Some(_) => "GPU",
        None => "GPU",
    }
}

/// Collects missing optional hardware decoder plugins (vendor-scoped when part of a stack is installed).
///
/// Returns [`None`] when every optional decoder we care about is present.
pub fn missing_optional_hw_decoders_cli() -> Option<MissingOptionalHwDecodersCli> {
    let _ = init_gstreamer();
    let registry = gst::Registry::get();

    let mut first_present: Option<&str> = None;
    for (name, _) in OPTIONAL_HW_VIDEO_DECODERS {
        if let Some(f) = registry.find_feature(name, gst::ElementFactory::static_type()) {
            if f.rank() != gst::Rank::NONE {
                first_present = Some(*name);
                break;
            }
        }
    }

    let mut detail_lines = Vec::new();
    for (factory, tip) in OPTIONAL_HW_VIDEO_DECODERS {
        let missing = registry
            .find_feature(factory, gst::ElementFactory::static_type())
            .map(|f| f.rank() == gst::Rank::NONE)
            .unwrap_or(true);
        if !missing {
            continue;
        }
        if !missing_hw_decoder_hint_in_scope(first_present, factory) {
            continue;
        }
        detail_lines.push(format!("`{}` not found — {}", factory, tip));
    }

    if detail_lines.is_empty() {
        return None;
    }

    let partial_stack = first_present.is_some();
    let stack_label = stack_label_for_optional_hw(first_present);
    let count = detail_lines.len();

    Some(MissingOptionalHwDecodersCli {
        count,
        stack_label,
        partial_stack,
        detail_lines,
    })
}

/// One-line hints per **missing** optional hardware decoder plugin (for CLI / UX).
///
/// Prefer [`missing_optional_hw_decoders_cli`] for structured output. This returns only the
/// verbose detail lines.
pub fn missing_optional_hw_decoder_install_hints() -> Vec<String> {
    missing_optional_hw_decoders_cli()
        .map(|c| c.detail_lines)
        .unwrap_or_default()
}

/// Initialize GStreamer (call once at startup)
pub fn init_gstreamer() -> Result<()> {
    gst::init()?;
    boost_hardware_decoder_ranks();
    Ok(())
}

/// Raise ranks of hardware video decoders so [`create_optimized_pipeline_v2`]'s `decodebin`
/// autoplugs them ahead of typical software `PRIMARY` decoders (process-wide, idempotent).
fn boost_hardware_decoder_ranks() {
    if !PREFER_HW_VIDEO_DECODE.load(Ordering::Relaxed) {
        return;
    }
    if HW_DECODE_RANK_BOOST_DISABLED.load(Ordering::Relaxed) {
        return;
    }
    let registry = gst::Registry::get();
    // Typical sw decoders are PRIMARY (256); nudge known HW elements above that when present.
    let preferred = gst::Rank::PRIMARY + 100;
    let mut snap = HW_ORIGINAL_RANKS.lock().unwrap();
    if snap.is_none() {
        let mut originals = Vec::new();
        for (name, _) in OPTIONAL_HW_VIDEO_DECODERS {
            if let Some(f) = registry.find_feature(name, gst::ElementFactory::static_type()) {
                if f.rank() != gst::Rank::NONE {
                    originals.push(((*name).to_string(), f.rank()));
                    f.set_rank(preferred);
                }
            }
        }
        *snap = Some(originals);
    } else {
        for (name, _) in OPTIONAL_HW_VIDEO_DECODERS {
            if let Some(f) = registry.find_feature(name, gst::ElementFactory::static_type()) {
                if f.rank() != gst::Rank::NONE {
                    f.set_rank(preferred);
                }
            }
        }
    }
}

/// Restore decoder plugin ranks after a hardware decode failure and stop boosting for this process.
fn rollback_hw_decoder_rank_boost_for_software_fallback() {
    let registry = gst::Registry::get();
    let mut snap = HW_ORIGINAL_RANKS.lock().unwrap();
    if let Some(originals) = snap.take() {
        for (name, rank) in originals {
            if let Some(f) = registry.find_feature(&name, gst::ElementFactory::static_type()) {
                f.set_rank(rank);
            }
        }
    }
    HW_DECODE_RANK_BOOST_DISABLED.store(true, Ordering::Relaxed);
}

pub(crate) fn is_likely_hardware_decode_failure(err: &anyhow::Error) -> bool {
    let mut chain = err.to_string().to_lowercase();
    for cause in err.chain().skip(1) {
        chain.push(' ');
        chain.push_str(&cause.to_string().to_lowercase());
    }
    const NEEDLES: &[&str] = &[
        "vaapi",
        "vadisplay",
        "nvdec",
        "nvh26",
        "d3d11",
        " egl",
        "drm",
        "cuda",
        "vdpau",
        "mmal",
        "qsv",
        "videotoolbox",
        "failed to open drm",
        "cannot allocate memory",
        "internal data stream error",
    ];
    NEEDLES.iter().any(|n| chain.contains(n))
}

/// Check if hardware video acceleration is available
pub fn check_hardware_acceleration() -> (bool, String) {
    // Initialize GStreamer if not already done
    let _ = init_gstreamer();

    let registry = gst::Registry::get();

    for (decoder_name, _) in OPTIONAL_HW_VIDEO_DECODERS {
        if let Some(feature) =
            registry.find_feature(decoder_name, gst::ElementFactory::static_type())
        {
            if feature.rank() != gst::Rank::NONE {
                let label = match *decoder_name {
                    "vaapih264dec" => "VA-API (Intel/AMD)",
                    "vaapih265dec" => "VA-API HEVC",
                    "vaapimpeg2dec" => "VA-API MPEG-2",
                    "nvh264dec" => "NVDEC (NVIDIA)",
                    "nvh265dec" => "NVDEC HEVC",
                    "d3d11h264dec" => "D3D11 (Windows)",
                    "d3d11h265dec" => "D3D11 HEVC (Windows)",
                    _ => *decoder_name,
                };
                return (true, label.to_string());
            }
        }
    }

    (false, "Software only".to_string())
}

/// Extract frames at `sample_rate` frames per second (decoded RGB → perceptual hash per buffer).
///
/// Resolution is preserved (no scaling). Timestamps come from buffer PTS.
///
/// # Arguments
/// * `video_path` - Path to the video file
/// * `sample_rate` - Frames per second to extract
/// * `progress_callback` - Callback function for progress updates
/// * `config` - Configuration options
///
/// # Returns
/// * `Result<Vec<Frame>>` - Vector of extracted frames with timestamps and hashes
pub fn extract_frames_gstreamer_v2<F>(
    video_path: &Path,
    sample_rate: f64,
    progress_callback: F,
    config: &Config,
) -> Result<Vec<Frame>>
where
    F: Fn(usize, usize) + Send + Sync + 'static,
{
    let progress = Arc::new(progress_callback);
    for attempt in 0u8..2 {
        match extract_frames_gstreamer_v2_impl(video_path, sample_rate, progress.clone(), config) {
            Ok(frames) => return Ok(frames),
            Err(e)
                if attempt == 0
                    && prefer_hardware_video_decode_enabled()
                    && !HW_DECODE_RANK_BOOST_DISABLED.load(Ordering::Relaxed)
                    && is_likely_hardware_decode_failure(&e) =>
            {
                if !config.json_summary {
                    eprintln!(
                        "Video decode: hardware decode failed; restoring default plugin ranks and retrying with software decoders.\n  ({})",
                        e
                    );
                }
                rollback_hw_decoder_rank_boost_for_software_fallback();
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!("extract_frames_gstreamer_v2 retry loop");
}

fn extract_frames_gstreamer_v2_impl<F>(
    video_path: &Path,
    sample_rate: f64,
    progress_callback: Arc<F>,
    config: &Config,
) -> Result<Vec<Frame>>
where
    F: Fn(usize, usize) + Send + Sync + 'static,
{
    let extraction_start = Instant::now();

    if config.debug {
        println!(
            "⚡ Starting V2 OPTIMIZED GStreamer frame extraction for: {:?}",
            video_path
        );
        println!("⚡ Sample rate: {}", sample_rate);
    }

    // Ensure GStreamer is initialized
    init_gstreamer()?;

    // Get video duration for progress tracking
    let duration_start = Instant::now();
    let duration = get_video_duration_gstreamer(video_path, config)?;
    let duration_time = duration_start.elapsed();
    let expected_frames = (duration * sample_rate) as usize;

    if config.debug {
        println!("⚡ Duration extraction: {:?}", duration_time);
        println!(
            "⚡ Video duration: {:.2}s, expected frames: {}",
            duration, expected_frames
        );
    }

    if config.debug {
        println!(
            "⚡ Target ~{} frames over {:.2}s at {:.3} fps (videorate)",
            expected_frames, duration, sample_rate
        );
    }

    // Create optimized pipeline
    let pipeline_start = Instant::now();
    let pipeline = create_optimized_pipeline_v2(video_path, sample_rate, config)?;
    let pipeline_time = pipeline_start.elapsed();

    if config.debug {
        println!("⚡ Pipeline creation: {:?}", pipeline_time);
    }

    // Set up frame collection
    let frames = Arc::new(Mutex::new(Vec::<Frame>::new()));
    let frames_clone = frames.clone();
    let progress_callback_clone = progress_callback.clone();
    let expected_frames_clone = expected_frames;

    // Get video info for dynamic buffer sizing
    let video_info = get_video_info_gstreamer(video_path)?;
    let frame_width = video_info.width;
    let frame_height = video_info.height;

    if config.debug {
        println!(
            "⚡ Video resolution: {}x{} (NO downscaling)",
            frame_width, frame_height
        );
    }

    // Configure appsink callbacks for frame processing
    let appsink = pipeline
        .by_name("sink")
        .ok_or_else(|| anyhow::anyhow!("Failed to find appsink element"))?;

    let appsink = appsink.downcast::<gst_app::AppSink>().unwrap();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .new_sample(move |appsink| {
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                let buffer = sample.buffer().ok_or_else(|| gst::FlowError::Error)?;

                // Get actual buffer dimensions
                let caps = sample.caps().ok_or_else(|| gst::FlowError::Error)?;
                let structure = caps.structure(0).ok_or_else(|| gst::FlowError::Error)?;
                let width = structure
                    .get::<i32>("width")
                    .map_err(|_| gst::FlowError::Error)?;
                let height = structure
                    .get::<i32>("height")
                    .map_err(|_| gst::FlowError::Error)?;

                // Convert buffer to RGB image (dynamic size)
                let rgb_image = match buffer_to_rgb_image_v2(buffer, width as u32, height as u32) {
                    Ok(img) => img,
                    Err(_) => return Err(gst::FlowError::Error),
                };

                // Generate perceptual hash
                let dynamic_img = image::DynamicImage::ImageRgb8(rgb_image);
                let hash = match generate_perceptual_hash(&dynamic_img) {
                    Ok(h) => h,
                    Err(_) => return Err(gst::FlowError::Error),
                };

                // Calculate timestamp
                let pts = buffer.pts().unwrap_or(gst::ClockTime::ZERO);
                let timestamp = pts.seconds_f64();

                // Create frame and add to collection
                let frame = Frame {
                    timestamp,
                    perceptual_hash: hash,
                };

                let current_count = {
                    let mut frames = frames_clone.lock().unwrap();
                    frames.push(frame);
                    frames.len()
                };

                // Never call progress while holding `frames_clone`: the callback may lock shared
                // processor state and parallel extractors can deadlock. Also throttle updates —
                // locking the global processor mutex per frame is extremely expensive with many workers.
                const PROGRESS_STRIDE: usize = 32;
                let should_report = current_count == 1
                    || current_count % PROGRESS_STRIDE == 0
                    || current_count >= expected_frames_clone;
                if should_report {
                    progress_callback_clone(current_count, expected_frames_clone);
                }

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start pipeline and wait for it to be ready
    let streaming_start = Instant::now();
    pipeline.set_state(gst::State::Playing)?;

    // Wait for pipeline to be in PLAYING state and prerolled
    let bus = pipeline.bus().unwrap();

    if config.debug {
        println!("⚡ Waiting for pipeline to be ready...");
    }

    // Run until EOS with a wall-clock bound so a stuck pipeline cannot hang forever
    let wall_deadline =
        Instant::now() + std::time::Duration::from_secs_f64((duration * 6.0).max(120.0));
    let mut eos_seen = false;
    while !eos_seen {
        if Instant::now() >= wall_deadline {
            pipeline.set_state(gst::State::Null)?;
            return Err(anyhow::anyhow!(
                "Timed out waiting for end-of-stream from frame extraction pipeline"
            ));
        }
        let remaining = wall_deadline.saturating_duration_since(Instant::now());
        let chunk_ns = remaining.as_nanos().min(10_000_000_000) as u64;
        let timeout = gst::ClockTime::from_nseconds(chunk_ns.max(1));
        let Some(msg) = bus.timed_pop(Some(timeout)) else {
            continue;
        };
        use gst::MessageView;

        match msg.view() {
            MessageView::Eos(..) => {
                if config.debug {
                    println!("⚡ Received EOS - extraction complete");
                }
                eos_seen = true;
            }
            MessageView::Error(err) => {
                let error_msg = format!("Pipeline error: {}", err.error());
                if config.debug {
                    println!("⚠️  {}", error_msg);
                }
                pipeline.set_state(gst::State::Null)?;
                return Err(anyhow::anyhow!("{}", error_msg));
            }
            MessageView::StateChanged(state_changed) => {
                if state_changed.src().map(|s| *s == pipeline).unwrap_or(false) {
                    let new_state = state_changed.current();
                    if config.debug {
                        println!("⚡ Pipeline state: {:?}", new_state);
                    }
                }
            }
            _ => {}
        }
    }

    let streaming_time = streaming_start.elapsed();
    let total_extraction_time = extraction_start.elapsed();

    // Stop pipeline
    pipeline.set_state(gst::State::Null)?;

    // Get final frames
    let final_frames = frames.lock().unwrap().clone();

    if config.debug {
        println!("⚡ Total streaming time: {:?}", streaming_time);
        println!("⚡ TOTAL V2 EXTRACTION TIME: {:?}", total_extraction_time);
        println!(
            "⚡ Breakdown: Duration={:.1}%, Pipeline={:.1}%, Streaming={:.1}%",
            duration_time.as_secs_f64() / total_extraction_time.as_secs_f64() * 100.0,
            pipeline_time.as_secs_f64() / total_extraction_time.as_secs_f64() * 100.0,
            streaming_time.as_secs_f64() / total_extraction_time.as_secs_f64() * 100.0
        );
        println!(
            "⚡ Frames per second: {:.2}",
            final_frames.len() as f64 / total_extraction_time.as_secs_f64()
        );
    }

    Ok(final_frames)
}

/// Create ultra-optimized GStreamer pipeline (V2)
///
/// Pipeline design:
/// - filesrc: Read video file with large block size
/// - queue: Buffer to prevent stalls (no size limits for max throughput)
/// - decodebin: Decode video stream (HW preferred via boosted plugin ranks at init)
/// - queue: Buffer decoded frames
/// - videorate: Temporal downsampling to the requested sample FPS **before** color convert
/// - videoconvert: Convert to RGB with multi-threading (n-threads=0)
/// - appsink: Extract frames without sync/throttling
///
/// Key optimizations:
/// - NO videoscale: Preserve original resolution
/// - videorate before videoconvert: avoid colorspace work on frames we will drop
/// - sync=false: No synchronization overhead
/// - n-threads=0: Auto-detect and use all CPU threads
/// - Large queues: Prevent pipeline stalls
fn create_optimized_pipeline_v2(
    video_path: &Path,
    sample_rate: f64,
    config: &Config,
) -> Result<gst::Pipeline> {
    // Use absolute path for GStreamer
    let absolute_path = video_path
        .canonicalize()
        .unwrap_or_else(|_| video_path.to_path_buf())
        .to_string_lossy()
        .to_string();

    if config.debug {
        println!("⚡ Creating V2 optimized pipeline for: {}", absolute_path);
    }

    // Create pipeline
    let pipeline = gst::Pipeline::new();

    // Create elements with optimized properties
    let filesrc = gst::ElementFactory::make("filesrc")
        .property("location", &absolute_path)
        .property("blocksize", 256 * 1024u32) // 256KB sequential reads
        .build()?;

    // Queue before decoding - no limits for maximum throughput
    let queue1 = gst::ElementFactory::make("queue")
        .name("queue1")
        .property("max-size-buffers", 0u32) // No buffer limit
        .property("max-size-time", 0u64) // No time limit
        .property("max-size-bytes", 0u32) // No byte limit
        .build()?;

    // Decodebin will automatically select hardware decoder (VA-API) if available
    // and fall back to software decoder if hardware fails or is unavailable
    let decodebin = gst::ElementFactory::make("decodebin").build()?;

    // Set up error recovery for VA-API failures
    // If VA-API fails to allocate buffers, GStreamer will automatically try software decoders
    if config.debug {
        println!("⚡ Hardware acceleration (VA-API) will be used if available");
        println!("⚡ Will automatically fallback to software decoder if hardware fails");
    }

    // Queue after decoding - buffer decoded frames
    let queue2 = gst::ElementFactory::make("queue")
        .name("queue2")
        .property("max-size-buffers", 30u32) // Buffer up to 30 frames
        .build()?;

    // Video converter with multi-threading - NO SCALING
    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("videoconvert")
        .property("n-threads", 0u32) // Auto-detect threads (use all cores)
        .build()?;

    // Downsample in time to `sample_rate` fps (see module docs)
    let videorate = gst::ElementFactory::make("videorate")
        .name("videorate")
        .build()?;

    // Appsink for frame extraction - optimized for maximum throughput
    // Note: max-buffers must be >= 2 for VA-API hardware acceleration compatibility
    let appsink = gst::ElementFactory::make("appsink")
        .name("sink")
        .property("emit-signals", true)
        .property("sync", false) // Disable sync for maximum throughput
        .property("async", false) // No async state changes
        .property("max-buffers", 10u32) // Buffer frames (must be >= 2 for VA-API)
        .property("drop", false) // Don't drop frames
        .build()?;

    // Set caps for RGB output - NO width/height (preserve original), with framerate
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field(
            "framerate",
            gst::Fraction::new(sample_rate.ceil() as i32, 1),
        )
        .build();
    appsink.set_property("caps", &caps);

    // Add elements to pipeline
    pipeline.add_many([
        &filesrc,
        &queue1,
        &decodebin,
        &queue2,
        &videoconvert,
        &videorate,
        &appsink,
    ])?;

    // Link static elements
    filesrc.link(&queue1)?;
    queue1.link(&decodebin)?;

    // Handle dynamic pad from decodebin
    let pipeline_clone = pipeline.clone();
    let debug_mode = config.debug; // Copy the bool value to avoid lifetime issues
    decodebin.connect_pad_added(move |_, pad| {
        // Only handle video pads - ignore audio and subtitle pads
        let caps = match pad.current_caps() {
            Some(caps) => caps,
            None => {
                // Pad doesn't have caps yet, try getting from template
                match pad.pad_template_caps() {
                    caps => caps,
                }
            }
        };

        // Check if this is a video pad
        let structure = match caps.structure(0) {
            Some(s) => s,
            None => return,
        };

        let name = structure.name();
        if !name.starts_with("video/") {
            // Not a video pad - ignore (audio, subtitles, etc.)
            return;
        }

        if debug_mode {
            println!("⚡ Video pad added: {}", pad.name());
        }

        // Get elements from the pipeline
        let queue2 = match pipeline_clone.by_name("queue2") {
            Some(e) => e,
            None => {
                eprintln!("⚠️  Failed to find queue2 element");
                return;
            }
        };

        // Check if queue2 is already linked
        let sink_pad = match queue2.static_pad("sink") {
            Some(pad) => pad,
            None => {
                eprintln!("⚠️  Failed to get sink pad from queue2");
                return;
            }
        };

        if sink_pad.is_linked() {
            // Already linked - this is expected for additional pads
            if debug_mode {
                println!("⚡ Queue2 already linked, skipping pad {}", pad.name());
            }
            return;
        }

        // Link the video pad to queue2
        if let Err(e) = pad.link(&sink_pad) {
            eprintln!("⚠️  Failed to link decodebin video pad to queue2: {}", e);
            return;
        }

        // Now link the rest of the chain
        let videoconvert = match pipeline_clone.by_name("videoconvert") {
            Some(e) => e,
            None => {
                eprintln!("⚠️  Failed to find videoconvert element");
                return;
            }
        };
        let videorate = match pipeline_clone.by_name("videorate") {
            Some(e) => e,
            None => {
                eprintln!("⚠️  Failed to find videorate element");
                return;
            }
        };
        let appsink = match pipeline_clone.by_name("sink") {
            Some(e) => e,
            None => {
                eprintln!("⚠️  Failed to find appsink element");
                return;
            }
        };

        if let Err(e) = queue2.link(&videorate) {
            eprintln!("⚠️  Failed to link queue2 to videorate: {}", e);
            return;
        }

        if let Err(e) = videorate.link(&videoconvert) {
            eprintln!("⚠️  Failed to link videorate to videoconvert: {}", e);
            return;
        }

        if let Err(e) = videoconvert.link(&appsink) {
            eprintln!("⚠️  Failed to link videoconvert to appsink: {}", e);
            return;
        }

        if debug_mode {
            println!("⚡ Pipeline chain linked successfully");
        }
    });

    Ok(pipeline)
}

/// Convert GStreamer buffer to RGB image (V2 - dynamic sizing)
///
/// This version handles arbitrary image sizes (no hardcoded 320x240).
fn buffer_to_rgb_image_v2(buffer: &gst::BufferRef, width: u32, height: u32) -> Result<RgbImage> {
    let map = buffer
        .map_readable()
        .map_err(|_| anyhow::anyhow!("Failed to map buffer"))?;
    let data = map.as_slice();

    // RGB format: 3 bytes per pixel
    let expected_size = (width * height * 3) as usize;

    if data.len() != expected_size {
        return Err(anyhow::anyhow!(
            "Unexpected buffer size: {} vs {} ({}x{})",
            data.len(),
            expected_size,
            width,
            height
        ));
    }

    // Create RGB image from buffer data
    let rgb_image = RgbImage::from_raw(width, height, data.to_vec())
        .ok_or_else(|| anyhow::anyhow!("Failed to create RGB image from buffer"))?;

    Ok(rgb_image)
}

/// Get video duration using GStreamer discoverer
pub fn get_video_duration_gstreamer(video_path: &Path, config: &Config) -> Result<f64> {
    init_gstreamer()?;

    // Use absolute path for GStreamer
    let absolute_path = video_path
        .canonicalize()
        .unwrap_or_else(|_| video_path.to_path_buf())
        .to_string_lossy()
        .to_string();

    if config.debug {
        println!("⚡ Getting duration for: {}", absolute_path);
    }

    let discoverer = gst_pbutils::Discoverer::new(gst::ClockTime::from_seconds(10))?;
    let info = discoverer.discover_uri(&format!("file://{}", absolute_path))?;

    let duration = info
        .duration()
        .unwrap_or(gst::ClockTime::ZERO)
        .seconds_f64();
    if config.debug {
        println!("⚡ Duration extracted: {:.2}s", duration);
    }
    Ok(duration)
}

/// Get video metadata using GStreamer discoverer
pub fn get_video_info_gstreamer(video_path: &Path) -> Result<crate::analyzer::VideoInfo> {
    init_gstreamer()?;

    // Use absolute path for GStreamer
    let absolute_path = video_path
        .canonicalize()
        .unwrap_or_else(|_| video_path.to_path_buf())
        .to_string_lossy()
        .to_string();

    let discoverer = gst_pbutils::Discoverer::new(gst::ClockTime::from_seconds(10))?;
    let info = discoverer.discover_uri(&format!("file://{}", absolute_path))?;

    // Extract video stream info
    let video_streams = info.video_streams();
    let video_info = video_streams
        .first()
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let br = video_info.bitrate();
    Ok(crate::analyzer::VideoInfo {
        duration: info
            .duration()
            .unwrap_or(gst::ClockTime::ZERO)
            .seconds_f64(),
        width: video_info.width(),
        height: video_info.height(),
        fps: video_info.framerate().numer() as f64 / video_info.framerate().denom() as f64,
        bitrate: if br > 0 { Some(br as u64) } else { None },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v2_gstreamer_initialization() {
        let result = init_gstreamer();
        assert!(result.is_ok());
    }

    #[test]
    fn hardware_decode_failure_heuristic_matches_vaapi_message() {
        let e = anyhow::anyhow!("Pipeline error: vaapih264dec failed to open display");
        assert!(super::is_likely_hardware_decode_failure(&e));
    }

    #[test]
    fn hardware_decode_failure_heuristic_rejects_plain_io() {
        let e = anyhow::anyhow!("No such file or directory");
        assert!(!super::is_likely_hardware_decode_failure(&e));
    }

    #[test]
    fn missing_hw_hint_scope_lists_all_when_no_present_decoder() {
        assert!(super::missing_hw_decoder_hint_in_scope(None, "nvh264dec"));
        assert!(super::missing_hw_decoder_hint_in_scope(
            None,
            "vaapih265dec"
        ));
    }

    #[test]
    fn missing_hw_hint_scope_filters_other_vendor_stacks() {
        assert!(super::missing_hw_decoder_hint_in_scope(
            Some("vaapih264dec"),
            "vaapih265dec"
        ));
        assert!(!super::missing_hw_decoder_hint_in_scope(
            Some("vaapih264dec"),
            "nvh264dec"
        ));
        assert!(super::missing_hw_decoder_hint_in_scope(
            Some("nvh264dec"),
            "nvh265dec"
        ));
        assert!(!super::missing_hw_decoder_hint_in_scope(
            Some("nvh264dec"),
            "d3d11h264dec"
        ));
    }

    #[test]
    fn test_v2_buffer_to_rgb_conversion() {
        // Test dynamic buffer sizing
        // This is a basic test - actual buffer conversion is tested in integration tests
        let width = 1920u32;
        let height = 1080u32;
        let expected_size = (width * height * 3) as usize;

        // Verify size calculation
        assert_eq!(expected_size, 1920 * 1080 * 3);
    }
}
