//! Parallel processing orchestration with state machine and synchronization points

use crate::analyzer::{get_video_info, EpisodeFrames, VideoInfo};
use crate::audio_extractor::{extract_audio_samples, EpisodeAudio};
use crate::audio_chromaprint::detect_audio_segments_chromaprint;
use crate::audio_correlation::detect_audio_segments_correlation;
use crate::audio_energy_bands::detect_audio_segments_energy_bands;
use crate::audio_fingerprint::detect_audio_segments_fingerprint;
use crate::audio_hasher::process_audio_samples;
use crate::audio_mfcc::detect_audio_segments_mfcc;
use crate::audio_spectral_v2::detect_audio_segments_spectral_v2;
use crate::gstreamer_extractor_v2::{extract_frames_gstreamer_v2, get_video_duration_gstreamer};
use crate::progress_display::spawn_progress_display;
use crate::segment_detector::{
    combine_audio_video_segments, detect_audio_segments, detect_common_segments,
    merge_overlapping_segments,
};
use crate::state_machine::{FileProcessor, ProcessingState};
use crate::synchronization::{ProcessingCoordinator, SyncPoint};
use crate::video_processor::cut_video_segments;
use crate::{Config, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Process multiple video files using the state machine with synchronization points
pub fn process_files_parallel(
    video_files: Vec<PathBuf>,
    config: Config,
) -> Result<Vec<FileProcessor>> {
    // Sort files alphabetically for consistent processing order
    let mut sorted_files = video_files;
    sorted_files.sort_by(|a, b| {
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

    // Create FileProcessor for each file
    let processors: Vec<FileProcessor> = sorted_files
        .into_iter()
        .map(|path| FileProcessor::new(path))
        .collect();

    // Create shared state
    let processors_arc = Arc::new(Mutex::new(processors));
    let coordinator = ProcessingCoordinator::new(processors_arc.clone(), config.clone());

    // Spawn progress display thread
    let progress_handle = spawn_progress_display(processors_arc.clone());

    // Main processing pipeline
    let result = process_pipeline(coordinator, processors_arc.clone(), &config);

    // Wait for progress display to finish
    progress_handle.join().unwrap();

    // Return final results
    let final_processors = processors_arc.lock().unwrap().clone();
    result.map(|_| final_processors)
}

/// Main processing pipeline with synchronization points
fn process_pipeline(
    coordinator: ProcessingCoordinator,
    processors_arc: Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    // Phase 1: Probe all files
    probe_all_files(&processors_arc, config)?;

    // SYNC POINT 1: Wait for all files to reach Probed state
    coordinator.wait_for_sync(SyncPoint::AfterProbed)?;

    // Phase 2: Extract frames in parallel (video and/or audio)
    if !config.audio_only {
        extract_frames_parallel(&processors_arc, config)?;
    }
    
    if config.enable_audio_matching || config.audio_only {
        extract_audio_parallel(&processors_arc, config)?;
    }

    // Phase 3: Analyze frames in parallel
    if !config.audio_only {
        analyze_frames_parallel(&processors_arc, config)?;
    }

    // SYNC POINT 2: Wait for all files to reach Analyzed state
    coordinator.wait_for_sync(SyncPoint::AfterAnalyzed)?;

    // Phase 4: Find repeated segments globally (audio and/or video)
    find_repeated_segments(&processors_arc, config)?;

    // Phase 5: Cut files in parallel
    cut_files_parallel(&processors_arc, config)?;

    Ok(())
}

/// Probe all video files to get metadata in parallel
fn probe_all_files(processors_arc: &Arc<Mutex<Vec<FileProcessor>>>, config: &Config) -> Result<()> {
    let file_paths = {
        let processors_guard = processors_arc.lock().unwrap();
        processors_guard
            .iter()
            .map(|p| p.file_path.clone())
            .collect::<Vec<_>>()
    };

    // Create thread pool with limited workers
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel_workers)
        .build()
        .unwrap();

    pool.install(|| {
        file_paths.par_iter().for_each(|file_path| {
            if let Err(e) = probe_single_file(file_path, processors_arc, config) {
                eprintln!("Failed to probe {:?}: {}", file_path, e);
            }
        });
    });

    Ok(())
}

/// Probe a single video file
fn probe_single_file(
    file_path: &PathBuf,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    // Transition to Probing state
    update_processor_state(processors_arc, file_path, |p| {
        p.transition_to(ProcessingState::Probing { progress: 0.0 });
    })?;

    if config.debug {
        println!(
            "🔄 Started probing: {}",
            file_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    // Add a small delay to make the state visible
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Probe the video
    match get_video_info(file_path) {
        Ok(video_info) => {
            // Update progress to 100%
            update_processor_state(processors_arc, file_path, |p| {
                p.update_probing(1.0);
            })?;

            // Add a small delay to make the state visible
            std::thread::sleep(std::time::Duration::from_millis(100));

            // Transition to Probed state
            let estimated_frames = estimate_frame_count(&video_info, config);
            update_processor_state(processors_arc, file_path, |p| {
                p.set_video_info(video_info);
                p.transition_to(ProcessingState::Probed {
                    frames_total: estimated_frames,
                });
            })?;

            if config.debug {
                println!(
                    "🔄 Completed probing: {} ({} frames estimated)",
                    file_path.file_name().unwrap_or_default().to_string_lossy(),
                    estimated_frames
                );
            }
        }
        Err(e) => {
            update_processor_state(processors_arc, file_path, |p| {
                p.fail(format!("Probing failed: {}", e));
            })?;
            return Err(e);
        }
    }

    Ok(())
}

/// Extract frames from all files in parallel
fn extract_frames_parallel(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    let file_paths = {
        let processors_guard = processors_arc.lock().unwrap();
        processors_guard
            .iter()
            .map(|p| p.file_path.clone())
            .collect::<Vec<_>>()
    };

    // Create thread pool with limited workers
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel_workers)
        .build()
        .unwrap();

    pool.install(|| {
        file_paths.par_iter().for_each(|file_path| {
            if let Err(e) = extract_single_file(file_path, processors_arc, config) {
                eprintln!("Failed to extract frames from {:?}: {}", file_path, e);
            }
        });
    });

    Ok(())
}

/// Extract frames from a single file
fn extract_single_file(
    file_path: &PathBuf,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    // Get video info
    let video_info = get_processor_video_info(processors_arc, file_path)?;

    // Transition to ExtractingVideo state
    let estimated_frames = estimate_frame_count(&video_info, config);
    update_processor_state(processors_arc, file_path, |p| {
        p.transition_to(ProcessingState::ExtractingVideo {
            frames_processed: 0,
            frames_total: estimated_frames,
        });
    })?;

    if config.debug {
        println!(
            "🔄 Started extracting video: {} ({} frames)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            estimated_frames
        );
    }

    // Extract frames with progress tracking
    let sample_rate = if config.quick { 0.5 } else { 5.0 };
    let frames =
        extract_frames_with_state_machine_progress(file_path, sample_rate, processors_arc, config)?;
    let actual_frame_count = frames.frames.len();

    // Transition to ExtractedVideo state
    update_processor_state(processors_arc, file_path, |p| {
        p.set_frames(frames);
        p.transition_to(ProcessingState::ExtractedVideo {
            frames_processed: actual_frame_count,
            frames_total: actual_frame_count,
        });
    })?;

    if config.debug {
        println!(
            "🔄 Completed extracting: {} ({} frames extracted)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            actual_frame_count
        );
    }

    Ok(())
}

/// Extract frames with progress tracking that updates the state machine using GStreamer
fn extract_frames_with_state_machine_progress(
    video_path: &PathBuf,
    sample_rate: f64,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<EpisodeFrames> {
    use std::time::Instant;

    let extraction_start = Instant::now();

    // Get video duration for progress calculation using GStreamer
    let duration_start = Instant::now();
    let duration = get_video_duration_gstreamer(video_path, config)?;
    let duration_time = duration_start.elapsed();
    let _expected_frames = (duration * sample_rate) as usize;

    if config.debug {
        println!("🕐 Duration extraction: {:?}", duration_time);
    }

    // Clone data needed for the callback
    let video_path_clone = video_path.clone();
    let processors_arc_clone = processors_arc.clone();
    let video_path_for_callback = video_path_clone.clone();

    // Extract frames using optimized V2 extractor with progress callback
    // This extractor automatically uses hardware acceleration (VA-API) if available,
    // and falls back to software decoding if hardware fails
    let frames = extract_frames_gstreamer_v2(
        &video_path,
        sample_rate,
        move |current, total| {
            // Update the state machine with current progress
            if let Err(e) = update_processor_state(
                &processors_arc_clone,
                &video_path_for_callback,
                |p| {
                    p.update_extracting_video(current, total);
                },
            ) {
                eprintln!("Failed to update processor state: {}", e);
            }
        },
        config,
    )?;

    let total_extraction_time = extraction_start.elapsed();

    if config.debug {
        println!("🕐 GStreamer extraction: {:?}", total_extraction_time);
        println!(
            "🕐 Breakdown: Duration={:.2}%, Extraction={:.2}%",
            duration_time.as_secs_f64() / total_extraction_time.as_secs_f64() * 100.0,
            (total_extraction_time - duration_time).as_secs_f64()
                / total_extraction_time.as_secs_f64()
                * 100.0
        );
    }

    Ok(EpisodeFrames {
        episode_path: video_path.clone(),
        frames,
    })
}

/// Extract audio from all files in parallel
fn extract_audio_parallel(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    let file_paths = {
        let processors_guard = processors_arc.lock().unwrap();
        processors_guard
            .iter()
            .map(|p| p.file_path.clone())
            .collect::<Vec<_>>()
    };

    if config.debug {
        println!("🎵 Starting audio extraction phase...");
    }

    // Create thread pool with limited workers
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel_workers)
        .build()
        .unwrap();

    pool.install(|| {
        file_paths.par_iter().for_each(|file_path| {
            if let Err(e) = extract_audio_single_file(file_path, processors_arc, config) {
                eprintln!("Failed to extract audio from {:?}: {}", file_path, e);
            }
        });
    });

    if config.debug {
        println!("🎵 Completed audio extraction phase");
    }

    Ok(())
}

/// Extract audio from a single file
fn extract_audio_single_file(
    file_path: &PathBuf,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    if config.debug {
        println!(
            "🎵 Extracting audio: {}",
            file_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    let sample_rate_hz = 22050; // 22.05 kHz mono audio
    let frame_rate = if config.quick { 0.5 } else { 1.0 }; // Audio frames per second

    // Extract raw audio samples with progress tracking
    let processors_clone = processors_arc.clone();
    let file_path_clone = file_path.clone();
    let audio_samples = extract_audio_samples(
        file_path,
        sample_rate_hz,
        move |current, total| {
            let _ = update_processor_state(&processors_clone, &file_path_clone, |p| {
                // Use ExtractingAudio state
                p.state = ProcessingState::ExtractingAudio {
                    samples_processed: current,
                    samples_total: total,
                };
            });
        },
    )?;

    // Process audio samples into audio frames with spectral hashes
    let audio_frames = process_audio_samples(&audio_samples, sample_rate_hz as f32, frame_rate)?;

    // Store audio frames in processor (including raw samples for advanced algorithms)
    let episode_audio = EpisodeAudio {
        episode_path: file_path.clone(),
        audio_frames,
        raw_samples: audio_samples.clone(), // Keep raw samples for chromaprint/MFCC/etc
        sample_rate: sample_rate_hz as f32,
    };

    if config.debug {
        println!(
            "🎵 Extracted {} audio frames from {}",
            episode_audio.audio_frames.len(),
            file_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    // Update processor with audio frames and transition to ExtractedAudio
    let audio_frame_count = episode_audio.audio_frames.len();
    update_processor_state(processors_arc, file_path, |p| {
        p.audio_frames = Some(episode_audio);
        p.transition_to(ProcessingState::ExtractedAudio {
            samples_processed: audio_frame_count,
            samples_total: audio_frame_count,
        });
    })?;

    if config.debug {
        println!(
            "🎵 Completed extracting audio: {} ({} frames)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            audio_frame_count
        );
    }

    Ok(())
}

/// Analyze frames from all files in parallel
fn analyze_frames_parallel(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    if config.debug {
        println!("🔄 Starting analysis phase...");
    }

    let file_paths = {
        let processors_guard = processors_arc.lock().unwrap();
        processors_guard
            .iter()
            .map(|p| p.file_path.clone())
            .collect::<Vec<_>>()
    };

    // Create thread pool with limited workers
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel_workers)
        .build()
        .unwrap();

    pool.install(|| {
        file_paths.par_iter().for_each(|file_path| {
            if let Err(e) = analyze_single_file(file_path, processors_arc, config) {
                eprintln!("Failed to analyze frames from {:?}: {}", file_path, e);
            }
        });
    });

    if config.debug {
        println!("🔄 Completed analysis phase...");
    }

    Ok(())
}

/// Analyze frames from a single file
fn analyze_single_file(
    file_path: &PathBuf,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    if config.debug {
        println!(
            "🔄 Started analyzing: {}",
            file_path.file_name().unwrap_or_default().to_string_lossy()
        );
    }

    // Get frames
    let frames = get_processor_frames(processors_arc, file_path)?;

    // Transition to Analyzing state
    let total_frames = frames.frames.len();
    update_processor_state(processors_arc, file_path, |p| {
        p.transition_to(ProcessingState::Analyzing {
            frames_analyzed: 0,
            frames_total: total_frames,
        });
    })?;

    // Add a small delay to make the state visible
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Perform actual frame analysis with progress tracking
    let analysis_results =
        analyze_frames_for_patterns_with_progress(&frames, processors_arc, file_path, config)?;

    // Transition to Analyzed state
    update_processor_state(processors_arc, file_path, |p| {
        p.set_analysis_results(analysis_results);
        p.transition_to(ProcessingState::Analyzed {
            frames_analyzed: total_frames,
            frames_total: total_frames,
        });
    })?;

    if config.debug {
        println!(
            "🔄 Completed analyzing: {} ({} frames analyzed)",
            file_path.file_name().unwrap_or_default().to_string_lossy(),
            total_frames
        );
    }

    Ok(())
}

/// Find repeated segments across all files (global analysis)
fn find_repeated_segments(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    // Get all frames and audio from all files
    let (all_frames, all_audio) = {
        let processors_guard = processors_arc.lock().unwrap();
        let mut frames = Vec::new();
        let mut audio = Vec::new();

        for processor in processors_guard.iter() {
            if let Some(ref file_frames) = processor.frames {
                frames.push(file_frames.clone());
            }
            if let Some(ref file_audio) = processor.audio_frames {
                audio.push(file_audio.clone());
            }
        }
        (frames, audio)
    };

    if all_frames.is_empty() && all_audio.is_empty() {
        return Ok(());
    }

    // Update all processors to FindingRepeated state
    {
        let mut processors_guard = processors_arc.lock().unwrap();
        for processor in processors_guard.iter_mut() {
            if !processor.state.is_failed() {
                processor.transition_to(ProcessingState::FindingRepeated { progress: 0.0 });
            }
        }
    }

    // Update progress to 10% (starting segment detection)
    {
        let mut processors_guard = processors_arc.lock().unwrap();
        for processor in processors_guard.iter_mut() {
            if !processor.state.is_failed() {
                processor.update_finding_repeated(0.1);
            }
        }
    }

    // Detect segments based on configuration
    let final_segments = if config.audio_only {
        // Audio-only mode
        if all_audio.is_empty() {
            Vec::new()
        } else {
            // Update progress to 30% (audio detection starting)
            {
                let mut processors_guard = processors_arc.lock().unwrap();
                for processor in processors_guard.iter_mut() {
                    if !processor.state.is_failed() {
                        processor.update_finding_repeated(0.3);
                    }
                }
            }
            
            let audio_segments = match config.audio_algorithm {
                crate::AudioAlgorithm::Chromaprint => {
                    detect_audio_segments_chromaprint(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::Mfcc => {
                    detect_audio_segments_mfcc(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::SpectralV2 => {
                    detect_audio_segments_spectral_v2(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::EnergyBands => {
                    detect_audio_segments_energy_bands(&all_audio, config, config.debug_dupes)?
                }
                // Legacy algorithms
                crate::AudioAlgorithm::SpectralHash => {
                    detect_audio_segments(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::CrossCorrelation => {
                    detect_audio_segments_correlation(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::Fingerprint => {
                    detect_audio_segments_fingerprint(&all_audio, config, config.debug_dupes)?
                }
            };
            // Update progress to 80% (audio detection complete)
            {
                let mut processors_guard = processors_arc.lock().unwrap();
                for processor in processors_guard.iter_mut() {
                    if !processor.state.is_failed() {
                        processor.update_finding_repeated(0.8);
                    }
                }
            }
            
            merge_overlapping_segments(audio_segments)
        }
    } else if config.enable_audio_matching {
        // Combined audio + video mode
        
        // Update progress to 20% (video detection starting)
        {
            let mut processors_guard = processors_arc.lock().unwrap();
            for processor in processors_guard.iter_mut() {
                if !processor.state.is_failed() {
                    processor.update_finding_repeated(0.2);
                }
            }
        }
        
        let video_segments = if all_frames.is_empty() {
            Vec::new()
        } else {
            detect_common_segments(&all_frames, config, config.debug_dupes)?
        };
        
        // Update progress to 50% (audio detection starting)
        {
            let mut processors_guard = processors_arc.lock().unwrap();
            for processor in processors_guard.iter_mut() {
                if !processor.state.is_failed() {
                    processor.update_finding_repeated(0.5);
                }
            }
        }
        
        let audio_segments = if all_audio.is_empty() {
            Vec::new()
        } else {
            // Use selected audio algorithm
            match config.audio_algorithm {
                crate::AudioAlgorithm::Chromaprint => {
                    detect_audio_segments_chromaprint(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::Mfcc => {
                    detect_audio_segments_mfcc(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::SpectralV2 => {
                    detect_audio_segments_spectral_v2(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::EnergyBands => {
                    detect_audio_segments_energy_bands(&all_audio, config, config.debug_dupes)?
                }
                // Legacy algorithms
                crate::AudioAlgorithm::SpectralHash => {
                    detect_audio_segments(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::CrossCorrelation => {
                    detect_audio_segments_correlation(&all_audio, config, config.debug_dupes)?
                }
                crate::AudioAlgorithm::Fingerprint => {
                    detect_audio_segments_fingerprint(&all_audio, config, config.debug_dupes)?
                }
            }
        };

        // Update progress to 80% (merging segments)
        {
            let mut processors_guard = processors_arc.lock().unwrap();
            for processor in processors_guard.iter_mut() {
                if !processor.state.is_failed() {
                    processor.update_finding_repeated(0.8);
                }
            }
        }

        // Combine and merge
        let combined = combine_audio_video_segments(video_segments, audio_segments);
        merge_overlapping_segments(combined)
    } else {
        // Video-only mode (default)
        
        // Update progress to 30% (video detection starting)
        {
            let mut processors_guard = processors_arc.lock().unwrap();
            for processor in processors_guard.iter_mut() {
                if !processor.state.is_failed() {
                    processor.update_finding_repeated(0.3);
                }
            }
        }
        
        if all_frames.is_empty() {
            Vec::new()
        } else {
            let video_segments = detect_common_segments(&all_frames, config, config.debug_dupes)?;
            
            // Update progress to 80% (merging)
            {
                let mut processors_guard = processors_arc.lock().unwrap();
                for processor in processors_guard.iter_mut() {
                    if !processor.state.is_failed() {
                        processor.update_finding_repeated(0.8);
                    }
                }
            }
            
            merge_overlapping_segments(video_segments)
        }
    };

    // Update all processors with duplicate segments
    {
        let mut processors_guard = processors_arc.lock().unwrap();
        for processor in processors_guard.iter_mut() {
            if !processor.state.is_failed() {
                let duplicates = final_segments
                    .iter()
                    .map(|s| (s.start_time, s.end_time))
                    .collect();
                processor.set_duplicates(duplicates);
                processor.set_common_segments(final_segments.clone());
                processor.transition_to(ProcessingState::Cutting { progress: 0.0 });
            }
        }
    }

    Ok(())
}

/// Cut files in parallel
fn cut_files_parallel(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    let file_paths = {
        let processors_guard = processors_arc.lock().unwrap();
        processors_guard
            .iter()
            .map(|p| p.file_path.clone())
            .collect::<Vec<_>>()
    };

    // Create thread pool with limited workers
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.parallel_workers)
        .build()
        .unwrap();

    pool.install(|| {
        file_paths.par_iter().for_each(|file_path| {
            if let Err(e) = cut_single_file(file_path, processors_arc, config) {
                eprintln!("Failed to cut file {:?}: {}", file_path, e);
            }
        });
    });

    Ok(())
}

/// Cut a single file
fn cut_single_file(
    file_path: &PathBuf,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    config: &Config,
) -> Result<()> {
    // Get duplicates
    let duplicates = get_processor_duplicates(processors_arc, file_path)?;

    if config.dry_run {
        // In dry run, just mark as done
        update_processor_state(processors_arc, file_path, |p| {
            p.complete(file_path.clone());
        })?;
        return Ok(());
    }

    // Create output directory
    std::fs::create_dir_all(&config.output_dir)?;

    // Generate output filename
    let filename = file_path.file_name().unwrap_or_default();
    let output_path = config.output_dir.join(filename);

    // Update progress
    update_processor_state(processors_arc, file_path, |p| {
        p.update_cutting(0.5);
    })?;

    // Cut the video
    cut_video_segments(file_path, &output_path, &duplicates)?;

    // Update progress and mark as done
    update_processor_state(processors_arc, file_path, |p| {
        p.update_cutting(1.0);
        p.complete(output_path);
    })?;

    Ok(())
}

/// Helper functions

fn update_processor_state<F>(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    file_path: &PathBuf,
    update_fn: F,
) -> Result<()>
where
    F: FnOnce(&mut FileProcessor),
{
    let mut processors_guard = processors_arc.lock().unwrap();
    if let Some(processor) = processors_guard
        .iter_mut()
        .find(|p| p.file_path == *file_path)
    {
        update_fn(processor);
    }
    Ok(())
}

fn get_processor_video_info(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    file_path: &PathBuf,
) -> Result<VideoInfo> {
    let processors_guard = processors_arc.lock().unwrap();
    if let Some(processor) = processors_guard.iter().find(|p| p.file_path == *file_path) {
        processor
            .video_info
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Video info not available"))
    } else {
        Err(anyhow::anyhow!("File not found in processors"))
    }
}

fn get_processor_frames(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    file_path: &PathBuf,
) -> Result<EpisodeFrames> {
    let processors_guard = processors_arc.lock().unwrap();
    if let Some(processor) = processors_guard.iter().find(|p| p.file_path == *file_path) {
        processor
            .frames
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Frames not available"))
    } else {
        Err(anyhow::anyhow!("File not found in processors"))
    }
}

fn get_processor_duplicates(
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    file_path: &PathBuf,
) -> Result<Vec<(f64, f64)>> {
    let processors_guard = processors_arc.lock().unwrap();
    if let Some(processor) = processors_guard.iter().find(|p| p.file_path == *file_path) {
        Ok(processor.duplicates.clone().unwrap_or_default())
    } else {
        Err(anyhow::anyhow!("File not found in processors"))
    }
}

fn estimate_frame_count(video_info: &VideoInfo, config: &Config) -> usize {
    let sample_rate = if config.quick { 0.5 } else { 5.0 };
    let duration = video_info.duration;
    (duration * sample_rate) as usize
}

/// Analyze frames for patterns with progress tracking
fn analyze_frames_for_patterns_with_progress(
    frames: &EpisodeFrames,
    processors_arc: &Arc<Mutex<Vec<FileProcessor>>>,
    file_path: &PathBuf,
    _config: &Config,
) -> Result<Vec<u64>> {
    use crate::hasher::RollingHash;

    let total_frames = frames.frames.len();
    let mut analysis_results = Vec::with_capacity(total_frames);

    // Use a window size of 5 for rolling hash
    let mut rolling_hash = RollingHash::new(5);

    for (i, frame) in frames.frames.iter().enumerate() {
        // Add frame hash to rolling hash
        if let Some(hash_value) = rolling_hash.add(frame.perceptual_hash) {
            analysis_results.push(hash_value);
        } else {
            // Window not full yet, use the frame hash directly
            analysis_results.push(frame.perceptual_hash);
        }

        // Update progress every 5 frames or on the last frame
        if i % 5 == 0 || i == total_frames - 1 {
            update_processor_state(processors_arc, file_path, |p| {
                p.update_analyzing(i + 1, total_frames);
            })?;

            // Add a small delay to make progress visible
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }

    Ok(analysis_results)
}

fn _analyze_frames_for_patterns(frames: &EpisodeFrames, _config: &Config) -> Result<Vec<u64>> {
    // TODO: Implement actual frame analysis
    // For now, return dummy data
    Ok(vec![0u64; frames.frames.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_frame_count() {
        let video_info = VideoInfo {
            duration: 100.0,
            width: 1920,
            height: 1080,
            fps: 30.0,
            bitrate: Some(5000000),
        };

        let dry_run_config = Config {
            dry_run: true,
            quick: false,
            ..Config::default()
        };
        let quick_config = Config {
            dry_run: false,
            quick: true,
            ..Config::default()
        };
        let normal_config = Config {
            dry_run: false,
            quick: false,
            ..Config::default()
        };

        let dry_run_frames = estimate_frame_count(&video_info, &dry_run_config);
        let quick_frames = estimate_frame_count(&video_info, &quick_config);
        let normal_frames = estimate_frame_count(&video_info, &normal_config);

        assert_eq!(dry_run_frames, 500); // 100 * 5.0 (same as normal mode)
        assert_eq!(quick_frames, 50); // 100 * 0.5
        assert_eq!(normal_frames, 500); // 100 * 5.0
    }
}
