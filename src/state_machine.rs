//! State machine for tracking video file processing progress

use crate::analyzer::{EpisodeFrames, VideoInfo};
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Represents the current processing state of a video file
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessingState {
    /// Initial state, waiting to start processing (no progress bar)
    Waiting,

    /// Probing video metadata
    Probing { progress: f64 },

    /// Video metadata obtained, ready to extract frames
    Probed { frames_total: usize },

    /// Extracting frames from video
    Extracting {
        frames_processed: usize,
        frames_total: usize,
    },

    /// Frame extraction complete
    Extracted {
        frames_processed: usize,
        frames_total: usize,
    },

    /// Analyzing extracted frames for patterns
    Analyzing {
        frames_analyzed: usize,
        frames_total: usize,
    },

    /// Frame analysis complete
    Analyzed {
        frames_analyzed: usize,
        frames_total: usize,
    },

    /// Finding repeated segments across all files
    FindingRepeated { progress: f64 },

    /// Cutting out repeated segments
    Cutting { progress: f64 },

    /// Processing complete successfully
    Done { output_path: PathBuf },

    /// Processing failed with error
    Failed { error: String },
}

impl ProcessingState {
    /// Check if this is a terminal state (Done or Failed)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            ProcessingState::Done { .. } | ProcessingState::Failed { .. }
        )
    }

    /// Check if this state is Failed
    pub fn is_failed(&self) -> bool {
        matches!(self, ProcessingState::Failed { .. })
    }

    /// Check if this state is Done
    pub fn is_done(&self) -> bool {
        matches!(self, ProcessingState::Done { .. })
    }

    /// Check if this state has reached the first sync point (Probed)
    pub fn reached_first_sync(&self) -> bool {
        matches!(
            self,
            ProcessingState::Probed { .. }
                | ProcessingState::Extracting { .. }
                | ProcessingState::Extracted { .. }
                | ProcessingState::Analyzing { .. }
                | ProcessingState::Analyzed { .. }
                | ProcessingState::FindingRepeated { .. }
                | ProcessingState::Cutting { .. }
                | ProcessingState::Done { .. }
        )
    }

    /// Check if this state has reached the second sync point (Analyzed)
    pub fn reached_second_sync(&self) -> bool {
        matches!(
            self,
            ProcessingState::Analyzed { .. }
                | ProcessingState::FindingRepeated { .. }
                | ProcessingState::Cutting { .. }
                | ProcessingState::Done { .. }
        )
    }

    /// Get progress as a percentage (0.0 to 1.0)
    pub fn progress(&self) -> f64 {
        match self {
            ProcessingState::Waiting => 0.0,
            ProcessingState::Probing { progress } => *progress,
            ProcessingState::Probed { .. } => 1.0,
            ProcessingState::Extracting {
                frames_processed,
                frames_total,
            } => {
                if *frames_total == 0 {
                    0.0
                } else {
                    *frames_processed as f64 / *frames_total as f64
                }
            }
            ProcessingState::Extracted { .. } => 1.0,
            ProcessingState::Analyzing {
                frames_analyzed,
                frames_total,
            } => {
                if *frames_total == 0 {
                    0.0
                } else {
                    *frames_analyzed as f64 / *frames_total as f64
                }
            }
            ProcessingState::Analyzed { .. } => 1.0,
            ProcessingState::FindingRepeated { progress } => *progress,
            ProcessingState::Cutting { progress } => *progress,
            ProcessingState::Done { .. } => 1.0,
            ProcessingState::Failed { .. } => 0.0,
        }
    }

    /// Get a human-readable name for this state
    pub fn name(&self) -> &str {
        match self {
            ProcessingState::Waiting => "Waiting",
            ProcessingState::Probing { .. } => "Probing",
            ProcessingState::Probed { .. } => "Probed",
            ProcessingState::Extracting { .. } => "Extracting",
            ProcessingState::Extracted { .. } => "Extracted",
            ProcessingState::Analyzing { .. } => "Analyzing",
            ProcessingState::Analyzed { .. } => "Analyzed",
            ProcessingState::FindingRepeated { .. } => "Finding Repeated",
            ProcessingState::Cutting { .. } => "Cutting",
            ProcessingState::Done { .. } => "Done",
            ProcessingState::Failed { .. } => "Failed",
        }
    }
}

/// Tracks the processing state and progress of a single video file
#[derive(Debug, Clone)]
pub struct FileProcessor {
    /// Path to the video file being processed
    pub file_path: PathBuf,

    /// Current processing state
    pub state: ProcessingState,

    /// Video metadata (populated after probing)
    pub video_info: Option<VideoInfo>,

    /// Extracted frames (populated after extraction)
    pub frames: Option<EpisodeFrames>,

    /// Analysis results as rolling hashes
    pub analysis_results: Option<Vec<u64>>,

    /// Duplicate segments found (start_time, end_time)
    pub duplicates: Option<Vec<(f64, f64)>>,

    /// Full segment information for summary reporting
    pub common_segments: Option<Vec<crate::segment_detector::CommonSegment>>,

    /// Time when processing started
    start_time: Instant,

    /// Time when current state was entered
    state_start_time: Instant,

    /// Time when processing completed (for terminal states)
    completion_time: Option<Instant>,
}

impl FileProcessor {
    /// Create a new FileProcessor for the given video file
    pub fn new(file_path: PathBuf) -> Self {
        let now = Instant::now();
        Self {
            file_path,
            state: ProcessingState::Waiting,
            video_info: None,
            frames: None,
            analysis_results: None,
            duplicates: None,
            common_segments: None,
            start_time: now,
            state_start_time: now,
            completion_time: None,
        }
    }

    /// Transition to a new state
    pub fn transition_to(&mut self, new_state: ProcessingState) {
        self.state = new_state;
        self.state_start_time = Instant::now();

        // Set completion time for terminal states
        if self.state.is_terminal() && self.completion_time.is_none() {
            self.completion_time = Some(Instant::now());
        }
    }

    /// Get the total elapsed time since processing started
    pub fn total_elapsed(&self) -> Duration {
        if let Some(completion) = self.completion_time {
            completion.duration_since(self.start_time)
        } else {
            Instant::now().duration_since(self.start_time)
        }
    }

    /// Get the elapsed time in the current state
    pub fn state_elapsed(&self) -> Duration {
        if let Some(completion) = self.completion_time {
            completion.duration_since(self.state_start_time)
        } else {
            Instant::now().duration_since(self.state_start_time)
        }
    }

    /// Get the filename without directory path
    pub fn filename(&self) -> String {
        self.file_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    }

    /// Check if processing is finished (Done or Failed)
    pub fn is_finished(&self) -> bool {
        self.state.is_terminal()
    }

    /// Set video info after probing
    pub fn set_video_info(&mut self, info: VideoInfo) {
        self.video_info = Some(info);
    }

    /// Set extracted frames
    pub fn set_frames(&mut self, frames: EpisodeFrames) {
        self.frames = Some(frames);
    }

    /// Set analysis results
    pub fn set_analysis_results(&mut self, results: Vec<u64>) {
        self.analysis_results = Some(results);
    }

    /// Set duplicate segments
    pub fn set_duplicates(&mut self, duplicates: Vec<(f64, f64)>) {
        self.duplicates = Some(duplicates);
    }

    /// Set common segments information
    pub fn set_common_segments(&mut self, segments: Vec<crate::segment_detector::CommonSegment>) {
        self.common_segments = Some(segments);
    }

    /// Update probing progress
    pub fn update_probing(&mut self, progress: f64) {
        if let ProcessingState::Probing { .. } = self.state {
            self.state = ProcessingState::Probing { progress };
        }
    }

    /// Update extraction progress
    pub fn update_extracting(&mut self, frames_processed: usize, frames_total: usize) {
        if let ProcessingState::Extracting { .. } = self.state {
            self.state = ProcessingState::Extracting {
                frames_processed,
                frames_total,
            };
        }
    }

    /// Update analysis progress
    pub fn update_analyzing(&mut self, frames_analyzed: usize, frames_total: usize) {
        if let ProcessingState::Analyzing { .. } = self.state {
            self.state = ProcessingState::Analyzing {
                frames_analyzed,
                frames_total,
            };
        }
    }

    /// Update finding repeated progress
    pub fn update_finding_repeated(&mut self, progress: f64) {
        if let ProcessingState::FindingRepeated { .. } = self.state {
            self.state = ProcessingState::FindingRepeated { progress };
        }
    }

    /// Update cutting progress
    pub fn update_cutting(&mut self, progress: f64) {
        if let ProcessingState::Cutting { .. } = self.state {
            self.state = ProcessingState::Cutting { progress };
        }
    }

    /// Mark processing as complete
    pub fn complete(&mut self, output_path: PathBuf) {
        self.transition_to(ProcessingState::Done { output_path });
    }

    /// Mark processing as failed
    pub fn fail(&mut self, error: String) {
        self.transition_to(ProcessingState::Failed { error });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_progression() {
        let mut processor = FileProcessor::new(PathBuf::from("test.mkv"));

        assert_eq!(processor.state, ProcessingState::Waiting);
        assert!(!processor.is_finished());

        processor.transition_to(ProcessingState::Probing { progress: 0.5 });
        assert_eq!(processor.state.name(), "Probing");
        assert_eq!(processor.state.progress(), 0.5);

        processor.transition_to(ProcessingState::Probed { frames_total: 100 });
        assert!(processor.state.reached_first_sync());

        processor.transition_to(ProcessingState::Extracting {
            frames_processed: 50,
            frames_total: 100,
        });
        assert_eq!(processor.state.progress(), 0.5);

        processor.complete(PathBuf::from("output.mkv"));
        assert!(processor.is_finished());
        assert!(processor.state.is_done());
    }

    #[test]
    fn test_sync_points() {
        let probed = ProcessingState::Probed { frames_total: 100 };
        assert!(probed.reached_first_sync());
        assert!(!probed.reached_second_sync());

        let analyzed = ProcessingState::Analyzed {
            frames_analyzed: 100,
            frames_total: 100,
        };
        assert!(analyzed.reached_first_sync());
        assert!(analyzed.reached_second_sync());

        let probing = ProcessingState::Probing { progress: 0.5 };
        assert!(!probing.reached_first_sync());
        assert!(!probing.reached_second_sync());
    }

    #[test]
    fn test_failure_state() {
        let mut processor = FileProcessor::new(PathBuf::from("test.mkv"));
        processor.fail("Test error".to_string());

        assert!(processor.is_finished());
        assert!(processor.state.is_failed());
        assert!(!processor.state.is_done());
    }
}
