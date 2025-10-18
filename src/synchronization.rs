//! Synchronization point coordination for state machine processing

use crate::state_machine::{FileProcessor, ProcessingState};
use crate::{Config, Result};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Represents synchronization points in the processing pipeline
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncPoint {
    /// After all files have been probed (reached Probed state)
    AfterProbed,
    /// After all files have been analyzed (reached Analyzed state)
    AfterAnalyzed,
}

/// Result of checking a synchronization point
#[derive(Debug, Clone, PartialEq)]
pub enum SyncResult {
    /// All files have reached the sync point, ready to proceed
    Ready,
    /// Some files are still processing, waiting for them
    Waiting { 
        /// Number of files that have reached the sync point
        ready_count: usize, 
        /// Total number of files
        total_count: usize 
    },
    /// One or more files have failed, abort all processing
    Failed { 
        /// List of failed files with their error messages
        failed_files: Vec<(String, String)> 
    },
}

/// Coordinates synchronization between multiple file processors
pub struct ProcessingCoordinator {
    processors: Arc<Mutex<Vec<FileProcessor>>>,
    config: Config,
}

impl ProcessingCoordinator {
    /// Create a new processing coordinator
    pub fn new(processors: Arc<Mutex<Vec<FileProcessor>>>, config: Config) -> Self {
        Self { processors, config }
    }
    
    /// Check if all files have reached the specified sync point
    pub fn check_sync_point(&self, sync_point: SyncPoint) -> Result<SyncResult> {
        let processors_guard = self.processors.lock().unwrap();
        let total_count = processors_guard.len();
        
        if total_count == 0 {
            return Ok(SyncResult::Ready);
        }
        
        let mut ready_count = 0;
        let mut failed_files = Vec::new();
        
        for processor in processors_guard.iter() {
            match &processor.state {
                ProcessingState::Failed { error } => {
                    failed_files.push((processor.filename(), error.clone()));
                }
                state if self.has_reached_sync_point(state, sync_point) => {
                    ready_count += 1;
                }
                _ => {
                    // File hasn't reached the sync point yet
                }
            }
        }
        
        // If any files failed, abort all processing (3a)
        if !failed_files.is_empty() {
            return Ok(SyncResult::Failed { failed_files });
        }
        
        // Check if all files have reached the sync point
        if ready_count == total_count {
            Ok(SyncResult::Ready)
        } else {
            Ok(SyncResult::Waiting { ready_count, total_count })
        }
    }
    
    /// Wait for all files to reach the specified sync point
    /// Returns an error if any file fails before reaching the sync point
    pub fn wait_for_sync(&self, sync_point: SyncPoint) -> Result<()> {
        let mut check_interval = Duration::from_millis(100);
        let max_interval = Duration::from_millis(1000);
        
        loop {
            match self.check_sync_point(sync_point)? {
                SyncResult::Ready => {
                    return Ok(());
                }
                SyncResult::Failed { failed_files } => {
                    let error_msg = failed_files
                        .iter()
                        .map(|(file, error)| format!("{}: {}", file, error))
                        .collect::<Vec<_>>()
                        .join(", ");
                    anyhow::bail!("Processing failed: {}", error_msg);
                }
                SyncResult::Waiting { ready_count, total_count } => {
                    if self.config.verbose {
                        println!(
                            "Waiting for sync point {:?}: {}/{} files ready",
                            sync_point, ready_count, total_count
                        );
                    }
                    
                    // Wait before checking again
                    std::thread::sleep(check_interval);
                    
                    // Gradually increase check interval to avoid busy waiting
                    check_interval = (check_interval * 2).min(max_interval);
                }
            }
        }
    }
    
    /// Check if a specific state has reached the given sync point
    fn has_reached_sync_point(&self, state: &ProcessingState, sync_point: SyncPoint) -> bool {
        match sync_point {
            SyncPoint::AfterProbed => state.reached_first_sync(),
            SyncPoint::AfterAnalyzed => state.reached_second_sync(),
        }
    }
    
    /// Get the current status of all processors
    pub fn get_status(&self) -> Result<Vec<(String, String, f64)>> {
        let processors_guard = self.processors.lock().unwrap();
        let mut status = Vec::new();
        
        for processor in processors_guard.iter() {
            let filename = processor.filename();
            let state_name = processor.state.name().to_string();
            let progress = processor.state.progress();
            status.push((filename, state_name, progress));
        }
        
        Ok(status)
    }
    
    /// Check if all processing is complete (all files in terminal states)
    pub fn is_all_complete(&self) -> Result<bool> {
        let processors_guard = self.processors.lock().unwrap();
        
        if processors_guard.is_empty() {
            return Ok(true);
        }
        
        Ok(processors_guard.iter().all(|p| p.is_finished()))
    }
    
    /// Check if any processing has failed
    pub fn has_failures(&self) -> Result<bool> {
        let processors_guard = self.processors.lock().unwrap();
        Ok(processors_guard.iter().any(|p| p.state.is_failed()))
    }
    
    /// Get count of files in each state
    pub fn get_state_counts(&self) -> Result<std::collections::HashMap<String, usize>> {
        let processors_guard = self.processors.lock().unwrap();
        let mut counts = std::collections::HashMap::new();
        
        for processor in processors_guard.iter() {
            let state_name = processor.state.name().to_string();
            *counts.entry(state_name).or_insert(0) += 1;
        }
        
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::{FileProcessor, ProcessingState};
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::Mutex;
    
    fn create_test_coordinator() -> (ProcessingCoordinator, Arc<Mutex<Vec<FileProcessor>>>) {
        let processors = Arc::new(Mutex::new(vec![
            FileProcessor::new(PathBuf::from("file1.mkv")),
            FileProcessor::new(PathBuf::from("file2.mkv")),
            FileProcessor::new(PathBuf::from("file3.mkv")),
        ]));
        
        let config = Config::default();
        let coordinator = ProcessingCoordinator::new(processors.clone(), config);
        (coordinator, processors)
    }
    
    #[test]
    fn test_sync_point_ready() {
        let (coordinator, processors) = create_test_coordinator();
        
        // Set all files to Probed state
        {
            let mut guard = processors.lock().unwrap();
            for processor in guard.iter_mut() {
                processor.transition_to(ProcessingState::Probed { frames_total: 100 });
            }
        }
        
        let result = coordinator.check_sync_point(SyncPoint::AfterProbed).unwrap();
        assert_eq!(result, SyncResult::Ready);
    }
    
    #[test]
    fn test_sync_point_waiting() {
        let (coordinator, processors) = create_test_coordinator();
        
        // Set only one file to Probed state
        {
            let mut guard = processors.lock().unwrap();
            guard[0].transition_to(ProcessingState::Probed { frames_total: 100 });
            // Others remain in Waiting state
        }
        
        let result = coordinator.check_sync_point(SyncPoint::AfterProbed).unwrap();
        match result {
            SyncResult::Waiting { ready_count: 1, total_count: 3 } => {},
            _ => panic!("Expected Waiting with 1/3 ready"),
        }
    }
    
    #[test]
    fn test_sync_point_failed() {
        let (coordinator, processors) = create_test_coordinator();
        
        // Set one file to failed state
        {
            let mut guard = processors.lock().unwrap();
            guard[0].fail("Test error".to_string());
        }
        
        let result = coordinator.check_sync_point(SyncPoint::AfterProbed).unwrap();
        match result {
            SyncResult::Failed { failed_files } => {
                assert_eq!(failed_files.len(), 1);
                assert_eq!(failed_files[0].0, "file1.mkv");
                assert_eq!(failed_files[0].1, "Test error");
            },
            _ => panic!("Expected Failed result"),
        }
    }
    
    #[test]
    fn test_second_sync_point() {
        let (coordinator, processors) = create_test_coordinator();
        
        // Set all files to Analyzed state
        {
            let mut guard = processors.lock().unwrap();
            for processor in guard.iter_mut() {
                processor.transition_to(ProcessingState::Analyzed {
                    frames_analyzed: 100,
                    frames_total: 100,
                });
            }
        }
        
        let result = coordinator.check_sync_point(SyncPoint::AfterAnalyzed).unwrap();
        assert_eq!(result, SyncResult::Ready);
    }
    
    #[test]
    fn test_status_reporting() {
        let (coordinator, processors) = create_test_coordinator();
        
        // Set different states
        {
            let mut guard = processors.lock().unwrap();
            guard[0].transition_to(ProcessingState::Probing { progress: 0.5 });
            guard[1].transition_to(ProcessingState::Probed { frames_total: 100 });
            guard[2].transition_to(ProcessingState::Extracting {
                frames_processed: 25,
                frames_total: 100,
            });
        }
        
        let status = coordinator.get_status().unwrap();
        assert_eq!(status.len(), 3);
        
        // Check that we have the expected states
        let state_names: Vec<&str> = status.iter().map(|(_, state, _)| state.as_str()).collect();
        assert!(state_names.contains(&"Probing"));
        assert!(state_names.contains(&"Probed"));
        assert!(state_names.contains(&"Extracting"));
    }
}
