//! Progress display management for state machine processing

use crate::state_machine::{FileProcessor, ProcessingState};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Spawn a progress display thread that shows all file processors
pub fn spawn_progress_display(
    processors: Arc<Mutex<Vec<FileProcessor>>>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let multi = MultiProgress::new();
        let mut bars: HashMap<PathBuf, ProgressBar> = HashMap::new();
        
        loop {
            let processors_guard = match processors.try_lock() {
                Ok(guard) => guard,
                Err(_) => {
                    // If we can't get the lock, wait a bit and try again
                    thread::sleep(Duration::from_millis(50));
                    continue;
                }
            };
            
            // Create/update progress bars for all files
            for processor in processors_guard.iter() {
                // Skip Waiting state - no progress bar shown
                if matches!(processor.state, ProcessingState::Waiting) {
                    continue;
                }
                
                let bar = bars.entry(processor.file_path.clone())
                    .or_insert_with(|| create_file_progress_bar(&multi));
                
                update_progress_bar(bar, processor);
            }
            
            // Check if all complete
            let all_finished = processors_guard.iter().all(|p| p.is_finished());
            
            drop(processors_guard);
            
            if all_finished {
                break;
            }
            
            thread::sleep(Duration::from_millis(100));
        }
        
        // Clean up progress bars
        multi.clear().unwrap();
    })
}

/// Create a progress bar for a single file
fn create_file_progress_bar(multi: &MultiProgress) -> ProgressBar {
    let pb = multi.add(ProgressBar::new(100));
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb
}

/// Update a progress bar based on file processor state
fn update_progress_bar(bar: &ProgressBar, processor: &FileProcessor) {
    let state_info = format_state_info(processor);
    let filename = processor.filename();
    
    // Format message based on state
    let message = match &processor.state {
        ProcessingState::Waiting => {
            // This should not happen as we skip Waiting state
            format!("{} - {}", state_info, filename)
        }
        ProcessingState::Probing { .. } => {
            format!("{}: {}", state_info, filename)
        }
        ProcessingState::Probed { frames_total } => {
            format!("[{}] {}: {}", frames_total, state_info, filename)
        }
        ProcessingState::Extracting { frames_processed, frames_total } => {
            format!("[{}/{}] {}: {}", frames_processed, frames_total, state_info, filename)
        }
        ProcessingState::Extracted { frames_processed, frames_total } => {
            format!("[{}/{}] {}: {}", frames_processed, frames_total, state_info, filename)
        }
        ProcessingState::Analyzing { frames_analyzed, frames_total } => {
            format!("[{}/{}] {}: {}", frames_analyzed, frames_total, state_info, filename)
        }
        ProcessingState::Analyzed { frames_analyzed, frames_total } => {
            format!("[{}/{}] {}: {}", frames_analyzed, frames_total, state_info, filename)
        }
        ProcessingState::FindingRepeated { .. } => {
            format!("{}: {}", state_info, filename)
        }
        ProcessingState::Cutting { .. } => {
            format!("{}: {}", state_info, filename)
        }
        ProcessingState::Done { .. } => {
            format!("✓ {}: {}", state_info, filename)
        }
        ProcessingState::Failed { error } => {
            format!("✗ {}: {} ({})", state_info, filename, error)
        }
    };
    
    // Set progress and length based on state
    let (progress, length) = get_progress_info(processor);
    bar.set_position(progress);
    bar.set_length(length);
    
    // Create progress bar visualization
    let progress_bar = if length > 0 {
        let filled = (progress as f64 / length as f64 * 40.0) as usize;
        let filled = filled.min(40); // Cap at 40 to prevent overflow
        let empty = 40 - filled;
        format!("[{}{}]", "#".repeat(filled), "-".repeat(empty))
    } else {
        "[----------------------------------------]".to_string()
    };
    
    // Create the full message with progress bar
    let full_message = format!("{} {}", progress_bar, message);
    bar.set_message(full_message);
}

/// Get progress information for a state
fn get_progress_info(processor: &FileProcessor) -> (u64, u64) {
    match &processor.state {
        ProcessingState::Waiting => (0, 100),
        ProcessingState::Probing { progress } => {
            let pos = (progress * 100.0) as u64;
            (pos, 100)
        }
        ProcessingState::Probed { .. } => (100, 100),
        ProcessingState::Extracting { frames_processed, frames_total } => {
            let total = *frames_total as u64;
            let pos = *frames_processed as u64;
            (pos, total.max(1)) // Avoid division by zero
        }
        ProcessingState::Extracted { frames_processed, frames_total } => {
            let total = *frames_total as u64;
            let pos = *frames_processed as u64;
            (pos, total.max(1))
        }
        ProcessingState::Analyzing { frames_analyzed, frames_total } => {
            let total = *frames_total as u64;
            let pos = *frames_analyzed as u64;
            (pos, total.max(1))
        }
        ProcessingState::Analyzed { frames_analyzed, frames_total } => {
            let total = *frames_total as u64;
            let pos = *frames_analyzed as u64;
            (pos, total.max(1))
        }
        ProcessingState::FindingRepeated { progress } => {
            let pos = (progress * 100.0) as u64;
            (pos, 100)
        }
        ProcessingState::Cutting { progress } => {
            let pos = (progress * 100.0) as u64;
            (pos, 100)
        }
        ProcessingState::Done { .. } => (100, 100),
        ProcessingState::Failed { .. } => (0, 100),
    }
}

/// Format elapsed time as MM:SS
fn _format_elapsed_time(duration: Duration) -> String {
    let total_secs = duration.as_secs();
    let minutes = total_secs / 60;
    let seconds = total_secs % 60;
    format!("[{:02}:{:02}]", minutes, seconds)
}

/// Format state information for display
fn format_state_info(processor: &FileProcessor) -> String {
    match &processor.state {
        ProcessingState::Waiting => "Waiting".to_string(),
        ProcessingState::Probing { .. } => "Probing".to_string(),
        ProcessingState::Probed { .. } => "Probed".to_string(),
        ProcessingState::Extracting { .. } => "Extracting".to_string(),
        ProcessingState::Extracted { .. } => "Extracted".to_string(),
        ProcessingState::Analyzing { .. } => "Analyzing".to_string(),
        ProcessingState::Analyzed { .. } => "Analyzed".to_string(),
        ProcessingState::FindingRepeated { .. } => "Finding Repeated".to_string(),
        ProcessingState::Cutting { .. } => "Cutting".to_string(),
        ProcessingState::Done { .. } => "Done".to_string(),
        ProcessingState::Failed { .. } => "Failed".to_string(),
    }
}

/// Print a summary of processing results
pub fn print_summary(processors: &[FileProcessor]) {
    let total_files = processors.len();
    let completed = processors.iter().filter(|p| p.state.is_done()).count();
    let failed = processors.iter().filter(|p| p.state.is_failed()).count();
    
    println!("\n=== Processing Summary ===");
    println!("Total files: {}", total_files);
    println!("Completed: {}", completed);
    println!("Failed: {}", failed);
    
    if failed > 0 {
        println!("\nFailed files:");
        for processor in processors.iter().filter(|p| p.state.is_failed()) {
            if let ProcessingState::Failed { error } = &processor.state {
                println!("  {}: {}", processor.filename(), error);
            }
        }
    }
    
    if completed > 0 {
        println!("\nCompleted files:");
        for processor in processors.iter().filter(|p| p.state.is_done()) {
            if let ProcessingState::Done { output_path } = &processor.state {
                println!("  {} -> {}", processor.filename(), output_path.display());
            }
        }
    }
    
    // Show timing information
    let total_time = processors.iter()
        .map(|p| p.total_elapsed())
        .max()
        .unwrap_or_default();
    
    println!("\nTotal processing time: {:02}:{:02}", 
        total_time.as_secs() / 60, 
        total_time.as_secs() % 60
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::{FileProcessor, ProcessingState};
    use std::path::PathBuf;
    use std::time::Duration;
    
    #[test]
    fn test_format_elapsed_time() {
        assert_eq!(_format_elapsed_time(Duration::from_secs(0)), "[00:00]");
        assert_eq!(_format_elapsed_time(Duration::from_secs(30)), "[00:30]");
        assert_eq!(_format_elapsed_time(Duration::from_secs(60)), "[01:00]");
        assert_eq!(_format_elapsed_time(Duration::from_secs(90)), "[01:30]");
        assert_eq!(_format_elapsed_time(Duration::from_secs(3661)), "[61:01]");
    }
    
    #[test]
    fn test_format_state_info() {
        let mut processor = FileProcessor::new(PathBuf::from("test.mkv"));
        
        assert_eq!(format_state_info(&processor), "Waiting");
        
        processor.transition_to(ProcessingState::Probing { progress: 0.5 });
        assert_eq!(format_state_info(&processor), "Probing");
        
        processor.transition_to(ProcessingState::Probed { frames_total: 100 });
        assert_eq!(format_state_info(&processor), "Probed");
        
        processor.transition_to(ProcessingState::Extracting { 
            frames_processed: 5, 
            frames_total: 20 
        });
        assert_eq!(format_state_info(&processor), "Extracting");
        
        processor.transition_to(ProcessingState::Analyzing { 
            frames_analyzed: 3, 
            frames_total: 8 
        });
        assert_eq!(format_state_info(&processor), "Analyzing");
        
        processor.transition_to(ProcessingState::Cutting { progress: 0.6 });
        assert_eq!(format_state_info(&processor), "Cutting");
        
        processor.complete(PathBuf::from("output.mkv"));
        assert_eq!(format_state_info(&processor), "Done");
        
        processor.fail("Test error".to_string());
        assert_eq!(format_state_info(&processor), "Failed");
    }
    
    #[test]
    fn test_get_progress_info() {
        let mut processor = FileProcessor::new(PathBuf::from("test.mkv"));
        
        // Test different states
        processor.transition_to(ProcessingState::Probing { progress: 0.5 });
        assert_eq!(get_progress_info(&processor), (50, 100));
        
        processor.transition_to(ProcessingState::Extracting {
            frames_processed: 25,
            frames_total: 100,
        });
        assert_eq!(get_progress_info(&processor), (25, 100));
        
        processor.transition_to(ProcessingState::Done { 
            output_path: PathBuf::from("output.mkv") 
        });
        assert_eq!(get_progress_info(&processor), (100, 100));
    }
    
    #[test]
    fn test_print_summary() {
        let mut processors = vec![
            FileProcessor::new(PathBuf::from("file1.mkv")),
            FileProcessor::new(PathBuf::from("file2.mkv")),
            FileProcessor::new(PathBuf::from("file3.mkv")),
        ];
        
        processors[0].complete(PathBuf::from("output1.mkv"));
        processors[1].fail("Test error".to_string());
        // processors[2] remains in Waiting state
        
        // This test just ensures the function doesn't panic
        print_summary(&processors);
    }
}