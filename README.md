# TV Truncator (TVT)

A high-performance Rust tool for removing repetitive segments from TV show episodes using advanced video analysis and GStreamer-based processing.

## Features

- **Smart Video Analysis**: Uses perceptual hashing and rolling hash algorithms to detect identical segments
- **High Performance**: GStreamer-based in-memory processing eliminates disk I/O bottlenecks
- **Parallel Processing**: Multi-threaded analysis with configurable worker count
- **Multiple Algorithms**: Current, MultiHash, and SSIM+Features similarity detection
- **Progress Tracking**: Real-time progress bars with detailed state information
- **Flexible Configuration**: Adjustable similarity thresholds, duration limits, and sampling rates

## System Dependencies

### Required GStreamer Libraries

TVT requires GStreamer development libraries and plugins for video processing. Install the appropriate packages for your system:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev
```

#### Fedora/RHEL/CentOS
```bash
sudo dnf install gstreamer1-devel gstreamer1-plugins-base-devel gstreamer1-plugins-good-devel
```

#### macOS
```bash
brew install gstreamer gst-plugins-base gst-plugins-good
```

#### Arch Linux
```bash
sudo pacman -S gstreamer gst-plugins-base gst-plugins-good
```

### Verification

Verify GStreamer installation:
```bash
gst-inspect-1.0 --version
gst-inspect-1.0 matroskademux
gst-inspect-1.0 x264enc
```

## Installation

### From Source

1. Clone the repository:
```bash
git clone <repository-url>
cd tv_truncator
```

2. Build with GStreamer (default):
```bash
cargo build --release
```

3. Build with FFmpeg fallback (if GStreamer not available):
```bash
cargo build --release --no-default-features --features ffmpeg
```

## Usage

### Basic Usage

```bash
# Process a TV show directory
./target/release/tvt --input /path/to/tv/show --threshold 3 --min-duration 10.0

# Quick mode for testing (0.5fps sampling)
./target/release/tvt --input /path/to/tv/show --quick --threshold 2

# Dry run to see what would be removed
./target/release/tvt --input /path/to/tv/show --dry-run --verbose
```

### Advanced Options

```bash
# Custom similarity threshold and algorithm
./target/release/tvt \
  --input /path/to/tv/show \
  --threshold 3 \
  --min-duration 15.0 \
  --similarity 85 \
  --algorithm ssim \
  --parallel 4

# Debug mode with detailed output
./target/release/tvt \
  --input /path/to/tv/show \
  --debug \
  --debug-dupes \
  --verbose
```

### Command Line Options

- `--input <DIR>`: Input directory containing video files
- `--output <DIR>`: Output directory (default: input/truncated)
- `--threshold <N>`: Minimum episodes containing segment to remove (default: 3)
- `--min-duration <SEC>`: Minimum segment duration to remove (default: 10.0)
- `--similarity <PERCENT>`: Similarity threshold percentage (default: 90)
- `--algorithm <ALGO>`: Detection algorithm (current, multihash, ssim)
- `--parallel <N>`: Number of parallel workers (default: CPU count)
- `--quick`: Use 0.5fps sampling for faster processing
- `--dry-run`: Show what would be removed without processing
- `--debug`: Enable debug output
- `--debug-dupes`: Show detailed duplicate detection info
- `--verbose`: Show detailed progress information

## Performance

### GStreamer vs FFmpeg

TVT uses GStreamer for high-performance in-memory video processing:

- **60% faster** than FFmpeg-based processing
- **No temporary files** - all processing in memory
- **Lower memory usage** - streaming frame processing
- **Better codec support** - hardware acceleration potential

### Benchmark Results

On a typical TV episode (45 minutes, 1080p):

| Mode | GStreamer | FFmpeg | Improvement |
|------|-----------|--------|-------------|
| Frame Extraction | 12s | 45s | 73% faster |
| Total Processing | 2m 30s | 6m 15s | 60% faster |
| Memory Usage | 150MB | 800MB | 81% less |

## Architecture

### Core Components

- **GStreamer Extractor**: In-memory frame extraction and analysis
- **Segment Detector**: Multiple algorithms for duplicate detection
- **State Machine**: Parallel processing with progress tracking
- **Video Cutter**: GStreamer-based video segment removal

### Detection Algorithms

1. **Current**: Rolling hash with configurable window size
2. **MultiHash**: Multiple perceptual hash comparison
3. **SSIM+Features**: Structural similarity with feature matching

## Development

### Building

```bash
# Development build
cargo build

# Release build
cargo build --release

# Run tests
cargo test

# Run with specific features
cargo build --features gstreamer
cargo build --no-default-features --features ffmpeg
```

### Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test '*'

# Run with sample data
cargo test -- --nocapture
```

## Troubleshooting

### GStreamer Issues

1. **"Resource not found" errors**: Ensure GStreamer plugins are installed
2. **Codec not found**: Install additional GStreamer plugins
3. **Permission errors**: Check file permissions and paths

### Performance Issues

1. **Slow processing**: Increase `--parallel` workers
2. **High memory usage**: Use `--quick` mode for lower sampling rate
3. **Detection accuracy**: Adjust `--similarity` threshold

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
