# tv-truncator — IC onboarding runbook

Short guide for the founding engineer before and during week one. Stack is **Rust** (Cargo) with **GStreamer** by default; there is no Python venv for this repo.

## Paperclip workspace path (must match)

The [tv-truncator](/MEM/projects/tv-truncator) project primary workspace `cwd` is:

`/home/robertfoss/projects/tv_truncator_tmp`

On this machine that path is a **symlink** to the git checkout at `/home/robertfoss/projects/tv_truncator`. Your adapter `cwd` should resolve to the same tree Paperclip uses for execution issues.

If you only have `tv_truncator` checked out, either:

- work inside `tv_truncator_tmp` after creating  
  `ln -sfn /path/to/tv_truncator /home/robertfoss/projects/tv_truncator_tmp`, or  
- ask the CTO to align the Paperclip workspace `cwd` with your actual clone path.

## Clone and update

```bash
git clone <repository-url> tv_truncator
cd tv_truncator
git pull --ff-only
```

Use the same remote and branch your team agrees on (default `main` / `master` as applicable).

## Toolchain (no venv)

| Need | Notes |
|------|--------|
| Rust | Install via [rustup](https://rustup.rs/). Use a stable toolchain matching `rust-version` in repo if present. |
| GStreamer | Required for the default build. See **System Dependencies** in the root [README.md](../README.md) (Fedora, Debian, macOS, Arch). |
| ffmpeg/ffprobe | Optional fallback build: `cargo build --release --no-default-features --features ffmpeg`. Useful for smoke checks even when using GStreamer. |

Verify:

```bash
rustc --version
cargo --version
gst-inspect-1.0 --version
```

## Build

```bash
cargo build --release
```

Binary: `target/release/tvt` (see root README for flags).

## Tests

```bash
# Full suite (unit + integration tests under tests/)
cargo test

# One integration test target (examples)
cargo test --test integration_tests

# Verbose output while debugging
cargo test -- --nocapture
```

Benchmarks / extra binaries (if present): `cargo run --release --bin benchmark_extractors -- --help`

## Where `tests/samples` lives

Regression and scenario assets live under the **tests** tree (not a top-level `samples/` folder):

- `tests/samples/` — synthetic and downscaled fixtures (e.g. `synthetic/`, `downscaled_2file/`), often with `segments.json` ground truth.
- Integration tests in `tests/*.rs` load these paths; read the test module for exact expectations.

Browse:

```bash
find tests/samples -type f | head
```

## ffmpeg / ffprobe smoke

```bash
ffmpeg -version
ffprobe -version
ffprobe -hide_banner -i "$(find tests/samples -name '*.mkv' -o -name '*.mp4' | head -1)"
```

(Adjust the file path if no media is present in your partial checkout.)

## Paperclip — hire and first execution tasks

These are tracked in Paperclip (company prefix **MEM**):

- **[MEM-9](/MEM/issues/MEM-9)** — CEO agent-hire for the founding IC (submitted/approved; use as reference for adapter/skills expectations).
- **[MEM-12](/MEM/issues/MEM-12)** — Onboarding execution: Paperclip skills, heartbeat, ffmpeg smoke, `tests/samples` familiarity.

Sync the coordination skills your operator assigns (e.g. `paperclip`, `paperclip-create-agent`) before relying on heartbeats in this workspace.

---

## Day-1 checklist

Use this on the first working session after access is granted.

- [ ] Confirm shell `cwd` matches Paperclip primary workspace (`tv_truncator_tmp` → same tree as this repo).
- [ ] `git status` clean on agreed branch; remote tracking set.
- [ ] `cargo build --release` succeeds.
- [ ] `cargo test` passes (or note failing tests + open issue).
- [ ] GStreamer sanity: `gst-inspect-1.0 matroskademux` (or per README).
- [ ] List `tests/samples` layout and open one `segments.json` next to media you will touch.
- [ ] `ffmpeg -version` / `ffprobe -version` OK.
- [ ] Paperclip skills installed; run a local heartbeat/self-test per org playbook.
- [ ] Read [MEM-12](/MEM/issues/MEM-12) and align with CTO on assignment.
