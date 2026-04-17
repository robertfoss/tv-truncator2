# Git commit policy (tv-truncator2)

This repository uses Git history as part of the review story: commits should be easy to read, bisect, and revert.

## Mandatory use (engineers)

**Board policy:** any commit that changes **application code** (`src/`), **tests** (`tests/**/*.rs` and related harness code), or **tracked fixtures/samples** (for example `tests/samples/`, golden manifests, or other sample assets checked into git) **must** follow this document for message format, atomicity, hygiene, and verification expectations.

Pure documentation edits under `docs/` alone are encouraged to follow the same conventions but are not the primary enforcement boundary.

## Goals

- Each commit should express **one coherent change** (or one tightly related bundle that would not make sense split).
- Messages should explain **what** changed and, when it is not obvious, **why**.
- Mainline history should stay **buildable and tested** at each commit when practical.

## Message format

Use a **short subject line** (aim for about 72 characters or fewer, hard cap ~100), in the **imperative mood**, with **no trailing period**:

- Good: `Fix audio segment boundary when overlap is partial`
- Avoid: `Fixed stuff.` / `WIP` / `updates`

Optionally use a **Conventional Commits**–style prefix when it helps navigation:

| Prefix     | Use for |
|------------|---------|
| `feat:`    | User-visible behavior or CLI changes |
| `fix:`     | Bug fixes |
| `test:`    | Test-only changes |
| `docs:`    | Documentation |
| `chore:`   | Tooling, CI, formatting, deps without behavior change |
| `refactor:`| Internal restructuring without intended behavior change |
| `perf:`    | Performance improvements |

Examples:

```text
fix: clamp MFCC window when sample rate changes

The extractor assumed a fixed frame size; align with GStreamer caps.
```

```text
docs: describe GStreamer plugin requirements on Fedora
```

### Body

Add a blank line after the subject, then a body when any of the following apply:

- The change is non-trivial or touches subtle behavior.
- There is important context (tradeoffs, rejected alternatives, ticket links).
- Reviewers need deployment or migration notes.

Keep lines in the body wrapped at a reasonable width (about 72 characters) for plain-text readability.

## Scope and hygiene

- Do **not** commit secrets, tokens, large generated binaries, or machine-local paths.
- Prefer **rebasing or squashing** on a feature branch before merge so intermediate “fix typo” noise does not land on main, unless those steps are intentionally preserved for bisect clarity.
- Match project verification before pushing: `cargo fmt`, `cargo clippy`, and `cargo test` (see `.cursorrules`).

## Automated / Paperclip agent commits

When a commit is produced by an automated Paperclip agent run, append **exactly** this trailer (no agent name in the trailer):

```text
Co-Authored-By: Paperclip <noreply@paperclip.ing>
```

Place it at the **end** of the commit message after the body.

## Pull requests

- The PR description should summarize intent and list notable risks or test gaps.
- Prefer a **small number of commits** with clear messages over one giant opaque commit, unless the change is intentionally atomic.
