# TinyTrace Implementation Plan

## Purpose

This document is the execution roadmap for the B.Tech project version of TinyTrace. It is designed to keep implementation aligned with the frozen architecture in [trace_lightwieght.md](trace_lightwieght.md), while also pushing the codebase toward research-grade quality and later publication readiness.

This plan removes week-based scheduling and instead focuses on phase gates, deliverables, and exit criteria.

## Current Reality Check

The current workspace already contains a runnable TinyTrace prototype, but it is not yet a fully architecture-faithful TinyTrace implementation.

### Already complete

- standalone TinyTrace codebase exists
- event schema `(timestamp, score, caption)` exists
- `time -> score -> caption` generation flow exists
- decoder-only LCEM-style prototype exists
- real video loading exists
- QVHighlights subset conversion exists
- TRACE-style QVHighlights highlight metrics exist
- one-sample prediction inspection exists
- downloader script for QVHighlights subset exists
- README and repo hygiene are in place

### Still incomplete / important gaps

- real MobileCLIP-S0 is integrated, frozen, and shape-verified
- spatial MobileCLIP features feed learned slot compression
- frame timestamps use TRACE-style discrete numeric embeddings
- prompt/event serialization is centralized and parser round-trips are tested
- decoder generation is intentionally restricted to batch size one until per-sequence head state is implemented
- the architecture-aligned model has not yet passed the required synthetic overfit gate

## Guiding Rules

1. The frozen architecture takes priority over convenience.
2. TRACE-master is a reference only, never a runtime dependency.
3. No large-scale real training begins before architecture validation succeeds.
4. Synthetic overfit is a required gate before serious real-data experiments.
5. Every phase should leave behind reproducible artifacts, not just code changes.

## M0 — Architecture and Tensor Contract

### Goal

Lock down the exact implementation contract before further architecture work.

### Tasks

- audit each implemented module against `trace_lightwieght.md`
- define required tensor contracts between:
  - frame loader
  - visual encoder
  - compression module
  - time encoder
  - prompt builder
  - LCEM decoder
  - parser
- document expected shapes and token ordering
- identify which parts are:
  - complete
  - partial
  - placeholder
  - missing

### Deliverable

- architecture/tensor contract document committed in repo

### Status

- `completed`

Notes:
- `ARCHITECTURE_CONTRACT.md` records the executable tensor and token contract

## Phase 1 — Architecture Alignment

### Goal

Make the implementation match the frozen TinyTrace architecture.

### Phase 1A — MobileCLIP integration

#### Tasks

- replace the custom conv visual encoder with MobileCLIP-S0
- keep MobileCLIP fully standalone inside TinyTrace
- freeze MobileCLIP parameters
- keep frozen BatchNorm behavior in evaluation mode during training
- implement official preprocessing:
  - bilinear resize preserving aspect ratio
  - center crop
  - `[0, 1]` tensor range (the official Apple v1 S0 transform has no mean/std normalization)
  - tensor shape expectations
- expose spatial features before global pooling
- adapt MobileCLIP output into TinyTrace visual token flow
- verify the output shape matches compression input requirements

#### Deliverable

- `Video -> MobileCLIP -> spatial features` works end to end

#### Status

- `completed`

### Phase 1B — TRACE-style token pipeline

#### Tasks

- keep the learned lightweight compression module for now
- replace continuous frame-time MLP approach with TRACE-style discrete 13-token numeric time encoding where required by the frozen design
- ensure prompt order matches:
  - `Visual Tokens + Time Tokens + Instruction Tokens -> LCEM`
- verify event parser still reconstructs:
  - timestamp
  - score
  - caption
- verify head switching remains `time -> score -> caption`

#### Deliverable

- full forward pass matching architecture specification

#### Status

- `completed`

Notes:
- each frame contributes four compressed visual tokens and six discrete time embeddings
- the real MobileCLIP-S0 path and full LCEM forward path are shape-verified

## Phase 2 — Engineering Stabilization

### Goal

Make the system reliable and maintainable.

### Tasks

- fix and standardize configuration loading
- restore config cleanly from checkpoints
- centralize serialization code paths
- add safe malformed-generation handling
- improve batch handling
- add padding and masking for variable-length videos
- clean duplicate logic between training, eval, and parser paths
- verify dependency list is complete
- decode JSON video samples lazily and cache decoded frames safely
- verify the MobileCLIP checkpoint checksum and pin the package revision
- add validation, resume, periodic checkpointing, loss history, and prediction snapshots

### Deliverable

- stable training and inference pipeline

### Status

- `completed` for the supported single-sample generation contract

Notes:
- config restoration, shared serialization, defensive parsing, dependencies, and variable-frame masking are implemented
- JSON construction is metadata-only; video decoding occurs in `__getitem__` and supports an atomic persistent frame cache
- MobileCLIP setup is pinned and the official checkpoint is checksum-verified
- training supports validation, best/latest/periodic checkpoints, resume, history, and prediction snapshots
- named optimizer groups separate compression, embeddings, LCEM, task heads, and MobileCLIP parameters
- linear warmup, cosine decay, gradient clipping, AMP, validation-selected checkpoints, and early stopping are implemented
- machine-readable training logs include per-task losses, target counts, learning rates, gradient norms, and throughput
- canonical training profiles reject unknown fields and preserve all optimization/stability settings
- training uses deterministic epoch-addressable ordering and validation never uses random frame fallback
- gradient accumulation handles complete and partial windows and keeps scheduler steps optimizer-aligned
- resumable checkpoints are versioned and include RNG, counters, selection state, run metadata, and bounded retention
- validation artifacts include raw token IDs, parsed output, ground truth, termination metadata, and parser warnings
- best-loss and best-primary-metric checkpoint roles are distinct and early stopping records its criterion and best step
- the zero-dropout baseline remains stable; dropout 0.1 is an isolated experimental configuration
- batched generation still requires independent per-sequence head-switching state

## Phase 3 — Tests Integrated With Development

### Goal

Build confidence continuously instead of postponing all tests.

### Required tests

- numeric tokenizer encode/decode
- event serialization/parsing
- MobileCLIP output shapes
- compression output shapes
- LCEM forward pass
- `time -> score -> caption` switching
- checkpoint save/load
- variable-length batch collation
- malformed generation safety

### Deliverable

- passing test suite for all core invariants

### Exit criterion

- all tests pass

### Status

- `completed`

Notes:
- 15 focused tests cover every required invariant listed above

## Phase 4 — Synthetic Validation Gate

### Goal

Prove the architecture can learn before spending time on real-video scale-up.

### Tasks

- create a tiny synthetic dataset with 4 to 8 samples
- train until the model nearly memorizes them
- inspect timestamp, score, and caption predictions directly
- if overfitting fails:
  - stop
  - debug architecture/training
  - do not proceed to real-data scale-up

### Deliverable

- clear evidence that TinyTrace can overfit tiny synthetic data

### Exit criterion

- near-perfect memorization on the tiny synthetic set

### Status

- `completed`

Notes:
- the architecture-aligned MobileCLIP version overfit 4 deterministic visual samples
- measured result: final loss `0.000692`, exact decoded matches `4/4`
- artifacts are written to the ignored `outputs-synthetic-overfit/` directory

## Phase 5 — Dataset Preparation

### Goal

Prepare a clean real-data subset only after architecture validation.

### Tasks

- validate dataset schema before large downloads
- validate timestamp and score ranges
- download 50 to 100 valid QVHighlights videos
- verify each video is readable
- regenerate TinyTrace JSON from clean valid clips
- create explicit train/validation splits
- keep initial experiments small and controlled

### Deliverable

- clean 50 to 100 video TinyTrace-ready subset

### Exit criterion

- train/val subsets exist and all files are decodable

### Status

- `partial`

Notes:
- downloader exists
- tiny real subset exists
- 50 to 100 valid clean subset is not prepared yet

## Phase 6 — Initial Training

### Goal

Validate the real-data pipeline on a controlled subset.

### Tasks

- train on 50 to 100 valid QVHighlights videos
- run 10 to 20 epochs, adjusted by convergence
- save checkpoints every epoch
- track training and validation loss
- save sample predictions after each epoch
- inspect failure patterns in:
  - timestamps
  - scores
  - captions

### Deliverable

- initial real-data training run with checkpoints and prediction examples

### Exit criterion

- training completes cleanly and produces interpretable prediction outputs

### Status

- `pending`

Notes:
- earlier smoke training used the superseded placeholder architecture
- architecture-aligned real-data training has not started

## Phase 7 — TRACE-Style Training Strategy

### Goal

Implement staged optimization closer to TRACE training logic.

### Tasks

- add stage controls before larger real-data training
- Stage 1:
  - freeze MobileCLIP and LCEM
  - train compression, time/score embeddings, and task heads
- Stage 2:
  - keep MobileCLIP frozen
  - jointly fine-tune LCEM and task modules
- evaluate after each stage
- compare Stage 1 and Stage 2 behavior

### Deliverable

- two-stage TinyTrace training pipeline

### Exit criterion

- both stages run reproducibly and improve prediction quality relative to smoke baseline

### Status

- `pending`

## Phase 8 — Evaluation and Thesis Artifacts

### Goal

Produce reproducible thesis-quality outputs.

### Tasks

- save final checkpoints
- save exact configuration files
- save dataset splits
- save training logs
- save sample predictions
- save QVHighlights metrics
- record parameter count
- record memory usage
- record inference latency
- produce qualitative examples for thesis figures
- prepare limitations and reproducibility notes

### Deliverable

- thesis-ready artifact bundle

### Exit criterion

- a third party can reproduce the reported runs using repo artifacts

### Status

- `pending`

## Milestones

| Milestone | Success Criteria | Status |
|---|---|---|
| M0 | Architecture/tensor contract documented | Completed |
| M1a | MobileCLIP integrated and frozen; spatial feature extraction works | Completed |
| M1b | MobileCLIP feature shapes verified against compression path | Completed |
| M2 | TRACE-style token pipeline implemented | Completed |
| M3 | Core tests pass | Completed (15 tests) |
| M4 | Model overfits 4–8 synthetic samples | Completed (4/4 exact) |
| M5 | 50–100 valid QVHighlights videos prepared | Pending |
| M6 | Initial real-data training completes with checkpoints and predictions | Pending |
| M7 | Two-stage training implemented | Pending |
| M8 | Evaluation metrics and thesis-ready artifacts generated | Pending |

## Immediate Next Actions

1. Validate QVHighlights schema, timestamp ranges, score ranges, and video paths.
2. Add per-sequence adaptive-head state before enabling batched generation.
3. Prepare explicit, deterministic train/validation split files.
4. Download and verify 50 to 100 matching QVHighlights clips.
5. Begin controlled real-data training only after every selected clip passes validation.

## Important Warning

Do not treat the current prototype as final TinyTrace.

The current codebase is useful because:

- it proves the pipeline can run
- it exposes the real-data failure modes
- it provides a stable base for the next architecture-faithful implementation steps

But before serious BTP experiments and before publication claims, MobileCLIP integration and architecture tightening are mandatory.

## Priority 4 — Representation Improvements

### Implementation status

- `implemented`: named 8/12/16/24/32 frame configurations
- `implemented`: named 20/48/64 caption-budget configurations
- `implemented`: explicit frame, caption, and generation safety limits
- `implemented`: deterministic temporal-coverage analysis without acquisition
- `implemented`: per-sample caption truncation metadata and aggregate logging
- `implemented`: patch-summary and compressed-slot diversity diagnostics
- `implemented`: trained-checkpoint latency, throughput, and GPU-memory benchmark
- `implemented`: single-variable ablation validation and sequential execution
- `implemented`: comprehensive Priority 4 tests
- `pending evidence`: execute real controlled runs and select a default from measured tradeoffs

### Decision rule

The code does not promote a candidate automatically. A comparison is incomplete
until quality metrics, training measurements, and inference benchmark artifacts
exist for both baseline and candidate. Frame experiments advance one rung at a
time; caption 48 and 64 are independent candidates against the 20-token
reference. MobileCLIP, LCEM capacity, tokenizer, preprocessing, split, seed,
optimizer, effective batch size, training duration, and checkpoint-selection
policy remain fixed.
