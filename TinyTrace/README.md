# TinyTrace

## QVHighlights Phase-A Training

The current **Phase A v3** trains only query-conditioned QVHighlights saliency.
It uniformly samples 128 frames and directly predicts 75 scores, one for each
two-second bin of a 150-second clip. It does not autoregressively generate
timestamp characters, score characters, captions, `<sync>` transitions, or
event boundaries; all five of those legacy loss weights are zero. This direct
head removes the termination and event-count collapse seen in the Phase-A-v1
run.

From the repository root, run the complete fail-closed GPU workflow with one
command:

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/run_phase_a_pipeline.py --device cuda
```

The pipeline verifies or publishes the immutable cleaned split (1,155 train and
132 validation videos after recording 67 invalid exclusions), precomputes
frozen MobileCLIP-S0 FP16 features for four clips, overfits those real videos,
compares the fit against its seeded untrained baseline, verifies query swaps
materially change scores and visual swaps/removal degrade temporal ranking, completes
the full feature cache, runs
exactly 100 optimizer steps as a smoke gate, and only then starts the fresh full run in
`TinyTrace/outputs-qvh-phase-a-v3-full`. It refuses non-empty gate and full-run
directories. Use `--skip-full` to run only the validation gates.

The full profile is
`TinyTrace/configs/train_qvhighlights_phase_a_v3.json`; the model config is
`TinyTrace/configs/tinytrace_qvhighlights_phase_a_v3.json`. MobileCLIP stays
frozen, the required feature cache avoids repeating its 128-frame forward pass
each epoch, the effective batch is 8 (batch 1, accumulation 8), and the learning
rate is `0.0001`. The Phase-A-only gradient cap is 5.0 after a real initial
batch measured a finite norm of about 4.37; the 100-step smoke gate rejects a
run when 50% or more optimizer updates still require clipping.

All 1,354 source filenames request a 150-second window. Phase A v3 probes the
files again at launch and excludes every clip below 149.5 seconds. This records
66 truncated downloads plus one probe failure. The earlier immutable v2 split
is preserved for audit only; it accepted 49 truncated clips whose sparse labels
happened to end before the missing media tail.

Training reports `qvh_mean_score_proxy_*` because the converted split contains
mean saliency scores. Those metrics are explicitly proxies. Exact official
QVHighlights Fair/Good/VeryGood mAP and Hit@1 require the original per-bin
scores from all three annotators; do not relabel the mean-score proxy as an
official result.

TinyTrace is a lightweight reimplementation of the TRACE causal event modeling idea for constrained environments. It keeps the TRACE event structure and generation order:

- `time -> score -> caption`
- structured event output
- causal autoregressive decoding

while replacing the heavy backbone with a smaller visual encoder and a compact decoder-only transformer.

That `time -> score -> caption` path is the legacy autoregressive path retained
for later Phase-B caption work. Current Phase A uses the direct 75-bin head
described above.

## What This Repo Contains

- lightweight TinyTrace model code
- training and evaluation scripts
- a synthetic sample dataset for smoke testing
- a converter for small QVHighlights subsets
- TRACE-style highlight evaluation for the QVHighlights setting

## Current Status

TinyTrace is currently an architecture-aligned prototype:

- MobileCLIP-S0 is the visual encoder and remains frozen during training
- pre-pooling MobileCLIP spatial features are compressed with learned slots
- each frame contributes TRACE-style fixed-width discrete time embeddings
- Phase A uses 128 frame groups plus the query and a direct 75-bin saliency head
- the legacy Phase-B prefix can additionally append autoregressive event tokens
- variable-length frame batches are padded and attention-masked
- Phase-A frame decoding is batched into a compact uint8 cache, then frozen MobileCLIP patch features are cached in FP16
- checkpoints include optimizer/config/history state and support resume
- training supports validation, best/latest checkpoints, 75-bin prediction snapshots, and conditioning diagnostics
- the Phase-A launcher enforces real-video overfit and exact 100-step gates before scaling
- focused architecture tests pass

It is not yet a final trained model. The pipeline must pass its overfit,
conditioning, and smoke gates on the target GPU before the full result can be
treated as a valid Phase-A run.

The MobileCLIP checkpoint is intentionally not committed. The setup helper
downloads Apple's official `mobileclip_s0.pt` and verifies its SHA-256 digest.

## TinyTrace Phase Plan

TinyTrace is meant to become a **general VTG edge model**, not only a
QVHighlights trainer. The target is to mimic TRACE's mental model while
remaining lightweight enough for constrained hardware.

The final TinyTrace interface should support both:

- `video + query -> query-relevant events`
- `video only -> general event sequence`

and the shared event structure remains:

- `timestamp`
- `score`
- `caption`

### Phase A: Query-Conditioned Grounding

**Status:** implemented and already trained in multiple iterations.

**Current dataset**

- `QVHighlights`

**Current input/output**

- input: `video + query`
- output: one dense 75-bin saliency curve over a 150-second clip

**What Phase A has already done**

- established the cleaned QVHighlights-based training split
- reformatted the raw QVHighlights data into TinyTrace Phase-A training JSON
- trained the direct dense 75-bin saliency head
- validated the run with overfit, conditioning, cache, and smoke gates
- produced reusable Phase-A checkpoints for warm-starting later phases

**Why this phase exists**

- QVHighlights is naturally strong for temporal grounding and saliency
- it teaches TinyTrace where important moments are for a given query
- it is a good first step, but it is not yet the full final TinyTrace behavior

### Phase B: Video-Only Event Generation

**Status:** planned next phase.

**Main goal**

- move TinyTrace beyond query-only supervision
- train the model to read a video and output structured events directly

**Target input/output**

- input: `video only`
- output: event sequence with `timestamp + caption`
- score may be weak/default at first when the dataset does not provide strong
  saliency labels

**Expected dataset direction**

- `ActivityNet Captions` as the main open-domain Phase-B dataset
- `YouCook2` as an optional helper or warmup dataset for cleaner instructional
  event structure

**How training should start**

- **not from scratch**
- warm-start from the **best Phase A checkpoint**

**Why this phase exists**

- the final TinyTrace model must support video-only inference
- caption-supporting datasets are needed for that behavior
- `ActivityNet Captions` better matches a general-domain TinyTrace than a
  cooking-only dataset
- TRACE also supports video-only dense captioning tasks, so TinyTrace should too

### Phase C: General VTG Unification

**Status:** planned after a stable Phase B.

**Main goal**

- combine query-conditioned and video-only training into one shared edge model

**Target input/output**

- `video + query -> query-relevant events`
- `video only -> general events`

**Expected dataset direction**

- `QVHighlights` for query-conditioned timestamp/score grounding
- `ActivityNet Captions` for main video-only timestamp/caption event generation
- `YouCook2` as optional helper data
- `Charades-STA` as an optional later retrieval-focused dataset

**Architecture direction**

- one shared lightweight video backbone
- one shared temporal/event modeling core
- optional query-conditioning branch
- shared event-centric representation across tasks

### Phase D: Final Research Model

**Status:** long-term target.

**Main goal**

- produce a compact TRACE-like TinyTrace model for edge environments

**Final expected behavior**

- input: `video only`
- output: structured events with `timestamp + score + caption`
- optional query mode: `video + query`

At a high level:

- Phase A teaches TinyTrace **where** the relevant moments are
- Phase B teaches TinyTrace **what happened**
- Phase C teaches TinyTrace to support **both query-conditioned and video-only VTG**
- Phase D aims for the final research-quality edge model

## Project Structure

Main code paths:

- `TinyTrace/tinytrace/` : model, data loader, tokenizers, parser
- `TinyTrace/scripts/train_tinytrace.py` : training
- `TinyTrace/scripts/run_phase_a_pipeline.py` : Phase-A-v3 data, overfit, conditioning, smoke, and full-run gates
- `TinyTrace/scripts/precompute_visual_features.py` : frozen MobileCLIP feature-cache builder
- `TinyTrace/scripts/eval_tinytrace.py` : inspect one sample prediction
- `TinyTrace/scripts/eval_tinytrace_vhd.py` : proxy or official QVHighlights metrics, depending on supplied labels
- `TinyTrace/scripts/prepare_qvhighlights_subset.py` : convert downloaded QVHighlights clips into TinyTrace JSON
- `TinyTrace/configs/tinytrace_baseline.json` : baseline config
- `TinyTrace/configs/tinytrace_qvhighlights_phase_a_v3.json` : current dense Phase-A model
- `TinyTrace/configs/train_qvhighlights_phase_a_v3.json` : current full Phase-A training profile
- `TinyTrace/data/sample_dataset.json` : minimal sample dataset
- `TinyTrace/trace_lightwieght.md` : project design specification

## Environment Setup

```bash
python3 -m venv TinyTrace/.venv
TinyTrace/.venv/bin/pip install -r TinyTrace/requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv TinyTrace/.venv
TinyTrace/.venv/Scripts/python.exe -m pip install -r TinyTrace/requirements.txt
TinyTrace/.venv/Scripts/python.exe TinyTrace/scripts/setup_mobileclip.py
```

For development, install the test dependencies and run the complete suite before
training:

```powershell
cd TinyTrace
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
.venv/Scripts/python.exe -m pytest -q
```

## Smoke Test With Synthetic Data

The examples from here through the historical 500-video launcher exercise the
legacy autoregressive/caption-capable path. They are useful for compatibility
testing, but they are not substitutes for the Phase-A-v3 pipeline above.

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --epochs 3 \
  --batch-size 8 \
  --dataset-size 128

PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py
```

## Train With A Small JSON Dataset

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/sample_dataset.json \
  --frame-cache-dir TinyTrace/.cache/frames

PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --dataset-json TinyTrace/data/sample_dataset.json
```

Each TinyTrace sample looks like:

```json
{
  "instruction": "localize events and describe them",
  "num_frames": 8,
  "events": [
    {
      "timestamp": [0.2, 1.4],
      "score": [3.6],
      "caption": "person starts activity"
    }
  ]
}
```

## QVHighlights: What You Need

For TinyTrace, QVHighlights is a good first real dataset because it matches TRACE's highlight-detection setting.

But the dataset setup has two different parts:

1. annotations
2. videos

You need both.

Files you already downloaded:

- `dataset/mt_fmt-8k.json`
- `dataset/val.caption_coco_format.json`

What they are for:

- `mt_fmt-8k.json` : training-style annotation source
- `val.caption_coco_format.json` : useful for evaluation/reference, not your first training file

Important:

- `mt_fmt-8k.json` alone is not enough
- videos alone are not enough
- TinyTrace training needs matching annotation rows and matching `.mp4` files

## Is Your Current QVHighlights Download Enough?

For first prototype work:

- yes, it is enough to start
- yes, it is enough to test the full TinyTrace pipeline

For proper training:

- no, it is not enough yet

Right now you have a few downloaded clips in:

- `qvhighlights/videos/train/`

TinyTrace currently filters bad/corrupted clips and builds a small clean subset from the usable ones.

## How To Prepare TinyTrace Training Data From QVHighlights

Put matching QVHighlights train clips here:

```bash
qvhighlights/videos/train/
```

To auto-download valid QVHighlights clips, run the downloader from the project root folder:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/download_qvhighlights_subset.py --target-count 50
```

This will:

- create `dataset/qvhighlights/videos/train/` if it does not exist
- download valid clips automatically
- save the matched annotation subset to `dataset/qvhighlights/mt_fmt-50-valid.json`

Then convert the downloaded clips into TinyTrace JSON:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/prepare_qvhighlights_subset.py \
  --source-json dataset/mt_fmt-8k.json \
  --video-dir qvhighlights/videos/train \
  --output-json TinyTrace/data/qvh_tinytrace_subset.json \
  --max-samples 8
```

This script:

- finds matching videos by filename
- skips unreadable/corrupted clips
- extracts the query text
- converts the annotation into TinyTrace event format

## Train On Real QVHighlights Clips

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --val-dataset-json TinyTrace/data/qvh_tinytrace_val.json \
  --epochs 10 \
  --batch-size 2 \
  --warmup-ratio 0.05 \
  --min-lr-ratio 0.1 \
  --amp auto \
  --accumulation-steps 2 \
  --early-stopping-patience 3 \
  --monitor val_loss \
  --output-dir TinyTrace/outputs-qvh
```

Real-data training does not silently substitute random frames. Use
`--allow-random-frames` only for deliberate training-data debugging; validation
always rejects random fallback frames. Training order is derived from the run
seed and epoch, while validation order is fixed. Resume a stopped scheduled run
with the same total `--epochs` value that was used when the run started:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --val-dataset-json TinyTrace/data/qvh_tinytrace_val.json \
  --epochs 10 \
  --resume TinyTrace/outputs-qvh/checkpoints/latest.pt \
  --output-dir TinyTrace/outputs-qvh
```

Each run writes `config.json`, `training_config.json`, `run_arguments.json`,
`run_metadata.json`, `optimizer_groups.json`, `training_log.jsonl`,
`history.json`, `run_summary.json`, `checkpoints/latest.pt`,
`checkpoints/best-loss.pt`, `checkpoints/best-primary-metric.pt`, the backwards-
compatible `checkpoints/best.pt`, bounded periodic checkpoints, and epoch
prediction JSON files. Prediction records contain raw generated token IDs,
parsed events, ground truth, parser warnings, generation length, and termination
reason.

The JSONL log includes per-task raw/weighted losses, target counts, named-group
learning rates, gradient norms, clipping events, micro-step/optimizer-step
counters, examples/frames/tokens throughput, peak CUDA memory, checkpoint
selection, and validation records. `--amp auto` selects BF16 or FP16 on
supported CUDA hardware and stays in FP32 on CPU. Use `--amp bf16` explicitly
to test CPU BF16.

The optimizer uses named groups for compression, embeddings, LCEM, task heads,
and MobileCLIP, with separate decay/no-decay groups. Linear warmup is followed
by cosine decay. Gradient accumulation divides each actual accumulation window
correctly, including the final partial window, and advances clipping, optimizer,
scaler, and scheduler state only at update boundaries.

Versioned resumable checkpoints restore optimizer, scheduler, AMP scaler, early
stopping, RNG, deterministic epoch order, counters, and selection state. Older
one/two-group optimizer checkpoints are migrated when possible, but legacy
checkpoints that did not save RNG/data-order state cannot guarantee an exact
continuation. Periodic retention is configured with `--checkpoint-keep`.

`configs/final_train_qvh500.json` is a validated historical v1 training profile;
unknown profile fields fail before training. The stable model baseline keeps
`dropout=0.0`. `configs/tinytrace_dropout_010.json` is the isolated experimental
dropout candidate and must not replace the baseline without a controlled
validation ablation.

For a quick smoke run:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --epochs 1 \
  --batch-size 1 \
  --output-dir TinyTrace/outputs-qvh-smoke
```

## Current Dataset Folder and Legacy-v1 Inputs

The final cleaned dataset prepared for TinyTrace is stored here:

```bash
final_qvhighlights_tinytrace/
```

It currently contains:

- `videos/train/` : 1,218 downloaded training clips
- `videos/val/` : 136 downloaded validation clips
- `annotations/tinytrace_train.json` and `tinytrace_val.json` : original
  Phase-A-v1 conversion inputs
- `annotations/qvh_raw_valid.json` : source saliency annotations
- `annotations/tinytrace_phase_a_v3_train.json` and
  `tinytrace_phase_a_v3_val.json` : current strict 1,155/132 Phase-A-v3 splits
- `annotations/phase_a_v3_exclusions.json` and
  `phase_a_v3_manifest.json` : the 67-row exclusion audit and immutable hashes
- `annotations/*phase_a_v2*` : preserved superseded publication; do not train it

You do not need to move this folder anywhere else. Keep it exactly under:

```bash
final_qvhighlights_tinytrace/
```

The command below is retained only to identify the legacy v1 path; do not use
it for the current Phase-A-v3 run:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_tinytrace.py \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_train.json \
  --epochs 10 \
  --batch-size 2 \
  --output-dir TinyTrace/outputs-qvh-final
```

To inspect one validation sample after training:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --checkpoint TinyTrace/outputs-qvh-final/tinytrace.pt \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_val.json \
  --sample-index 0
```

The legacy event checkpoint cannot be passed to the current Phase-A evaluator.
That evaluator intentionally accepts only direct 75-bin Phase-A-v3
checkpoints; it no longer manufactures clip scores from generated event
strings. Use the current dense command in the evaluation section below.

## Legacy Autoregressive Final Training Setup

Before final training, TinyTrace needs:

1. the Python dependencies from `TinyTrace/requirements.txt`
2. the pretrained `MobileCLIP-S0` checkpoint

Install dependencies:

```bash
cd /home/vikaspal/Desktop/Traceall
TinyTrace/.venv/bin/pip install -r TinyTrace/requirements.txt
```

Download and verify the MobileCLIP checkpoint:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/setup_mobileclip.py
```

That will place the pretrained checkpoint at:

```bash
TinyTrace/checkpoints/mobileclip_s0.pt
```

This checkpoint is the pretrained weight file for the MobileCLIP visual backbone.
TinyTrace does not learn those visual features from scratch. It starts from this
already-trained MobileCLIP model, freezes it, and trains the lightweight
TinyTrace parts on top of it.

## Legacy One-File Final Training Run

Use this training profile file:

```bash
TinyTrace/configs/final_train_qvh500.json
```

Edit that file if you want to change:

- epochs
- batch size
- learning rate
- device
- output folder
- dataset paths

Then run final training with one command:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/run_training_profile.py \
  --profile TinyTrace/configs/final_train_qvh500.json
```

Even simpler, you can use the dedicated final-training launcher:

```bash
cd /home/vikaspal/Desktop/Traceall
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/train_final_qvh500.py
```

If the MobileCLIP checkpoint is missing, this launcher will first try to
download it automatically and then start training.

This profile currently trains on:

- `final_qvhighlights_tinytrace/annotations/tinytrace_train.json`
- validates on `final_qvhighlights_tinytrace/annotations/tinytrace_val.json`

## Check One Video's Prediction

To inspect what TinyTrace currently predicts for one video:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python TinyTrace/scripts/eval_tinytrace.py \
  --checkpoint TinyTrace/outputs-qvh-smoke/checkpoints/best.pt \
  --dataset-json TinyTrace/data/qvh_tinytrace_subset.json \
  --sample-index 0
```

It prints:

- `ground_truth`
- `predicted`

Both are event lists with:

- `timestamp`
- `score`
- `caption`

## TRACE-Style Metrics For QVHighlights

TinyTrace has two deliberately separate QVHighlights evaluation modes:

- the mean-score training proxy, always named `qvh_mean_score_proxy_*`;
- exact official Fair/Good/VeryGood mAP and Hit@1, available only when every
  evaluated query has the original 75-bin labels from all three annotators.

The evaluator requires exactly 75 predictions per query and does not silently
truncate mismatched arrays. If official three-annotator ground truth is absent,
it must report only the proxy; a proxy score is not a conference-comparable
official result. See `TRAINING_REFERENCE.md` for the current Phase-A-v3
commands and label contract.

After the Phase-A-v3 pipeline has produced a checkpoint and feature cache, run
the mean-score proxy evaluation with:

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/eval_tinytrace_vhd.py \
  --checkpoint TinyTrace/outputs-qvh-phase-a-v3-full/checkpoints/best-primary-metric.pt \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_phase_a_v3_val.json \
  --visual-feature-cache-dir TinyTrace/.cache/mobileclip_qvh-phase-a-v3-128-fp16 \
  --require-visual-feature-cache \
  --device cuda \
  --save-path TinyTrace/outputs-qvh-phase-a-v3-full/qvh_val_proxy.json
```

To add exact official metrics, append
`--official-ground-truth PATH/TO/highlight_val_release.jsonl`. That file must
contain the original `relevant_clip_ids` and all three `saliency_scores` for
every evaluated query.

With genuine three-annotator labels, official output includes:

- `HL-min-Fair-mAP`
- `HL-min-Fair-Hit1`
- `HL-min-Good-mAP`
- `HL-min-Good-Hit1`
- `HL-min-VeryGood-mAP`
- `HL-min-VeryGood-Hit1`

## Priority 4 Representation Ablations

Priority 4 keeps MobileCLIP-S0, the four-layer width-192 LCEM, tokenization,
preprocessing, and optimizer policy fixed. It exposes two controlled experiment
families:

- frames: `8 -> 12 -> 16 -> 24 -> 32`
- caption target tokens: `20`, `48`, and `64`

The baseline configuration is not changed automatically. Frame comparisons are
sequential, while 48- and 64-token caption candidates are each compared with
the 20-token reference. Caption candidates change only the caption limit and
the minimally required dependent generation budget.

Analyze temporal spacing and caption truncation using existing annotations (no
video acquisition is performed):

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python \
  TinyTrace/scripts/analyze_representation_profiles.py \
  --dataset-json final_qvhighlights_tinytrace/annotations/tinytrace_val.json \
  --output TinyTrace/outputs-representation/coverage.json
```

Create a dry-run manifest for the first sequential frame comparison:

```bash
PYTHONPATH=TinyTrace TinyTrace/.venv/bin/python \
  TinyTrace/scripts/run_representation_ablation.py \
  --training-profile TinyTrace/configs/final_train_qvh500.json \
  --kind frame --baseline-frames 8 \
  --output-root TinyTrace/outputs-ablation-frame-08-12
```

Review the generated profiles and manifest, then add `--execute` to train the
baseline and candidate in order and benchmark their end-to-end latency, stage
latencies, inference memory, throughput, and feature diversity. Caption runs
use `--kind caption --caption-candidate 48` (or `64`). Every completed run must
contain `run_summary.json`, `history.json`, and
`representation_benchmark.json` before it is eligible for a decision.

Dataset samples, epoch metrics, and prediction artifacts now include explicit
caption-target truncation metadata. Configuration validation bounds experiments
to 128 frames, 64 caption tokens, and 512 generated tokens. The legacy
representation runner still restricts its controlled ladder to 32 frames;
current Phase A is the separately reviewed 128-frame configuration.

## Recommended Next Step

For Phase A, run `scripts/run_phase_a_pipeline.py` in a fresh workspace. Do not
start the 1,155-video full run unless its four-video overfit, conditioning, and
exact 100-optimizer-step smoke gates all pass. The 8→12→16→24→32 procedure
above is retained only for the separate legacy autoregressive representation
ablation; it does not override the reviewed 128-frame Phase-A-v3 profile.
