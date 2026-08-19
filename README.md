# TinyTrace Phase B v2: Dense Video Event Captioning

This is the authoritative implementation, validation, execution, and handoff
plan for TinyTrace Phase B v2. The historical specification remains in
[README_PHASE_B_V2.md](README_PHASE_B_V2.md); v2 implementation decisions should
follow this master README.

## CURRENT STATUS

```text
Phase 1 / MobileCLIP feature extraction: COMPLETE
ActivityNet MobileCLIP cache: ALREADY GENERATED ON ANOTHER PC
V1: PRESERVED BASELINE
V2 implementation: NOT YET COMPLETE
Stage 0: NEXT
Full training: NOT YET STARTED
Training will occur on a separate GPU PC
```

The existing cache is a read-only input. Stage 0 must map and validate it before
training. Do not regenerate it as part of v2.

## 1. Project Overview

TinyTrace Phase B v2 is a compact, video-only dense event captioning system for
untrimmed ActivityNet videos. For each video it must find meaningful events,
regress numerical start/end times, assign confidence, and generate a fluent
caption grounded in the visual content of that event.

Expected output:

```json
[
  {
    "start": 12.4,
    "end": 21.8,
    "score": 0.94,
    "caption": "A person lifts weights in a gym."
  },
  {
    "start": 24.1,
    "end": 35.6,
    "score": 0.89,
    "caption": "The person performs another set of exercises."
  }
]
```

Timestamps are numerical detector outputs, not characters or text tokens. The
localization and language subsystems have separate heads, supervision, losses,
metrics, and checkpoints before they are trained jointly.

### V1 and V2 coexist

```text
V1 = preserved comparison baseline
V2 = new target architecture
```

V1 uses the existing TinyTrace character-level autoregressive decoder to emit
structured timestamps, scores, and captions. V2 instead uses a temporal
Transformer, DETR-style event queries, Hungarian matching, event-specific visual
pooling, a trainable visual-to-text bridge, and `google/flan-t5-small`.

V1 source, configurations, checkpoints, and run directories must remain
untouched. V2 uses separate modules, configurations, tests, runs, and
checkpoints. The V1 character decoder must not be reused as the V2 captioner.

## 2. V1 to V2 Transition

Completed preprocessing produced the visual evidence needed by V2:

```text
ActivityNet video
      ↓
sampled frames
      ↓
MobileCLIP-S0
      ↓
cached frame-level spatial features
```

Each expected cache entry contains:

```text
patch_features: [T, 64, 1024]
frame_times:    [T]
```

`T` is the number of sampled frames. The 64 values are an 8×8 spatial feature
map flattened into tokens, each with 1024 channels. `frame_times` maps the
ordered features to seconds in the source video.

Preprocessing did not produce event boundaries, a trained event detector, final
captions, or dense event JSON. As an analogy, Phase 1 gives V2 its visual
representation—its “eyes.” V2 still learns **WHEN** an event occurs and **WHAT**
happened.

## 3. Existing Artifacts and Their Status

| Artifact | Status | V2 role | Read/Write |
| --- | --- | --- | --- |
| MobileCLIP-S0 checkpoint | Available | Provenance and optional raw-frame verification | Read-only |
| ActivityNet MobileCLIP feature cache | **ALREADY EXISTS ON ANOTHER PC** | Primary `[T,64,1024]` visual input plus `[T]` times | **Read-only** |
| ActivityNet annotations | Required | Durations, splits, segments, captions | Read-only source |
| ActivityNet videos | Available with preprocessing assets | Audit/reference; normal V2 training uses cache | Read-only |
| V1 checkpoints | Preserved | Baseline comparison only | Read-only |
| V1 run directory | Preserved | Baseline metrics and predictions | Read-only |
| V2 run directory | Not created | Manifests, reports, checkpoints, predictions, exports | Writable, Git-ignored |
| FLAN-T5 Small checkpoint | To be cached on training PC | Pretrained tokenizer and caption model | Read-only base; tuned copies in V2 checkpoints |

Never confuse these artifacts:

```text
MobileCLIP model weights
    ≠ MobileCLIP feature cache
    ≠ V2 temporal detector checkpoint
    ≠ FLAN-T5 checkpoint
```

Current cache filenames may be hashes of video path, file metadata, frame count,
preprocessing, checkpoint digest, and format version. Stage 0 must therefore
create a durable `video_id → visual_feature_path` mapping. A cache directory
alone does not prove annotation alignment.

## 4. Final V2 Architecture

```text
Video
  │
  ├── sampled frames                         (already processed)
  ├── MobileCLIP-S0                          (already processed)
  │       ↓
  │   existing read-only cache
  │       ↓
  │   [T,64,1024] + frame_times [T]
  │
  ├── spatial pooling/projection
  │       ↓
  │   frame features [T,D]
  ├── normalized time + position embeddings
  ├── temporal Transformer
  ├── multi-scale temporal features
  ├── learned event queries
  │       ├── event confidence
  │       └── normalized center + duration → [start,end]
  │
  └── event-specific temporal features
          ↓
      visual-to-text adapter
          ↓
      FLAN-T5 Small conditioning tokens
          ↓
      caption
```

| Component | Input → output | Responsibility |
| --- | --- | --- |
| Cache loader | cache → `[T,64,1024]`, `[T]` | Load and validate visual evidence without decoding video |
| Spatial pooling | `[T,64,1024]` → `[T,1024]` | Aggregate patches into frame representations |
| Projection | `[T,1024]` → `[T,D]` | Adapt MobileCLIP channels to temporal hidden size |
| Time/position encoding | features + times → `[T,D]` | Preserve order and normalized video time |
| Temporal Transformer | `[T,D]` + mask → `[T,D]` | Model full-video temporal relationships |
| Multi-scale module | `[T,D]` → multiple scales | Combine boundary detail and long context |
| Event detector | temporal memory + `Q` queries → scores/segments | Predict an unordered event set |
| Event pooler | temporal memory + segment → event tokens | Retain evidence for one event only |
| Visual-to-text adapter | event tokens → 512-D tokens | Translate vision to FLAN-T5 conditioning |
| FLAN-T5 Small | conditioning + target tokens → caption | Generate natural-language descriptions |

`D`, event-query count `Q`, layer count, temporal scales, loss weights,
confidence threshold, overlap filtering, and generation settings are **TO BE
TUNED FROM VALIDATION** and saved in the resolved configuration.

### Cache constraint

The default V2 path uses fixed cached features, so MobileCLIP remains frozen.
Unfreezing it cannot update existing tensors. Any MobileCLIP fine-tuning is a
separate later experiment requiring raw frames and on-the-fly forward passes;
it must not silently replace the cache-based baseline. Selected FLAN-T5 layers
can be unfrozen without invalidating the visual cache.

## 5. End-to-End Data Flow for One Video

For a 60-second video:

```text
32/64 cached frame entries
    ↓
[T,64,1024] patch features + [T] times
    ↓ spatial pooling + Linear(1024,D)
[T,D] frame sequence
    ↓ time embeddings + temporal Transformer
event query predictions
    ↓
event 1 = [10,20] seconds, confidence 0.93
event 2 = [30,42] seconds, confidence 0.87
```

For event 1, select or softly pool temporal features in/near `[10,20]`, map the
event tokens to FLAN-T5's hidden dimension, and generate caption 1. Repeat using
`[30,42]` for event 2. Filter low-confidence and duplicate/high-overlap segments,
convert normalized boundaries to seconds, sort chronologically, and export the
ordered list. If no query passes the configured threshold, return an empty list;
do not fabricate an event.

## 6. Master Roadmap

```text
Stage 0 — Data + reproducibility preparation
Stage 1 — Temporal event localization
Stage 2 — Visual-to-text captioning
Stage 3 — Joint dense event captioning
Stage 4 — Final evaluation + export
Stage 5 — Edge optimization after quality acceptance
```

## Stage 0 — Data and Reproducibility Preparation

### PLAN

Build a clean, reproducible V2 data pipeline around the existing read-only
ActivityNet cache. This stage happens before training.

### WHY

Video IDs, durations, event annotations, frame times, and visual tensors must
refer to the same timeline. Stage 0 prevents silent cache misses, corrupt data,
split leakage, and invalid segments from contaminating every later result.

### INPUT

- transferred ActivityNet MobileCLIP cache;
- ActivityNet `train.json` and `val_1.json`;
- cache-generation metadata and preprocessing config;
- MobileCLIP-S0 checkpoint digest for provenance;
- V2 template configuration and source revision.

Canonical manifest item:

```json
{
  "video_id": "...",
  "duration": 123.4,
  "split": "train",
  "events": [
    {"start": 10.2, "end": 18.7, "caption": "..."}
  ],
  "visual_feature_path": "..."
}
```

Loaded representation:

```text
visual_features: [T,64,1024]
frame_times:     [T]
segments:        [N,2] normalized to [0,1]
captions:        list[str] of length N
frame_mask:      [T]
event_mask:      [N] when padded
```

Seconds remain available for reports and export; detector targets use
`start/duration` and `end/duration`.

### ACTION

1. Create a separate V2 namespace and ignored `phase_b_activitynet_v2_run/`.
2. Define a resolved configuration with paths, cache contract, shapes, model
   parameters, loss weights, seeds, and stage flags.
3. Build a stable `video_id → cache path` index; never depend on directory order.
4. Convert all valid ActivityNet events without V1's six-event truncation.
5. Reject missing, empty, reversed, non-finite, negative, or out-of-duration
   segments; reject empty captions and normalize whitespace.
6. Sort events and detect duplicate video IDs.
7. Preserve train/validation assignments and test for leakage.
8. Validate each cache: existence, keys, format, tensor types, dimensions,
   finite values, frame count, monotonic times, and duration range.
9. Report missing/corrupt cache entries; never use random frames or regenerate.
10. Implement the V2 dataset loader and collator with frame/event masks.
11. Save skipped samples with machine-readable reasons.
12. Save resolved config, package/Python/PyTorch/CUDA versions, seed, Git
    revision, manifest digest, and cache provenance.
13. Run dataset, loader, and collator smoke tests before GPU training.

### DESCRIBE

The manifest is the stable join between annotations and opaque cache filenames.
The loader safely deserializes the cache and returns ordered features and times.
The collator pads to the longest batch sample and emits masks so padding cannot
affect attention, matching, pooling, losses, or metrics.

Inspect at least 5–10 real entries and record shape, dtype, mean, standard
deviation, min/max, L2 norms, NaN/Inf counts, zero/constant checks, frame-time
ordering, and—where practical—adjacent/distant cosine similarity. This detects
corruption or collapse; it does not prove detector quality.

### OUTPUT

```text
phase_b_activitynet_v2_run/
├── configs/resolved_config.json
├── manifests/activitynet_v2_manifest.json
├── reports/dataset_validation.json
├── reports/skipped_samples.json
└── metadata/reproducibility.json
```

### VALIDATION

- Every retained sample passes tensor, timestamp, duration, and event checks.
- Train/validation video-ID intersection is empty.
- Manifest/cache coverage is reported by split.
- At least 5–10 real entries pass numerical sanity checks.
- Loader/collator tests verify deterministic ordering, shapes, masks, padding.
- Identical inputs reproduce identical manifests and digests.

### DEPENDENCIES

The cache must be transferred or mounted, annotations and generation metadata
must be available, and PyTorch must safely load the cache payloads.

### CHECKPOINT

Stage 0 is ready only when `dataset_validation.json` passes and every retained
manifest row resolves to a validated cache entry. This report is the stage gate;
Stage 0 produces no model checkpoint.

### DO NOT

- Do not regenerate or modify MobileCLIP features.
- Do not modify V1.
- Do not start detector or caption training.
- Do not truncate events, change splits, or skip problems silently.
- Do not substitute random RGB frames for missing cache entries.

### RISKS

- Transferred path/stat-based hashes may not resolve.
- Cache files may lack embedded video IDs.
- Annotation and decoded-video durations may differ.
- 32 samples may be coarse for short actions in long videos.
- A copied directory may be incomplete.

## Stage 1 — Temporal Event Localization

### PLAN

Teach the model **WHEN** events happen. Freeze MobileCLIP and FLAN-T5. Train the
feature adapter, temporal Transformer, multi-scale module, event queries, and
detector heads.

### WHY

Caption training can hide localization failures. First prove that the model can
find events and regress valid boundaries before asking it to describe them.

### INPUT

- Stage 0 validated manifest and read-only cache;
- `[B,T,64,1024]` patch features, `[B,T]` times, and frame masks;
- `[B,N,2]` normalized ground-truth segments and event masks;
- resolved Stage 1 configuration.

### ACTION

1. Implement spatial pooling over 64 patch tokens.
2. Implement `Linear(1024,D)` and LayerNorm.
3. Add normalized continuous-time and sequence-position embeddings.
4. Implement a masked temporal Transformer.
5. Build multi-scale temporal features and masks.
6. Add `Q` learned event-query embeddings and query decoding.
7. Predict event/no-event confidence.
8. Predict normalized center and positive duration.
9. Convert center/duration to clipped, non-reversed `[start,end]`.
10. Implement Hungarian matching.
11. Implement event-existence, boundary L1, and temporal gIoU losses.
12. Implement confidence-aware duplicate/overlap filtering for inference.
13. Add localization metrics and qualitative prediction samples.
14. Add atomic checkpoints, exact resume, NaN/gradient checks, and early stopping.

### DESCRIBE

The temporal feature adapter is:

```text
[T,64,1024]
      ↓ spatial pooling
[T,1024]
      ↓ Linear(1024,D)
[T,D]
      ↓ LayerNorm
[T,D]
```

It is needed because cached frames contain 64 spatial tokens and 1024 channels,
which need not equal the temporal model dimension. A learned projection is a
normal adapter, not an incompatibility.

Event queries predict a set rather than an autoregressive sequence. Every query
emits:

```text
confidence
normalized center
normalized duration
```

Hungarian matching creates one-to-one supervision. For example:

```text
ground truth: A, B, C
predictions:  query 1, query 2, query 3, query 4, ...
                     ↓ class + boundary matching cost
assignment:   A↔query 3, B↔query 1, C↔query 4
unmatched:    supervised as no-event
```

Losses are applied after assignment so multiple queries are not rewarded for
the same annotated event.

### OUTPUT

```json
[
  {"start": 10.2, "end": 18.7, "score": 0.91}
]
```

Stage 1 also outputs metrics, matched examples, history, and checkpoints.

### VALIDATION

- Precision, recall, and F1 at configured temporal IoU thresholds.
- Mean IoU for matched events.
- Mean/median start and end absolute error in seconds.
- Temporal mAP where practical.
- Zero invalid, reversed, or out-of-range segments after conversion.
- Confidence and boundaries vary across videos and queries.
- Predictions do not systematically cluster at time zero.

IoU/confidence thresholds and detector-selection score are **TO BE TUNED FROM
VALIDATION** and frozen before final evaluation.

### DEPENDENCIES

Stage 0 must pass. Dataset masks, normalized boundaries, temporal IoU, gIoU, and
segment conversion must have unit tests.

### CHECKPOINT

- `latest.pt`: written atomically after each completed epoch with model,
  optimizer, scheduler, scaler, epoch/step, RNG states, and config/manifest IDs.
- `best-temporal.pt`: updated only when the declared validation localization
  score improves; never selected from training loss alone.

### DO NOT

- Do not train captions or load FLAN-T5 into the loss path.
- Do not emit timestamps as text.
- Do not modify cached features or unfreeze MobileCLIP.

### RISKS

- Class imbalance may favor no-event.
- Bad masks may let padding influence attention or matching.
- Center/duration may collapse near zero.
- Sparse sampling may limit boundary precision.
- Duplicate queries may require confidence-aware filtering.

## Stage 2 — Visual-to-Text Captioning

### PLAN

Teach the model **WHAT** happened. Initially use ground-truth event segments so
caption learning is isolated from detector errors.

### WHY

FLAN-T5 knows language but cannot interpret raw MobileCLIP tensors. Ground-truth
segments first establish a clean visual-to-language signal; predicted segments
are introduced later.

### INPUT

- Stage 1 temporal checkpoint;
- ground-truth segments and full captions;
- event-aligned temporal features;
- pinned `google/flan-t5-small` tokenizer and checkpoint.

### ACTION

1. Implement hard-mask and/or differentiable event-specific pooling.
2. Preserve multiple event tokens if one pooled vector loses action detail.
3. Implement a visual-to-text projection/MLP or cross-attention adapter.
4. Map event features to FLAN-T5 Small's 512-dimensional hidden space.
5. Pass conditioning tokens through a documented `inputs_embeds` and attention
   mask interface to the FLAN-T5 encoder.
6. Tokenize full captions and report truncation explicitly.
7. Compute caption cross-entropy with padded labels ignored.
8. Implement free generation for validation.
9. Add EOS stopping, maximum length, repetition penalty, and no-repeat n-grams.
10. Implement temporally matched caption metrics and qualitative reports.

### DESCRIBE

```text
MobileCLIP/temporal event features
              ≠
FLAN-T5 token embeddings
```

A trainable bridge is mandatory:

```text
event temporal features [K,D]
        ↓ projection/MLP or cross-attention
visual conditioning tokens [K',512]
        ↓ attention mask
FLAN-T5 encoder → FLAN-T5 decoder → caption tokens
```

Initially keep MobileCLIP frozen, freeze the Stage 1 temporal encoder long
enough to isolate the bridge, freeze most FLAN-T5 weights, and train the bridge
plus selected final decoder layers. Gradual temporal/FLAN-T5 unfreezing is
allowed only when controlled validation improves generated captions. Pretrained
components use lower learning rates.

Caption evaluation is not exact string matching. Example:

> Ground truth: “A man is drinking coffee from his cup in hand.”

> Prediction: “A person is holding a cup.”

First match the predicted event to ground truth using temporal IoU. Only a
temporally matched event receives caption evaluation. The prediction deserves
partial overlap credit but not full credit because it misses “drinking.”

“A man is drinking coffee” and “A person drinks coffee” can both be good despite
different wording. Use multiple signals:

- BLEU measures n-gram precision but can undervalue paraphrases.
- METEOR better tolerates word-form variation but remains reference dependent.
- CIDEr rewards informative phrases supported by references.
- Qualitative/semantic analysis catches hallucination, missing actions,
  repetition, identity errors, and weak visual grounding.

No single caption metric is sufficient.

### OUTPUT

```json
{
  "start": 10.2,
  "end": 18.7,
  "caption": "A person drinks from a cup."
}
```

### VALIDATION

- BLEU, METEOR, and CIDEr under the declared matched-event protocol.
- Match coverage reported beside caption scores; unmatched events are visible.
- Captions inspected across videos and events within the same video.
- Empty output, premature EOS, repetition, generic collapse, and truncation
  reported.
- Free generation compared with teacher-forced token loss every epoch.

### DEPENDENCIES

Stage 1 representation must be stable; segment pooling/masks must be tested; and
FLAN-T5 model/tokenizer revisions must be pinned.

### CHECKPOINT

`best-caption.pt` is updated when the declared generated-caption score on
temporally matched events improves. It includes the bridge, trainable FLAN-T5
weights, needed temporal state, tokenizer/config identity, and generation rules.

### DO NOT

- Do not feed 1024-D MobileCLIP tensors directly to FLAN-T5.
- Do not use the V1 character tokenizer or decoder.
- Do not select a checkpoint from teacher-forced loss alone.
- Do not score captions as localized when their events are unmatched.
- Do not silently truncate captions.

### RISKS

- FLAN-T5 may ignore a weak adapter.
- Whole-video pooling may cause generic captions.
- Too much freezing may prevent grounding; too much unfreezing may overfit.
- Teacher forcing may hide poor free generation.

## Stage 3 — Joint Dense Event Captioning

### PLAN

Join localization and captioning so the model learns **WHEN + WHAT** and uses
predicted segments at inference.

### WHY

Stage 2 uses clean ground-truth segments, but deployment uses detector segments.
Joint training reduces this mismatch while preserving the diagnostic value of
the separately trained systems.

### INPUT

- `best-temporal.pt` and `best-caption.pt` initialization;
- validated cache and annotations;
- matched predicted segments, event features, and caption targets.

### ACTION

1. Connect detector → predicted event → pooler → adapter → FLAN-T5.
2. Begin with a recorded mixture of ground-truth and matched predictions.
3. Increase predicted-segment use only as localization stabilizes.
4. Pool caption evidence from the corresponding predicted event.
5. Optimize confidence, localization, caption, and alignment jointly.
6. Use separate optimizer groups and lower rates for pretrained components.
7. Monitor per-loss gradients, NaNs, and early stopping signals.
8. Retain temporal, caption, and combined best checkpoints independently.

### DESCRIBE

```text
cached MobileCLIP features
    ↓ temporal Transformer
event detector
    ↓ matched predicted event
event-specific feature pooling
    ↓ visual-to-text adapter
FLAN-T5 Small
    ↓ caption
```

Joint loss:

```text
L = λevent × event-existence loss
  + λbox   × boundary L1 loss
  + λgIoU  × temporal generalized-IoU loss
  + λtext  × caption token loss
  + λalign × visual-text alignment loss
```

- `event` teaches queries whether an event exists.
- `box` penalizes normalized boundary errors.
- `gIoU` rewards temporal overlap, including non-overlap cases.
- `text` teaches captions for matched event-caption pairs.
- `align` encourages corresponding event visuals and text to share useful
  semantics; its formulation is **DECISION REQUIRED**.

Loss weights are **TO BE TUNED FROM VALIDATION**, recorded in the resolved
config, and never silently changed mid-run. Selected FLAN-T5 layers may be
unfrozen only after a validation improvement. MobileCLIP stays frozen in the
cache-based pipeline; raw-frame fine-tuning would be a separate experiment.

### OUTPUT

```json
[
  {
    "start": 12.4,
    "end": 21.8,
    "score": 0.94,
    "caption": "A person lifts weights in a gym."
  }
]
```

### VALIDATION

- Report localization and caption metrics separately.
- Evaluate captions only on temporally matched events.
- Report match coverage, empty predictions, duplicates, and invalid segments.
- Use a fixed representative subset each epoch; reserve full validation for
  selected checkpoints.
- Select by a declared combined score, not training loss.

### DEPENDENCIES

Stages 1 and 2 must pass independently. Predicted-segment matching, pooling,
optimizer groups, and loss scaling must be tested.

### CHECKPOINT

- `latest.pt`: exact joint resume state after every completed epoch.
- `best-temporal.pt`: best localization result, retained separately.
- `best-caption.pt`: best matched-event caption result, retained separately.
- `best-combined.pt`: recommended full system under the frozen combined
  validation protocol.

### DO NOT

- Do not start joint training with random detector and captioner weights.
- Do not replace ground-truth segments with unstable predictions all at once.
- Do not let caption loss erase localization quality.
- Do not claim caption improvement while match coverage deteriorates.

### RISKS

- Detector mistakes may poison caption conditioning.
- Caption gradients may destabilize boundaries.
- One loss may dominate the others.
- Joint memory use may require smaller batches or gradient accumulation.

## Stage 4 — Final Evaluation and Export

### PLAN

Answer: **Is V2 actually better than V1?** Compare both on a fixed protocol and
export reproducible predictions.

### WHY

Localization and captioning can fail independently. Final evaluation must expose
both dimensions rather than hiding them inside one unexplained number.

### INPUT

- frozen validation manifest;
- V1 baseline predictions/checkpoint;
- V2 best temporal, caption, and combined checkpoints;
- fixed filtering, matching, generation, and metric settings.

### ACTION

1. Run complete validation for selected checkpoints.
2. Compare V1 and V2 on identical retained video IDs and annotations.
3. Produce temporal, caption, dense-event, qualitative, and failure reports.
4. Define/freeze any combined score before selecting the final checkpoint.
5. Export chronological JSON with model/config provenance.
6. Re-run a deterministic subset to verify reproducibility.

### DESCRIBE

Temporal comparison includes precision, recall, F1, mAP where practical, mean
IoU, and start/end error. Caption comparison includes BLEU, METEOR, CIDEr,
semantic consistency, and qualitative grounding. Caption metrics apply only to
temporally matched events, with match coverage beside them.

A combined score is allowed only after its components, directions, scaling,
normalization, and weights are defined. Its formula is **DECISION REQUIRED** and
never replaces the separate metric tables.

### OUTPUT

```json
{
  "video_id": "v_example",
  "events": [
    {
      "start": 12.4,
      "end": 21.8,
      "normalized_start": 0.103,
      "normalized_end": 0.182,
      "confidence": 0.94,
      "caption": "A person lifts weights in a gym."
    }
  ],
  "model_version": "tinytrace-phase-b-v2"
}
```

### VALIDATION

- Validate schema, finite values, chronological order, and duration range.
- Use the same population and metrics for V1 and V2.
- Reproduce predictions from the checkpoint and resolved config.
- Return no fabricated event when all scores are below threshold.

### DEPENDENCIES

Stage 3 must be stable, V1 must be evaluable under the same protocol, and metric
and export tests must pass.

### CHECKPOINT

`best-combined.pt`, immutable config, tokenizer/generation settings, validation
report, and predictions form the recommended release bundle. Best temporal and
caption checkpoints remain separately available.

### DO NOT

- Do not tune thresholds after inspecting final results.
- Do not compare V1 and V2 on different samples.
- Do not publish an undefined combined score.
- Do not omit temporal match coverage from caption results.

### RISKS

- Metric implementations may differ from the baseline.
- Filtering thresholds may materially change precision/recall.
- A combined score may conceal a subsystem regression.
- Qualitative examples may be cherry-picked unless sampled deterministically.

## Stage 5 — Edge Optimization After Quality Acceptance

### PLAN

Optimize the accepted model for deployment only after caption and timestamp
quality meet the acceptance criteria.

### WHY

Early compression makes it harder to distinguish architecture failures from
deployment degradation. Quality comes first; optimization comes second.

### INPUT

Accepted `best-combined.pt`, frozen evaluation set/report, and target hardware
constraints (**DECISION REQUIRED**).

### ACTION

Benchmark FP16, INT8, ONNX, TensorRT, frame-count reduction, latency, peak
memory, model size, and throughput. Compare every variant with the accepted
unoptimized baseline.

### DESCRIBE

Treat each optimization as an ablation. Accept it only when speed/memory gains
and temporal/caption quality deltas are both measured.

### OUTPUT

Hardware-specific artifacts and benchmark reports with explicit quality deltas.

### VALIDATION

Re-run the frozen temporal, caption, dense-event, schema, and reproducibility
checks for every deployment candidate.

### DEPENDENCIES

Stage 4 acceptance and target runtime/hardware selection.

### CHECKPOINT

Approve an optimized build only after it passes the frozen quality-regression
gate. Numerical tolerances are **TO BE TUNED** before benchmarking.

### DO NOT

- Do not optimize before quality acceptance.
- Do not replace quality metrics with latency alone.
- Do not overwrite the full-precision checkpoint.

### RISKS

- Quantization may harm boundaries or rare-word generation.
- Fewer frames may miss short events.
- Unsupported operators may alter adapter/generation behavior.

## 7. Loss and Evaluation Contract

| Area | Training signal | Validation signal |
| --- | --- | --- |
| Event existence | Matched-query classification | Precision, recall, F1 |
| Boundaries | Normalized L1 + temporal gIoU | IoU, mAP, start/end error |
| Caption | Token cross-entropy | BLEU, METEOR, CIDEr, qualitative grounding |
| Dense event | Joint/alignment losses | Matched-event captions plus match coverage |

Metric thresholds, combined-score normalization, alignment loss, query count,
confidence threshold, and overlap threshold are **TO BE TUNED FROM VALIDATION**.

## 8. Complete Pre-Training Checklist

- [ ] Existing ActivityNet MobileCLIP cache available
- [ ] Cache path configured
- [ ] Cache verified read-only
- [ ] Cache-to-video mapping recovered and validated
- [ ] ActivityNet annotations available
- [ ] Full valid event lists retained
- [ ] V2 manifest generated
- [ ] Train/validation split verified
- [ ] No leakage
- [ ] Invalid annotations reported
- [ ] Missing/corrupt cache entries reported
- [ ] Dataset validation report passes
- [ ] Dataset loader works
- [ ] Collator works
- [ ] Feature shape `[T,64,1024]` verified
- [ ] Frame-time order/range verified
- [ ] Feature statistics checked for 5–10 videos
- [ ] Temporal feature adapter works
- [ ] Time embeddings work
- [ ] Temporal Transformer forward pass works
- [ ] Multi-scale masks work
- [ ] Event queries work
- [ ] Center/duration conversion tested
- [ ] Hungarian matcher tested
- [ ] Localization losses tested
- [ ] Localization metrics implemented
- [ ] Event feature pooling works
- [ ] FLAN-T5 tokenizer works
- [ ] Visual-to-text adapter works
- [ ] FLAN-T5 forward pass works
- [ ] Caption padding and ignore index tested
- [ ] Caption generation works
- [ ] Caption metrics implemented
- [ ] Temporally matched caption protocol tested
- [ ] Checkpoint save/load works
- [ ] Resume restores complete state
- [ ] Best checkpoints use validation
- [ ] JSON export works
- [ ] Small end-to-end smoke test passes
- [ ] Reproducibility metadata saved
- [ ] V2 run directory ignored
- [ ] V1 remains untouched
- [ ] READY FOR GPU TRAINING

The last item is checked only after the relevant Stage 0 and component smoke
tests pass.

## 9. Training PC Handoff

Full training occurs on a separate GPU PC. First finish Stage 0 and validate the
pipeline; transfer does not replace validation.

### MUST TRANSFER

- V2 source code and tests;
- V2 template and Stage 0 resolved configs;
- validated V2 manifest;
- complete ActivityNet MobileCLIP cache;
- cache mapping and generation metadata;
- ActivityNet annotations used by the manifest;
- MobileCLIP-S0 checkpoint and SHA-256 provenance;
- Stage 0 validation/skipped reports;
- dependency/environment report and Git revision;
- Stage 1/2 checkpoints when resuming a later stage.

### CAN BE REGENERATED

- Python environment from pinned dependencies;
- FLAN-T5 Small base snapshot when the exact revision remains available;
- temporary logs and evaluation caches;
- predictions reproduced from a preserved checkpoint/config.

The ActivityNet MobileCLIP cache is already complete and is not in this list.

### MUST REMAIN READ-ONLY

- ActivityNet MobileCLIP cache;
- ActivityNet raw annotations and videos;
- MobileCLIP-S0 and FLAN-T5 base checkpoints;
- V1 source, configs, checkpoints, predictions, and run directory;
- immutable Stage 0 manifest once a training run starts.

### Expected training-PC layout

```text
TinyTraceProject/
├── README.md
├── README_PHASE_B_V2.md
├── TinyTrace/
│   ├── tinytrace/                         # V1 + separate V2 modules
│   ├── scripts/                           # separate V2 launchers
│   ├── configs/                           # V2 templates
│   └── tests/
├── external_inputs/                       # mounted/read-only
│   ├── activitynet/
│   │   ├── annotations/
│   │   │   ├── train.json
│   │   │   └── val_1.json
│   │   ├── mobileclip_cache/
│   │   │   └── <hashed-entry>.pt
│   │   └── cache_mapping.json
│   ├── mobileclip/mobileclip_s0.pt
│   └── flan_t5_small/                     # optional local snapshot
└── phase_b_activitynet_v2_run/            # writable, Git-ignored
    ├── configs/
    ├── manifests/
    ├── metadata/
    ├── reports/
    ├── checkpoints/
    ├── predictions/
    └── exports/
```

Paths may be configured differently, but roles and read/write rules do not
change. Never hard-code machine-specific absolute paths in tracked source.

## 10. Training Order

```text
0. Pass Stage 0 data/reproducibility validation
1. Stage 1 localization training
2. Evaluate localization; select best-temporal.pt
3. Stage 2 captioning with ground-truth segments
4. Evaluate generated captions; select best-caption.pt
5. Stage 3 joint training with gradually introduced predicted segments
6. Complete final evaluation
7. Compare V1 versus V2 using the same protocol
8. Only then perform edge optimization
```

Do not train everything jointly from scratch. Each isolated stage provides a
diagnostic checkpoint and prevents one failing subsystem from hiding another.

## 11. Failure Modes and Debugging Guide

### Detector finds events at time zero

Possible causes: incorrect time embeddings or normalization, bad center/duration
conversion, matcher errors, loss imbalance, or padding treated as evidence.
Inspect raw boundaries before clipping, assignments, frame times, masks, and
per-loss gradients.

### Detector predicts no events

Possible causes: no-event imbalance, threshold too high, matcher failure,
invalid targets, or unstable queries. Inspect pre-threshold logits and matched
recall before changing inference settings.

### Detector predicts duplicate events

Possible causes: weak one-to-one assignment, incorrect class cost, excess
queries, or missing filtering. Verify matcher uniqueness before relying on
post-processing.

### Captions repeat generic phrases

Possible causes: adapter not learning, FLAN-T5 over-frozen, whole-video rather
than event pooling, caption imbalance, truncation, or permissive generation.
Shuffle visual features as a diagnostic: if captions barely change, FLAN-T5 is
probably ignoring vision.

### Good captions but wrong timestamps

This is primarily localization. Inspect Stage 1 metrics, segment normalization,
matching, masks, and temporal resolution before modifying FLAN-T5.

### Good timestamps but generic captions

This is primarily visual-text conditioning. Inspect event pooling, adapter
activations, conditioning masks, caption diversity, and FLAN-T5 freezing.

### Good token loss but bad generated captions

Teacher forcing may hide exposure, decoding, repetition, or generalization
problems. Run free generation every epoch on a fixed subset; do not select the
caption checkpoint from token loss alone.

### Caption describes the wrong event

Check that matched query indices, segment masks, pooled features, and caption
targets keep identical ordering. This often indicates an indexing bug.

### NaN or exploding loss

Inspect cache finiteness, zero-duration targets, gIoU edge cases, empty masks,
mixed precision, learning rates, and per-component gradients. Stop and retain
diagnostics rather than continuing a corrupted run.

### Cache has zero hits after transfer

The hash may depend on old absolute paths or file metadata. Recover the original
mapping, validate payloads directly, and construct the V2 manifest from verified
paths. Do not immediately regenerate or rename opaque entries.

## 12. Checkpoint and Reproducibility Rules

- Write checkpoints atomically.
- `latest.pt` is for resume, not a quality claim.
- Never overwrite best temporal, caption, and combined checkpoints with each
  other.
- Store model, optimizer, scheduler, scaler, epoch/step, RNG states, resolved
  config digest, manifest digest, source revision, and best metric state.
- Refuse incompatible resume when architecture, manifest, or tokenizer identity
  differs unless an explicit migration exists.
- Save metric direction and selection formula with each best checkpoint.
- Keep generated V2 artifacts inside the ignored V2 run root.

## 13. Acceptance Criteria

V2 is accepted only when:

1. Event segments are not systematically clustered at zero.
2. Predicted events align with actual activities.
3. Captions are grammatical.
4. Captions are visually grounded in matched events.
5. Captions differ appropriately across videos and events.
6. There is no severe repetitive-caption collapse.
7. Temporal metrics are competitive with or better than V1.
8. Caption metrics are competitive with or better than V1 under the same
   matched-event protocol.
9. Match coverage and unmatched events are reported.
10. Exported JSON is valid, complete, ordered, and reproducible.
11. Checkpoint save and exact resume work.
12. V1 remains unchanged.
13. The selected model can be exported and run for inference.

No arbitrary numerical threshold is declared here. Confidence, overlap, metric,
regression-tolerance, and combined-score thresholds are **TO BE TUNED FROM
VALIDATION**, frozen before final evaluation, and saved in configuration.

## 14. Final One-Page Flow

```text
PHASE 1 / EXISTING PREPROCESSING
ActivityNet
   ↓
MobileCLIP-S0
   ↓
[T,64,1024] + frame_times
   ↓
(existing cache — already available on another PC, read-only)

                ↓

STAGE 0
cache + annotations
   ↓
V2 manifest + stable video/cache mapping
   ↓
tensor/timestamp/split validation
   ↓
clean reproducible dataset

                ↓

STAGE 1 — WHEN?
MobileCLIP features
   ↓
spatial pooling + projection
   ↓
time embeddings + temporal Transformer
   ↓
multi-scale features + event queries
   ↓
Hungarian matching
   ↓
event confidence + numerical boundaries
   ↓
temporal metrics + best-temporal.pt

                ↓

STAGE 2 — WHAT?
ground-truth event segment
   ↓
event-specific pooling
   ↓
visual-to-text adapter
   ↓
FLAN-T5 Small
   ↓
generated caption
   ↓
matched caption metrics + best-caption.pt

                ↓

STAGE 3 — WHEN + WHAT
predicted/matched events
   ↓
event-specific features
   ↓
FLAN-T5 captions
   ↓
joint localization + captioning
   ↓
best-combined.pt

                ↓

STAGE 4
V1 versus V2 on the same protocol
   ↓
separate temporal + caption reports
   ↓
reproducible JSON export

                ↓

STAGE 5 — POST-QUALITY ONLY
FP16 / INT8 / ONNX / TensorRT / frame reduction
   ↓
latency, memory, and quality-regression benchmarks
```

## WHAT WE DO NEXT

The immediate implementation task is:

```text
Stage 0:
V2 ActivityNet manifest + dataset loader + validation
using the existing MobileCLIP cache read-only.
```

Do not regenerate MobileCLIP features and do not start training. First transfer
or mount the cache, recover its mapping, validate real entries, build the
immutable manifest, and pass the Stage 0 gate. Then move the prepared project
and validated inputs to the GPU PC for Stage 1 localization training.
