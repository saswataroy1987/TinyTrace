# TinyTrace Phase B v2: Dense Video Event Captioning

## Goal

Build the strongest compact TinyTrace Phase B model for **video-only dense event
captioning**:

- find every meaningful event in an untrimmed video;
- predict accurate start and end timestamps for each event;
- generate a fluent, visual-grounded caption for each event;
- use MobileCLIP as the visual backbone;
- keep the architecture small enough to be practical, but do **not** sacrifice
  caption or timestamp quality merely to meet a specific edge-device budget.

The existing `phase_b_activitynet_v1_run` is a baseline. It uses a custom,
character-level decoder. It remains preserved for comparison, but it is not the
target architecture for this goal.

## Target Output

For one video, the model returns an ordered list of events:

```json
[
  {"start": 12.4, "end": 21.8, "score": 0.94, "caption": "A person lifts weights in a gym."},
  {"start": 24.1, "end": 35.6, "score": 0.89, "caption": "The person performs another set of exercises."}
]
```

Timestamps are numerical model outputs, not character strings generated inside
a caption. This is a deliberate separation: temporal localization and natural
language are different problems and should be supervised separately.

## Final v2 Architecture

```text
video
  │
  ├─ uniform + boundary-aware frame sampling (initially 32–64 frames)
  │
  ├─ MobileCLIP image encoder (pretrained; initially frozen)
  │      └─ one visual feature per sampled frame
  │
  ├─ temporal encoder
  │      ├─ positional/time embeddings
  │      ├─ temporal Transformer layers
  │      └─ multi-scale temporal features
  │
  ├─ set-based event detector (learned event queries)
  │      ├─ event confidence
  │      ├─ normalized centre + duration
  │      └─ temporal segment [start, end]
  │
  └─ event-conditioned captioner
         ├─ pool temporal features only inside each detected segment
         ├─ project event/video features into text-conditioning tokens
         └─ FLAN-T5 Small subword decoder → caption
```

### Components

| Component | Choice | Responsibility |
|---|---|---|
| Visual backbone | MobileCLIP | Compact pretrained frame representation. |
| Temporal model | Small multi-scale Transformer | Relate frames across a full video and preserve temporal position. |
| Event detector | DETR-style learned event queries | Predict a variable set of event segments and confidence scores. |
| Caption decoder | `google/flan-t5-small` | Generate grammatical captions from pretrained subword language knowledge. |
| Visual-to-text bridge | trainable projection/cross-attention adapter | Ground each text caption in its event segment's visual features. |

FLAN-T5 Small is the chosen language starting point because it is an
encoder–decoder text model with a compact 512-dimensional architecture and a
32,128-token subword vocabulary. It replaces the v1 character decoder; it does
not replace MobileCLIP. See the [official configuration](https://huggingface.co/google/flan-t5-small/blob/main/config.json).

## Why v2 Should Improve Over v1

| v1 limitation | v2 correction |
|---|---|
| Decoder learns language from individual characters. | A pretrained subword decoder already knows words and sentence structure. |
| Timestamps are emitted as text tokens. | A dedicated regression head predicts numerical event boundaries. |
| Captions and event boundaries compete in one autoregressive sequence. | Detection and captioning have separate heads and losses, then are joined per event. |
| Teacher-forced training can hide repetitive free generation. | Subword decoding, controlled generation, and validation on generated captions reduce this failure mode. |
| One-scale frame context makes long-video boundaries difficult. | Multi-scale temporal features and set-based event matching model multiple events. |

## Data Contract

Phase B v2 uses the cache-verified ActivityNet samples only. The raw dataset,
frame cache, MobileCLIP feature cache, and v1 checkpoints are never modified.

Each training item must provide:

```json
{
  "video_id": "...",
  "duration": 123.4,
  "events": [
    {"start": 10.2, "end": 18.7, "caption": "..."}
  ],
  "visual_feature_path": "..."
}
```

Preparation rules:

1. Use only successfully decoded and feature-cached videos.
2. Reject invalid, empty, reversed, or out-of-duration annotations.
3. Normalize event boundaries to `[0, 1]` for detector training while retaining
   seconds for reports and final predictions.
4. Sort events chronologically.
5. Preserve the fixed train/validation split; do not leak videos between them.
6. Keep full captions; FLAN-T5 tokenization, rather than a character limit,
   determines sequence length. Truncation will be reported rather than silent.

## Training Plan

### Stage 0 — Reproducible v2 setup

- Create a new `phase_b_activitynet_v2_run/` work root.
- Reuse the existing verified MobileCLIP cache read-only.
- Save an immutable resolved config, package versions, seed, source revision, and
  dataset manifest in the v2 run folder.
- Add automatic resume from `latest.pt`; never overwrite best checkpoints.

### Stage 1 — Temporal event localization

- Freeze MobileCLIP and FLAN-T5.
- Train the temporal encoder and event detector using ActivityNet event segments.
- Use Hungarian/set matching between predicted queries and ground-truth events.
- Optimize event existence, normalized boundary L1, and temporal generalized-IoU
  losses.
- Select the best detector by validation temporal localization metrics.

### Stage 2 — Visual-to-text grounding

- Use ground-truth segments initially, pool their temporal features, and train
  the projection/adapter plus FLAN-T5 caption decoder.
- Keep most pretrained text weights frozen at first; train the bridge and the
  final decoder layers so visual features learn to condition language.
- Optimize token-level caption loss with padded tokens ignored.

### Stage 3 — Joint dense captioning

- Unfreeze selected upper MobileCLIP/FLAN-T5 layers only if validation improves.
- Feed matched predicted segments to the captioner as training stabilizes.
- Jointly optimize localization, event confidence, caption, and visual-text
  alignment losses.
- Use a lower learning rate and early stopping based on a combined validation
  score, not training loss alone.

### Stage 4 — Final evaluation and export

- Run full validation only for the best checkpoints; use a fixed representative
  validation subset during every epoch for fast feedback.
- Export clean JSON predictions with captions, seconds, normalized timestamps,
  confidence, and model version.
- Save the best caption checkpoint, best temporal checkpoint, and best combined
  checkpoint separately.

## Losses and Metrics

### Training losses

```text
L = λevent × event-existence loss
  + λbox   × (boundary L1 + temporal gIoU)
  + λtext  × caption token cross-entropy
  + λalign × visual-text alignment loss
```

Loss weights will be defined in the v2 configuration and tuned from validation
results; they will not be silently changed mid-run.

### What is measured

| Area | Validation signal |
|---|---|
| Event detection | Precision/recall and F1 at temporal IoU thresholds. |
| Timestamp quality | Mean IoU, start/end absolute error, and temporal mAP where available. |
| Caption quality | METEOR, CIDEr, BLEU, and qualitative matched-event samples. |
| Dense event quality | Caption metrics only on temporally matched events, plus a combined score. |
| Failure safety | Invalid-video count, skipped sample manifest, NaN/gradient checks, and resume checks. |

An improving token loss alone is not enough. The model is accepted only when
generated captions differ appropriately across videos and align to valid
detected segments.

## Generation Rules

- Detector emits up to a configured maximum number of event queries.
- Low-confidence and duplicate/high-overlap segments are removed with
  confidence-aware temporal filtering.
- The captioner sees the corresponding event segment features, not only a
  whole-video global feature.
- Use constrained maximum caption length, EOS stopping, repetition penalties,
  and no-repeat n-gram controls during inference.
- Do not fabricate an event when no query confidence passes the threshold.

## Checkpoints and Safety

- `latest.pt`: resume checkpoint after each completed epoch.
- `best-temporal.pt`: best timestamp/localization score.
- `best-caption.pt`: best caption score on matched events.
- `best-combined.pt`: final recommended checkpoint.
- Every generated artifact remains under the ignored v2 work directory.
- Raw videos, feature caches, v1 outputs, and previous checkpoints are read-only
  inputs to v2.

## Acceptance Criteria

Before calling v2 successful, we will verify:

1. Captions are grammatical and visually distinct across different videos.
2. Generated captions do not collapse into repeated high-frequency words.
3. Event segments span the actual actions rather than clustering at time zero.
4. Validation caption scores and temporal metrics improve against v1.
5. The output JSON is valid, complete, chronologically ordered, and reproducible
   from the selected checkpoint.

## Scope

This plan prioritizes quality. Edge optimization (ONNX/TensorRT, FP16/INT8,
frame-count reduction, and latency benchmarking) is deliberately postponed
until the caption-and-timestamp model reaches the acceptance criteria above.
