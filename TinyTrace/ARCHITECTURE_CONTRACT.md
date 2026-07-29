# TinyTRACE Architecture Contract

This file records the executable interfaces for current QVHighlights Phase A
and the retained autoregressive Phase-B path. It complements, but does not
replace, `trace_lightwieght.md`.

## Mode separation

- **Phase A v3** is query-conditioned dense saliency prediction. It consumes
  128 uniformly sampled frames and returns 75 scores, one per two-second bin.
  It has no autoregressive timestamp, score, caption, `<sync>`, boundary, or
  event-count target.
- **Legacy autoregressive/Phase B** retains TRACE's structured
  `time -> score -> caption` event-token protocol for later caption training.

A checkpoint/result must name its mode; metrics from the two output protocols
are not interchangeable.

## Visual path

1. Input frames: `[B, T, 3, H, W]`, floating point RGB in `[0, 1]`.
2. MobileCLIP-S0 preprocessing: bilinear resize to `256 x 256`. The official
   MobileCLIP v1 transform applies resize, center crop, and tensor conversion;
   it does not apply an additional normalization transform.
3. Frozen MobileCLIP-S0 spatial tower: use `forward_embeddings`,
   `forward_tokens`, and `conv_exp`, before the global-pooling head.
4. Expected S0 spatial output: `[B*T, 1024, 8, 8]`.
5. Flattened patch features: `[B*T, 64, 1024]`.
6. Learned slot compression: `[B*T, 64, 1024] -> [B*T, 4, d_model]`.

For Phase A v3, the frozen spatial output is precomputed and stored as versioned
FP16 patch features. Training requires that cache and does not silently fall
back to random frames or a different MobileCLIP representation. The compact
uint8 decoded-frame cache is an intermediate used while building the feature
cache. Chunked MobileCLIP extraction bounds preprocessing memory without
changing feature order.

Calling MobileCLIP's public `encode_image()` is not valid for this path because
it returns one globally pooled embedding and removes the spatial token axis.

## Shared decoder prefix

Each scalar frame timestamp is serialized as fixed-width `0000.0`, producing
six IDs from the shared 13-token time vocabulary. `<sync>` is not included in
frame-time metadata because it is reserved for output head switching.

For each frame, tokens are concatenated in this order:

`[4 compressed visual slots, 6 discrete frame-time embeddings]`

The frame groups are flattened in chronological order and followed by the
query/instruction tokens. Legacy autoregressive training then appends
teacher-forced/generated event tokens:

`[frame 1 visual+time, ..., frame T visual+time] + instruction + events`

Phase A v3 stops the causal input at the query/instruction. It fuses the
query-conditioned hidden representation with temporally interpolated frame
features and learned bin embeddings, then produces one bounded score per bin:

`[B, 128, visual/time] + query -> dense saliency head -> [B, 75]`

Bin `i` maps directly to `[2*i, 2*(i+1))` seconds. Phase A optimizes balanced
Smooth-L1 regression, binary relevance at score `>=3`, and within-video
pairwise ranking. Its active loss weights are normalized; all legacy
autoregressive task weights are exactly zero in the v3 config.

The dense contract assumes the complete 150-second QVHighlights window. All
source filenames request that duration; v3 excludes media shorter than 149.5
seconds and re-probes retained media before launch. A short file must not be
stretched across 75 bins because relative interpolation would misalign every
absolute two-second target.

## Frozen-module invariant

All MobileCLIP parameters have `requires_grad=False`, and its BatchNorm layers
remain in evaluation mode even when the parent TinyTRACE model is training.
`require_visual_feature_cache=true` is therefore compatible only with
`stage2_start_epoch=0`; partial visual fine-tuning must use a separate uncached
experiment and is not part of Phase A v3.
