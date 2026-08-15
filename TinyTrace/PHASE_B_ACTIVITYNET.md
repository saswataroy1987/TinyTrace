# Phase B ActivityNet

This document explains how to run TinyTrace Phase B after downloading
`ActivityNet Captions`.

## Goal

Phase B moves TinyTrace from query-conditioned Phase A grounding toward
video-only structured event generation.

Input:

- `video only`

Output:

- `timestamp + score + caption`

The score field is weakly supervised in this first Phase-B setup because
ActivityNet Captions provides timestamped captions but not QVHighlights-style
saliency scores.

## Required Assets

1. TinyTrace repo
2. `final_next_phase_assets/`
3. ActivityNet Captions annotations and videos
4. `mobileclip_s0.pt`

## Expected Local Layout

```text
~/TinyTrace/
├── TinyTraceRepo/
│   ├── TinyTrace/
│   └── final_next_phase_assets/
└── activitynet_captions/
    ├── train.json
    ├── val_1.json
    └── videos/
```

Alternative annotation layout also works:

```text
activitynet_captions/
├── annotations/
│   ├── train.json
│   └── val_1.json
└── videos/
```

## One-Command Preparation + Training

From the repo root:

```bash
python3 TinyTrace/scripts/run_phase_b_activitynet.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --device cuda
```

This will:

1. convert raw ActivityNet Captions annotations into TinyTrace Phase-B JSON
2. point the converted JSON at the local `videos/` directory
3. build the Phase-B model/training profile
4. warm-start from:
   - `final_next_phase_assets/phase_a_bootstrap/phase_a_v3_best_primary_metric.pt`
5. precompute frozen MobileCLIP visual features
6. launch Phase-B training

## Prepare Only

If you want only the converted dataset and generated configs first:

```bash
python3 TinyTrace/scripts/run_phase_b_activitynet.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --prepare-only
```

## Reuse Existing Feature Cache

If feature caching already finished in the same work directory:

```bash
python3 TinyTrace/scripts/run_phase_b_activitynet.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --device cuda \
  --skip-feature-cache
```

## Output Locations

The runner writes into:

- `./phase_b_activitynet_v1_run/annotations/`
- `./phase_b_activitynet_v1_run/configs/`
- `./phase_b_activitynet_v1_run/cache/`
- `./phase_b_activitynet_v1_run/outputs-activitynet-phase-b-v1/`

## Notes

- The official Phase-B warm-start source is the Phase-A-v3 bootstrap checkpoint.
- `mobileclip_s0.pt` is still required, but it can be downloaded again on the
  training laptop.
- ActivityNet Captions is the main open-domain Phase-B dataset.
- `YouCook2` can later be added as optional helper data, not as the only Phase-B
  dataset.
