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

## Recommended Launcher

Use the local launcher first:

- `TinyTrace/scripts/run_phase_b_local.py`

It will:

1. create or reuse a virtual environment
2. install Python dependencies
3. install a CUDA-enabled PyTorch build when `--device cuda` is used
4. download and verify `mobileclip_s0.pt` if it is missing
5. verify that CUDA is visible
6. call the Phase-B ActivityNet runner

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
python3 TinyTrace/scripts/run_phase_b_local.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --device cuda
```

This will:

1. create or reuse the local venv
2. install the required runtime dependencies
3. download MobileCLIP if needed
4. verify GPU visibility
5. convert raw ActivityNet Captions annotations into TinyTrace Phase-B JSON
6. point the converted JSON at the local `videos/` directory
7. build the Phase-B model/training profile
8. warm-start from:
   - `final_next_phase_assets/phase_a_bootstrap/phase_a_v3_best_primary_metric.pt`
9. precompute frozen MobileCLIP visual features
10. launch Phase-B training

## Prepare Only

If you want only the converted dataset and generated configs first:

```bash
python3 TinyTrace/scripts/run_phase_b_local.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --prepare-only
```

## Reuse Existing Feature Cache

If feature caching already finished in the same work directory:

```bash
python3 TinyTrace/scripts/run_phase_b_local.py \
  --dataset-root /path/to/activitynet_captions \
  --work-root ./phase_b_activitynet_v1_run \
  --device cuda \
  --skip-feature-cache \
  --skip-install
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
- The lower-level runner `TinyTrace/scripts/run_phase_b_activitynet.py` still
  exists, but the preferred entrypoint on a fresh laptop is
  `TinyTrace/scripts/run_phase_b_local.py`.
