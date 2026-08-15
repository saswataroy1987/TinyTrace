# Phase B Repo Layout

This file defines what should stay in GitHub for the next TinyTrace phase and
what should remain local-only.

## Push To GitHub

These are the files we want in the repo:

- `TinyTrace/tinytrace/`
- `TinyTrace/scripts/`
- `TinyTrace/configs/`
- `TinyTrace/tests/`
- `TinyTrace/README.md`
- `TinyTrace/LOCAL_PHASE_A.md`
- `TinyTrace/PHASE_B_REPO_LAYOUT.md`
- small JSON/MD/TXT metadata files that describe dataset preparation

In short: **code, configs, tests, and documentation** should go to GitHub.

## Keep Local Only

These should not be pushed:

- `TinyTrace_kaggle_dataset_ready/`
- `TinyTrace_next_phase_assets/`
- `phase_a_v5_local_run/`
- `local_phase_b_assets/`
- `temp_local_assets/`
- `dataset/activitynet_captions/`
- `dataset/youcook2/`
- raw videos
- feature caches
- training outputs
- large checkpoints not needed as repo source files

In short: **datasets, raw videos, caches, and run outputs** should stay local.

## Recommended Local Layout

Use this layout on the training machine:

```text
~/TinyTrace/
├── TinyTraceRepo/                  # cloned GitHub repo
├── TinyTrace_kaggle_dataset_ready/ # existing Phase-A-ready bundle
├── local_phase_b_assets/
│   ├── activitynet_captions/
│   │   ├── annotations/
│   │   └── videos/
│   └── youcook2/
│       ├── annotations/
│       └── videos/
└── temp_local_assets/
```

## Phase B Warm-Start Source

The official warm-start source for Phase B should be the validated Phase A v3
checkpoint:

- `TinyTrace_kaggle_dataset_ready/bootstrap/phase_a_v3_best_primary_metric.pt`

This is preferred over incomplete local Phase-A-v5 checkpoint folders.

## Goal

After this cleanup, GitHub should contain only the reusable Phase B code path.
The machine-local folders should contain only the large assets required to run
training.
