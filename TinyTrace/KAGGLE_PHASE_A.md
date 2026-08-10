# Kaggle Phase A

## Goal

After uploading one Kaggle dataset bundle and cloning the repository, Phase A
training should start with one command.

This Kaggle flow supports two dataset modes:

1. a **final direct-training bundle** that already contains the dense Phase-A
   train/val JSON files; or
2. a **raw-source bundle** that contains the cleaned pre-Phase-A source files
   and is rebuilt into dense labels on Kaggle.

In both cases, the launcher audits the dataset, precomputes frozen MobileCLIP
features, and launches a fresh **v4 warm-start** training run.

## Recommended uploaded Kaggle dataset bundle

The preferred uploaded Kaggle dataset should contain:

```text
tinytrace-kaggle-input/
├── vendor/                         # optional, for offline Kaggle installs
│   └── mobileclip-....whl
├── checkpoints/
│   └── mobileclip_s0.pt
├── bootstrap/
│   └── phase_a_v3_best_primary_metric.pt
├── final_phase_a_v4/
│   ├── annotations/
│   │   ├── tinytrace_phase_a_v4_train.json
│   │   ├── tinytrace_phase_a_v4_val.json
│   │   ├── phase_a_v4_manifest.json
│   │   └── phase_a_v4_exclusions.json
│   └── videos/
│       ├── train/
│       └── val/
└── optional_source_audit/
    └── final_qvhighlights_tinytrace/
        └── annotations/
            ├── qvh_raw_valid.json
            ├── tinytrace_train.json
            └── tinytrace_val.json
```

The launcher also supports the older raw-source-only layout:

```text
tinytrace-kaggle-input/
├── checkpoints/mobileclip_s0.pt
├── bootstrap/phase_a_v3_best_primary_metric.pt
└── final_qvhighlights_tinytrace/
    ├── annotations/
    │   ├── qvh_raw_valid.json
    │   ├── tinytrace_train.json
    │   └── tinytrace_val.json
    └── videos/
        ├── train/
        └── val/
```

## Why the bootstrap checkpoint is recommended

The v4 run is configured as a **fresh warm-start**, not a resume:

- model weights initialize from the best v3 checkpoint
- optimizer, scheduler, and early-stopping state all start fresh
- output artifacts are written into a new run directory

This is the safest path to improve Phase A without mixing old run state into the
new experiment.

## Repo-side launcher

Use:

- `scripts/setup_kaggle_env.py`
- `scripts/package_phase_a_dataset.py`
- `scripts/run_phase_a_kaggle.py`

`package_phase_a_dataset.py` is the local helper you can use before uploading
to Kaggle so the final bundle already matches what the launcher expects.

The Kaggle launcher:

1. validates the uploaded Kaggle dataset layout;
2. either reuses the uploaded dense Phase-A JSONs directly or rebuilds them
   from raw cleaned source files;
3. writes a Kaggle-specific model config and training profile;
4. audits the training dataset;
5. precomputes MobileCLIP features;
6. starts the v4 warm-start training run.

## Kaggle environment setup

If Kaggle internet is enabled, `setup_kaggle_env.py` can install MobileCLIP
from GitHub automatically.

If Kaggle internet is disabled, upload a local MobileCLIP wheel or source
archive under `vendor/` inside the Kaggle dataset bundle, then run the same
setup script with `--dataset-root`.

## Notes

- The uploaded dataset can stay read-only under `/kaggle/input/...`.
- Generated annotations, caches, configs, and outputs are written under
  `TinyTrace/.kaggle_phase_a_v4/` by default.
- If you already precomputed the feature cache inside the working session, rerun
  the launcher with `--skip-feature-cache`.
