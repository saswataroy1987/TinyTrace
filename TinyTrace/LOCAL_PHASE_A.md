# Local Phase A

This is the recommended way to run TinyTrace Phase A on a local GPU machine
such as an RTX 4050 workstation.

## What you need

1. the latest TinyTrace GitHub repo
2. a local copy of `TinyTrace_kaggle_dataset_ready`

Recommended layout:

```text
~/work/TinyTrace/
├── TinyTraceRepo/
└── TinyTrace_kaggle_dataset_ready/
```

## One-command local run

From the cloned repo root:

```bash
cd ~/work/TinyTrace/TinyTraceRepo
python3 TinyTrace/scripts/run_phase_a_local.py \
  --dataset-root ../TinyTrace_kaggle_dataset_ready
```

This script will:

1. create a local virtual environment if it does not already exist
2. install a CUDA-enabled PyTorch build suitable for modern NVIDIA GPUs
3. install TinyTrace runtime dependencies and MobileCLIP
4. verify CUDA visibility
5. launch the TinyTrace Phase-A v5 warm-start run

If the venv already exists, it is reused automatically.

## Important options

Choose a custom work directory:

```bash
python3 TinyTrace/scripts/run_phase_a_local.py \
  --dataset-root ../TinyTrace_kaggle_dataset_ready \
  --work-root ./phase_a_v5_local_run
```

Reuse an existing feature cache after a failed run in the same work directory:

```bash
python3 TinyTrace/scripts/run_phase_a_local.py \
  --dataset-root ../TinyTrace_kaggle_dataset_ready \
  --work-root ./phase_a_v5_local_run \
  --skip-feature-cache \
  --skip-install
```

Force a clean venv rebuild:

```bash
python3 TinyTrace/scripts/run_phase_a_local.py \
  --dataset-root ../TinyTrace_kaggle_dataset_ready \
  --force-recreate-venv
```

## Output locations

- working caches and checkpoints: `--work-root`
- exported final artifacts: `<work-root>/exported_artifacts`

The exported artifact directory contains the final configs, annotations,
metrics, and checkpoints that matter for review.
