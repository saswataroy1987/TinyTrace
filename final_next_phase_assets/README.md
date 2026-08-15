# Final Next Phase Assets

This folder contains the small set of Phase-A artifacts that should travel with
the GitHub repo and be reused when starting the next TinyTrace phase on another
laptop.

## Included

- `phase_a_bootstrap/phase_a_v3_best_primary_metric.pt`
  - official Phase-A warm-start checkpoint for Phase B
  - safe to use as the default initialization source

- `configs/tinytrace_qvhighlights_phase_a_v3.json`
  - Phase-A v3 model config used for the validated bootstrap run

- `configs/train_qvhighlights_phase_a_v3.json`
  - Phase-A v3 training profile for reference and reproducibility

## Not Included

These remain local-only and should not be pushed in this folder:

- raw datasets
- raw videos
- feature caches
- large temporary run folders
- `checkpoints/mobileclip_s0.pt`

`mobileclip_s0.pt` is still required for training, but it is intentionally kept
outside this GitHub-carried folder because it is a large external checkpoint.

## Intended Use

On another laptop:

1. clone or pull the TinyTrace repo
2. keep this `final_next_phase_assets/` folder from GitHub
3. place the downloaded next-phase dataset locally
4. use `phase_a_bootstrap/phase_a_v3_best_primary_metric.pt` as the Phase-B
   warm-start checkpoint

## Why Phase A v3

Phase A v3 is the chosen bootstrap source because it has a proper saved
checkpoint format and is already the established warm-start artifact in the
project assets.
