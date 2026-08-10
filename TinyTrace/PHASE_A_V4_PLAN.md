# TinyTrace Phase A v4 Retraining Plan

## Goal

Phase A v4 should improve temporal grounding quality enough to become a stronger
foundation for later TRACE-style timestamp and score generation.

This phase still trains **dense query-conditioned saliency over 75 two-second
bins**. It does **not** switch to autoregressive timestamp/score tokens yet.
That switch belongs to the next stage after Phase A is stronger.

## Why we are retraining Phase A

The completed Phase A v3 run is a technical success, but not yet a strong final
result:

- it clearly beats constant baselines
- it learns query/video conditioning
- it remains weak on exact temporal localization
- it is not yet strong enough to be the cleanest base for structured event
  generation

The biggest safe improvement path is to tighten **data quality and auditability**
first, then rerun Phase A with a controlled profile.

## Decision

For the next run:

1. Do **not** train from scratch in the sense of inventing a new target format.
2. Do **rebuild and re-audit** the Phase A dataset from raw
   `qvh_raw_valid.json`.
3. Do **run a new Phase A training cycle** after the dataset audit passes.
4. Treat the current v3 checkpoint as a **reference/baseline**, not as the only
   path forward.
5. Do **not** jump directly into full TRACE-style caption training yet.

## Data sources

Use these files as the authoritative input set:

- `final_qvhighlights_tinytrace/annotations/qvh_raw_valid.json`
- `final_qvhighlights_tinytrace/annotations/tinytrace_train.json`
- `final_qvhighlights_tinytrace/annotations/tinytrace_val.json`

These are the pre-Phase-A cleaned source files.

## Target format

Rebuild the current dense Phase A contract:

- 150.0 second clip window
- 75 bins
- 2.0 seconds per bin
- direct source mapping `int(source_time / 2.0)`
- no TRACE `-1` generated-time offset
- exclude media shorter than 149.5 seconds

The canonical implementation is:

- `scripts/prepare_phase_a_qvhighlights.py`

## Required pre-training checks

Before launching the next run:

1. Rebuild dense annotations from raw source.
2. Run dataset audit and inspect summary output.
3. Manually inspect a few examples with:
   - sparse positives
   - dense positives
   - long positive runs
   - multiple separated positive regions
4. Confirm the positive-bin ratio and train/val counts.
5. Confirm excluded clips are only missing/probe-failed/truncated media.

## Training recommendation

Use the same broad Phase A objective as v3:

- dense score regression
- relevance supervision
- within-video ranking

Keep the model family unchanged for this retraining cycle unless a hard
debugging signal shows otherwise. We want cleaner evidence before making larger
architecture changes.

## Checkpoint policy

For Phase A retraining itself, use a **fresh output directory** and train the
run cleanly from initialization so it remains directly comparable to v3.

Keep these v3 checkpoints as baselines:

- `outputs-qvh-phase-a-v3-full/checkpoints/best-primary-metric.pt`
- `outputs-qvh-phase-a-v3-full/checkpoints/best-loss.pt`

After we obtain a stronger Phase A model, **that** checkpoint should initialize
the next timestamp/score event-training stage.

## Success criteria for the next Phase A run

The next Phase A run should satisfy all of these:

1. Reproduce the fail-closed data contract without annotation errors.
2. Beat the current v3 best `qvh_mean_score_proxy_Good_mAP` of `36.2`.
3. Improve temporal ranking/localization, not only loss.
4. Keep conditioning diagnostics healthy.
5. Preserve deterministic, auditable artifacts.

## What comes after a successful Phase A rerun

After a stronger Phase A model is available:

1. Initialize a bridge stage from that checkpoint.
2. Re-enable structured timestamp and score supervision.
3. Keep caption loss off or minimal at first.
4. Move to full TRACE-like `time -> score -> caption` training only after the
   bridge stage behaves well.
