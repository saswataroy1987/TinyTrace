# Final Causal Run

This is the final isolated lightweight TRACE-inspired run. It does not modify
MobileCLIP, the validated cache, the ActivityNet manifest, Stage 1, Stage 3,
v3, or any B1/B2/C1 directory.

## Architecture

Frozen cached MobileCLIP patches -> eight learned cross-attention slots per
cached frame -> six learned continuous-time tokens per cached frame -> FLAN-T5
encoder -> one autoregressive structured event sequence. The decoder emits
events in chronological target order as `<EVENT> <START> <T...> <END> <T...>
<CAPTION> ... </EVENT>`, ending with `<END_EVENTS>`.

The model receives only cached video frames, timestamps, and an instruction on
the encoder side. Ground-truth event boundaries/captions exist only in the
teacher-forced decoder target. Later event tokens therefore condition on
earlier event tokens through normal FLAN-T5 autoregressive decoder history.

## Preflight Only

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/train_final_causal_activitynet.py \
  --model-config TinyTrace/configs/tinytrace_final_causal_activitynet.json \
  --stage0-config phase_b_activitynet_v2_run/configs/resolved_config.json \
  --manifest phase_b_activitynet_v2_run/manifests/activitynet_v2_manifest.json \
  --cache-root phase_b_activitynet_v1_run/cache/mobileclip_activitynet_phase_b_v1 \
  --output-root stage2_final_causal --audit-root stage2_audit \
  --device cuda --batch-size 2 --sanity-only
```

## Final Training

Run this only after the preflight succeeds. It uses AdamW, seed 7, batch size
2, 15 epochs, bridge/time LR `1e-4`, FLAN LR `2e-5`, and audits every two
epochs.

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/train_final_causal_activitynet.py \
  --model-config TinyTrace/configs/tinytrace_final_causal_activitynet.json \
  --stage0-config phase_b_activitynet_v2_run/configs/resolved_config.json \
  --manifest phase_b_activitynet_v2_run/manifests/activitynet_v2_manifest.json \
  --cache-root phase_b_activitynet_v1_run/cache/mobileclip_activitynet_phase_b_v1 \
  --output-root stage2_final_causal --audit-root stage2_audit \
  --device cuda --epochs 15 --batch-size 2 \
  --bridge-learning-rate 1e-4 --flan-learning-rate 2e-5 \
  --audit-every 2 --seed 7
```

Outputs are written under `stage2_final_causal/`: resolved config, mandatory
preflight, logs/history, checkpoints, controlled real/shuffled audits, and
`FINAL_REPORT.md`. That directory is intentionally git-ignored because it
contains generated checkpoints and audit outputs.

## Inference

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/infer_final_causal_activitynet.py \
  --model-config TinyTrace/configs/tinytrace_final_causal_activitynet.json \
  --stage0-config phase_b_activitynet_v2_run/configs/resolved_config.json \
  --manifest phase_b_activitynet_v2_run/manifests/activitynet_v2_manifest.json \
  --cache-root phase_b_activitynet_v1_run/cache/mobileclip_activitynet_phase_b_v1 \
  --checkpoint stage2_final_causal/checkpoints/best-causal.pt \
  --video-id v_-E2dqOULQgY \
  --output stage2_final_causal/inference/v_-E2dqOULQgY.json \
  --device cuda
```
