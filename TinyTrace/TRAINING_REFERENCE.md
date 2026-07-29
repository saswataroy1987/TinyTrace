# TinyTrace Training Configuration Reference

This is the single reference for configuring TinyTrace model construction,
datasets, training, validation, checkpointing, logging, and controlled
experiments. It documents the executable contracts in `tinytrace/config.py`,
`tinytrace/training.py`, `scripts/train_tinytrace.py`, and the Priority 4
ablation utilities.

The companion [`configs/training_reference.jsonc`](configs/training_reference.jsonc)
is a commented, non-executable example. Existing JSON profiles remain the
actual launcher inputs and are intentionally unchanged.

## How to read this document

- **Code default** is the value used when the field is omitted.
- **Approved baseline** is the stable frozen-MobileCLIP research reference.
- **Accuracy / speed / memory** describe the expected direction, not a
  guaranteed measurement.
- **Retrain** means an existing trained checkpoint should not be used to claim
  results for the changed setting. `No` means the value changes only paths,
  logging, evaluation frequency, or runtime behavior.
- **Priority** identifies where the parameter became part of the current
  engineering contract: P1 correctness/architecture, P2 optimization, P3
  stability/reproducibility, or P4 representation ablation.

Three configurations must not be confused:

1. **Current QVHighlights Phase A v3** uses
   `configs/tinytrace_qvhighlights_phase_a_v3.json` with 128 uniformly sampled
   frames and predicts one query-conditioned saliency score for each of the 75
   two-second bins in a 150-second clip. It is the profile to use for current
   timestamp/saliency work.
2. The **legacy autoregressive/Phase-B architecture reference** uses frozen
   MobileCLIP-S0, 8 frames, 20 caption characters, width 192, and 4 LCEM
   layers. Its `time -> score -> caption` token protocol remains available for
   later caption experiments, but it is not the current Phase-A objective.
3. `configs/final_train_qvh500.json` is a historical operational profile. It
   sets `stage2_start_epoch=6`, enabling experimental partial MobileCLIP
   fine-tuning, and must not be resumed as a Phase-A-v3 run.

## Current QVHighlights Phase A v3

Phase A is a dense highlight-ranking task, not autoregressive event text
generation. The query and 128 frame groups are encoded once; a dedicated head
returns a `[batch, 75]` saliency curve. Bin `i` represents
`[2*i, 2*(i+1))` seconds and is supervised with the mean QVHighlights saliency
score in `[0, 4]`. This removes the character timestamp/score termination and
event-boundary failure modes observed in the v1 run.

The following autoregressive losses are exactly zero in the v3 model config:
`time_loss_weight`, `score_loss_weight`, `caption_loss_weight`,
`sync_loss_weight`, and `boundary_loss_weight`. Phase A therefore does **not**
train timestamp characters, score characters, captions, `<sync>` transitions,
or an event-count/boundary decision. Its active losses are dense score
regression, relevance classification, and within-video saliency ranking.

The executable files are:

- model: `configs/tinytrace_qvhighlights_phase_a_v3.json`;
- full profile: `configs/train_qvhighlights_phase_a_v3.json`;
- immutable train split:
  `../final_qvhighlights_tinytrace/annotations/tinytrace_phase_a_v3_train.json`
  (1,155 videos);
- immutable validation split:
  `../final_qvhighlights_tinytrace/annotations/tinytrace_phase_a_v3_val.json`
  (132 videos);
- audit files: `phase_a_v3_exclusions.json` and
  `phase_a_v3_manifest.json` in the same annotation directory.

All 1,354 filenames request exactly 150 seconds. The preparation step excludes
67 invalid source rows from the original 1,218/136 split: 66 media files are
shorter than 149.5 seconds and one cannot be probed. This includes 49 truncated
files that the superseded v2 publication missed because their sparse positive
labels ended before the missing media tail. Source annotations and v2 artifacts
remain unchanged. The v3 manifest records counts and SHA-256 hashes; launch
preflight also re-probes every media duration so stale/truncated replacements
cannot enter training.

MobileCLIP-S0 remains frozen. Decode 128 frames once into the versioned compact
uint8 frame cache, then precompute versioned FP16 spatial patch features. The
training profile requires that feature cache, uses a micro-batch of 1 with 8
accumulation steps, and uses `lr=0.0001`.

From the repository root, the fail-closed workflow is one command:

```bash
PYTHONPATH=TinyTrace python TinyTrace/scripts/run_phase_a_pipeline.py --device cuda
```

It verifies the immutable split and pinned checkpoint, caches four clips,
overfits those real videos, requires a large improvement over the seeded
untrained/constant baselines, checks that query swaps materially change logits
and scores while video swaps/removal degrade temporal ranking, then completes
the feature cache, runs
exactly 100 optimizer steps on real data, and starts the full run only after
every gate passes. It refuses non-empty gate/full output
directories. Add `--skip-full` to stop after the gates, or
`--skip-feature-cache` only when a complete compatible cache already exists.

Training-time validation uses explicitly named
`qvh_mean_score_proxy_*` metrics because the converted data contains mean
saliency scores. These are useful checkpoint-selection proxies, not official
QVHighlights results. Exact official Fair/Good/VeryGood mAP and Hit@1 require
the original per-bin scores from all three QVHighlights annotators and must be
reported only through the official evaluator when those labels are supplied.

The remainder of this document describes shared fields and the legacy
autoregressive baseline. Where a row says “approved baseline,” it refers to
that legacy/Phase-B reference unless the row explicitly names Phase A v3.

## Dataset and data loading

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `train_dataset_json` | `""` | Explicit real train JSON | Annotation path. Empty selects deterministic synthetic data of `dataset_size`; real training must provide a JSON list. Change for a new split/version. | Dataset-dependent; split changes invalidate comparisons. | Larger data increases epoch time. | Neutral. | Yes | Stable | P1 |
| `val_dataset_json` | `""` | Explicit, fixed validation JSON | Validation annotation path. Required for real validation and when early stopping is enabled. Never tune on test data. | Better selection reliability; no direct model effect. | Validation adds runtime. | Temporary validation activations only. | No, but reevaluate/reselect | Stable | P2 |
| `dataset_size` | `128` | `128` only for synthetic smoke data | Number of synthetic examples when no training JSON is supplied. Ignored for real JSON datasets. Use 4–8 for overfit gates and 128 for smoke runs. | Diagnostic only. | Linear synthetic epoch cost. | Neutral per batch. | Yes for the experiment | Stable | P1 |
| `frame_cache_dir` | `".cache/frames"` | Dedicated versioned cache directory | Stores decoded/preprocessed frame tensors. Use a separate location per dataset or storage policy. Cache identity already includes source metadata, frame count, image size, and format version. | Neutral when valid. | Usually improves repeated epochs substantially. | Neutral; consumes disk/RAM cache, not GPU memory. | No | Stable | P1 |
| `visual_feature_cache_dir` | `""` | Phase A: dedicated versioned FP16 cache | Stores frozen MobileCLIP spatial patch features. Phase A precomputes it once so 128-frame MobileCLIP inference is not repeated every epoch. It is incompatible with visual-backbone fine-tuning. | Neutral when identity/shape checks pass. | Material training speedup. | Reduces training GPU work; consumes disk. | No for identical frozen features | Stable Phase-A path | P1 |
| `require_visual_feature_cache` | `false` | `true` for Phase A v3 | Fail instead of decoding/re-encoding when a compatible feature entry is absent. Requires `visual_feature_cache_dir` and `stage2_start_epoch=0`. | Prevents mixed/missing evidence. | Avoids unexpected slow fallback. | Neutral. | No | Stable safety switch | P1 |
| `allow_random_frames` | `false` | `false` | Allows random frames when a JSON item lacks `video_path`/`frames_path`. Only acceptable for explicit synthetic debugging. | Severe harm on real training. | Avoids decode but produces meaningless evidence. | Similar batch memory. | Yes | Stable safety switch | P1 |
| `num_workers` | `0` | `2` in current final profile | DataLoader workers; integer `>=0`. Start with 2 on Windows and test 2/4/8 on the target storage. Persistent workers are enabled when nonzero. | Neutral if ordering remains deterministic. | More workers can hide decode latency; too many can slow I/O. | GPU neutral; increases host memory. | No | Stable, hardware-tuned | P3 |
| `batch_size` | `8` | `2` for real 256px video training | Micro-batch size, positive integer. Raise until GPU utilization is good without OOM; preserve effective batch with accumulation when reducing it. | Can affect optimization/noise. | Usually improves throughput until saturation. | Approximately linear increase. | Yes | Stable, hardware-tuned | P2 |
| `seed` | `7` | `7` | Seeds Python, PyTorch, sampler generators, and synthetic data. Keep fixed for controlled comparisons; repeat promising results with additional seeds. | No expected mean change; exposes variance. | Neutral. | Neutral. | Yes for exact comparison | Stable | P3 |
| `deterministic` | `true` | `true` | Enables deterministic algorithms and loader ordering. CUDA launchers automatically set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before child training so cuBLAS matrix multiplication satisfies PyTorch's deterministic contract. Disable only for a measured performance experiment. | Improves reproducibility, not expected mean quality. | May reduce throughput. | Usually neutral. | Yes for strict comparison | Stable | P3 |

Legacy dataset JSON items may specify `num_frames` between 1 and `max_frames`,
plus `video_path` or `frames_path`, `instruction`, and `events`. Phase-A-v3
items additionally require `task_mode="highlight"`, a non-empty query, exactly
75 `dense_saliency_scores`, a recorded duration of at least 149.5 seconds, and
the two-second bin metadata. Short decoded
sequences are padded and masked. Random fallback is never approved for real
training or validation.

## Representation and MobileCLIP

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `image_size` | `256` | `256` | Square MobileCLIP input size; positive integer. The verified S0 spatial contract is for 256. Almost never change. | Unknown; changing breaks the verified feature contract. | Larger is slower in MobileCLIP. | Larger increases activations. | Yes | Stable/frozen | P1 |
| `max_frames` | `8` | Legacy baseline: `8`; Phase A v3: `128` | Maximum sampled frames, valid `1..128`. The historical P4 autoregressive ladder remains `8→12→16→24→32`; current Phase A deliberately uses 128 uniform samples for two-second saliency coverage. | More frames improve the chance of observing short highlights. | MobileCLIP and LCEM work grow roughly linearly; use the frozen feature cache for Phase A. | Increases activations and attention memory. | Yes | 128 approved for Phase A v3 | P4 |
| `visual_hidden_dim` | `1024` | `1024` | MobileCLIP-S0 spatial channel width. Must match `[B*T,64,1024]`; do not tune. | Incorrect values fail or corrupt the interface. | Neutral when correct. | Projection size depends on it. | Yes/checkpoint-incompatible | Stable/frozen | P1 |
| `compressed_visual_tokens` | `4` | `4` | Learned slots per frame; positive integer. Do not change before diagnostics demonstrate a compression bottleneck. | More slots may retain detail but can overfit. | More LCEM prefix tokens slow attention. | Increases activation/attention memory. | Yes/checkpoint-incompatible | Stable; future experiment | P1 |
| `time_tokens_per_frame` | `6` | `6` | Fixed `0000.0` TRACE frame-time width. Only 6 is valid. | Changing breaks time encoding. | Neutral when correct. | Neutral. | Yes/checkpoint-incompatible | Stable/frozen | P1 |
| `mobileclip_model_name` | `"mobileclip_s0"` | `"mobileclip_s0"` | Pinned visual backbone identity. Do not replace within TinyTrace. | Backbone-dependent and architecture-breaking. | Backbone-dependent. | Backbone-dependent. | Yes/checkpoint-incompatible | Stable/frozen | P1 |
| `mobileclip_checkpoint` | `"checkpoints/mobileclip_s0.pt"` | Same | Path to official pretrained weights. Change only to relocate the identical verified checkpoint. | Wrong weights severely harm quality. | No effect. | No effect. | No if bytes identical; otherwise yes | Stable | P1 |
| `mobileclip_checkpoint_sha256` | Official 64-char digest | Same | Integrity digest. Empty disables verification but is not recommended. Change only for a deliberately versioned checkpoint. | Prevents silent weight mismatch. | Small one-time startup cost. | Neutral. | No if bytes identical | Stable | P1 |
| `freeze_visual_encoder` | `true` | `true` | Initial MobileCLIP trainability. Keep true for the approved baseline. Partial unfreezing is controlled separately by stage-2 settings. | Freezing is stable; unfreezing may adapt or forget. | Unfreezing slows backward pass. | Unfreezing materially increases memory. | Yes | Stable baseline; false experimental | P1 |
| `mobileclip_apply_normalization` | `false` | `false` | Optional mean/std normalization. The pinned Apple S0 transform does **not** use it. Almost never change. | Enabling shifts pretrained features and is expected to hurt. | Negligible. | Negligible. | Yes/cache invalidation required | Stable/frozen | P1 |
| `mobileclip_image_mean` | `[0.48145466,0.4578275,0.40821073]` | Same, inactive | Three finite channel values used only when normalization is explicitly enabled. | Experimental feature-distribution change. | Negligible. | Negligible. | Yes/cache invalidation required | Compatibility-only | P1 |
| `mobileclip_image_std` | `[0.26862954,0.26130258,0.27577711]` | Same, inactive | Three positive channel values used only when normalization is enabled. | Experimental feature-distribution change. | Negligible. | Negligible. | Yes/cache invalidation required | Compatibility-only | P1 |

At the legacy baseline, the visual-time prefix is `8 × (4 + 6) = 80`
tokens. The historical named frame profiles produce 80, 120, 160, 240, and
320 prefix tokens respectively. Phase A v3 uses
`128 × (4 + 6) = 1,280` prefix tokens and stays within the declared 2,048
position capacity.

## Model and LCEM

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `d_model` | `192` | `192` | LCEM width; positive, even, divisible by `num_heads`. Increase only after representation and fine-tuning evidence shows under-capacity. | May improve underfitting; raises overfitting risk. | Wider layers are slower. | Increases parameters and activations. | Yes/checkpoint-incompatible | Stable baseline; scaling experimental | P1 |
| `num_layers` | `4` | `4` | Number of decoder blocks. A future isolated depth test may compare 4 vs 6. | May improve underfitting; can overfit. | Roughly linear decoder cost. | Increases activations/parameters. | Yes/checkpoint-incompatible | Stable baseline; scaling experimental | P1 |
| `num_heads` | `6` | `6` | Attention heads; positive and must divide `d_model`. Change only with a justified architecture experiment. | Usually secondary to width/depth. | Can alter kernel efficiency. | Usually similar at fixed width. | Yes/checkpoint-incompatible | Stable | P1 |
| `mlp_ratio` | `4` | `4` | Decoder MLP expansion ratio; positive integer. Do not tune before capacity evidence. | Higher may add capacity/overfitting. | Higher is slower. | Higher increases parameters/activations. | Yes/checkpoint-incompatible | Stable | P1 |
| `dropout` | `0.0` | Legacy baseline: `0.0`; Phase A v3: `0.1` | Decoder attention/MLP/input dropout, valid `[0,1)`. The reviewed Phase-A-v3 profile uses 0.1; do not project that decision onto the legacy caption baseline without a separate ablation. | May improve generalization, often raises train loss. | Small training slowdown; inference unchanged. | Usually neutral. | Yes | Mode-specific | P3 |
| `max_position_embeddings` | `2048` | `2048` | Positional capacity. Must cover maximum training and inference sequences. Raise only when a justified representation/protocol change requires it. | No gain unless current capacity is exceeded. | No material runtime effect below used length. | Larger positional buffer is small; actual sequence length dominates. | Usually yes/checkpoint contract | Stable | P1 |
| `time_loss_weight` | `1.0` | `1.0` | Non-negative multiplier for timestamp-token loss. Change one weight at a time after gradient/task diagnostics. | Can trade timestamp quality against other tasks. | Neutral. | Neutral. | Yes | Stable baseline; tuning experimental | P1 |
| `score_loss_weight` | `1.0` | `1.0` | Score-token loss multiplier. | Can trade score quality against other tasks. | Neutral. | Neutral. | Yes | Stable baseline; tuning experimental | P1 |
| `caption_loss_weight` | `1.0` | `1.0` | Caption character loss multiplier. | Can trade caption quality against numeric tasks. | Neutral. | Neutral. | Yes | Stable baseline; tuning experimental | P1 |
| `sync_loss_weight` | `1.0` | `1.0` | Shared multiplier for time/score/caption synchronization terms. | Affects valid phase transitions/completion. | Neutral. | Neutral. | Yes | Stable baseline; tuning experimental | P1 |
| `boundary_loss_weight` | `1.0` | `1.0` | Event-boundary/EOS decision loss multiplier. | Affects event counts and stopping behavior. | Neutral. | Neutral. | Yes | Stable baseline; tuning experimental | P1 |

### Dense Phase-A head fields

These fields are active only when `phase_a_dense_saliency=true`. They do not
change the legacy autoregressive caption path.

| Parameter | Code default | Phase-A-v3 value | Contract |
|---|---:|---:|---|
| `phase_a_dense_saliency` | `false` | `true` | Select the direct query-conditioned dense saliency head. |
| `phase_a_bin_count` | `75` | `75` | Number of output bins; the QVHighlights contract is exactly 75. |
| `phase_a_bin_size_seconds` | `2.0` | `2.0` | Seconds represented by each direct bin. |
| `phase_a_max_score` | `4.0` | `4.0` | Upper bound of the dense saliency target/prediction scale. |
| `phase_a_positive_threshold` | `3.0` | `3.0` | Threshold used for relevance supervision and human-readable event runs. |
| `saliency_regression_loss_weight` | `1.0` | `1.0` | Balanced Smooth-L1 score regression. |
| `saliency_relevance_loss_weight` | `0.5` | `0.5` | Binary relevance loss for targets at least 3. |
| `saliency_ranking_loss_weight` | `0.25` | `0.25` | Within-video pairwise ranking loss. |
| `saliency_positive_weight` | `4.77` | `4.68` | Train-only negative/positive-bin ratio for v3 (exactly 4.68293643, rounded). Recompute for a different dataset. |
| `visual_encoder_chunk_size` | `32` | `16` | Positive maximum number of frames per MobileCLIP forward. Phase-A feature precomputation uses 16 to bound peak memory. |

## Event protocol, text, and generation

This section is the legacy autoregressive/Phase-B protocol. In dense Phase A,
the prompt is still tokenized, but event timestamp, score, caption, `<sync>`,
and boundary targets are not serialized or optimized.

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `text_vocab_size` | `256` | `256` | Character/byte-like text vocabulary size. Do not change without tokenizer migration. | Changes language modeling contract. | Larger head can be slower. | Increases embeddings/heads. | Yes/checkpoint-incompatible | Stable/frozen | P1 |
| `max_text_len` | `48` | Legacy: `48`; Phase A v3: `256` | Maximum instruction characters; non-negative. Phase A retains longer QVHighlights queries while remaining within the verified position budget. | May preserve longer instructions. | Longer prompt increases attention cost. | Increases activations. | Yes for comparable training | Stable | P1 |
| `max_caption_tokens` | `20` | `20` | Maximum caption target/generation characters, valid `0..64`. Approved candidates are 48 and 64 with matching generation budgets. | Larger reduces target truncation and may improve captions. | Can materially increase autoregressive latency. | Training activations and inference KV/state grow. | Yes | Experimental above 20 | P4 |
| `min_caption_tokens` | `5` | `5` | Minimum generated caption length before `<sync>` is allowed; `0..max_caption_tokens`. Change only for measured empty/overlong-caption failures. | Can prevent empty captions or force noise. | Higher may generate more steps. | Small sequence increase. | Yes for consistent behavior | Stable | P1 |
| `max_events` | `3` | `3` | Maximum serialized/generated events; positive integer. Change only if dataset analysis proves truncation and budgets are recomputed. | More can improve recall or increase false events. | More autoregressive steps. | Longer sequences increase memory. | Yes | Stable | P1 |
| `max_generated_tokens` | `128` | Legacy: `128`; Phase A v3: `1` | Legacy hard generation cap, valid `1..512` and at least its required event budget. Dense Phase A performs one direct forward and sets this inactive compatibility field to 1. Use 202 for legacy caption-48 and 250 for caption-64. | Too low truncates legacy structured output; it does not control dense predictions. | Controls legacy worst-case latency only. | Controls legacy sequence memory only. | Yes for legacy; no dense effect | Mode-specific | P1 |
| `timestamp_value_count` | `2` | `2` | Start/end values per event; non-negative, but the current architecture requires the two-value timestamp contract. | Changing breaks localization semantics. | Changes sequence length. | Changes sequence memory. | Yes/checkpoint/protocol break | Stable/frozen | P1 |
| `score_value_count` | `1` | `1` | Scores per event. Keep one. | Changing breaks task semantics. | Changes sequence length. | Changes sequence memory. | Yes/checkpoint/protocol break | Stable/frozen | P1 |
| `time_vocab` | 13 TRACE symbols | Same | Exact ordered tuple `<sync>,<sep>,0..9,.`; validation rejects any change. | Changing breaks encoding/parsing. | Neutral. | Neutral. | Yes/checkpoint/protocol break | Stable/frozen | P1 |
| `score_vocab` | 13 TRACE symbols | Same | Same fixed numeric vocabulary for scores. | Changing breaks encoding/parsing. | Neutral. | Neutral. | Yes/checkpoint/protocol break | Stable/frozen | P1 |
| `pad_token_id` | `0` | `0` | Text padding ID; must be unique and inside text vocabulary. | Incorrect value corrupts masking. | Neutral. | Neutral. | Yes/checkpoint break | Stable/frozen | P1 |
| `bos_token_id` | `1` | `1` | Beginning-of-sequence ID. | Incorrect value breaks prompting. | Neutral. | Neutral. | Yes/checkpoint break | Stable/frozen | P1 |
| `eos_token_id` | `2` | `2` | End-of-sequence ID. | Incorrect value breaks stopping/parsing. | Can alter generated length. | Sequence-dependent. | Yes/checkpoint break | Stable/frozen | P1 |
| `video_token_id` | `3` | `3` | Instruction/video boundary marker. | Incorrect value breaks conditioning. | Neutral. | Neutral. | Yes/checkpoint break | Stable/frozen | P1 |
| `instruction_token_offset` | `4` | `4` | Reserved text-token offset inside vocabulary. Keep fixed with tokenizer. | Protocol-dependent. | Neutral. | Neutral. | Yes/checkpoint break | Stable/frozen | P1 |

Derived contracts are not user parameters: the baseline requires 118 tokens to
represent three maximum-length events plus EOS; `max_generated_tokens=128`
provides margin. Configuration loading rejects position or generation budgets
that cannot represent the declared maximum.

## Training run control

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `train_script` | `"scripts/train_tinytrace.py"` | Repository training script | Launcher target. Change only when intentionally using a versioned alternative entry point. | Code-dependent. | Code-dependent. | Code-dependent. | Usually yes | Stable | P3 |
| `model_config` | `"configs/tinytrace_baseline.json"` | Baseline model JSON | Model configuration path. Ablations substitute a named profile without editing the baseline. | Configuration-dependent. | Configuration-dependent. | Configuration-dependent. | Yes | Stable | P3 |
| `output_dir` | `"outputs"` | Unique directory per run | Root for logs, summaries, predictions, and checkpoints. Never reuse an unrelated completed run directory. | Neutral. | Disk-dependent. | Neutral. | No | Stable | P2 |
| `device` | `"cpu"` | `"cuda"` for real training | PyTorch device string such as `cpu`, `cuda`, or `cuda:0`. Verify availability before launch. | Small numerical variation possible. | CUDA is dramatically faster. | Uses selected accelerator memory. | No, but reevaluate parity | Stable | P2 |
| `epochs` | `3` | `10` in current final profile | Positive maximum epoch count. Set with convergence and early stopping in mind. | Too few underfit; too many can overfit. | Linear upper-bound runtime. | Neutral per step. | Yes | Stable, experiment-specific | P2 |
| `max_steps_per_epoch` | `0` | `0` | `0` means all batches; positive values create smoke/debug limits. Never use a limited value for a claimed full run. | Limits learning/data exposure. | Reduces runtime. | Neutral. | Yes | Stable diagnostic | P3 |
| `max_optimizer_steps` | `0` | Full Phase A: `0`; smoke: `100` | Exact global optimizer-step cap after accumulation. `0` means uncapped. Use this, not a micro-batch limit, for reproducible overfit/smoke gates. | Diagnostic cap only. | Bounds gate runtime exactly. | Neutral. | Yes | Stable diagnostic | P3 |

## Optimizer

TinyTrace uses AdamW with named responsibility groups:
`compression`, `embeddings`, `lcem`, `task_heads`, and `mobileclip`. Each is
split into `decay` (matrix/tensor weights) and `no_decay` (biases/scales).
Duplicate or unassigned parameters fail early.

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `lr` / `learning_rate` | `0.0003` | Legacy baseline: `0.0003`; Phase A v3: `0.0001` | Peak LR for task modules; finite and `>0`. Phase A lowers it after persistent clipping in the failed v1 run. Tune only after inspecting warmup, gradient norms, and validation. | Too high destabilizes; too low undertrains. | Higher may converge faster but can diverge. | Neutral. | Yes | `0.0001` approved for Phase A v3 | P2 |
| `weight_decay` | `0.01` | `0.01` | AdamW decay for parameters with rank `>=2`; finite non-negative. Biases/norm scales receive zero decay. | May improve generalization; excess underfits. | Negligible. | Neutral. | Yes | Stable | P2 |
| `gradient_clip` | `1.0` | Legacy: `1.0`; Phase A v3: `5.0` | Global pre-step norm cap; non-negative. `0` disables effective clipping. A real 128-frame Phase-A initial batch produced a finite norm of about 4.37, so retaining 1.0 would clip immediately; the 100-step gate still rejects a clipping rate of 50% or more. | Prevents unstable updates; too low suppresses learning. | Small norm-computation cost. | Small transient cost. | Yes | Mode-specific, smoke-gated | P2 |
| `stage2_visual_lr_scale` / `visual_lr_scale` | `0.1` | `0.05` in current final profile; inactive in frozen baseline | Multiplier applied to the MobileCLIP optimizer groups. Use only with approved partial unfreezing; recommended conservative range `0.01..0.1`. | Can enable adaptation or catastrophic forgetting. | Neutral forward; unfreezing controls backward cost. | Neutral alone. | Yes | Experimental | P3 |

## Scheduler

The scheduler is linear warmup followed by cosine decay and advances only after
a successful optimizer step. Total steps account for gradient accumulation.

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `warmup_ratio` | `0.05` | `0.05` | Fraction of total optimizer steps used for linear warmup; `[0,1]`, but must yield fewer than total steps. Typical range 3–10%. | Stabilizes early training; excess delays learning. | Same step count. | Neutral. | Yes | Stable baseline; tuning experimental | P2 |
| `min_lr_ratio` | `0.1` | `0.1` | Final LR as a fraction of peak after cosine decay; `[0,1]`. | Lower can refine convergence; too low may stop useful learning. | Same step count. | Neutral. | Yes | Stable | P2 |

## Precision: AMP, BF16, and FP16

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `amp` / `amp_mode` | `"off"` | `"off"` reference; `"auto"` recommended after parity smoke | One of `off`, `auto`, `fp16`, `bf16`. `auto` selects BF16 on supported CUDA, otherwise FP16; on CPU it disables AMP. FP16 CUDA uses gradient scaling. | Usually near parity; must verify finite loss and metrics. | Often faster on modern GPUs. | Usually materially lower activation memory. | Yes for controlled comparison | Stable implementation; hardware-conditional | P2 |

Prefer BF16 on supported hardware. Use FP16 only with the built-in scaler and
verify skipped/non-finite steps. Never change precision halfway through an
unrecorded run.

## Gradient accumulation

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `accumulation_steps` | `1` | `1` | Positive micro-batches per optimizer step. Effective batch is `batch_size × accumulation_steps` for one device. Raise when memory prevents the desired effective batch. | Preserving effective batch improves comparison fairness. | More micro-steps per update; throughput may fall. | Lowers required memory versus equal physical batch. | Yes | Stable | P3 |

The final partial window is normalized correctly; clipping, optimizer, and
scheduler steps occur only at accumulation boundaries.

## Checkpointing and resume

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `resume` | `""` | `""` for new run | Path to a complete checkpoint. Resume restores model, optimizer, scheduler, scaler, RNG, counters, early stopping, and history. Use only with compatible model/training contracts. | Preserves continuation when exact. | Adds startup I/O. | Neutral. | No; continues same run | Stable | P3 |
| `save_every` | `5` | `1` for important real runs | Periodic checkpoint interval in epochs; `0` disables periodic epoch snapshots. `latest` and best roles are still maintained. | Neutral; improves recoverability. | More disk I/O. | Neutral. | No | Stable | P2 |
| `checkpoint_keep` | `3` | `3` | Positive count of periodic epoch checkpoints retained. Does not remove role checkpoints such as latest/best. | Neutral. | Disk-management only. | Neutral. | No | Stable | P2 |

Checkpoint roles are `latest.pt`, `best-loss.pt`,
`best-primary-metric.pt`, compatibility alias `best.pt`, bounded periodic
`epoch-XXXX.pt`, and inference artifact `tinytrace.pt`. Checkpoint format and
optimizer format are currently version 3 and are not configurable.

## Early stopping and checkpoint selection

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `early_stopping_patience` | `0` | `0` reference | Non-negative eligible epochs without improvement; `0` disables stopping. For longer validated runs, 3–5 is a reasonable starting range. | Can prevent overfitting; too short stops noise. | Can reduce total runtime. | Neutral. | Yes | Stable, experiment-specific | P2 |
| `early_stopping_min_delta` | `0.0` | `0.0` | Required improvement magnitude; finite non-negative. Choose in units of the monitored metric. | Filters noise; too high ignores real gains. | May stop sooner. | Neutral. | Yes | Stable | P2 |
| `early_stopping_min_epochs` | `1` | `1` reference | Positive epoch before bad epochs count. For 2,000 videos, use at least 5. Early stopping also remains inactive during warmup. | Prevents premature stopping. | Sets minimum runtime. | Neutral. | Yes | Stable | P2 |
| `monitor` | `"val_loss"` | Legacy: `"val_loss"`; Phase A v3: `"qvh_mean_score_proxy_Good_mAP"` | Selects the checkpoint metric. Dense Phase A supports the explicitly named mean-score proxy Good mAP/Hit1 monitors. Do not rename a proxy as official QVHighlights. Structured metrics require validation prediction at that epoch. | Determines selected checkpoint. | Structured monitors increase validation cost. | Temporary inference memory. | No, but checkpoint choice/evaluation changes | Stable | P2 |
| `monitor_mode` | `"min"` | Legacy: `"min"`; Phase A v3: `"max"` | Must match metric direction: minimize losses/MAEs, maximize IoU/recall/exact match. Invalid combinations are rejected. | Incorrect direction would select bad weights; validation prevents it. | Neutral. | Neutral. | No | Stable | P2 |

## Validation

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `metrics_every` | `1` | `1` | Epoch interval for full structured generation metrics; `0` disables periodic metrics unless the monitor requires them. Raise to 2–5 for large/slow validation sets while still computing teacher-forced validation loss each epoch. | Less frequent feedback, no direct training change under `val_loss`. | Can substantially reduce validation time. | Neutral peak; less frequent use. | No | Stable | P3 |

Validation uses deterministic order, `model.eval()`, no gradients, real inputs
only, and the same parser/metrics implementation for baseline and candidate.
Changing the validation split or metric code requires reevaluating all compared
checkpoints.

## Logging

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `log_every` | `25` | `25` | Console progress interval in micro-steps; `0` disables periodic console lines. Machine-readable epoch/step logging remains. | Neutral. | Very frequent logging can slow training. | Neutral. | No | Stable | P2 |

Fixed machine-readable artifacts include `events.jsonl`, `history.json`,
`metrics.json`, per-epoch metric JSON, `run_metadata.json`,
`run_summary.json`, and `failure.json` on failure. Logs include decomposed and
weighted task losses, target counts, learning rates per optimizer group,
gradient norms/clipping, examples/frames/tokens throughput, elapsed time,
accelerator peak memory, caption-budget truncation, selection state, source
revision, dependencies, seed, and hardware.

## Prediction artifacts

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `prediction_every` | `1` | `1` | Epoch interval for saved prediction examples; `0` disables periodic files. | Neutral; enables qualitative debugging. | Generation adds validation time. | Temporary inference memory. | No | Stable | P2 |
| `prediction_samples` | `2` | `2` | Positive number of deterministic examples stored per prediction artifact. Increase to 4–8 for long runs if disk/latency are acceptable. | Neutral. | More examples take longer. | Peak unchanged; output disk grows. | No | Stable | P2 |

Legacy autoregressive predictions include sample/source identity, checkpoint
identity, ground truth, raw generated token IDs, parsed events, termination,
caption truncation, and parser warnings. Dense Phase-A predictions instead
include the full 75-value `pred_saliency_scores` curve, the corresponding
`qvh_mean_score_targets`, thresholded human-readable runs, and curve-diversity
diagnostics; no autoregressive termination claim applies.

## Staged-training experiment parameters

| Parameter | Code default | Approved baseline | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `stage2_start_epoch` | `0` | `0` frozen reference; current final profile uses `6` | `0` disables partial MobileCLIP fine-tuning. Positive epoch enables the declared strategy from that epoch. Do not enable in a representation-only ablation. | May improve grounding or cause forgetting. | Slower training after transition; inference unchanged. | Material training-memory increase. | Yes | Experimental | P3 |
| `stage2_unfreeze_strategy` | `"conv_exp"` | `"conv_exp"`, inactive | Named MobileCLIP final expansion/stage policy. Currently use only `conv_exp`; arbitrary strings may fail when activated. | May adapt final visual features. | Adds backward work. | Increases gradients/optimizer activation state. | Yes | Experimental | P3 |

## Priority 4 ablation parameters

These are CLI parameters for `scripts/run_representation_ablation.py`; they are
not fields in a normal training profile.

| Parameter | CLI default | Approved use | Description, valid values, and when to change | Accuracy | Speed | GPU memory | Retrain | Status | Priority |
|---|---:|---:|---|---|---|---|---|---|---|
| `training_profile` | Required | Frozen controlled profile | Base training profile cloned for both arms. | Holds training conditions constant. | Profile-dependent. | Profile-dependent. | Yes (runner trains arms) | Stable | P4 |
| `kind` | Required | `frame` or `caption` | Declares the sole independent variable. | Depends on candidate. | Depends on candidate. | Depends on candidate. | Yes | Stable | P4 |
| `baseline_frames` | `8` | `8`, then accepted rung only | Frame baseline in `8,12,16,24`; runner requires the immediately next ladder value. | Controls temporal evidence. | Higher baselines cost more. | Higher baselines use more memory. | Yes | Experimental | P4 |
| `caption_candidate` | `48` | `48` or `64` | Caption candidate compared independently with the 20-token reference. | May reduce truncation. | Longer generation. | Longer sequences. | Yes | Experimental | P4 |
| `output_root` | Required | Unique experiment directory | Stores cloned profiles, manifest, runs, benchmark, and report. | Neutral. | Disk-only. | Neutral. | No | Stable | P4 |
| `execute` | `false` | Review dry run first | Without it, only the controlled manifest/profiles are written; with it, baseline and candidate train sequentially and are benchmarked. | Neutral itself. | Enables long work. | Run-dependent. | No | Stable safety switch | P4 |
| `benchmark_samples` | `2` | `2+` representative samples | Number of fixed validation samples per measured iteration. Increase for more stable systems estimates. | No training effect. | Linear benchmark time. | Peak usually unchanged. | No | Stable | P4 |
| `benchmark_warmup` | `1` | `1–3` | Unmeasured warmup iterations before latency sampling. | Neutral. | Adds benchmark time. | Can stabilize allocator/cache state. | No | Stable | P4 |
| `benchmark_repeats` | `5` | `5–20` | Measured iterations. Increase for lower timing noise. | Neutral. | Linear benchmark time. | Neutral peak. | No | Stable | P4 |

The standalone benchmark also requires `config`, `checkpoint`, `dataset_json`,
`frame_cache_dir`, `output`, and `device`. It reports decode/preprocessing,
MobileCLIP, compression, generation, median/p90 end-to-end latency, generated
tokens/s, peak accelerator memory, and feature diversity. Use identical warmup,
repetitions, device, cache state, and samples for every comparison.

## Recommended current configuration for QVHighlights Phase A

Do not copy the legacy examples below for Phase A. Use the checked-in model and
training profile named in the Phase-A-v3 section, and launch them through
`scripts/run_phase_a_pipeline.py`. The critical effective settings are 128
frames, 75 direct two-second bins, a frozen required FP16 visual-feature cache,
zero autoregressive task weights, `lr=0.0001`, a measured Phase-A gradient
clip of `5.0`, batch 1, accumulation 8, and no full-run optimizer-step cap.

## Recommended legacy autoregressive baseline configuration

This is the approved reference, not an instruction to overwrite an existing
profile:

```json
{
  "model": {
    "image_size": 256,
    "max_frames": 8,
    "compressed_visual_tokens": 4,
    "time_tokens_per_frame": 6,
    "d_model": 192,
    "num_layers": 4,
    "num_heads": 6,
    "mlp_ratio": 4,
    "dropout": 0.0,
    "max_caption_tokens": 20,
    "max_events": 3,
    "max_generated_tokens": 128,
    "freeze_visual_encoder": true,
    "mobileclip_apply_normalization": false
  },
  "training": {
    "device": "cuda",
    "epochs": 10,
    "batch_size": 2,
    "accumulation_steps": 1,
    "lr": 0.0003,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.1,
    "amp": "off",
    "monitor": "val_loss",
    "monitor_mode": "min",
    "early_stopping_patience": 0,
    "save_every": 1,
    "checkpoint_keep": 3,
    "prediction_every": 1,
    "prediction_samples": 2,
    "metrics_every": 1,
    "num_workers": 2,
    "stage2_start_epoch": 0,
    "seed": 7,
    "deterministic": true,
    "allow_random_frames": false
  }
}
```

Before replacing `amp="off"` with `auto`, run a fixed-batch AMP parity smoke
test on the exact target GPU.

## Historical proposed 2,000-video autoregressive training

This is a proposed long-run starting point. It assumes a fixed train split of
approximately 2,000 verified videos and a separate deterministic validation
split. Hardware-sensitive fields must be confirmed with a 50–100-step smoke
run.

```json
{
  "train_dataset_json": "PATH/TO/tinytrace_train_2000.json",
  "val_dataset_json": "PATH/TO/tinytrace_val.json",
  "output_dir": "TinyTrace/outputs-qvh-2000-frozen-v1",
  "frame_cache_dir": "TinyTrace/.cache/frames_qvh2000_v1",
  "model_config": "TinyTrace/configs/tinytrace_baseline.json",
  "device": "cuda",
  "epochs": 15,
  "batch_size": 2,
  "accumulation_steps": 4,
  "lr": 0.0003,
  "weight_decay": 0.01,
  "gradient_clip": 1.0,
  "warmup_ratio": 0.05,
  "min_lr_ratio": 0.1,
  "amp": "auto",
  "early_stopping_patience": 3,
  "early_stopping_min_delta": 0.0,
  "early_stopping_min_epochs": 5,
  "monitor": "val_loss",
  "monitor_mode": "min",
  "save_every": 1,
  "checkpoint_keep": 3,
  "prediction_every": 1,
  "prediction_samples": 4,
  "metrics_every": 2,
  "num_workers": 2,
  "log_every": 50,
  "max_steps_per_epoch": 0,
  "stage2_start_epoch": 0,
  "stage2_visual_lr_scale": 0.05,
  "stage2_unfreeze_strategy": "conv_exp",
  "seed": 7,
  "deterministic": true,
  "allow_random_frames": false,
  "resume": ""
}
```

Why these differences from the normal baseline:

- effective batch size is 8 (`2 × 4`) without requiring a physical batch of 8;
- `amp=auto` is recommended for long-run efficiency only after parity passes;
- patience 3 after at least 5 epochs prevents obviously wasted late epochs;
- full structured generation every 2 epochs reduces validation cost while
  teacher-forced validation loss remains available every epoch;
- MobileCLIP stays frozen so this run establishes the clean 2,000-video
  reference before any partial fine-tuning experiment.

If the target GPU cannot hold batch 2, use batch 1 and accumulation 8. If it
comfortably holds a larger micro-batch, increase it while preserving effective
batch 8. Do not simultaneously change frames, captions, dropout, or model size.

## Long-run launch checklist

- [ ] Commit or record the exact source revision; record any dirty changes.
- [ ] Confirm the chosen profile parses with `TrainingProfile.from_json`.
- [ ] Confirm the model JSON parses with `TinyTraceConfig.from_json`.
- [ ] Verify the official MobileCLIP checkpoint exists and its SHA-256 matches.
- [ ] Confirm every train/validation video exists, decodes, and has valid event timestamps.
- [ ] Confirm train and validation IDs are disjoint and split files are immutable.
- [ ] Set `allow_random_frames=false`.
- [ ] Use a new `output_dir`; never overwrite a reported run.
- [ ] Use a cache directory tied to the dataset/preprocessing version and ensure enough disk space.
- [ ] For Phase A, verify all 1,336 cleaned samples have compatible frozen FP16 visual features and keep `stage2_start_epoch=0`.
- [ ] Confirm `max_frames`, caption budget, and generation budget match the intended named profile.
- [ ] Confirm the effective batch size: micro-batch × accumulation × device count.
- [ ] Run 50–100 optimizer steps and inspect finite losses, gradient norms, clipping, LR, throughput, and memory.
- [ ] Verify AMP parity on the target GPU before enabling it for the long run.
- [ ] Confirm scheduler total steps and warmup steps printed in run metadata are plausible.
- [ ] Confirm validation ordering is deterministic and validation uses no random fallback.
- [ ] Confirm checkpoint monitor/direction are correct and structured metrics are available if selected.
- [ ] Load the smoke checkpoint and verify exact resume of the next LR/step.
- [ ] Inspect prediction artifacts appropriate to the mode: 75-bin curves and conditioning diversity for Phase A; raw IDs/parser/termination for autoregressive Phase B.
- [ ] Estimate total epoch and validation-generation time before committing the full run.
- [ ] Ensure checkpoint retention and output disk capacity are sufficient.
- [ ] Record the hypothesis and the single independent variable for any ablation.

## Parameters that should almost never change

Treat these as architecture/protocol constants unless planning an explicit
checkpoint-format and dataset migration:

- `mobileclip_model_name`, `visual_hidden_dim`, `image_size`;
- MobileCLIP preprocessing and normalization fields;
- `time_tokens_per_frame`, `timestamp_value_count`, `score_value_count`;
- numeric vocabularies and all special token IDs;
- `text_vocab_size` and tokenizer contract;
- legacy Phase-B event order `time -> score -> caption`;
- baseline compressor slots, width, depth, and head structure without prior
  evidence of under-capacity.

## Parameters intended for controlled experimentation

Change only one conceptual factor at a time:

- `max_caption_tokens`: 20 vs 48 or 64, with dependent generation budget;
- `max_frames`: historical autoregressive ladder 8→12→16→24→32; Phase A v3 is the separately approved 128-frame profile;
- `dropout`: 0.0 vs 0.1 candidate;
- `stage2_start_epoch`, unfreeze strategy, and visual LR scale only in a
  dedicated fine-tuning experiment;
- task loss weights only after loss/gradient domination diagnostics;
- LR, warmup, decay floor, and weight decay in isolated optimization studies;
- decoder depth/width only after representation and fine-tuning evidence.

Hardware-only tuning may adjust `batch_size`, `accumulation_steps`,
`num_workers`, AMP mode, logging frequency, benchmark repetitions, and cache
location, but every change must still be recorded with the run.
