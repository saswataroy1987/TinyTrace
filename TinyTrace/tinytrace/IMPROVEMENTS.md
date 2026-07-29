# TinyTrace Internal Engineering Improvement Specification

**Audience:** TinyTrace maintainers, machine-learning research engineers, and AI coding agents  
**Document type:** Internal design and execution specification  
**Scope:** Repository implementation, model architecture, optimization, validation, evaluation, and deployment engineering  
**Out of scope:** Dataset acquisition, collection, annotation, preparation, or recommendations to increase dataset size

## Executive Summary

TinyTrace is a lightweight causal event model for videos. Its current processing path is:

```text
Video
  -> frame sampling and decoding
  -> MobileCLIP-S0 spatial feature extraction
  -> learned slot compression
  -> discrete frame-time embeddings
  -> lightweight causal event model (LCEM)
  -> structured event generation
```

The output protocol is an architectural invariant. Every event must be generated in exactly this order:

```text
time -> score -> caption
```

The current baseline samples at most eight frames. Each frame is encoded by the frozen MobileCLIP-S0 image tower before global pooling, producing 64 spatial features of width 1024 for a 256-by-256 input. A learned slot compressor reduces those features to four `d_model`-wide visual tokens. Six fixed-width numeric time tokens are appended to every frame's visual tokens. The resulting frame groups are flattened chronologically and placed before the instruction and event tokens. A four-layer, six-head, width-192 causal Transformer processes the complete sequence. Independent time, score, and text heads implement structured generation, while a synchronization token switches the active head.

This design already has several important strengths: it retains spatial MobileCLIP information, provides explicit temporal metadata, constrains numeric formats, supports variable-length frame masking, and remains small enough for edge-oriented work. Its weaknesses are chiefly temporal coverage, limited caption budget, a decoder and tokenizer that may become bottlenecks, an incomplete staged fine-tuning strategy, and missing optimization and systems measurements required to make reliable design decisions.

The purpose of this specification is to improve accuracy and engineering quality without losing TinyTrace's defining constraints. MobileCLIP must remain the visual encoder. The decoder must remain lightweight. No large language model may be introduced. Complexity increases must be evidence-driven and evaluated against latency and memory, not accuracy alone.

An implementation agent following this document should deliver, in order: configurable and tested temporal sampling; better generation budgets without uncontrolled sequence growth; a robust optimization stack; a conservative partial-MobileCLIP fine-tuning path; phase-specific validation; decomposed task metrics; and reproducible latency, throughput, and memory reports. Future deployment optimizations are documented separately and must not be mixed into the immediate implementation phases.

## Design Philosophy

### Why TinyTrace exists

TRACE-style causal event modeling is attractive because it expresses several video-understanding tasks through one structured autoregressive protocol. A full-scale implementation, however, can require a large visual backbone and a large language decoder. That cost conflicts with local inference, research on limited hardware, real-time processing, and edge deployment. TinyTrace exists to test how much of the structured modeling benefit can be retained with a compact visual-language stack.

The objective is therefore not maximum benchmark performance at any computational cost. The objective is the best attainable accuracy under explicit constraints: a MobileCLIP encoder, a small task-specific decoder, bounded memory, practical latency, deterministic structured output, and clean reproducibility.

### Why the architecture must remain lightweight

Model size affects more than checkpoint storage. It determines activation memory, optimizer-state memory, memory bandwidth, startup time, thermal behavior, and the range of devices on which inference is possible. Increasing decoder width increases attention projections, MLP parameters, activations, and output-head computation. Increasing sequence length increases attention work approximately quadratically. Increasing the number of sampled frames increases both MobileCLIP work and LCEM prefix length. These costs interact, so apparently modest independent changes can multiply total latency.

Every architecture proposal must therefore include a cost estimate and a measurement plan. An improvement that adds 1% task accuracy but doubles end-to-end latency is not automatically acceptable. The correct decision depends on the declared deployment budget and should be recorded as a measured tradeoff.

### Why latency matters

Video models process repeated visual inputs. Even when the LCEM is small, decoding many frames through MobileCLIP can dominate wall-clock time. Autoregressive generation then invokes the decoder repeatedly, making output length another direct latency multiplier. Latency must be split into at least frame decoding, MobileCLIP feature extraction, visual compression, LCEM prefill, and autoregressive token generation. A single aggregate number hides the actual bottleneck and leads to poorly targeted optimization.

Latency measurements must include warm runs, synchronization on asynchronous accelerators, fixed hardware metadata, and percentile statistics. Mean latency alone is insufficient for real-time systems because occasional long frames or generations can violate the service budget.

### Why memory usage matters

Training memory determines achievable batch size and whether mixed precision or gradient accumulation is required. Inference memory determines the minimum viable device. Peak memory must be measured, not inferred only from parameter count, because intermediate MobileCLIP feature maps, the LCEM attention sequence, cached features, and generated-token tensors contribute materially.

Memory regressions should be treated as architecture regressions. Every change that affects frame count, image size, visual slots, decoder width, layer count, or generation budget must report its effect on peak training and inference memory.

### Why edge deployment matters

Edge compatibility imposes practical requirements: predictable operations, bounded sequence lengths, modest parameter counts, limited dynamic control flow, and export-friendly components. The immediate Python implementation may use a generation state machine, but core tensor paths should avoid unnecessary Python-side per-token or per-element loops when a vectorized equivalent is straightforward. Future ONNX or device-runtime export will be easier if tensor shapes and phase transitions are explicit.

No improvement may silently assume a datacenter GPU. CPU execution must remain functional. Accelerator-only optimizations should have a correct CPU fallback. The project should preserve a small default configuration even if optional research configurations are added.

## Current Architecture Review

### Executable architecture

The current default configuration uses 256-by-256 RGB frames, at most eight sampled frames, four compressed visual tokens per frame, and six numeric frame-time embeddings per frame. The resulting visual-temporal prefix contains 80 tokens for an eight-frame video:

```text
8 frames * (4 visual slots + 6 time tokens) = 80 prefix tokens
```

MobileCLIP-S0 is used before global pooling. The adapter calls the image tower's embedding, token, and expansion stages, producing a spatial tensor expected to have shape `[B*T, 1024, 8, 8]`. Flattening produces `[B*T, 64, 1024]`. Learned attention slots compress the 64 spatial positions into four width-192 tokens.

Frame timestamps are serialized in fixed-width form such as `0001.5`. Each character is embedded from the 13-symbol numeric vocabulary. Visual slots and time tokens are interleaved per frame, and frame groups remain chronological.

The LCEM is a pre-normalized decoder-only Transformer with sinusoidal positional encoding, four blocks, six attention heads, width 192, and MLP expansion ratio four. A causal mask prevents access to future positions. A key-padding mask hides padded frame groups and padded text positions. Three heads predict text/synchronization, time symbols, and score symbols.

Generation is constrained by a state machine. Timestamp and score formats are masked according to their current character position. Synchronization transitions from time to score, score to caption, and caption to an event boundary. At the boundary the decoder chooses EOS or the first timestamp symbol of another event.

### Strengths

The visual path preserves spatial information instead of using MobileCLIP's globally pooled embedding. This is important for localized activities and prevents the compression module from receiving a single information bottleneck. The learned slots offer a controlled token budget and an explicit place to trade accuracy against latency.

The frame-time path is discrete and interpretable. Unlike an unconstrained timestamp MLP, fixed-width numeric embeddings align frame metadata with the output numeric representation and preserve exact textual structure.

The decoder separates task heads. Time, score, and caption tokens have different statistical structures; independent heads avoid forcing all outputs through an unnecessarily large unified projection. Numeric format masks prevent many invalid sequences at inference time. Defensive parsing prevents malformed generations from crashing evaluation.

The configuration and checkpoint contract is explicit. The MobileCLIP dependency revision and official checkpoint digest are pinned. Frozen BatchNorm layers remain in evaluation mode. Variable frame counts are padded and masked. JSON video decoding is lazy and supports a persistent frame cache.

### Weaknesses and bottlenecks

Eight uniformly sampled frames can miss short events and provide weak boundary evidence. Raising frame count, however, increases both visual encoder work and LCEM sequence length, so temporal improvement must be implemented as a measured configuration sweep rather than an unconditional jump.

Four compressed slots may discard fine-grained spatial evidence, especially when multiple objects or regions matter in one frame. The current compressor is a single learned-query attention operation followed by an MLP. It has no iterative refinement, temporal interaction, or explicit diversity regularizer. This simplicity is appropriate initially, but its information loss must be measured.

The caption path uses a byte/character-oriented vocabulary and a default maximum of 20 caption tokens. Twenty characters are often insufficient for a useful description. Character decoding also requires more autoregressive steps than subword decoding. Increasing only `max_generated_tokens` will not solve per-caption truncation because `max_caption_tokens` independently masks further caption characters.

The LCEM has zero default dropout and no documented learning-rate scheduler or warmup. The training loop supports gradient clipping, validation, checkpointing, resume, and prediction snapshots, but optimization policy remains basic. The best checkpoint is selected by aggregate validation loss, which may hide degradation in timestamp localization or captions.

Generation supports batch size one because the active head and phase position are global Python state. This is correct but limits throughput. Batched generation requires independent state for every sequence; it must not be implemented by forcing all samples to change heads together.

## Root Cause Analysis

### Insufficient temporal coverage

**Current behavior.** At most eight frames represent the video. Uniform timestamps span the duration, and each selected frame is processed independently by MobileCLIP before LCEM fusion.

**Why it hurts accuracy.** Short events can occur entirely between selected frames. Event boundaries may be represented by only one nearby frame or none. Static appearance can indicate what is present but not reliably when an action begins or ends.

**Training effect.** The decoder receives ambiguous or missing evidence while still being penalized for exact timestamp characters. It can minimize loss by learning duration priors, frequent event positions, or instruction correlations rather than visual temporal grounding.

**Inference effect.** Timestamps become coarse, regress toward common positions, or remain plausible but disconnected from actual transitions. Captions may still look reasonable, hiding the temporal failure.

**Expected improvement.** A controlled increase to 12 or 16 frames should improve temporal recall and boundary evidence. The gain must be verified separately from the additional compute cost.

### Frame sampling behavior

**Current behavior.** Frame timestamps are uniformly spaced from zero to a safe end time. This is deterministic and simple but treats every interval as equally informative.

**Why it hurts accuracy.** Uniform sampling can waste frames on static regions and undersample rapid changes. It also has boundary sensitivity: small duration or decode changes can move a selected frame across an action transition.

**Training effect.** Repeated static evidence can dominate the visual prefix. The model may learn redundant tokens while receiving little motion evidence.

**Inference effect.** Events containing brief motion are disproportionately missed. Results can vary when video metadata or decoding changes slightly.

**Expected improvement.** The immediate phase should improve configurability and determinism, not introduce a complex sampler. Motion-aware sampling belongs in future work after the uniform baseline is thoroughly measured.

### Caption truncation

**Current behavior.** `max_caption_tokens` defaults to 20 and tokens are character-like. Once the limit is reached, generation masks all normal text tokens and forces termination through synchronization.

**Why it hurts accuracy.** Twenty characters may terminate descriptions before the object, action qualifier, or outcome is expressed. Reference captions longer than the budget are structurally impossible to reproduce.

**Training effect.** Training can include longer serialized captions while inference enforces a shorter limit unless serialization also clips consistently. Any mismatch creates a train-inference discrepancy.

**Inference effect.** Captions are incomplete, end mid-word, or become overly terse. Caption metrics and qualitative usefulness suffer.

**Expected improvement.** Raise the default only after measuring caption-length distribution already presented to the model. A first controlled setting is 48 or 64 character tokens, accompanied by a total generation-budget calculation and latency report.

### Total token generation budget

**Current behavior.** Generation stops after `max_generated_tokens=128`, even if the state machine has not emitted EOS. Up to three events may be produced.

**Why it hurts accuracy.** The theoretical event budget includes timestamp characters, separators, synchronization symbols, score characters, caption characters, and a final EOS or next-event transition. Increasing caption allowance without increasing total allowance can simply move truncation to the outer loop.

**Training effect.** There is no direct training effect, but evaluation becomes inconsistent with teacher-forced capacity when valid targets exceed inference limits.

**Inference effect.** Outputs can end without EOS and may be discarded or partially parsed. Long multi-event outputs are most affected.

**Expected improvement.** Derive and validate a minimum safe generation budget from configuration fields. Warn or fail during configuration validation if the total budget cannot represent `max_events` at the chosen phase lengths.

### Decoder capacity

**Current behavior.** The decoder uses width 192, four layers, six heads, and MLP ratio four.

**Why it may hurt accuracy.** The decoder must fuse up to 80 visual-time prefix tokens, interpret instructions, maintain event phase, model numeric boundaries, and produce captions. Capacity may become inadequate after frame or caption budgets increase.

**Training effect.** Under-capacity appears as both training and validation losses plateauing at high values. It should not be confused with regularization or data limitations.

**Inference effect.** Outputs can be grammatically weak, repetitive, or insensitive to visual differences. Numeric and caption tasks may compete for shared hidden representations.

**Expected improvement.** No immediate size increase is justified without evidence. If training loss remains high after optimization and correctness checks, test one dimension at a time: first six layers at width 192, then width 256 with a compatible head count. Measure parameter, latency, and memory changes.

### Character tokenizer limitations

**Current behavior.** Text is represented through a fixed 256-entry character/byte-like vocabulary.

**Why it hurts accuracy and efficiency.** Words require many decoding steps, semantic sharing across related words is weak, and captions consume a large fraction of the autoregressive budget.

**Training effect.** Long effective sequences reduce examples processed per unit compute. Learning lexical semantics from individual characters is harder for a small decoder.

**Inference effect.** Generation is slower and more exposed to spelling errors. The benefit is simple deterministic coverage and no unknown-word problem.

**Expected improvement.** Tokenizer migration may eventually improve sequence efficiency, but it changes vocabulary, checkpoints, serialization, parsing assumptions, and generation lengths. It is postponed until the character baseline is optimized and measured.

### Optimization policy

**Current behavior.** AdamW, constant learning rate, weight decay, and gradient clipping are available. There is no warmup, decay schedule, automatic mixed precision, early stopping, or parameter-group learning-rate policy.

**Why it hurts accuracy.** A constant initial learning rate can destabilize randomly initialized task modules. A constant final learning rate can prevent convergence. One learning rate is unsuitable when partially fine-tuning pretrained MobileCLIP and training new decoder layers simultaneously.

**Training effect.** Loss can spike early, oscillate late, or overwrite useful pretrained features. Compute may be wasted after validation stops improving.

**Inference effect.** Inference code is unchanged, but the selected weights are worse or less stable across seeds.

**Expected improvement.** Linear warmup, cosine decay, parameter groups, mixed precision where safe, and early stopping should make training more stable and reproducible.

### Overfitting and regularization

**Current behavior.** Decoder dropout defaults to zero. Weight decay is present, but there is no configurable label smoothing, stochastic depth, or phase-specific regularization.

**Why it hurts accuracy.** A small task dataset can be memorized by the decoder even when MobileCLIP is frozen. Aggregate training loss may continue falling while validation quality degrades.

**Training effect.** The train-validation gap grows, captions memorize surface forms, and boundary decisions become poorly calibrated.

**Inference effect.** Predictions are brittle and fail to generalize. Increasing architecture size would worsen this without stronger evidence and regularization.

**Expected improvement.** Start with decoder dropout 0.1 and monitor the gap. Do not stack several regularizers simultaneously; controlled experiments are required to attribute gains.

### Validation and checkpoint selection

**Current behavior.** The trainer records aggregate training and validation loss and selects the best checkpoint by the monitored aggregate loss.

**Why it hurts accuracy assessment.** The loss sums multiple task-head terms. A caption-loss improvement can hide worse timestamps, or a numeric-head improvement can hide degenerate captions.

**Training effect.** Early stopping and best-checkpoint selection can optimize the wrong balance.

**Inference effect.** The reported checkpoint may not be the best model for the declared task or composite metric.

**Expected improvement.** Log decomposed loss terms and task metrics. Select a primary checkpoint criterion in configuration and retain both lowest-loss and best-composite checkpoints when they differ.

### MobileCLIP domain adaptation

**Current behavior.** All MobileCLIP parameters are frozen and its BatchNorm modules remain in evaluation mode.

**Why it may hurt accuracy.** Generic image-text pretraining may not optimally represent the frame appearance and action cues needed for event localization. The final MobileCLIP blocks may benefit from conservative task adaptation.

**Training effect.** The slot compressor and LCEM must adapt around a fixed feature distribution. They cannot correct missing task-specific emphasis inside the encoder.

**Inference effect.** Visual features remain stable and efficient, but potentially less discriminative for the target task.

**Expected improvement.** After a stable frozen baseline, unfreezing only the last eligible encoder stage with a much smaller learning rate may improve visual grounding. Full unfreezing is not the default because it raises memory and catastrophic-forgetting risk.

### Preprocessing correctness

**Current behavior.** Frames are converted to floating RGB in `[0,1]`, resized with bilinear interpolation while preserving aspect ratio, and center-cropped to 256 by 256. The official Apple MobileCLIP v1 S0 transform does not add CLIP mean/std normalization.

**Why errors hurt accuracy.** Stretching aspect ratio, changing channel order, supplying `[0,255]`, or adding an unsupported normalization changes the feature distribution seen by pretrained MobileCLIP.

**Training effect.** Downstream modules attempt to compensate for systematically shifted features, slowing optimization and reducing transfer quality.

**Inference effect.** Features differ across code paths, especially if cached and uncached preprocessing are inconsistent.

**Expected improvement.** Preserve the current verified transform, add reference-equivalence tests, and version cache identities when preprocessing changes.

## Guiding Principles

1. **Preserve the event protocol.** The order `time -> score -> caption` is not a tunable choice. Serialization, loss routing, generation, parsing, and metrics must agree on it. Any change requires round-trip and head-transition tests.

2. **Keep MobileCLIP as the encoder.** Alternative visual backbones are outside the TinyTrace identity. Improvements may alter which MobileCLIP blocks are trainable or how its spatial features are compressed, but may not replace it.

3. **Do not introduce a large language model.** The LCEM is a compact, task-specific causal decoder. Importing a general-purpose LLM would invalidate latency, memory, and edge goals.

4. **Measure before increasing complexity.** A model-size increase is allowed only after correctness, optimization, and representation bottlenecks are isolated. Every increase requires an ablation against the previous configuration.

5. **Optimize before redesigning.** Warmup, schedules, regularization, batching, and generation constraints should be corrected before introducing a more complex compressor or decoder.

6. **Treat efficiency as a first-class metric.** Every report must include parameter count, peak memory, latency, and throughput beside accuracy. Regressions require an explicit decision, not silent acceptance.

7. **Preserve reproducibility.** Save complete configuration, dependency revision, checkpoint digest, seed, hardware description, training history, and evaluation output. Resume must restore optimizer and scheduler state.

8. **Keep modules independently testable.** Sampling, preprocessing, MobileCLIP extraction, compression, time encoding, decoder, generation, parsing, and metrics need narrow interfaces and unit tests.

9. **Do not silently break checkpoints.** Changes to vocabulary, parameter names, tensor shapes, or serialization must increment a checkpoint-format version. Provide a migration path when practical and a clear incompatibility error otherwise.

10. **Prefer bounded behavior.** Frame count, input resolution, visual slots, caption length, event count, and total generation length must have validated limits. Unbounded dynamic sequences conflict with reliable edge inference.

11. **Use the simplest adequate solution.** Complexity must solve a demonstrated failure. Motion-aware sampling, iterative compressors, and tokenizer migration remain future work until the baseline indicates they are needed.

## Engineering Principles

The following principles govern implementation decisions, experiment review, and acceptance into the TinyTrace baseline. They complement the architecture-specific guiding principles above by defining the order in which engineering work must occur.

### Correctness before optimization

An incorrect fast implementation is not an improvement. Serialization, target shifting, label routing, causal masking, padding, generation state transitions, parsing, checkpoint restoration, and metric denominators must be correct before speed or memory work begins. Optimization often changes execution order, precision, batching, caching, or control flow; applying it to an unverified path makes failures harder to isolate.

Correctness must be demonstrated with narrow unit tests and at least one end-to-end invariant. For example, batched generation needs unit tests for independent phase state and an end-to-end comparison showing that each batch member produces the same result it produces when decoded alone. AMP needs finite-gradient checks and a short reference comparison against float32. A claim of correctness based only on decreasing loss is insufficient.

### Optimization before scaling

Warmup, scheduling, stable loss balance, dropout, efficient batching, and correct checkpoint selection must be addressed before increasing frames, decoder width, or layer count. Otherwise additional parameters or input tokens may compensate for a training defect, producing an expensive model whose actual bottleneck remains unresolved.

Optimization work is accepted only when it reports both convergence behavior and final structured metrics. Faster loss reduction without better selected-checkpoint quality is not enough. Conversely, a method that reaches equal quality with fewer optimizer steps or lower peak memory is a valid improvement even if the final accuracy is unchanged.

### Evidence before complexity

Every additional module, trainable stage, dynamic policy, or configuration option creates maintenance and deployment cost. Complexity is justified only by a measured failure and an isolated experiment showing that the proposed mechanism addresses it. A feature-diversity diagnostic should precede a more complex slot compressor. Timestamp error relative to frame spacing should precede temporal modules. Caption truncation statistics should precede tokenizer migration.

The evidence record must name the failure, metric, baseline, proposed independent variable, expected result, measured result, systems cost, and decision. If the evidence does not support the mechanism, the code should not become part of the default path.

### Reproducibility before benchmarking

Benchmark numbers are meaningful only when another run can reconstruct the model, weights, configuration, inputs, seed, checkpoint-selection rule, metrics, and hardware environment. Before comparing architecture variants, checkpoint and run metadata must be complete, validation must be deterministic, and metric code must be tested.

A result that cannot be reproduced must be labeled exploratory. It must not be used to change defaults, support thesis conclusions, or justify additional complexity.

### Benchmark before claiming improvement

An improvement claim requires comparison against a named baseline under controlled conditions. The report must include task-separated quality, structured validity, latency, throughput, and peak memory. “Training loss decreased” is not an architecture improvement claim. “The model looks better” is useful qualitative evidence but not a sufficient decision criterion.

Claims must specify the evaluation scope and uncertainty. Small changes should be repeated across seeds when practical. If only one run is possible, the limitation must be stated and the change should remain experimental unless its effect is large and mechanistically clear.

### Preserve a stable reference path

Experimental options must not erase the last verified baseline. Keep a frozen-MobileCLIP, four-layer, width-192 reference configuration that can be rerun after changes. New features should be disabled by default until their acceptance criteria pass. This provides a known-good route for regression diagnosis and prevents an accumulation of interacting unverified changes.

## Research Methodology

TinyTrace development is research engineering. Every architectural or optimization conclusion must come from a controlled comparison rather than from a sequence of unrelated runs.

### Controlled conditions

For a baseline/candidate comparison, hold the following constant unless one item is the declared independent variable:

- random seed and all seeded random-number generators;
- exact training and validation split identifiers;
- preprocessing, frame timestamps, and cached-feature version;
- model initialization policy and pretrained checkpoint;
- hardware model, device count, precision, and relevant runtime settings;
- optimizer, learning-rate schedule, effective batch size, and training duration;
- validation interval and deterministic prediction subset;
- checkpoint-selection criterion;
- parser and evaluation implementation;
- metric definitions, matching rules, and denominators;
- latency warmup, repetitions, synchronization, and batch size.

If a factor cannot be held constant, record it as a confounder and do not present the comparison as a clean causal conclusion. For example, changing frame count may require a smaller micro-batch due to memory. The effective batch size should then be restored with gradient accumulation; otherwise both representation and optimization changed.

### Seed policy

Use a declared canonical seed for rapid iteration and regression checks. Once a candidate shows a meaningful gain, repeat the baseline and candidate with additional fixed seeds when resources permit. Report each run and aggregate statistics; do not report only the best seed. Seed values belong in configuration and checkpoint metadata.

Determinism has limits across hardware and library kernels. Record deterministic-algorithm settings and known nondeterministic operations. “Same seed” is not a substitute for storing the full run environment.

### Baseline discipline

Every experiment must identify an immutable baseline configuration and checkpoint-selection rule before training starts. A baseline cannot be retroactively replaced because a different historical run scored better. If a bug is discovered, invalidate affected comparisons, fix the bug, and establish a new versioned baseline.

The experiment record should include a concise hypothesis in advance. Example: “Increasing sampled frames from 8 to 12 will reduce median timestamp boundary error because maximum uniform sampling interval decreases, at an expected cost of approximately 50% more MobileCLIP frame work.” The expected direction and cost make the outcome interpretable even when the hypothesis is rejected.

### Identical evaluation

Baseline and candidate checkpoints must be evaluated by the same command and code revision whenever possible. Metric changes require reevaluating both models. Checkpoint selection must use the same criterion; selecting one model by lowest loss and another by best timestamp metric invalidates a direct comparison.

Quality and systems evaluation should be separated operationally but linked by run identifier. Systems measurements should use the same frame count and generation limits declared by the model configuration. Report actual generated lengths because autoregressive latency can differ even under the same maximum.

### Interpretation and decision

Classify outcomes as accepted, rejected, inconclusive, or follow-up required. “Accepted” means predefined quality and efficiency criteria pass. “Rejected” means the hypothesis failed or cost exceeded limits. “Inconclusive” means noise or a confounder prevents attribution. “Follow-up required” is reserved for a result that supports the mechanism but needs a second isolated test.

Do not merge several inconclusive changes and hope their combination works. Resolve each variable independently first.

## Experiment Design & Ablation Strategy

### Single-independent-variable rule

TinyTrace must never introduce multiple architectural or optimization changes in one primary ablation. Each experiment changes exactly one independent variable while preserving every controllable condition described in the research methodology.

This rule is critical because model components interact. Increasing frames and decoder size together may improve timestamps, but the experiment cannot determine whether the gain came from temporal evidence, capacity, or their interaction. Adding warmup, cosine decay, dropout, and a new learning rate in one run can produce a better curve while revealing nothing about which change mattered. Unattributed gains cannot support a reliable design, cannot guide rollback, and cannot predict behavior on edge hardware.

Interactions may be studied only after individual effects are known. A factorial or interaction experiment should be explicitly labeled and compare all necessary combinations, including the unchanged baseline and each isolated main effect.

### Example: frame-count experiment

```text
Experiment ID: FRAME-12

Baseline:
  max_frames = 8

Candidate:
  max_frames = 12

Held constant:
  image size, MobileCLIP weights, compression slots, decoder architecture,
  dropout, optimizer, scheduler, effective batch size, training steps,
  generation limits, seed, split, checkpoint criterion, and metrics

Measure:
  timestamp localization metrics
  score metrics
  caption metrics
  structured validity
  median/p90 latency
  peak training and inference memory
  videos per second and generated tokens per second

Decision:
  accept, reject, inconclusive, or follow-up required according to criteria
```

If 12 frames are accepted, they become the baseline for a separate 12-to-16 comparison. Do not train an 8-frame baseline once and then compare it casually with 12, 16, 24, and 32-frame candidates created under changing code or schedules. Revalidate the active baseline when material infrastructure changes.

### Required ablation sequence

The following variables must be tested independently:

1. **Correctness and infrastructure changes.** These should ideally preserve outputs. Establish parity before using the revised infrastructure for research comparisons.
2. **Warmup.** Compare no warmup with one declared warmup policy while leaving the post-warmup learning-rate behavior constant.
3. **Cosine learning-rate decay.** Compare constant or existing decay against cosine while retaining the accepted warmup and identical peak learning rate.
4. **Dropout.** Compare the baseline rate with one candidate rate; do not change width, schedule, or weight decay concurrently.
5. **Caption budget.** Change only `max_caption_tokens` and the minimally required total generation budget. Treat the total-budget adjustment as a dependent consistency change, not a separate architecture intervention.
6. **Frame count.** Follow the sequential configurable profiles and preserve effective batch size.
7. **Partial MobileCLIP fine-tuning.** Compare the frozen baseline against one declared trainable final stage with separate, fixed optimizer groups.
8. **Decoder depth or width.** Test only after earlier phases. Change depth first while holding width, then width in a separate experiment if needed.

The sequence does not mean every candidate must be adopted. A rejected variable is documented and the active baseline remains unchanged.

### Metric families

Every architectural experiment must report:

- timestamp accuracy, boundary error, and temporal IoU;
- score error and task-relevant score quality;
- caption quality, length, and truncation;
- structured validity, EOS behavior, and parse success;
- total and decomposed validation loss;
- parameter and trainable-parameter counts;
- median and tail latency;
- peak training and inference memory;
- FPS or videos per second under a fixed definition;
- generated tokens per second and actual output length;
- convergence steps and elapsed training time.

An optimization-only experiment may have a primary convergence or efficiency goal, but it still reports final structured quality to prove no regression.

### Complete experiment table template

Use one row per run, not only one row per aggregated experiment.

| Field | Baseline run | Candidate run | Requirement |
|---|---|---|---|
| Experiment ID | | | Unique and immutable |
| Run ID | | | Links artifacts and logs |
| Hypothesis | | | Written before execution |
| Independent variable | Reference value | Candidate value | Exactly one variable |
| Controlled variables | | | List or config-diff proof |
| Source revision | | | Exact commit/worktree state |
| Configuration artifact | | | Saved full configuration |
| Checkpoint initialization | | | Same policy |
| Random seed | | | Same seed per paired comparison |
| Hardware/runtime | | | Identical or confounder declared |
| Precision | | | Identical |
| Effective batch size | | | Identical |
| Training steps/epochs | | | Identical stopping policy |
| Checkpoint criterion | | | Identical |
| Selected checkpoint step | | | Reported, not forced equal |
| Total validation loss | | | Same implementation |
| Time loss | | | Same implementation |
| Score loss | | | Same implementation |
| Caption loss | | | Same implementation |
| Timestamp metrics | | | Include boundary error/tIoU |
| Score metrics | | | Include MAE/RMSE as applicable |
| Caption metrics | | | Include truncation/length |
| Parse success | | | Include failures in denominator |
| Median/p90 latency | | | Same benchmark protocol |
| Peak train memory | | | Same measurement method |
| Peak inference memory | | | Same measurement method |
| Throughput/FPS | | | Fixed batch and frame definition |
| Parameter counts | | | Total and trainable |
| Observed confounders | | | Explicitly documented |
| Result across seeds | | | Individual and aggregate values |
| Decision | | | Accept/reject/inconclusive/follow-up |
| Decision rationale | | | Links criteria to measurements |

### Ablation acceptance

Before running, declare a primary metric, regression guardrails for other task metrics, and systems limits. Do not invent acceptance thresholds after seeing results. A candidate that improves its primary metric but violates latency or memory guardrails is not accepted as the edge baseline; it may remain an optional quality profile.

Small gains near run-to-run variation are inconclusive. Repeat them. Large efficiency regressions require explicit approval even when quality improves. Negative results are retained because they prevent future agents from repeating unsupported changes.

## Multi-Task Loss Design

TinyTrace learns three ordered but statistically different tasks:

```text
time -> score -> caption
```

Timestamp symbols require exact numeric structure and boundary sensitivity. Scores are shorter numeric sequences with their own scale and frequency. Captions are longer, higher-entropy character sequences. Synchronization and event-boundary losses add further transition objectives. Raw loss magnitudes and token counts therefore need not be comparable.

### Weighted objective

At the conceptual level, optimize:

```text
L_total = L_time + lambda_score * L_score + lambda_caption * L_caption
```

Synchronization and boundary terms must also be defined explicitly: either assigned to the phase they terminate or exposed as separately weighted components. The implementation must document the chosen decomposition. Do not hide several terms under an ambiguous aggregate name.

This equation does not prescribe lambda values. Weight selection is an empirical research problem. Fixed values should never be copied from an unrelated model without examining TinyTrace's component scales, token counts, gradients, and target metrics.

### Why balancing matters

The task producing the most tokens or largest gradients can dominate shared LCEM updates. Character captions are typically longer than score sequences, while the implementation may currently sum independently averaged cross-entropies. Either design can dominate for different reasons: token-summed loss favors frequent phases, while an unweighted sum of independently averaged terms gives a short auxiliary term equal top-level influence regardless of token count.

Domination can make aggregate loss look healthy while one task stalls. A falling caption loss may obscure unchanged timestamp accuracy. Excessive time weighting may improve numeric formatting while degrading semantic captions because shared representations specialize too strongly.

### Required instrumentation

Return every component as a named tensor before aggregation. Log raw component loss, weighted contribution, valid target count, and per-component accuracy where meaningful. Periodically measure gradient norms contributed by each primary task to shared decoder parameters. This can be done on a diagnostic batch with separate backward passes or an equivalent gradient-analysis utility; it need not run every training step.

Track task metrics beside losses. Low time-token loss does not guarantee good event boundaries, and low character loss does not guarantee useful complete captions.

### Detecting domination

Indicators include:

- one weighted term contributes most of `L_total` throughout training;
- shared-parameter gradient norm from one task is orders of magnitude larger;
- aggregate loss improves while another component and its task metric remain flat;
- changing a task weight changes unrelated task quality sharply;
- one head becomes overconfident while event transitions or other heads degrade;
- best aggregate-loss checkpoint differs substantially from best structured-metric checkpoint.

Magnitude alone does not prove harmful domination. A difficult task may legitimately contribute more. Diagnose through gradients and task outcomes, not loss scale only.

### Weight-tuning methodology

First train with the current documented aggregation and collect component losses, target counts, gradient diagnostics, and task metrics. Form a hypothesis about which task is under- or overrepresented. Then change exactly one task weight in an isolated experiment. Keep optimizer, schedule, seed, architecture, frame count, and checkpoint selection fixed.

Evaluate multiple seeds for small effects. Select weights by a predefined composite or primary task objective with regression guardrails, not by minimum training loss. Record raw and weighted losses so future changes remain interpretable.

Dynamic weighting methods are not an immediate recommendation. They introduce additional state and failure modes. Consider them only if careful fixed-weight ablations cannot produce stable balance and the diagnostics demonstrate persistent gradient conflict.

### Failure cases and debugging

- **Timestamp metrics stall while captions improve:** inspect timestamp target counts, time-head gradients, frame evidence, and weighted time contribution before increasing its weight.
- **Captions become generic after increasing numeric emphasis:** compare shared-layer gradient direction and restore the previous weight as a control.
- **Score loss is noisy:** verify score formatting, range, label offsets, and sample count before reweighting.
- **Total loss changes discontinuously after weighting:** confirm logged values distinguish raw and weighted components and that resume restores weights.
- **A task weight appears to help only one seed:** classify the result as inconclusive rather than changing the default.
- **Transition validity degrades:** inspect sync and boundary terms separately; primary task weights may not address transition supervision.

### Verification and acceptance

Unit tests must construct controlled logits and prove that each component uses only its intended positions, weighting changes only the total aggregation, ignored/padded targets contribute nothing, and gradients reach the expected head and shared decoder.

An accepted loss design must provide stable finite components, prevent a single task from unintentionally overwhelming shared gradients, improve the declared task balance across validation metrics, and preserve structured validity. The chosen formulation and weights must be stored in configuration and checkpoints.

## Implementation Priority

The roadmap below is ordered by dependency. A later priority may be prototyped for investigation, but it must not replace the baseline before prerequisites pass.

1. **Correctness.** Verify serialization, preprocessing, shape contracts, masks, loss routing, generation state, parsing, metrics, and resume behavior. All later measurements depend on these being correct; a defect here invalidates every experiment built on it.

2. **Optimization.** Add observable loss components, optimizer groups, warmup, cosine scheduling, and correct gradient handling. Optimization must be stable before interpreting representational or capacity bottlenecks.

3. **Training stability.** Establish dropout policy, AMP parity, accumulation correctness, early stopping, deterministic validation, complete checkpoints, and run logging. This creates repeatable experiments and prevents noise from being mistaken for architecture gain.

4. **Representation improvements.** Run controlled frame-count and caption-budget ablations, preserving the MobileCLIP and decoder baseline. Better evidence and representable outputs should be pursued before adding parameters.

5. **Partial MobileCLIP fine-tuning.** Adapt only the final declared encoder stage using conservative parameter groups after the frozen baseline converges reliably. This tests whether representation specialization is the remaining limitation.

6. **Model capacity increase.** Increase decoder depth or width only if optimized training, improved temporal representation, validation, and partial fine-tuning still show under-capacity. Capacity is the final near-term optimization because it permanently raises inference cost and can hide upstream defects.

7. **Future research.** Distillation, motion-aware selection, tokenizer migration, quantization, pruning, export runtimes, and advanced compression follow only after TinyTrace V2 has a stable accuracy-efficiency baseline. These items require their own research plans and must not be bundled into V2 correctness work.

## Phase 1 — Video Representation Improvements

### Objective

Improve temporal coverage while preserving MobileCLIP compatibility and keeping compute growth explicit. The immediate work is configurable uniform sampling, preprocessing equivalence, cache correctness, and frame-count ablation. It is not a redesign of the visual encoder.

### Frame-count strategy

Eight frames should remain the low-cost reference until measurements exist, but it must not be treated as permanently optimal. The implementation must support a sequential family of configurable profiles:

```text
8 -> 12 -> 16 -> 24 -> 32 frames
```

This sequence is an experimental ladder, not a recommendation to adopt 32 frames. Compare 8 against 12 first. Only if 12 provides a defensible quality-efficiency tradeoff should it become the reference for 16. Apply the same rule before 24 and 32. Do not jump immediately to 48 frames or introduce arbitrary high-frame configurations: MobileCLIP work grows approximately linearly with frame count, and the LCEM prefix grows by ten tokens per frame under the current four-slot/six-time-token contract.

The baseline prefix lengths make the systems tradeoff concrete:

| Frames | Visual-time prefix tokens | Relative decoded-frame work versus 8 |
|---:|---:|---:|
| 8 | 80 | 1.0x |
| 12 | 120 | 1.5x |
| 16 | 160 | 2.0x |
| 24 | 240 | 3.0x |
| 32 | 320 | 4.0x |

The frame-work column is a first-order estimate, not a latency prediction. Decoder attention cost also grows with total sequence length, memory behavior can be nonlinear, and batching may change. Measure rather than extrapolate final performance.

The implementation should expose named experiment configurations rather than editing source defaults. Each profile must differ only in frame count during its primary ablation. For each profile, record:

- number of decoded frames;
- number of MobileCLIP calls or flattened frame batch size;
- visual-temporal prefix length;
- end-to-end latency;
- peak inference and training memory;
- timestamp, score, and caption metrics.

The chosen default must come from this ablation, not from the assumption that more frames are always better. If 24 frames provide negligible timestamp gain over 16 while materially increasing latency or memory, retain 16. If 12 is the best edge profile and 24 is useful for a quality-oriented profile, preserve both as explicitly named configurations rather than silently moving the global default.

The sampler must generate monotonically nondecreasing timestamps within the decodable video interval. Exact duplicate timestamps should be avoided when duration permits. Videos too short to provide distinct requested timestamps must have a documented policy: either allow duplicates with a warning/metadata flag or reduce the valid frame count and pad through the existing mask. Reducing and masking is preferable because duplicate frames masquerade as additional evidence.

### Temporal coverage verification

Add sampler unit tests using synthetic durations, including zero or invalid duration, very short duration, ordinary duration, and duration near a floating-point boundary. Tests should assert first/last safe timestamp behavior, monotonic order, valid count, and deterministic output.

Add an analysis utility that accepts already available evaluation inputs and reports the temporal spacing implied by each configured frame count. It must not perform acquisition or preparation. Its purpose is to expose theoretical resolution and connect timestamp error to frame spacing.

### MobileCLIP preprocessing

The authoritative path is RGB float tensors in `[0,1]`, aspect-preserving bilinear resize, and center crop to the configured square resolution. Do not add OpenAI CLIP mean/std normalization without direct evidence from the pinned Apple implementation. Do not stretch rectangular frames into a square. Do not mix PIL and tensor interpolation paths without equivalence tests.

Implement a preprocessing contract test that compares representative rectangular tensors against the pinned official transform within a documented numerical tolerance. Include constant images, horizontal and vertical gradients, and non-square dimensions. Verify dtype, range, channel order, output shape, and deterministic results.

The frame cache identity must incorporate a preprocessing-version string, not only path, file metadata, frame count, and image size. Otherwise a future transform correction can accidentally reuse stale tensors. Introduce a constant such as `FRAME_CACHE_FORMAT_VERSION = 2`, include it in the cache hash, and store it in the cache payload for inspection. Loading must validate expected keys, tensor ranks, dtype, frame count, and image dimensions. Invalid cache entries should be ignored and regenerated safely.

### Visual feature verification

For MobileCLIP-S0 at 256 square input, assert `[B*T, 64, 1024]` patch features and `[B, T, 4, d_model]` compressed tokens under the baseline. Tests should fail clearly if a dependency update changes spatial shape. Do not silently reshape an unexpected feature map.

Measure feature diversity across frames and slots. A diagnostic should report mean cosine similarity between frame-level patch summaries and between compressed slots. Very high similarity across all slots can indicate compressor collapse. This is diagnostic only in Phase 1; do not add a diversity loss until collapse is demonstrated.

### Configuration changes

Add explicit validation to `TinyTraceConfig`. At minimum:

- `max_frames >= 1`;
- `image_size > 0`;
- `compressed_visual_tokens >= 1`;
- `time_tokens_per_frame` matches the numeric formatter;
- estimated prefix plus maximum text length does not exceed positional capacity;
- requested frame count and generation budget are bounded by declared safety limits.

Configuration validation must run after JSON loading and checkpoint restoration. Invalid settings should fail before video decoding or model allocation.

### Expected impact and risks

Increasing frames should primarily improve timestamp quality and recall of brief actions. Caption gains may occur when previously missed visual evidence becomes visible. Risks include linear MobileCLIP latency growth, longer LCEM prefixes, higher activation memory, and diminishing returns from redundant frames.

### Acceptance criteria

- Frame profiles 8, 12, 16, 24, and 32 run without source-code changes.
- Sampling is deterministic, monotonic, duration-safe, and unit-tested.
- Official preprocessing equivalence is tested.
- Cache format is versioned and validates payloads.
- Shape tests cover every supported frame profile.
- An ablation reports quality, latency, and memory for each profile.
- The selected default is justified by a recorded quality-efficiency tradeoff.

### Things not to do

- Do not replace MobileCLIP.
- Do not add mean/std normalization that the pinned official transform does not use.
- Do not globally increase image resolution before frame-count effects are understood.
- Do not introduce optical flow or a second video backbone in this phase.
- Do not claim more frames are better without timestamp and systems measurements.

## Phase 2 — Decoder Improvements

### Immediate decoder work

Keep width 192 and four layers for the first improved baseline. Add configuration validation, decomposed loss logging, generation-budget validation, and per-sequence batched generation state before scaling capacity.

The current generation state uses Python strings and scalar counters. Replace it with per-sequence tensors or a small explicit state structure containing active mode, phase-token count, event count, finished flag, and current sequence length. At every decoding step, route each unfinished sequence to the correct head independently. Finished sequences should append padding while other sequences continue. Stop when all sequences finish or reach the maximum budget.

Tests must include a batch where one sample finishes at the first boundary while another starts a second event. Another test must force samples into different modes on the same iteration. Do not implement batching by synchronizing transitions across the batch.

### Caption and generation budgets

Increase caption capacity through a controlled configuration, initially testing 20, 48, and 64 character tokens. Ensure serialization and inference apply compatible truncation rules. Prefer rejecting or explicitly marking a target that exceeds the training contract rather than silently training one length and decoding another.

Add a function that computes a conservative required generation budget:

```text
per-event timestamp symbols
+ timestamp synchronization
+ score symbols
+ score synchronization
+ maximum caption tokens
+ caption synchronization
+ boundary decision allowance
```

Multiply by `max_events` and include final EOS. Validate `max_generated_tokens` against this estimate. The function must derive numeric phase lengths from configuration rather than hard-code current widths.

### Capacity scaling policy

Decoder capacity is the final near-term optimization, not the first. Do not increase depth, width, head count, or MLP ratio until all of the following have been completed and measured:

1. correctness and target-alignment tests pass;
2. warmup, scheduling, gradient handling, and loss balance are stable;
3. frame representation and caption-budget experiments are complete;
4. validation and checkpoint selection use decomposed metrics;
5. the frozen MobileCLIP baseline is converged;
6. conservative partial MobileCLIP fine-tuning has been evaluated;
7. evidence still indicates decoder under-capacity.

Increasing parameters too early can hide other bottlenecks. A larger decoder may memorize dataset priors when temporal evidence is missing, compensate for poor optimization, or reduce training loss while leaving timestamp grounding unchanged. It also establishes a higher latency and memory baseline that later engineers may mistake for necessary complexity.

Only consider capacity changes if the fully optimized model underfits: training loss remains materially high, tiny-set memorization still works, gradients are healthy, input evidence is present, all task losses—not only one—show a capacity-like plateau, and longer scheduled training does not improve it. Validate that the failure is not caption truncation, inadequate frame coverage, loss domination, or frozen-feature mismatch.

The first capacity experiment should add layers while retaining width 192. This increases depth without changing every embedding and projection shape. Compare four and six layers. If both underfit, test width 256 with an evenly divisible head count such as eight in a separate ablation based on the accepted depth baseline. Do not change depth and width in the same experiment, and do not combine decoder scaling with a new frame profile or fine-tuning policy.

For every candidate report total/trainable parameters, LCEM-only latency, end-to-end latency, peak memory, validation metrics, convergence behavior, and at least three seeds when resources permit. A larger decoder should only replace the baseline if gains are repeatable, specifically address the diagnosed under-capacity, and fit the declared deployment budget. Otherwise retain the smaller model even if the larger one achieves a marginal peak score.

### Tokenizer strategy

The character tokenizer remains the immediate baseline because it is deterministic, compact in vocabulary size, and handles arbitrary text without unknown tokens. Its disadvantages are long sequences and weak word-level semantics.

BPE or SentencePiece migration is postponed. It would alter text embeddings, output heads, checkpoint compatibility, serialized sequence lengths, caption limits, parser assumptions, and metric normalization. Before migration, instrument actual characters per caption, generated steps per caption, spelling-error patterns, and percentage of inference latency spent in caption decoding. Only proceed later if character decoding is a measured bottleneck.

### Acceptance criteria

- Generation supports independent batched state transitions.
- Batch-one outputs remain unchanged under deterministic test logits.
- Caption and total generation budgets are validated for consistency.
- Truncation is explicit and counted in evaluation reports.
- Loss is logged independently for time, score, caption, synchronization, and boundary terms.
- Any capacity increase has an isolated ablation and systems report.

## Phase 3 — MobileCLIP Fine-Tuning Strategy

### Why freezing first is correct

The frozen encoder establishes a stable feature distribution, minimizes training memory, preserves pretrained knowledge, and isolates whether the compressor and LCEM can learn the task. It is the correct Stage 1 baseline and must remain available even after fine-tuning support is added.

### Catastrophic forgetting risk

MobileCLIP was pretrained on broad image-text signals. Applying the decoder's learning rate to the full image tower can rapidly distort useful features, especially when downstream supervision is narrower. BatchNorm statistics can also drift if training mode is enabled indiscriminately. Forgetting may appear as lower training loss but worse validation and poor general visual discrimination.

### Stage 1: frozen visual encoder

Freeze every MobileCLIP parameter and keep all BatchNorm layers in evaluation mode. Train slot compression, time and score embeddings, text embeddings as configured, LCEM blocks, and task heads. Use this stage until validation no longer improves under the scheduled learning rate.

Record the best Stage 1 checkpoint and optimizer-independent model state. Stage 2 must initialize from that checkpoint, not from a separately randomized model.

### Stage 2: conservative partial unfreezing

Identify the last semantically meaningful MobileCLIP-S0 stage through named modules rather than brittle numeric parameter slicing. Add a configuration field listing or selecting the trainable stage. Initially unfreeze only the last encoder block or stage and, if necessary, its final expansion layer. Keep early embeddings and early backbone blocks frozen.

Use separate optimizer groups. A reasonable starting ratio is:

```text
new task modules: 1e-4 to 3e-4
LCEM during joint tuning: 5e-5 to 1e-4
unfrozen MobileCLIP stage: 1e-6 to 1e-5
```

Exact values require validation. The MobileCLIP group should have its own configurable weight decay. BatchNorm should remain in evaluation mode unless a specific experiment demonstrates that updating statistics is safe and beneficial.

### Verification

At stage construction, log every trainable module and parameter count. Add assertions that intended frozen parameters have no gradients and intended trainable parameters receive finite gradients after a backward pass. Save fine-tuning policy in the checkpoint configuration.

Compare Stage 2 against the best Stage 1 checkpoint using identical evaluation code. Monitor feature drift by computing cosine similarity between frozen-baseline and fine-tuned features for a fixed diagnostic batch. Large abrupt drift is a warning sign.

### Acceptance criteria

- Stage 1 and Stage 2 are independently selectable and resumable.
- Trainable parameter groups are explicit and logged.
- Scheduler and optimizer states restore correctly for both stages.
- Frozen parameters remain unchanged byte-for-byte during Stage 1.
- Only declared final MobileCLIP modules update during Stage 2.
- Stage 2 improves a declared validation metric without unacceptable latency or memory regression.
- Full-encoder unfreezing is not enabled by default.

## Phase 4 — Training Improvements

### Scheduler and warmup

Implement linear warmup followed by cosine decay. Define warmup in optimizer steps, not epochs, so behavior remains stable when batch size changes. A starting range is 3% to 10% of total steps, with a minimum that avoids an abrupt first update. The scheduler must step after a successful optimizer update and must not advance on skipped mixed-precision steps.

Store scheduler type, total planned steps, warmup steps, current scheduler state, and current global optimizer step in checkpoints. On resume, verify that the requested run is compatible with the restored schedule. If total epochs are extended, use an explicit documented schedule-extension policy rather than silently reconstructing a different curve.

### Parameter groups

Build optimizer groups by module responsibility: compression, embeddings, LCEM, task heads, and optionally unfrozen MobileCLIP. Log group name, learning rate, weight decay, and parameter count. Exclude biases and normalization scale parameters from weight decay unless an experiment specifies otherwise. Detect duplicate or unassigned trainable parameters and fail early.

### Gradient clipping and diagnostics

Retain configurable global-norm clipping, initially 1.0. Log the pre-clip norm at a reasonable interval. Persistent clipping indicates a learning-rate or loss-scale problem and should not be hidden. Log non-finite gradients and abort with checkpointed diagnostics instead of continuing corrupted training.

### Dropout

Test decoder dropout 0.1 against the current zero-dropout baseline. Apply dropout consistently in attention, MLP, and input embedding paths as configured. Do not add multiple new regularizers in the same ablation. If validation improves while training loss rises modestly, the tradeoff is expected. If both degrade, restore zero or test 0.05.

### Mixed precision

Add automatic mixed precision for supported accelerators. Prefer `bfloat16` where hardware supports it; otherwise use `float16` with gradient scaling. Keep loss accumulation and sensitive reductions in float32. CPU mode must remain correct without AMP.

Verify numerical parity on a fixed small batch: compare finite loss, output shapes, and short optimization behavior between float32 and AMP. Exact equality is not expected, but divergence or NaNs are unacceptable. Report speed and memory changes before making AMP the default for a hardware profile.

### Gradient accumulation

Support accumulation to reach a target effective batch size when memory is limited. Divide loss by accumulation steps before backward. Clip gradients and step optimizer/scheduler only at accumulation boundaries. Handle the final partial accumulation window explicitly. Log both micro-step and optimizer-step counters.

### Early stopping

Implement configurable early stopping after a minimum epoch/step threshold. Monitor a declared validation criterion with a `min_delta` and patience. Save the best checkpoint independently of early stopping state. Early stopping must not trigger during warmup or before the minimum training duration.

The default monitored quantity may initially be validation loss, but the framework must allow a composite task metric after Phase 6. Record the stopping reason and best step in the run summary.

### Checkpointing

Checkpoints must include:

- checkpoint format version;
- model state;
- optimizer state;
- scheduler state;
- gradient scaler state when used;
- complete model and training configuration;
- epoch, micro-step, optimizer step, and accumulation state;
- best monitored value and early-stopping state;
- random states for Python and PyTorch, including accelerator RNG where available;
- dependency and source revision metadata when available;
- metric history or a reference to the canonical history artifact.

Use atomic writes. Keep `latest`, `best-loss`, and `best-primary-metric` roles distinct. Periodic checkpoints should be bounded by a retention policy to prevent uncontrolled disk use. Never delete a checkpoint required by an active resume or reported result.

### Validation loop

Validation must run under `model.eval()` and `torch.no_grad()` or inference mode. Frozen MobileCLIP behavior must remain correct after returning to training. Log aggregate and decomposed loss. Run structured generation on a deterministic subset at configured intervals and save raw token IDs, parsed events, ground truth, truncation flags, generation length, and parser warnings.

Validation data order must be deterministic. Evaluation code should not rely on random fallback frames. Cached and uncached paths must produce equivalent tensors.

### Logging

At minimum, log:

- train and validation total loss;
- each task loss component;
- learning rate for each optimizer group;
- gradient norm and clipping frequency;
- epoch, optimizer step, examples processed, and elapsed time;
- tokens and frames processed per second;
- peak memory where supported;
- generation truncation and parse-failure rates;
- current best checkpoint criteria.

Write machine-readable JSON or JSON Lines in addition to console output. Logging failures should be surfaced but should not corrupt checkpoints.

### Hyperparameter philosophy

Change one conceptual factor at a time. Begin with the existing width and depth, then add scheduler/warmup, then dropout, then AMP, and only later partial MobileCLIP tuning. Use configuration files named by experiment purpose. Do not overwrite baseline configurations with unverified values.

Report seeds and variability. A single favorable run is insufficient for a permanent default when the gain is small. Prefer robust settings that work across several runs over fragile peak results.

### Common mistakes

- Advancing the scheduler per batch when gradient accumulation means no optimizer step occurred.
- Loading model weights on resume but reinitializing optimizer, scaler, or scheduler.
- Applying the decoder learning rate to MobileCLIP.
- Updating frozen BatchNorm statistics by calling `train()` without the existing override.
- Computing validation with dropout enabled.
- Selecting a checkpoint on validation examples that are also used for hyperparameter decisions without recording that fact.
- Summing task losses without logging their individual scales.
- Treating AMP overflow-skipped steps as completed optimizer steps.

### Acceptance criteria

- Warmup and cosine decay are unit-tested at boundary steps.
- Resume reproduces the next learning rate and optimizer step.
- AMP and float32 both complete a smoke run with finite loss.
- Gradient accumulation matches a non-accumulated reference within tolerance.
- Early stopping behavior is deterministic in a scripted metric test.
- Checkpoints restore all declared state.
- Training emits complete machine-readable logs and prediction artifacts.

## Phase 5 — Validation Strategy

### Synthetic overfit gate

Retain the existing four-to-eight-sample deterministic overfit gate. It validates serialization, head routing, causal alignment, optimizer updates, generation transitions, and parser round trips. Require exact decoded matches, not only low teacher-forced loss. Run this gate after changes to tokenization, loss alignment, decoder masks, generation, or checkpoint restoration.

If the model cannot overfit, stop. Do not compensate with a larger decoder. Inspect target shifting, label types, padding masks, frozen features, optimizer membership, learning rate, and generation constraints.

### Small real-input memorization gate

Using only already available, manually managed inputs, verify that a tiny fixed set can be memorized. This differs from synthetic validation because it exercises decoding, preprocessing, caching, MobileCLIP features, and realistic captions. The specification does not prescribe acquisition or preparation.

Success requires loss reduction and qualitative/event-level correctness. If synthetic overfit passes but this gate fails, investigate visual feature separability, preprocessing, temporal sampling, and target representability before decoder capacity.

### General validation

Track training and validation curves for total and task-specific losses. Save deterministic predictions throughout training so regressions can be inspected chronologically. Include examples with zero/one/multiple parsed events where supported, short and maximum-length captions, padded frame sequences, and boundary decisions.

Define stop-and-debug conditions:

- any NaN or infinite loss;
- parser failure rate increases sharply;
- all predictions collapse to EOS or maximum-length output;
- generated timestamps violate formatting despite masks;
- train loss does not decrease during the overfit gate;
- validation metrics change but saved predictions do not correspond to the reported checkpoint;
- cached and uncached features differ beyond tolerance.

### Acceptance criteria

- Synthetic exact-match overfit gate passes after relevant changes.
- A tiny real-input memorization check passes before broad experiments.
- Validation is deterministic from the same checkpoint and seed.
- Prediction artifacts contain raw and parsed forms plus failure metadata.
- Task-loss curves and structured metrics are available for checkpoint decisions.

## Phase 6 — Evaluation Improvements

### Timestamp evaluation

Report event timestamp quality independently. At minimum compute temporal intersection-over-union for matched events, recall/precision at declared tIoU thresholds, and start/end absolute error. Document the event-matching algorithm and behavior when event counts differ. Numeric parse failures count as failures, not missing observations silently excluded from denominators.

Also report timestamp error relative to sampling interval. This helps determine whether temporal error is limited by eight-frame coverage or by decoder prediction.

### Score evaluation

Report mean absolute error and root mean squared error for matched event scores. If scores are ordinal or thresholded for highlight metrics, report the relevant rank or threshold metrics separately. Clamp only if the task contract requires it; otherwise out-of-range generations should be counted and diagnosed.

### Caption evaluation

Use established caption metrics only with documented text normalization. Also report exact match where meaningful, average generated length, truncation rate, repetition indicators, and qualitative samples. Since character decoding can end mid-word, count forced maximum-caption termination separately from normal synchronization.

### Structured event evaluation

Report parser success, complete-event rate, event-count error, EOS rate, maximum-generation-budget rate, and head-transition validity. These engineering metrics explain why task accuracy changes.

### Systems benchmarking

Benchmark the following stages separately and end to end:

1. frame decode and preprocessing;
2. MobileCLIP feature extraction;
3. slot compression and LCEM prefill;
4. autoregressive generation;
5. parsing;
6. complete video-to-events latency.

Record hardware, operating system, Python and PyTorch versions, device precision, thread count, batch size, frame count, image size, output length, and warmup procedure. Accelerator timings must synchronize before and after measured regions.

Report median, p90, and p99 latency where enough repetitions exist; videos or frames per second; generated tokens per second; parameter count; checkpoint size; peak resident CPU memory; peak accelerator allocated/reserved memory; and utilization where reliable tooling is available.

For edge-oriented evaluation, include batch-one CPU latency and memory even if GPU training is used. Accuracy alone is insufficient because a model that misses its latency or memory budget fails the project objective.

### Acceptance criteria

- Timestamp, score, and caption metrics are separate.
- Structured validity and truncation metrics are reported.
- Benchmark results are reproducible from a command and saved configuration.
- Stage-level and end-to-end latency are both available.
- CPU batch-one measurements are included.
- Every architecture ablation includes quality and systems deltas.

## Phase 7 — Debugging Workflow

Use this decision sequence. Do not jump directly to architecture scaling.

```text
Model cannot memorize synthetic samples
  -> verify serialization and next-token shift
  -> verify label-type routing and synchronization targets
  -> verify causal/padding masks
  -> verify optimizer contains trainable parameters
  -> inspect gradients and learning rate
  -> only then inspect decoder capacity

Synthetic memorization passes, real-input memorization fails
  -> compare cached and uncached frames
  -> verify RGB/range/resize/crop
  -> inspect MobileCLIP feature variance
  -> inspect slot diversity
  -> verify target length is representable

Training is unstable or non-finite
  -> inspect raw loss components
  -> inspect gradient norms
  -> reduce learning rate or extend warmup
  -> validate AMP/scaler behavior
  -> verify numeric masks do not create all-negative-infinity rows

Training improves, validation degrades
  -> quantify train-validation gap by task
  -> inspect prediction history
  -> add modest dropout or stronger stopping
  -> avoid increasing decoder capacity

Captions are bad but timestamps are good
  -> inspect caption truncation rate
  -> verify character target alignment
  -> increase caption budget within latency constraints
  -> assess decoder under-capacity
  -> postpone tokenizer migration until measured

Timestamps are bad but captions are plausible
  -> inspect temporal sampling coverage
  -> compare error with frame interval
  -> verify frame-time numeric encoding
  -> inspect time-head loss and generation masks
  -> test controlled frame-count increase

Scores are bad while other outputs are good
  -> inspect score normalization/range contract
  -> verify score label offsets and sync target
  -> inspect per-head loss scale
  -> evaluate score error independently

Outputs terminate immediately
  -> inspect boundary and EOS logits
  -> verify minimum phase/caption masks
  -> inspect synchronization loss and target types

Outputs never terminate
  -> inspect EOS boundary calibration
  -> inspect max-event and max-token logic
  -> measure caption-sync generation
  -> verify boundary loss examples exist

Batched generation differs from batch-one
  -> inspect per-sequence mode and counters
  -> verify finished-sequence masking
  -> compare stepwise logits and appended tokens

Latency regresses
  -> separate frame, encoder, prefill, and decode timing
  -> compare frame count and sequence length
  -> compare generated-token count
  -> inspect accidental gradient tracking or missing inference mode
```

For every debugging session, save the failing configuration, minimal reproducer, observed tensor shapes, relevant log excerpt, and final root cause. Convert fixed bugs into regression tests.

## Things That Should NOT Be Implemented Yet

The following items are intentionally postponed. Their presence in future work is not authorization to implement them in the current baseline. Each has prerequisites that must be demonstrated before an implementation proposal is accepted.

### Tokenizer migration

Do not replace the character tokenizer yet. It would break text embedding and output-head compatibility, change sequence lengths, modify caption-budget semantics, and require new serialization and parser validation. First complete caption-budget ablations, measure the percentage of inference time spent decoding captions, quantify truncation and spelling failures, and establish a checkpoint-versioning plan. Migration is considered only if these measurements show that character tokenization is a material quality or latency bottleneck.

### Motion-aware sampling

Do not implement optical flow, learned frame selection, or heuristic motion sampling yet. Uniform frame-count ablations through the supported profiles must first establish how much error is attributable to temporal sparsity. A proposed motion method then needs an isolated comparison at identical frame count and must include sampler overhead. Without that baseline, any gain cannot be separated from simply processing different or more visual evidence.

### Knowledge distillation

Do not add a teacher model or distillation losses until the supervised TinyTrace optimization and multi-task loss design are stable. Distillation would introduce teacher quality, temperature, alignment, and loss-weight variables, obscuring basic model behavior. Prerequisites are reproducible Stage 1/Stage 2 training, decomposed task metrics, a fixed student baseline, and a teacher-output contract compatible with structured events.

### Quantization

Do not quantize an architecture whose quality-efficiency baseline is still changing. Quantization can obscure numerical bugs and creates hardware-specific accuracy tradeoffs. Prerequisites are a frozen TinyTrace V2 model, reference outputs, operator coverage analysis, target device declaration, and a benchmark that measures actual latency and memory rather than checkpoint size alone.

### Pruning

Do not prune decoder heads, channels, layers, or MobileCLIP parameters before profiling shows where redundant compute exists. Unstructured sparsity may reduce parameter count without reducing wall-clock latency. Prerequisites are a stable trained baseline, layer/head sensitivity analysis, hardware support for the chosen sparsity, and a fine-tuning and regression plan.

### TensorRT

Do not introduce a TensorRT path while the model interface and generation state remain under revision. TensorRT is target-specific and should not drive core architecture decisions. Prerequisites are a stable one-step inference interface, successful ONNX or equivalent graph representation, declared NVIDIA deployment hardware, numerical-parity tests, and a maintained framework fallback.

### ONNX export

Do not prioritize ONNX export before batched generation state, tensor contracts, and supported dynamic axes are stable. Premature export creates adapters for interfaces likely to change. Prerequisites are a versioned inference API, deterministic reference tensors, operator compatibility review, and numerical-parity acceptance thresholds.

### Adaptive slot compression

Do not implement dynamic slot counts, iterative slot attention, or input-dependent token pruning until diagnostics demonstrate that the fixed four-slot compressor is a bottleneck. Adaptive token counts complicate batching, positional behavior, export, and latency predictability. Prerequisites are slot-diversity measurements, fixed-slot ablations, and a defined upper bound that preserves edge behavior.

### Larger decoder

Do not enlarge the decoder during correctness, optimization, stability, frame, or MobileCLIP-adaptation work. A larger model can mask missing evidence and training defects while permanently increasing memory and latency. Prerequisites are completion of Priorities 1–5, persistent underfitting across task losses, stable gradients, successful memorization gates, and an isolated depth-first capacity ablation with predefined efficiency guardrails.

These postponements should be enforced during review. If an agent proposes one of these items, the proposal must link evidence that every prerequisite has passed. Otherwise it belongs in an experiment backlog, not the implementation plan.

## Phase 8 — Future Improvements

The items in this phase are not immediate recommendations. Implement them only after Phases 1–7 produce a stable measured baseline.

### Knowledge distillation

A stronger offline teacher could provide soft targets for event phases or hidden representations while TinyTrace remains the deployed student. Distillation may improve accuracy without inference cost, but it complicates training, requires carefully aligned vocabularies and event structures, and can transfer teacher bias. Consider it only after ordinary supervised optimization is stable.

### Motion-aware frame sampling

Use inexpensive visual-difference or motion scores to allocate frames to changing regions. This could improve short-event recall at the same frame budget. Risks include preprocessing overhead, nondifferentiable selection, instability, and missing semantically important static context. Compare against uniform sampling at identical frame count and include sampler latency.

### Temporal representation improvements

Potential directions include lightweight temporal mixing before slot compression, temporal positional biases, or a small temporal convolution over per-frame summaries. These may improve motion reasoning but add latency and complicate the clean per-frame MobileCLIP path. Introduce only one temporal mechanism at a time after frame-count experiments isolate the limitation.

### Tokenizer migration

BPE or SentencePiece can reduce caption steps and improve lexical modeling. The migration requires a checkpoint-format break or explicit conversion, new serialization tests, parser compatibility review, embedding/head changes, and new caption budgets. It is appropriate only when caption decoding is a measured quality or latency bottleneck.

### Quantization

Post-training dynamic quantization or quantization-aware training may reduce model size and CPU latency. MobileCLIP and LCEM components may have different quantization sensitivity. Start with decoder linear layers and report task-specific degradation. Do not claim success from file-size reduction alone; measure latency on target hardware.

### ONNX export

Exporting feature extraction and LCEM forward paths can improve portability. Autoregressive state-machine export is more difficult and may need a one-step decoder API with external state management. Export should include numerical parity tests and documented dynamic axes.

### TensorRT or device-specific runtimes

TensorRT may improve NVIDIA inference but is not a general edge solution. Pursue it only after ONNX stability and when target hardware is declared. Maintain a framework reference implementation for correctness comparisons.

### Pruning

Structured pruning of LCEM heads, MLP channels, or layers may reduce latency when supported by hardware. Unstructured sparsity often fails to produce wall-clock gains. Pruning requires fine-tuning and task-specific regression measurement.

### Future compressor ideas

Iterative slot attention, diversity regularization, or adaptive token counts could retain more information. These ideas increase complexity or dynamic behavior. Consider them only if diagnostics show slot collapse or a clear compression bottleneck after temporal coverage is fixed.

## Success Criteria for TinyTrace V2

TinyTrace V2 is successful only when it improves the complete research system while preserving the project's lightweight identity. Benchmark accuracy is necessary evidence but is not the sole definition of success.

### Machine-learning success

- Temporal localization improves against the versioned baseline under identical evaluation, or the existing quality is achieved at a meaningfully lower systems cost.
- Score quality does not regress beyond predefined guardrails.
- Caption quality improves or remains stable while truncation and invalid termination are reduced.
- Structured parse success, complete-event rate, and EOS behavior remain reliable.
- Multi-task loss contributions are observable and no task unintentionally dominates shared training.
- Improvements repeat across declared seeds or are explicitly labeled exploratory.

### Optimization success

- Training begins without destructive update spikes and converges under a documented warmup/decay policy.
- Gradients remain finite, clipping frequency is observable, and AMP behavior is verified where enabled.
- Training and validation task losses are stable and interpretable.
- Early stopping and checkpoint selection choose reproducible checkpoints under a declared criterion.
- Stage 1 and any accepted Stage 2 fine-tuning can resume without changing optimizer or scheduler semantics.

### Engineering success

- Configuration validation fails early on incompatible frame, token, decoder, or generation settings.
- Unit, integration, synthetic-overfit, and tiny real-input gates pass.
- Validation is deterministic under the documented environment.
- Checkpoints restore model, optimizer, scheduler, scaler, RNG, stage, and selection state.
- Logs and predictions are machine-readable and traceable to a run and source revision.
- Batched generation preserves per-sample state and batch-one parity.
- Metrics include malformed and truncated outputs in their denominators.

### Efficiency and deployment success

- End-to-end latency, stage latency, throughput, and peak memory are measured on declared hardware.
- No default change exceeds predefined latency or memory limits without an explicit profile-level decision.
- A batch-one CPU path remains functional and benchmarked.
- MobileCLIP remains the visual encoder and the LCEM remains task-specific and lightweight.
- No large language model or unbounded dynamic architecture is introduced.
- The selected default represents a measured quality/latency/memory tradeoff; optional quality profiles are named separately.

### Research success

- Every accepted change has a named baseline and a single-variable ablation.
- Experiment artifacts preserve configuration, seed, split identity, checkpoint rule, metrics, hardware, and source revision.
- Negative and inconclusive results are recorded.
- Claims distinguish quality improvement, efficiency improvement, and tradeoff changes.
- Another engineer can reproduce the selected result from repository artifacts and documented commands.

### TinyTrace V2 release gate

V2 is a **Go** only when all correctness gates pass, the selected configuration meets declared quality and edge-efficiency guardrails, training and evaluation reproduce, and no postponed feature was introduced without satisfying its prerequisites. V2 is a **No-Go** if accuracy rises through an unexplained confounded experiment, latency or memory is unmeasured, checkpoint selection is inconsistent, validation is nondeterministic, or the architecture no longer fits the MobileCLIP-plus-lightweight-LCEM constraint.

## Phase-by-Phase Acceptance and Go/No-Go Criteria

### Phase 1

**Implementation complete:** Configurable uniform frame profiles, sampler validation, official preprocessing-equivalence tests, cache versioning, and shape diagnostics exist.  
**Verification complete:** Frame-count ablation includes timestamp quality, latency, and memory.  
**Testing complete:** Duration edge cases, preprocessing, cache invalidation, and tensor shapes pass.  
**Expected output:** A justified default temporal profile and reproducible report.  
**Go:** At least one profile improves quality within the declared efficiency budget, or the eight-frame baseline is retained with evidence.  
**No-Go:** Preprocessing differs from the pinned official transform, caching is stale, or systems cost is unmeasured.

### Phase 2

**Implementation complete:** Independent batched generation, budget validation, decomposed loss, and controlled caption limits exist.  
**Verification complete:** Batch-one parity and different-mode batch tests pass.  
**Testing complete:** Phase transitions, EOS, multiple events, forced limits, and malformed outputs are covered.  
**Expected output:** Valid structured generation with known truncation behavior and improved throughput.  
**Go:** Structured validity is maintained or improved without unacceptable latency.  
**No-Go:** Batch members share state, checkpoint compatibility changes silently, or truncation remains unreported.

### Phase 3

**Implementation complete:** Frozen and partial-unfreeze stages are selectable, logged, and resumable.  
**Verification complete:** Gradient and parameter-delta audits prove only intended modules update.  
**Testing complete:** Stage transition and optimizer-group restoration pass.  
**Expected output:** Stage comparison with feature drift and task metrics.  
**Go:** Partial tuning produces repeatable gain without catastrophic drift.  
**No-Go:** Full encoder updates accidentally, validation declines, or memory exceeds budget.

### Phase 4

**Implementation complete:** Scheduler, warmup, AMP, accumulation, early stopping, full checkpoint state, and structured logs exist.  
**Verification complete:** Resume and numeric parity smoke tests pass.  
**Testing complete:** Scheduler boundaries, accumulation, scaler restoration, early stopping, and atomic saves are covered.  
**Expected output:** Stable reproducible runs and complete artifacts.  
**Go:** Optimization improves convergence or efficiency without quality regression.  
**No-Go:** Resume changes the trajectory unexpectedly or non-finite failures are hidden.

### Phase 5

**Implementation complete:** Synthetic and tiny real-input gates are automated.  
**Verification complete:** Exact decoded outputs are inspected.  
**Testing complete:** Determinism and stop conditions are covered.  
**Expected output:** Saved evidence for both memorization gates.  
**Go:** Both gates pass.  
**No-Go:** Proceeding to broader training when either gate fails.

### Phase 6

**Implementation complete:** Task-separated metrics and systems benchmark commands exist.  
**Verification complete:** Metrics are checked on hand-constructed examples and timing methodology is documented.  
**Testing complete:** Empty, malformed, perfect, and partially matched predictions are covered.  
**Expected output:** Machine-readable accuracy-efficiency report.  
**Go:** Results are reproducible and denominators include failures.  
**No-Go:** Only aggregate accuracy is reported or benchmark metadata is missing.

### Phase 7

**Implementation complete:** Decision-tree diagnostics are reflected in scripts/tests where practical.  
**Verification complete:** Known failures lead to actionable diagnostics.  
**Testing complete:** Fixed bugs become regression tests.  
**Expected output:** Root-cause records and minimal reproducers.  
**Go:** Failures can be isolated by subsystem.  
**No-Go:** Architecture is scaled to mask an unresolved correctness issue.

## Final Implementation Checklist

### Research methodology and experiment control

- [ ] Assign every experiment an immutable experiment ID.
- [ ] Assign every run a unique artifact-linked run ID.
- [ ] Write the hypothesis before execution.
- [ ] Name the exact baseline configuration.
- [ ] Declare exactly one independent variable.
- [ ] Generate a full baseline/candidate configuration diff.
- [ ] Keep seed fixed for each paired comparison.
- [ ] Keep split identifiers fixed.
- [ ] Keep hardware, precision, and runtime settings fixed.
- [ ] Keep effective batch size fixed.
- [ ] Keep training duration and stopping policy fixed.
- [ ] Keep checkpoint-selection criterion fixed.
- [ ] Keep parser and metric revisions fixed.
- [ ] Declare primary metric before execution.
- [ ] Declare quality regression guardrails before execution.
- [ ] Declare latency and memory guardrails before execution.
- [ ] Record unavoidable confounders.
- [ ] Repeat promising small effects across seeds.
- [ ] Report individual runs rather than only the best seed.
- [ ] Classify the result as accepted, rejected, inconclusive, or follow-up required.
- [ ] Preserve negative and inconclusive experiment records.
- [ ] Run interaction experiments only after isolated main effects are known.

### Foundations and configuration

- [ ] Add a checkpoint format version.
- [ ] Add centralized `TinyTraceConfig.validate()`.
- [ ] Validate configuration after JSON loading.
- [ ] Validate configuration after checkpoint restoration.
- [ ] Validate positive image size and frame count.
- [ ] Validate decoder width is divisible by head count.
- [ ] Validate positional-encoding capacity.
- [ ] Validate caption and total generation budgets.
- [ ] Validate numeric widths against formatters.
- [ ] Preserve unknown-field rejection.
- [ ] Record source/dependency revisions in run metadata.

### Video and preprocessing

- [ ] Extract timestamp selection into a testable sampler function.
- [ ] Support 8-, 12-, 16-, 24-, and 32-frame configuration profiles.
- [ ] Compare adjacent frame profiles sequentially.
- [ ] Preserve effective batch size during frame ablations.
- [ ] Record prefix length for every frame profile.
- [ ] Select the default frame profile from measured quality/latency/memory tradeoffs.
- [ ] Test ordinary-duration uniform sampling.
- [ ] Test very-short-duration sampling.
- [ ] Test invalid and zero duration handling.
- [ ] Test monotonic and bounded timestamps.
- [ ] Define duplicate-timestamp policy.
- [ ] Preserve RGB float `[0,1]` contract.
- [ ] Test rectangular aspect-preserving resize.
- [ ] Test center crop.
- [ ] Compare against pinned official MobileCLIP preprocessing.
- [ ] Add preprocessing/cache format version.
- [ ] Include format version in cache key.
- [ ] Validate cached tensor keys, ranks, shapes, and dtype.
- [ ] Regenerate invalid cache entries atomically.
- [ ] Test concurrent cache writers.

### MobileCLIP and compression

- [ ] Preserve checkpoint SHA-256 verification.
- [ ] Assert baseline MobileCLIP patch shape.
- [ ] Assert compressed slot shape for each frame profile.
- [ ] Verify frozen parameters receive no gradients.
- [ ] Verify frozen BatchNorm stays in evaluation mode.
- [ ] Add feature variance diagnostic.
- [ ] Add compressed-slot cosine-similarity diagnostic.
- [ ] Record MobileCLIP and compressor latency separately.
- [ ] Avoid compressor redesign until diagnostics justify it.

### Decoder and generation

- [ ] Keep `time -> score -> caption` invariant documented in code.
- [ ] Replace global generation mode with per-sequence state.
- [ ] Track per-sequence phase position.
- [ ] Track per-sequence event count.
- [ ] Track per-sequence finished state.
- [ ] Pad finished sequences while others continue.
- [ ] Stop when all sequences finish.
- [ ] Test sequences in different modes simultaneously.
- [ ] Test different event counts in one batch.
- [ ] Test batch-one generation parity.
- [ ] Derive minimum generation budget from configuration.
- [ ] Track forced caption termination.
- [ ] Track maximum-token termination.
- [ ] Track EOS termination.
- [ ] Log parser warnings without crashing.
- [ ] Preserve numeric format masks.
- [ ] Test every timestamp format position.
- [ ] Test every score format position.
- [ ] Test event-boundary EOS versus next timestamp.

### Loss and optimization

- [ ] Return named loss components.
- [ ] Define the documented multi-task aggregation equation.
- [ ] Store all task weights in configuration and checkpoints.
- [ ] Log raw loss components before weighting.
- [ ] Log weighted contribution of every component.
- [ ] Log valid target counts by component.
- [ ] Add diagnostic shared-gradient norms by primary task.
- [ ] Verify task weights change only intended aggregation behavior.
- [ ] Test padded/ignored targets contribute to no task loss.
- [ ] Test each task reaches its intended head and shared decoder.
- [ ] Tune one task weight per isolated experiment.
- [ ] Select task weights using declared validation metrics and guardrails.
- [ ] Log text loss.
- [ ] Log timestamp loss.
- [ ] Log score loss.
- [ ] Log synchronization losses.
- [ ] Log boundary loss.
- [ ] Verify next-token shift alignment.
- [ ] Add named optimizer parameter groups.
- [ ] Detect duplicate optimizer parameters.
- [ ] Detect unassigned trainable parameters.
- [ ] Exclude normalization and biases from weight decay where configured.
- [ ] Implement step-based linear warmup.
- [ ] Implement cosine decay.
- [ ] Test first, warmup-end, and final learning rates.
- [ ] Add gradient-norm logging.
- [ ] Preserve configurable gradient clipping.
- [ ] Abort and diagnose non-finite gradients.
- [ ] Add configurable decoder dropout experiments.
- [ ] Add AMP with bfloat16 preference.
- [ ] Add float16 gradient scaling fallback.
- [ ] Keep CPU float32 fallback.
- [ ] Add gradient accumulation.
- [ ] Handle partial final accumulation window.
- [ ] Step scheduler only with optimizer.
- [ ] Add deterministic early stopping.

### Staged MobileCLIP training

- [ ] Represent training stage in configuration.
- [ ] Keep fully frozen Stage 1 default.
- [ ] Select final MobileCLIP stage by stable module name.
- [ ] Create low-learning-rate encoder parameter group.
- [ ] Log trainable/frozen parameter counts by module.
- [ ] Assert Stage 1 MobileCLIP state is unchanged.
- [ ] Assert Stage 2 updates only declared modules.
- [ ] Keep BatchNorm policy explicit.
- [ ] Initialize Stage 2 from best Stage 1 checkpoint.
- [ ] Save stage and optimizer groups in checkpoints.
- [ ] Measure feature drift after Stage 2.

### Checkpointing and resume

- [ ] Save model state.
- [ ] Save optimizer state.
- [ ] Save scheduler state.
- [ ] Save AMP scaler state.
- [ ] Save complete configuration.
- [ ] Save epoch, micro-step, optimizer step, and global example count.
- [ ] Save early-stopping state.
- [ ] Save best metrics and criteria.
- [ ] Save Python RNG state.
- [ ] Save CPU PyTorch RNG state.
- [ ] Save accelerator RNG state where applicable.
- [ ] Use atomic checkpoint writes.
- [ ] Distinguish latest, best-loss, and best-metric checkpoints.
- [ ] Add bounded periodic-checkpoint retention.
- [ ] Verify resume restores exact next learning rate.
- [ ] Verify resume restores intended trainable modules.
- [ ] Fail clearly on incompatible checkpoint versions.

### Validation

- [ ] Keep synthetic exact-match overfit test automated.
- [ ] Run overfit gate after serialization changes.
- [ ] Run overfit gate after decoder-mask changes.
- [ ] Run overfit gate after loss-routing changes.
- [ ] Run overfit gate after tokenizer changes.
- [ ] Add tiny real-input memorization gate.
- [ ] Make validation ordering deterministic.
- [ ] Disable random-frame fallback for validation.
- [ ] Run validation in evaluation/inference mode.
- [ ] Restore training mode correctly after validation.
- [ ] Save raw generated token IDs.
- [ ] Save parsed event outputs.
- [ ] Save ground truth alongside predictions.
- [ ] Save generation lengths and termination reasons.
- [ ] Save parse errors and warnings.
- [ ] Compare predictions from the exact reported checkpoint.

### Metrics

- [ ] Implement deterministic event matching.
- [ ] Report timestamp tIoU.
- [ ] Report timestamp threshold precision/recall.
- [ ] Report start and end absolute error.
- [ ] Report error relative to frame interval.
- [ ] Report score MAE.
- [ ] Report score RMSE.
- [ ] Report caption metrics with documented normalization.
- [ ] Report generated caption length.
- [ ] Report caption truncation rate.
- [ ] Report complete-event rate.
- [ ] Report event-count error.
- [ ] Report parser success rate.
- [ ] Report EOS termination rate.
- [ ] Report maximum-budget termination rate.
- [ ] Include malformed generations in denominators.
- [ ] Unit-test metrics on hand-constructed cases.

### Systems measurement

- [ ] Benchmark frame decoding/preprocessing.
- [ ] Benchmark MobileCLIP extraction.
- [ ] Benchmark compression and LCEM prefill.
- [ ] Benchmark autoregressive decoding.
- [ ] Benchmark complete video-to-events path.
- [ ] Synchronize accelerator timings.
- [ ] Include warmup runs.
- [ ] Report median latency.
- [ ] Report p90 and p99 latency when applicable.
- [ ] Report frames/videos per second.
- [ ] Report generated tokens per second.
- [ ] Report total and trainable parameters.
- [ ] Report checkpoint size.
- [ ] Report peak CPU memory.
- [ ] Report peak accelerator allocated/reserved memory.
- [ ] Record hardware and software metadata.
- [ ] Include batch-one CPU results.
- [ ] Save benchmark results as machine-readable artifacts.

### Documentation and engineering hygiene

- [ ] Keep architecture contract synchronized with implementation.
- [ ] Document every checkpoint-breaking change.
- [ ] Add migration code or explicit incompatibility errors.
- [ ] Keep experiment configurations separate from defaults.
- [ ] Add type hints to new public functions.
- [ ] Add narrow unit tests for each new module.
- [ ] Convert every fixed regression into a test.
- [ ] Keep generated outputs and large weights ignored by Git.
- [ ] Avoid duplicate serialization or metric logic.
- [ ] Preserve standalone operation without TRACE runtime dependencies.
- [ ] Keep postponed features disabled until their prerequisites pass.
- [ ] Maintain a versioned stable reference configuration.
- [ ] Evaluate TinyTrace V2 against engineering and systems release gates, not accuracy alone.

## Final Notes

TinyTrace should evolve through measured, reversible steps. The immediate path is not a larger model; it is a more observable and better-optimized implementation. Temporal coverage, caption limits, loss decomposition, scheduling, validation, and systems benchmarking must be corrected before architecture scale is reconsidered.

Every implementation change should answer six questions in its pull request or experiment record:

1. What behavior changed?
2. Why was the change necessary?
3. How was it implemented without violating the architecture invariants?
4. What accuracy or reliability impact was expected?
5. What latency, memory, compatibility, or optimization risks were introduced?
6. Which tests and measurements verify the result?

The project succeeds when it produces reliable structured events under a practical edge-oriented compute budget. Accuracy, latency, memory, validity, and reproducibility are jointly required. None should be optimized in isolation.
