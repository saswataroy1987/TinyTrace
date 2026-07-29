# TinyTRACE
## Software Architecture & Design Specification
### A Lightweight Implementation of Causal Event Modeling for Video Temporal Grounding

**Document Class:** Research Architecture Specification (Implementation Blueprint)
**Target Deliverable:** B.Tech Thesis — TinyTRACE
**Status:** FROZEN DESIGN — Repository Audit Recorded (2026-07-12)
**Base Reference:** Guo et al., "TRACE: Temporal Grounding Video LLM via Causal Event Modeling" (arXiv:2410.05643)

---

## Document Control

| Field | Value |
|---|---|
| Project Name | TinyTRACE |
| Document Type | Architecture & Design Specification (ADS) |
| Audience | Implementers (human + Codex agent), Thesis Committee |
| Design State | Frozen (Reverse engineering in progress; sections still marked TODO where unverified) |
| Parent Work | TRACE (Tencent PCG / CUHK-Shenzhen, 2024) |
| Revision Policy | Architecture does not change unless TRACE source code proves a stated assumption incorrect |

---

## How to Read This Document

This specification is organized into seven parts. Parts I–III establish *what* TinyTRACE is and *why* it is designed the way it is. Part IV and V describe *how* the system behaves at training time and inference time, at the level of tensor and token flow, without being a literal code listing. Part VI documents the design rationale and trade-off analysis behind every frozen decision. Part VII records the repository-audit findings and the remaining implementation-open items relative to the TRACE reference implementation.

Every module description in Part III follows an identical template — Purpose, Inputs, Outputs, Pipeline, Mathematics, Why TRACE uses it, Why TinyTRACE keeps/replaces it, Advantages, Limitations, Future Improvements, Assumptions, TODO — so that any reader can locate the same category of information for any module without re-reading prose.

Wherever a detail is not explicitly stated in the TRACE paper (arXiv:2410.05643) and has not yet been verified against the TRACE public repository, this document marks it as:

> **TODO (Verify from TRACE implementation)**

Each such marker should be treated as an open implementation item. No implementation detail should be invented to fill these gaps; unresolved points should remain explicitly marked until verified against the TRACE codebase (`https://github.com/gyxxyg/TRACE`) or otherwise justified.

---

# PART I — INTRODUCTION AND PROJECT FRAMING

## 1.1 Motivation

Video Temporal Grounding (VTG) — the family of tasks that includes dense video captioning, moment retrieval, and video highlight detection — requires a model to reason about *when* things happen in a video, not merely *what* happens. Conventional video-LLM approaches treat this as an undifferentiated natural-language generation problem: the model is asked to emit a paragraph of prose that happens to contain numbers that are timestamps. This is structurally mismatched to the problem, because video is not prose — it is an ordered sequence of discrete events, each with a start/end time, a salience/importance, and a semantic description.

TRACE (Guo et al., 2024) addresses this mismatch by introducing **causal event modeling**: a formal reformulation of VTG as autoregressive generation over a structured sequence of events, where each event is explicitly decomposed into a timestamp, a salience score, and a caption, and is generated using task-specific encoders and decoding heads rather than a single undifferentiated language-modeling head. TRACE demonstrates that this structural alignment between model architecture and video structure yields substantial performance gains over prior video-LLMs.

TRACE, however, is built on a 7-billion-parameter Mistral backbone together with a full CLIP ViT-L vision encoder. This is appropriate for a research system aimed at maximizing benchmark performance, but it is not suitable for constrained compute environments — a single GPU, a laptop, or eventually mobile/edge deployment. There is currently no small-footprint reference implementation that preserves TRACE's structural contribution (causal event modeling, task-interleaved sequences, adaptive head switching) while replacing its two heaviest components with lightweight counterparts.

This creates the concrete engineering and research opportunity that motivates this project.

## 1.2 Problem Statement

> **Can the causal event modeling framework introduced by TRACE be faithfully implemented using a lightweight vision encoder and a lightweight decoder, while preserving TRACE's event representation, mathematical formulation, and task-interleaved generation strategy?**

This is fundamentally a systems/architecture question, not a "beat TRACE on benchmarks" question. The thesis is not attempting to outperform a 7B-parameter model trained on millions of samples across multiple GPU-days. The thesis is attempting to show that the *architectural idea* — modeling video as a causally-generated sequence of (timestamp, score, caption) events using task-specialized heads — is separable from the specific choice of a 7B LLM backbone, and can be instantiated at a fraction of the parameter count while remaining a faithful, controlled reimplementation of the same framework.

## 1.3 Research Objective

The single, narrowly-scoped research objective of this project is:

> Develop a lightweight implementation of TRACE's causal event modeling framework.

This objective is deliberately narrow. It is restated here because it is easy, over the course of a long implementation effort, to drift toward adjacent and more exciting-sounding goals. This document exists partly to prevent that drift. The following are explicitly **not** objectives of this project, even though they are topically adjacent:

- Improving upon TRACE's benchmark numbers.
- Redesigning the event representation (timestamp + score + caption).
- Building a CCTV / surveillance analytics system.
- Performing general Video Question Answering.
- Real-time or edge deployment (this is future work, not Version 1 scope).

## 1.4 Scope

**In scope for TinyTRACE Version 1:**

- A lightweight vision-language encoder (MobileCLIP) replacing CLIP ViT-L.
- A lightweight decoder-only transformer (LCEM — Lightweight Causal Event Modeling Module) replacing Mistral-7B.
- Preservation of TRACE's event representation: `Event = (Timestamp, Score, Caption)`.
- Preservation of TRACE's mathematical formulation: `P(e_k | e_1:k-1, I, F)`.
- Preservation of the overall pipeline shape: frame sampling → visual encoding → visual token compression → time encoding → prompt construction → autoregressive decoding → adaptive head switching → event parsing.
- A repository-audit process to recover unpublished implementation details from the official TRACE repository.

**Out of scope for TinyTRACE Version 1** (see Freeze 1 in the companion Frozen Decisions Ledger, reproduced in §1.7):

- CCTV / surveillance use cases.
- General-purpose Video Question Answering as a primary task.
- Person tracking, action recognition, or object detection as standalone capabilities.
- Real-time / streaming deployment.
- Any modification to the event representation (e.g., adding actor/action/object/location fields) — reserved for a hypothetical Version 2.

## 1.5 TRACE Overview

TRACE is a task-interleaved video LLM built around a single theoretical idea: a video is a sequence of events, `V = {e_1, e_2, ..., e_K}`, where each event `e_k = (t_k, s_k, c_k)` consists of a timestamp `t_k`, a salient score `s_k`, and a textual caption `c_k`. TRACE models the joint distribution over an event sequence autoregressively, conditioning each event on all previous events, a textual instruction `I`, and the video frames `F`:

```
P(e_k | e_1:k-1, I, F) = P(t_k | e_1:k-1, I, F)
                        · P(s_k | t_k, e_1:k-1, I, F)
                        · P(c_k | s_k, t_k, e_1:k-1, I, F)
```

To implement this, TRACE does three things that distinguish it from prior video-LLMs:

1. **Separated multi-task processing.** Distinct encoders/tokenizers and distinct decoding heads exist for visual frames, timestamps, salient scores, and text. Time and score each use a small 13-token vocabulary (digits 0–9, a decimal point, a `<sep>` token, and a `<sync>` token) whose embeddings are initialized from the LLM's own token embeddings.
2. **Task-interleaved sequence modeling.** The full input sequence to the LLM backbone (Mistral-7B-v0.2) is built by interleaving visual tokens, instruction tokens, and — for each event, in order — time tokens, score tokens, and text tokens, following exactly the factorization order of the causal event modeling equation above.
3. **Adaptive head-switching.** Because different segments of the output sequence are produced by different decoding heads (time head, score head, text head), TRACE needs a mechanism to know, at generation time, which head to consult for the next token. This is done by watching for the `<sync>` token, whose appearance triggers a cyclic switch: time head → score head → text head → (next event) time head → ...

Visually, each video frame is encoded by a frozen CLIP ViT-L into 576 tokens, which are compressed via Slot-Based Compression down to 8 tokens per frame, and then concatenated with 6 time-encoding tokens per frame (derived from the frame's timestamp, with `<sync>`/`<sep>` stripped) to form the per-frame visual input.

## 1.6 TinyTRACE Overview

TinyTRACE is the lightweight reimplementation described by this document. It preserves every structural element of TRACE listed in §1.5 — the event representation, the factorized probability, the task-interleaved sequence order, and the adaptive head-switching mechanism — while replacing exactly two components:

| Component | TRACE | TinyTRACE |
|---|---|---|
| Visual Encoder | CLIP ViT-L (Radford et al., 2021) | MobileCLIP |
| Reasoning / Decoder Backbone | Mistral-7B-v0.2 (Jiang et al., 2023) | LCEM (Lightweight Causal Event Modeling Module) — original contribution of this thesis |

Everything else in the pipeline — the compression stage, the time/score tokenizer design, the task-interleaved ordering, and the adaptive head-switching logic — is intended to be reproduced as faithfully as the TRACE source code allows, subject to the audit findings recorded in Part VII.

## 1.7 Frozen Design Decisions (Summary Ledger)

The following table is the canonical, non-negotiable summary of design decisions for TinyTRACE Version 1. It is reproduced from the project's Frozen Decisions Ledger and is repeated here so that this Architecture Specification is self-contained. **No section of this document may contradict this table.** If reverse engineering reveals that a frozen decision rests on an incorrect assumption, the correct action is to update the assumption and document the change explicitly — not to silently redesign around it.

| # | Decision | Status |
|---|---|---|
| 1 | Research scope limited to a lightweight implementation of TRACE's causal event modeling framework; CCTV, VQA, surveillance, tracking, detection, real-time deployment are out of scope | FROZEN |
| 2 | Event representation remains `(Timestamp, Score, Caption)` | FROZEN |
| 3 | Mathematical formulation `P(e_k \| e_1:k-1, I, F)` is preserved unchanged | FROZEN |
| 4 | Overall pipeline shape (Video → Frames → Visual Encoder → Visual Tokens → Time Encoding → Prompt → Autoregressive Decoder → Event Tokens → Parser) is preserved | FROZEN |
| 5 | Visual encoder is MobileCLIP (CLIP replacement) | FROZEN |
| 6 | Reasoning module is LCEM, a lightweight decoder transformer (Mistral-7B replacement) | FROZEN |
| 7 | LCEM architecture family is decoder-only Transformer, not GRU, not LSTM, not TinyLlama, not SmolLM | FROZEN |
| 8 | Decoder generates an autoregressive event-token sequence, parsed post-hoc into structured events | FROZEN |
| 9 | Adaptive head switching is preserved conceptually; exact mechanism TODO | FROZEN (impl. TODO) |
| 10 | Prompt construction follows Visual Tokens + Time Tokens + Instruction Tokens → Decoder; exact mechanism TODO | FROZEN (impl. TODO) |
| 11 | Visual token compression follows MobileCLIP → Compression → Visual Tokens; exact mechanism TODO | FROZEN (impl. TODO) |

Items allowed to change after reverse engineering (Freeze 13): tensor shapes, token counts, prompt format, token serialization, special tokens, decoder interface, compression module internals, teacher forcing details, loss implementation, training loop, inference loop.

Items **not** allowed to change (Freeze 14) without proof that an assumption is fundamentally wrong: research objective, event representation, mathematical formulation, MobileCLIP as the visual encoder, LCEM as the reasoning module, the overall pipeline shape, and the fact of autoregressive event generation.

## 1.8 High-Level Architecture

At the highest level of abstraction, TinyTRACE is a single forward pipeline with one autoregressive loop at its center:

```
                         ┌─────────────────────────────────────────────────────┐
                         │                    TinyTRACE                        │
                         │                                                     │
  Input Video            │   ┌───────────┐   ┌────────────┐   ┌─────────────┐  │
  ─────────────►         │   │  Frame    │──►│ MobileCLIP │──►│   Visual    │  │
                         │   │ Sampling  │   │  Encoder   │   │Token Builder│  │
                         │   └───────────┘   └────────────┘   └──────┬──────┘  │
                         │                                           │         │
                         │   ┌───────────┐                           ▼         │
                         │   │   Time    │◄──────────────────────────┘         │
                         │   │  Encoder  │                                     │
                         │   └─────┬─────┘                                     │
                         │         ▼                                          │
                         │   ┌───────────────┐        ┌─────────────────┐     │
  Text Instruction  ────►│   │    Prompt     │───────►│      LCEM       │     │
                         │   │  Construction │        │ (Decoder-only   │     │
                         │   └───────────────┘        │  Transformer)   │     │
                         │                             └────────┬────────┘    │
                         │                                      │             │
                         │                             ┌────────▼────────┐    │
                         │                             │ Adaptive Head   │    │
                         │                             │   Switching     │    │
                         │                             │ (Time/Score/    │    │
                         │                             │  Text heads)    │    │
                         │                             └────────┬────────┘    │
                         │                                      ▼             │
                         │                             ┌─────────────────┐    │
                         │                             │  Event Tokens   │    │
                         │                             └────────┬────────┘    │
                         │                                      ▼             │
                         │                             ┌─────────────────┐    │
  Structured Events  ◄───┼─────────────────────────────│  Event Parser   │    │
  (Timestamp, Score,     │                             └─────────────────┘    │
   Caption)              │                                                    │
                         └─────────────────────────────────────────────────────┘
```

This diagram is elaborated module-by-module in Part III. The autoregressive loop (LCEM ↔ Adaptive Head Switching ↔ Event Tokens, feeding back into the prompt/context for the next token) is elaborated separately in Part V, since it is the part of the system whose behavior differs meaningfully between training (teacher-forced, parallel) and inference (sequential, one token at a time).

---

# PART II — MATHEMATICAL FOUNDATION

This part establishes, equation by equation, the mathematics that both TRACE and TinyTRACE implement. For every equation, this document explains its meaning, its purpose within the system, which TRACE module realizes it, which TinyTRACE module realizes it, and any implementation notes relevant to a lightweight setting. This mapping table (§2.9) is the single most important artifact in Part II — it is the contract that TinyTRACE must satisfy for the reimplementation to be a legitimate lightweight version of TRACE rather than an unrelated architecture that happens to also process video.

## 2.1 Event Representation

**Equation:**

```
V = {e_1, e_2, ..., e_K} = {(t_k, s_k, c_k) | 1 ≤ k ≤ K}
```

**Meaning.** A video `V` is not modeled as a single monolithic object; it is modeled as an ordered set of `K` discrete events. Each event `e_k` bundles together three heterogeneous pieces of information: a timestamp `t_k` (when the event occurs), a salient score `s_k` (how important/highlight-worthy the event is), and a caption `c_k` (what happens, in natural language).

**Purpose.** This equation is the foundational data-structure decision of the entire framework. Every downstream design choice — separate encoders, separate heads, the interleaved token order, adaptive head switching — exists to serve this factorization. Without this equation, there would be no reason to treat timestamps, scores, and captions as distinct "tasks."

**TRACE Module.** Realized implicitly by the dataset annotation schema (see Appendix A of the TRACE paper — each training example carries parallel `times`, `scores`, and `conversations`/caption fields) and by the Event Parser at inference time.

**TinyTRACE Module.** Realized identically by the Event Parser module (Part III, §3.8) and by the TinyTRACE training data schema, which must mirror TRACE's `{times, scores, conversations}` JSON structure.

**Implementation Note.** This representation is frozen (Freeze 2) and must not be extended with additional fields (actor, action, object, location) in Version 1.

## 2.2 Causal Event Modeling — Full Factorization

**Equation:**

```
P(e_k | e_1:k-1, I, F) = P(t_k, s_k, c_k | e_1:k-1, I, F)
                        = P(t_k | e_1:k-1, I, F)
                        · P(s_k | t_k, e_1:k-1, I, F)
                        · P(c_k | s_k, t_k, e_1:k-1, I, F)
```

**Meaning.** This is the central equation of the entire framework (Eq. 2 in the TRACE paper). It states that the probability of generating the next event, given everything generated so far plus the video and instruction, factorizes into three sequential conditional probabilities — first the timestamp, then the score (conditioned additionally on the timestamp just generated), then the caption (conditioned additionally on both the timestamp and the score just generated). This mirrors the chain rule of probability applied to the triple `(t_k, s_k, c_k)`, with a *fixed* generation order (time → score → caption) chosen by the TRACE authors, who note that the theoretical framework does not require this particular order but that they selected one order for practical implementation.

**Purpose.** This equation converts "video understanding" from an unstructured language-generation problem into a specific, structured autoregressive factorization that mirrors the actual structure of video (events ordered in time, each with sub-components). It is what allows TRACE to use task-specific encoders/heads while still being trainable end-to-end with a single autoregressive objective (the product of conditionals becomes a sum of log-probabilities, i.e., a standard cross-entropy-style training signal per sub-task token stream).

**TRACE Module.** Realized jointly by: the task-interleaved sequence order (§3.2.2 of the TRACE paper) which places tokens in the sequence `[t_k tokens][s_k tokens][c_k tokens]` per event, and by the LLM backbone's autoregressive next-token prediction, which — because the sequence order matches the factorization order — automatically implements the chain rule above.

**TinyTRACE Module.** Realized identically by the Prompt Builder (which must construct the same `[time][score][caption]` per-event token order) and by LCEM's autoregressive next-token prediction over that sequence.

**Implementation Note.** Because the factorization is realized *purely through token ordering* rather than through separate probability heads with explicit multiplicative combination, faithfully reproducing this equation in TinyTRACE reduces to faithfully reproducing the token ordering scheme. This is why Part III places heavy emphasis on the exact ordering used by the Prompt Builder module.

## 2.3 Inter-Event Ordering

**Equation (informal):** Events are ordered by generation index `k = 1, ..., K` such that `e_k` is generated after `e_1, ..., e_{k-1}` are fully generated, and the paper further specifies that `event tokens are sequenced according to the events' occurrence time`.

**Meaning.** Beyond the intra-event factorization in §2.2, TRACE additionally imposes that consecutive events in the sequence must appear in chronological order of their occurrence in the video. This means the model is not just generating an arbitrary set of events; it is generating a chronologically sorted list.

**Purpose.** This constrains the search space and gives the autoregressive decoder a monotonic structure to exploit — later events are conditioned on the timestamps of strictly earlier events, so the model can learn regularities like "the next event's start time must be ≥ the previous event's end time."

**TRACE Module.** Data preprocessing / annotation ordering, reflected implicitly in training targets.

**TinyTRACE Module.** Must be reproduced identically in the TinyTRACE data preprocessing pipeline. TODO (Verify from TRACE implementation) — whether any post-hoc sorting or validation step is applied to enforce this ordering, or whether it is assumed to already hold in the raw annotations.

## 2.4 Autoregressive Token-Level Generation

**Equation:**

```
P(x_1, ..., x_N) = ∏_{i=1}^{N} P(x_i | x_1, ..., x_{i-1})
```

**Meaning.** This is the standard decomposition used by all autoregressive language models: the probability of a full token sequence is the product of next-token conditional probabilities. TRACE's Eq. 2 (§2.2 above) is a *structured special case* of this general decomposition, where the tokens `x_1, ..., x_N` happen to be grouped into semantically meaningful sub-sequences (visual, instruction, time, score, text) rather than being a homogeneous stream of natural-language subwords.

**Purpose.** This is the mechanism that lets a single Transformer decoder, trained with a single next-token-prediction objective, implement the entire causal event modeling framework, provided the token sequence is constructed in the right order (§2.2, §2.3) and the right decoding head is consulted for each token (§2.6 Adaptive Head Switching).

**TRACE Module.** Mistral-7B-v0.2 decoder stack, operating over the interleaved token sequence of Figure 3 in the TRACE paper.

**TinyTRACE Module.** LCEM decoder stack (Part III, §3.6), operating over the same interleaved token sequence, at reduced width/depth.

## 2.5 Decoder Mathematics — Transformer Internals

This subsection documents the standard Transformer-decoder mathematics that both Mistral-7B (TRACE) and LCEM (TinyTRACE) are built from. These equations are *general Transformer theory*, not TRACE-specific inventions, and are included here because every module description in Part III references them.

### 2.5.1 Scaled Dot-Product Self-Attention

**Equation:**

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

where `Q = X W_Q`, `K = X W_K`, `V = X W_V` are learned linear projections of the input sequence `X`, and `d_k` is the key dimension.

**Meaning.** For every position in the sequence, attention computes a weighted average over the *value* vectors of all other positions, where the weights are determined by the similarity (dot product) between that position's *query* vector and every other position's *key* vector. The `√d_k` scaling prevents the dot products from growing too large in magnitude as dimensionality increases, which would otherwise push the softmax into regions with vanishing gradients.

**Purpose.** This is the mechanism by which the decoder mixes information across the sequence — allowing, for instance, a caption token being generated for event 3 to attend back to the visual tokens, the instruction tokens, and the timestamp/score tokens of event 3 as well as all tokens of events 1 and 2.

**TRACE Module.** Standard multi-head self-attention layers inside Mistral-7B-v0.2 (a Llama-family architecture using grouped-query attention and RoPE positional encoding). TODO (Verify from TRACE implementation) — whether TRACE modifies Mistral's default attention implementation (e.g., masking rules specific to task-interleaved tokens) or uses it unmodified.

**TinyTRACE Module.** LCEM's self-attention layers (Part III, §3.6), implemented as standard causal multi-head (or grouped-query, TBD) self-attention at reduced hidden size, layer count, and head count.

**Causal Masking.** Because generation is autoregressive, attention must be masked so that position `i` cannot attend to positions `j > i`. This is implemented via an additive mask of `-∞` (or a very large negative number) added to the pre-softmax attention logits at all `j > i` positions, which drives the corresponding softmax weight to zero.

### 2.5.2 Multi-Head Attention

**Equation:**

```
MultiHead(X) = Concat(head_1, ..., head_h) W_O
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
```

**Meaning.** Rather than computing a single attention distribution, the model computes `h` parallel attention "heads," each operating in a lower-dimensional subspace, and concatenates their outputs before a final linear projection `W_O`. This allows different heads to specialize in different types of relationships (e.g., local syntactic patterns vs. long-range dependencies).

**Purpose.** Increases representational capacity of the attention mechanism without a proportional increase in the sequence-length-dependent computational cost of a single very-high-dimensional attention operation.

**TinyTRACE Module.** LCEM uses a reduced number of heads `h_tiny ≪ h_mistral` and reduced head dimension, per the parameter budget discussed in Part VI.

### 2.5.3 Cross-Attention

**Equation (general form):**

```
CrossAttention(Q_dec, K_enc, V_enc) = softmax( Q_dec K_encᵀ / √d_k ) V_enc
```

**Meaning.** Standard cross-attention lets a decoder query a *separate* encoder's key/value representations. **Important architectural note:** TRACE does not use a classical encoder-decoder cross-attention architecture. Visual tokens (from CLIP/MobileCLIP, post-compression) are injected directly into the decoder's own input sequence as if they were additional tokens, and the *decoder's self-attention* mechanism (§2.5.1) is what allows text/time/score tokens to "attend to" visual tokens — because they all sit in the same sequence. This is the standard "vision tokens as prefix/interleaved tokens" pattern used by LLaVA-style multimodal LLMs, not an encoder-decoder cross-attention pattern.

**Purpose of documenting this here.** To prevent a design error: an implementer might assume TRACE/TinyTRACE has a dedicated cross-attention module bridging vision and language, and start engineering one. It does not, as far as the paper text and figures indicate. TODO (Verify from TRACE implementation) — confirm that no auxiliary cross-attention layers exist anywhere in the released TRACE code (e.g., inside the Slot-Based Compression module, which may itself use cross-attention internally to compress 576 tokens to 8 — see §3.3).

**TRACE Module.** N/A at the LLM-backbone level (self-attention only, over an interleaved sequence). Possibly present *inside* the Slot-Based Compression module — TODO (Verify from TRACE implementation).

**TinyTRACE Module.** LCEM performs self-attention only over the interleaved sequence, mirroring TRACE. The Visual Token Builder's Compression sub-stage may use cross-attention internally — TODO (Verify from TRACE implementation), see §3.3.5.

### 2.5.4 Position-Wise Feed-Forward Network

**Equation:**

```
FFN(x) = W_2 · σ(W_1 x + b_1) + b_2
```

where `σ` is a nonlinearity (SwiGLU in Mistral/Llama-family models; TODO (Verify from TRACE implementation) — confirm Mistral-7B-v0.2's exact activation and gating variant).

**Meaning.** After the attention sub-layer mixes information *across* sequence positions, the feed-forward sub-layer transforms each position's representation *independently*, typically expanding to a higher intermediate dimension and projecting back down. This is where most of a Transformer's parameter count and much of its per-token "computation" (as opposed to "mixing") resides.

**Purpose.** Provides the non-linear transformation capacity needed for the model to represent complex functions of its (already attention-mixed) representations.

**TinyTRACE Module.** LCEM's feed-forward sub-layers, at reduced intermediate dimension. Choice of activation function is an open implementation parameter — TODO (Verify from TRACE implementation) whether to mirror Mistral's SwiGLU or select a lighter alternative (e.g., GELU) for LCEM; current default assumption is SwiGLU to preserve architectural fidelity, but this is explicitly listed as changeable per Freeze 13 (decoder interface / internals may be adapted for lightness).

### 2.5.5 Residual Connections

**Equation:**

```
x_{l+1} = x_l + Sublayer(Norm(x_l))
```

(pre-normalization residual form, as used in Llama/Mistral-family models)

**Meaning.** Each sub-layer's output is added back to its input rather than replacing it, and normalization is applied *before* the sub-layer rather than after (pre-norm). This is the standard modern Transformer residual pattern and is critical for stable training of deep stacks.

**Purpose.** Prevents vanishing gradients in deep networks and allows the "identity path" to dominate early in training, easing optimization.

**TinyTRACE Module.** LCEM uses the same pre-norm residual pattern, at reduced depth (fewer stacked decoder blocks).

### 2.5.6 Layer Normalization / RMSNorm

**Equation (RMSNorm, used by Llama/Mistral-family models):**

```
RMSNorm(x) = x / RMS(x) · g,   RMS(x) = sqrt( (1/d) Σ x_i² + ε )
```

**Meaning.** Normalizes activations by their root-mean-square magnitude (rather than mean-and-variance as in classical LayerNorm), then applies a learned per-channel gain `g`. RMSNorm is cheaper than LayerNorm (no mean subtraction) and is standard in the Llama/Mistral family.

**TRACE Module.** RMSNorm, inherited from Mistral-7B-v0.2's architecture. TODO (Verify from TRACE implementation) — confirm no custom normalization is added for the task-specific token streams.

**TinyTRACE Module.** RMSNorm, matched to LCEM's decoder blocks for architectural consistency with the parent model family (this is a reasonable default given Freeze 7 specifies "decoder transformer," not a specific normalization scheme — marked as an adaptable implementation detail).

## 2.6 Adaptive Head-Switching Mechanism (Mathematical View)

**Equation (informal, state-machine form):**

```
head(i) = TimeHead     if state = TIME
        = ScoreHead    if state = SCORE
        = TextHead     if state = TEXT

state ← next(state)  whenever token_i == <sync>
next(TIME) = SCORE, next(SCORE) = TEXT, next(TEXT) = TIME
```

**Meaning.** Because different segments of the output sequence are produced by different vocabularies and different decoding heads, the model needs an explicit, deterministic rule for choosing which head produces the logits for the *next* token during generation. TRACE implements this rule as a finite-state machine with three states (Time, Score, Text) that cycles forward every time a `<sync>` token is emitted, in the fixed order Time → Score → Text → Time → ... This directly operationalizes the factorization order of Eq. 2 (§2.2) at generation time.

**Purpose.** Without this mechanism, the model would need a single shared vocabulary/head spanning digits, `<sep>`/`<sync>` tokens, and the full text vocabulary, which the TRACE ablation study explicitly shows performs far worse ("w/o independent encoder/heads" — Fail to Follow Instruction, Table 3 of the TRACE paper). Adaptive head switching is what allows separate, small, task-specialized heads to be composed into a single coherent generation loop.

**TRACE Module.** A generation-time controller (outside of, or wrapping, the LLM forward pass) that inspects generated tokens for `<sync>` and redirects subsequent logit computation to the appropriate head (Time Head / Score Head / Text Head), as illustrated in Figure 4 of the TRACE paper.

**TinyTRACE Module.** The Adaptive Head Switching module (Part III, §3.7) — must reproduce this exact three-state cyclic behavior. TODO (Verify from TRACE implementation) — exact implementation: is this controller implemented as (a) a Python-level generation loop wrapper around `model.generate()`-style logic, (b) a custom `LogitsProcessor`, or (c) baked into a modified forward pass that computes all three heads' logits every step and masks two of them? The three approaches have materially different engineering implications for a lightweight reimplementation and must be resolved before final implementation, not assumed.

## 2.7 Loss Functions

**Equation (general form — token-level negative log-likelihood, applied per task stream):**

```
L = − Σ_i log P(x_i | x_<i)
  = − Σ_{i ∈ time tokens}  log P_TimeHead(x_i | x_<i)
    − Σ_{i ∈ score tokens} log P_ScoreHead(x_i | x_<i)
    − Σ_{i ∈ text tokens}  log P_TextHead(x_i | x_<i)
```

**Meaning.** Because Eq. 2 factorizes `P(e_k | ...)` into a product of three conditional probabilities computed by three different heads, and because a product of probabilities becomes a *sum* of negative log-probabilities, the overall training loss is naturally a sum of three per-task cross-entropy losses (over time tokens, score tokens, and text tokens respectively), computed with teacher forcing over the full interleaved sequence. Tokens belonging to visual/instruction segments are not part of the generation target and are excluded from the loss (standard "prompt masking").

**Purpose.** This is the training-time realization of the mathematical framework in §2.2 — it is what actually updates the weights of the vision compression layer, the time/score encoders and heads, and the LLM backbone.

**TRACE Module.** TODO (Verify from TRACE implementation) — the paper does not give the exact loss weighting between the three task streams (whether they are summed unweighted, averaged per-task then summed, or given explicit task-loss coefficients). Given that Stage 1 trains task modules independently and Stage 2 fine-tunes the LLM backbone jointly (§3.3 of the TRACE paper), it is also unclear whether the *relative* weighting of time/score/text loss changes between stages. This remains a high-priority open implementation item since it materially affects training stability at small model scale.

**TinyTRACE Module.** Same structural decomposition (sum of per-task cross-entropy over teacher-forced sequences), with the exact weighting scheme TODO pending TRACE source inspection, defaulting conservatively to unweighted summed cross-entropy across all non-masked tokens until verified otherwise.

## 2.8 Two-Stage Training Objective (Structural, not a new equation)

TRACE's training is not a single end-to-end optimization from scratch; it proceeds in two stages (§3.3 of the TRACE paper):

- **Stage 1 — Task Module Initialization.** The vision compression layer, time encoder/head, and score encoder/head are trained while the vision encoder (CLIP) and LLM backbone (Mistral) remain frozen. This is effectively a projection/alignment pretraining stage — teaching the lightweight task-specific modules to produce embeddings the (frozen, already-capable) LLM can interpret.
- **Stage 2 — Instruction Tuning.** The LLM backbone and task modules are fine-tuned jointly (vision encoder remains frozen) on VTG instruction-tuning data, using a much lower learning rate (5e-6 vs. 1e-3 in Stage 1) for two epochs vs. one.

**TinyTRACE Module.** LCEM training should mirror this two-stage structure (Part IV, §4.1), since the same rationale applies at small scale: it is unreasonable to expect a randomly initialized decoder to simultaneously learn (a) how to interpret freshly-initialized time/score tokenizers and a freshly-initialized visual compression module, and (b) how to perform the full causal event modeling task, in a single optimization stage. TODO (Verify from TRACE implementation) — whether TinyTRACE's much smaller LCEM (trained from scratch rather than initialized from a pretrained 7B LLM) requires an even earlier Stage 0 (e.g., pure language-modeling pretraining of LCEM's text head on general text, since TRACE benefits from Mistral's pretrained linguistic knowledge and LCEM will not have this by default). This is flagged as an open research question in Part VI.

## 2.9 Mathematical-to-Module Mapping Table

This table is the master cross-reference for Part II. Every row states an equation, its purpose, and the corresponding module in both TRACE and TinyTRACE. It should be treated as the single source of truth when there is any apparent conflict between prose descriptions elsewhere in this document.

| Equation | Purpose | TRACE Module | TinyTRACE Module | Impl. Status |
|---|---|---|---|---|
| Event representation `(t_k,s_k,c_k)` | Structure the output space | Data schema + Event Parser | Data schema + Event Parser | Frozen |
| `P(e_k \| e_1:k-1,I,F)` factorization | Core causal event modeling objective | Task-interleaved sequence + Mistral | Prompt Builder + LCEM | Frozen |
| Inter-event chronological order | Enforce temporally valid event sequences | Data preprocessing | Data preprocessing | TODO |
| Autoregressive token decomposition | Enables single-model training via next-token prediction | Mistral decoder | LCEM decoder | Frozen |
| Scaled dot-product self-attention | Sequence mixing | Mistral self-attn layers | LCEM self-attn layers | Frozen (standard) |
| Multi-head attention | Representational capacity | Mistral multi-head attn | LCEM multi-head attn (reduced) | Adaptable |
| Cross-attention (vision↔text) | N/A — not used at backbone level | Not present at backbone | Not present at backbone | TODO confirm |
| Feed-forward (SwiGLU) | Per-token nonlinear transform | Mistral FFN | LCEM FFN (reduced, activation TBD) | Adaptable |
| Residual connections | Training stability | Mistral pre-norm residual | LCEM pre-norm residual | Frozen (standard) |
| RMSNorm | Activation normalization | Mistral RMSNorm | LCEM RMSNorm | Adaptable |
| Adaptive head-switching state machine | Route logits to correct head at gen time | `<sync>`-triggered controller | Same controller pattern | TODO impl. detail |
| Sum of per-task cross-entropy | Training signal | Teacher-forced loss over 3 heads | Same structure | TODO weighting |
| Two-stage training | Stable optimization order | Stage 1 (modules) / Stage 2 (LLM+modules) | Mirrors TRACE; possible Stage 0 | TODO (Stage 0 open Q) |


---

# PART III — COMPLETE SYSTEM ARCHITECTURE

## 3.1 Overall Pipeline (Restated with Data Shapes)

```
Input Video (raw file, arbitrary length/fps)
   │
   ▼
[Frame Sampling]  ── produces T sampled frames + T timestamps
   │
   ▼
[MobileCLIP Encoder]  ── each frame → raw patch/token embeddings
   │
   ▼
[Visual Token Builder]  ── compress per-frame embeddings → 8 visual tokens/frame (TRACE value; TinyTRACE TODO)
   │
   ▼
[Time Encoder]  ── per-frame timestamp → 6 time tokens/frame (TRACE value; TinyTRACE TODO)
   │
   ├── concatenate 8 visual + 6 time tokens per frame ──► per-frame visual input (14 tokens/frame in TRACE)
   │
   ▼
[Prompt Builder]  ── assembles [F][I][t_1][s_1][c_1][t_2][s_2][c_2]...  per Eq. 2 ordering
   │
   ▼
[LCEM]  ── decoder-only Transformer, autoregressive over the assembled sequence
   │
   ▼
[Adaptive Head Switching]  ── routes each generation step to Time/Score/Text head based on <sync> state
   │
   ▼
Event Tokens (raw token stream: digits, <sep>, <sync>, text subwords)
   │
   ▼
[Event Parser]  ── decodes token stream → structured list of {timestamp, score, caption}
   │
   ▼
Output: [{t_1, s_1, c_1}, {t_2, s_2, c_2}, ..., {t_K, s_K, c_K}]
```

## 3.2 Data Flow Summary

| Stage | Input Shape (TRACE, as documented) | Output Shape (TRACE, as documented) | TinyTRACE Shape |
|---|---|---|---|
| Frame Sampling | Video, target frame count (128 in TRACE) | T frames + T timestamps | Same count assumption; TODO whether TinyTRACE reduces T for lightness (see Part VI) |
| MobileCLIP Encoder | 1 frame (image) | 576 tokens/frame (CLIP ViT-L value) | TODO (Verify MobileCLIP's native patch/token count — architecture-dependent, not necessarily 576) |
| Visual Token Builder (Compression) | 576 tokens/frame | 8 tokens/frame | TODO (Verify from TRACE implementation) whether 8 is a fixed hyperparameter of Slot-Based Compression or tunable; TinyTRACE may retune this |
| Time Encoder | 1 timestamp scalar (or pair, for spans) | 6 time tokens/frame (after removing `<sync>`/`<sep>`) | Same scheme assumed; TODO confirm digit-formatting convention (4 whole-number digits + dot + 1 fractional digit, per §3.2.1 of TRACE paper) |
| Prompt Builder | Per-frame visual+time tokens, instruction text, prior event tokens | Single interleaved token sequence | Same structural assembly |
| LCEM | Interleaved token sequence (length TODO — TRACE model max length is 4096) | Per-position hidden states | Reduced hidden size / depth; TODO define TinyTRACE max sequence length |
| Adaptive Head Switching | Hidden state at current generation step + running `<sync>` state | Logits over active head's vocabulary | Same state machine |
| Event Parser | Flat token stream with `<sep>`/`<sync>` delimiters | Structured event list | Same parsing contract |

## 3.3 Module: Frame Sampling

**Purpose.** Convert an input video of arbitrary length and frame rate into a fixed-size, temporally-representative set of `T` frames, each associated with a timestamp, suitable for encoding by the vision tower.

**Inputs.**
- Raw video file (arbitrary duration, fps, resolution).
- Target frame count `T` (TRACE: 128, used identically for both Stage 1 and Stage 2 training, per Table 7 of the TRACE paper).
- Sampling strategy selector (uniform vs. clip-then-random-within-clip).

**Outputs.**
- `T` decoded RGB frames.
- `T` corresponding timestamps (seconds, float), used downstream by the Time Encoder.

**Pipeline.**
1. Determine video duration and total native frame count.
2. Select `T` sample points according to the active sampling strategy:
   - *Uniform* — evenly spaced across the full duration (TRACE Stage 1: "we uniformly sample 128 frames from each video").
   - *Split-then-random-within-clip* — divide the video into `T` equal-length clips and sample one frame at a random offset within each clip (TRACE Stage 2: "the content is uniformly divided into 128 clips, with one frame randomly sampled from each clip").
3. Decode the selected frames and record their timestamps.

**Mathematics.** No learned parameters. Given duration `D` and clip index `j ∈ {0, ..., T-1}`, clip boundaries are `[jD/T, (j+1)D/T)`; uniform sampling picks the clip midpoint, random-within-clip sampling picks `t_j ~ Uniform(jD/T, (j+1)D/T)`.

**Why TRACE uses it.** Fixed-frame-count sampling makes downstream tensor shapes deterministic (critical for batching), while the two different strategies for Stage 1 vs. Stage 2 appear to serve different purposes: Stage 1 (module initialization) benefits from a simple, deterministic scheme; Stage 2's randomized-within-clip sampling is a data augmentation technique that exposes the model to slightly different temporal offsets across epochs, improving robustness to exact frame-boundary effects.

**Why TinyTRACE keeps/replaces it.** Keeps the same two-strategy scheme structurally, since it is a training-data-preparation technique independent of model size. TODO (Verify from TRACE implementation) whether `T=128` is itself tied to the LLM's `Model Max Length = 4096` (since `128 frames × 14 tokens/frame = 1792` visual+time tokens, leaving budget for instruction and event tokens within 4096) — if so, TinyTRACE's `T` must be chosen jointly with LCEM's own max sequence length rather than copied verbatim.

**Advantages.** Deterministic tensor shapes; simple; strategy directly ported from TRACE with no architectural risk.

**Limitations.** Uniform/random sampling can miss very short, high-salience events if `T` is too small relative to video length; no content-adaptive sampling (e.g., scene-change detection) is used.

**Future Improvements.** Content-adaptive or motion-adaptive frame sampling (explicitly deferred — would be a Version 2 research direction, not Version 1 scope per Freeze 1).

**Assumptions.** `T` for TinyTRACE will likely be smaller than 128 to fit within a smaller LCEM context window and a smaller compute budget; the exact value is a hyperparameter to be tuned experimentally, not a frozen decision.

**TODO.**
- TODO (Verify from TRACE implementation) — exact decoding library/backend used by TRACE for frame extraction (e.g., decord, OpenCV, PyAV) and any preprocessing (resize/crop/normalize) applied before the vision encoder.
- TODO (Verify from TRACE implementation) — whether frame sampling differs between training and the inference/generation code path in the released repository.

## 3.4 Module: MobileCLIP (Visual Encoder)

**Purpose.** Replace TRACE's frozen CLIP ViT-L/14-336 vision tower with a lightweight vision-language encoder that preserves the same design philosophy — a contrastively-pretrained image encoder whose embedding space is already aligned with natural language — while drastically reducing parameter count and inference latency.

**Inputs.** A single RGB frame (or a batch of `T` frames), resized/normalized per MobileCLIP's expected input format. TODO (Verify from TRACE implementation) — TRACE uses `openai/clip-vit-large-patch14-336`, i.e., 336×336 input resolution; TinyTRACE must determine MobileCLIP's native input resolution (typically 256×256 for MobileCLIP variants) and decide whether to resize inputs to match or adapt downstream token counts accordingly.

**Outputs.** A set of visual embedding tokens per frame (patch-level or token-level embeddings, pre-compression). TRACE's CLIP ViT-L produces 576 tokens per frame (24×24 patch grid at 336×336 / patch size 14). MobileCLIP's native output token count is architecture-dependent and will generally differ.

**Pipeline.**
1. Preprocess frame (resize, normalize per MobileCLIP's training statistics).
2. Forward pass through MobileCLIP's image tower.
3. Extract patch/token-level embeddings (not just a single pooled CLS/global embedding, since TRACE requires multiple spatial tokens per frame as input to its compression module).

**Mathematics.** MobileCLIP itself is a full learned vision-transformer/hybrid-CNN encoder; its internal mathematics are out of scope for this document (it is used as a pretrained, largely frozen component, mirroring TRACE's treatment of CLIP). The relevant interface-level fact is: `MobileCLIP: R^{H×W×3} → R^{N_patches × d_model}`.

**Why TRACE uses CLIP.** CLIP ViT-L is a strong, widely-validated, contrastively pretrained vision-language encoder whose embedding space is already loosely aligned with text, which eases the burden on the downstream compression + LLM stages of learning a good vision-to-language bridge from a relatively modest amount of instruction-tuning data.

**Why TinyTRACE replaces it with MobileCLIP.** MobileCLIP was explicitly selected over the alternative considered (Tiny VideoMAE) because MobileCLIP preserves CLIP's core design philosophy (image encoder + contrastive language alignment) while VideoMAE is a video-native masked-autoencoder representation model with a fundamentally different pretraining objective (self-supervised reconstruction, not vision-language contrastive alignment). Using MobileCLIP keeps the CLIP→MobileCLIP substitution a controlled, single-variable change (model size/efficiency) rather than a confound of both size *and* pretraining paradigm. This directly serves the project's controlled-comparison research objective (§1.2).

**Advantages.** Order-of-magnitude fewer parameters than CLIP ViT-L; mobile-optimized inference; retains vision-language alignment property needed for the downstream LLM to interpret visual tokens with limited additional training.

**Limitations.** Weaker representational capacity than CLIP ViT-L; likely coarser spatial resolution or fewer output tokens per frame, which changes the compression module's required compression ratio (§3.5); potential domain gap between MobileCLIP's pretraining data and TRACE's video-frame distribution.

**Future Improvements.** Evaluate multiple MobileCLIP variants (different size tiers) for the accuracy/latency trade-off; consider lightweight fine-tuning of MobileCLIP on video-frame data (out of scope for Version 1 given the vision encoder is kept frozen, mirroring TRACE's own choice to keep CLIP frozen throughout both training stages).

**Assumptions.** MobileCLIP remains **frozen** during both TinyTRACE training stages, mirroring TRACE's treatment of CLIP (frozen in both Stage 1 and Stage 2, per Table 7 of the TRACE paper — "Vision Encoder" row is identical across both stages, implying it is not part of the trained parameter set).

**TODO.**
- TODO (Verify from TRACE implementation) — exact CLIP checkpoint preprocessing pipeline (crop strategy, normalization constants) to replicate as closely as possible for MobileCLIP.
- TODO (Verify from TRACE implementation) — confirm CLIP is indeed frozen throughout (the paper's Table 7 lists "Vision Encoder" as a fixed setting identical across stages but does not use the word "frozen" explicitly next to it; cross-check against source code's `requires_grad` settings).
- TODO — select specific MobileCLIP variant/checkpoint (e.g., MobileCLIP-S0/S1/S2/B) based on parameter budget determined in Part VI.

## 3.5 Module: Visual Token Builder (Compression)

**Purpose.** Reduce the number of visual tokens per frame from MobileCLIP's native output count down to a small, fixed number (TRACE: 8 tokens/frame) suitable for concatenation into a long multi-frame sequence without exceeding the decoder's context window.

**Inputs.** Per-frame patch/token embeddings from MobileCLIP, shape `[N_patches, d_model]` per frame.

**Outputs.** Compressed per-frame visual tokens, shape `[8, d_model']` per frame (TRACE value; TinyTRACE value TODO), ready for concatenation with time tokens (§3.6) and injection into the prompt sequence.

**Pipeline (TRACE, as documented — "Slot-Based Compression").** TRACE explicitly cites "Slot-Based Compression (Guo et al., 2024)" — i.e., the same technique introduced in VTG-LLM, a prior work by an overlapping set of authors — as the mechanism reducing 576 tokens/frame down to 8. The TRACE paper itself does not re-derive the mechanics of Slot-Based Compression; it is treated as an imported building block.

**Mathematics.** TODO (Verify from TRACE implementation). Slot-based compression techniques in the broader literature (e.g., Slot Attention, Perceiver-style latent bottlenecks) typically operate by having a small, fixed number of learned "slot" query vectors attend (via cross-attention) over the full set of input tokens, iteratively refining slot representations — but the *exact* mechanism used by Guo et al.'s Slot-Based Compression (referenced, not re-specified, in the TRACE paper) must be verified directly from the cited implementation before TinyTRACE's compression module can be considered a faithful reproduction rather than a plausible guess.

**Why TRACE uses it.** Without compression, 128 frames × 576 tokens/frame = 73,728 visual tokens — far beyond any feasible context window (TRACE's model max length is 4096). Compression to 8 tokens/frame brings this down to 1024 visual tokens for 128 frames, a tractable budget alongside time tokens, instruction tokens, and event tokens.

**Why TinyTRACE keeps/replaces it.** The *need* for compression is architecture-independent (any long-video pipeline needs it), so TinyTRACE keeps a compression stage in the pipeline shape (Freeze 4). However, the specific compression ratio (8 tokens/frame) was tuned for CLIP ViT-L's 576-token output and Mistral's 4096-token context; with MobileCLIP potentially producing a different native token count and LCEM potentially using a different (likely smaller) context window, the *target* compressed token count per frame is an open hyperparameter, not something to copy verbatim. The *mechanism* (slot-based attention compression), however, should be reproduced as faithfully as the verified TRACE implementation allows, since it is a specific architectural choice, not a size-scaling parameter.

**Advantages.** Learned, content-adaptive compression (as opposed to naive average-pooling) should retain more task-relevant visual information per token; fixed output token count regardless of input resolution simplifies downstream sequence-length budgeting.

**Limitations.** Adds a nontrivial trainable module that must be initialized (Stage 1) before the rest of the system can be trained meaningfully; aggressive compression (576→8, a 72× reduction) risks discarding fine-grained visual detail relevant to precise temporal grounding.

**Future Improvements.** Adaptive compression ratio based on video complexity or scene-change density (Version 2+ direction, out of scope now).

**Assumptions.** Current working assumption is that the compression module is trained (not frozen) starting in Stage 1, alongside the time/score encoders/heads, per §3.3 of the TRACE paper's description of Stage 1 module initialization.

**TODO.**
- TODO (Verify from TRACE implementation) — the precise architecture of Slot-Based Compression: number of attention iterations, slot initialization scheme (learned vs. random), whether slots are shared across frames or frame-specific.
- TODO (Verify from TRACE implementation) — whether compression operates independently per-frame or has any cross-frame/temporal component (the TRACE paper's description strongly suggests per-frame independence — "each frame being encoded into 576 visual tokens... reduce the number of visual tokens to 8 per frame" — but this must be confirmed against source, not assumed from prose alone).
- TODO — determine TinyTRACE's target per-frame token count once MobileCLIP's native output shape and LCEM's context budget are fixed (Part VI).

## 3.6 Module: Time Encoder

**Purpose.** Convert numeric timestamps (and, by the same shared architecture, salient scores — see §3.6.5 note) into a small sequence of discrete tokens that the LLM backbone can process using its native token-embedding and attention machinery, while also producing a compact "time token" representation that is fused directly into each frame's visual input.

**Inputs.**
- For per-frame time fusion: a single timestamp scalar (the frame's sampled time, seconds).
- For event-level time generation: a timestamp scalar or a `[start, end]` pair, to be tokenized into the output event sequence.

**Outputs.**
- Per-frame fusion path: 6 time tokens per frame (after stripping `<sync>`/`<sep>` from the general tokenization scheme described below).
- Event-generation path: full timestamp token sequence including `<sep>` (between start/end) and `<sync>` (end of the time task for this event).

**Pipeline (as documented in the TRACE paper, §3.2.1).**
1. A dedicated tokenizer vocabulary of 13 tokens is used: digit tokens `⟨0⟩...⟨9⟩`, a decimal-point token `⟨.⟩`, a separator token `⟨sep⟩` (marks the boundary between two timestamps, e.g., start/end), and `⟨sync⟩` (marks the end of the entire time-task sub-sequence).
2. Each timestamp is formatted to a fixed length: 4 whole-number digits + 1 decimal point + 1 fractional digit (e.g., timestamp `10.23` → digits `0,0,1,0,.,2` — note: the TRACE paper's worked example tokenizes `10.23` as `⟨0⟩⟨0⟩⟨1⟩⟨0⟩⟨.⟩⟨2⟩`, i.e., only **one** fractional digit is kept, meaning the example value `10.23` is truncated/rounded to `10.2` at the token level — implementers must replicate this exact formatting convention, not a more "sensible" one, to remain faithful).
3. `⟨sep⟩` is inserted between the two timestamps of a span (start, end); `⟨sync⟩` terminates the whole timestamp sub-sequence.
4. Token embeddings for this 13-token vocabulary are initialized from the LLM's own token embedding table (i.e., not trained fully from scratch — the digit/punctuation tokens borrow initial embeddings from the LLM's existing embedding space, presumably its embeddings for the corresponding characters/subwords).
5. For the per-frame visual-fusion path specifically, the `⟨sync�"and `⟨sep⟩` tokens are stripped, leaving exactly 6 tokens per frame (matching the 4-digit + dot + 1-digit = 6-token format for a *single* timestamp, since each frame has only one timestamp, not a span).

**Mathematics.** A learned encoder (architecture unspecified beyond "encoder" in the paper text — TODO) maps the 6-token (or full, with `<sep>`/`<sync>`, for spans) tokenized timestamp into embedding vectors compatible with the LLM's hidden dimension; a matching **decoding head** maps LLM hidden states back to a distribution over the same 13-token vocabulary during generation.

**Why TRACE uses a dedicated time encoder/tokenizer rather than raw text digits.** The TRACE paper positions this design choice as building on VTG-LLM's finding (Guo et al., 2024, cited within the same paper) that dedicated time tokens with their own embeddings/heads avoid the "time token quantization error" and instruction-following failures seen when timestamps are represented as ordinary text (the ablation "w/o independent encoder/heads" in Table 3 fails to follow instructions entirely). A small, closed vocabulary (13 tokens) for numeric content is easier for a decoding head to learn precisely than an open text vocabulary of thousands of subword tokens, most of which are irrelevant to representing digits.

**Why TinyTRACE keeps this design.** This is exactly the kind of task-specialized-head design that the causal event modeling framework depends on (§2.6); removing it would not be "simplifying," it would be undermining the very mechanism the whole architecture is built around. It is therefore preserved as-is, structurally, in TinyTRACE.

**Advantages.** Small, easy-to-learn vocabulary; numerically precise (fixed-width formatting avoids ambiguity); shares embedding initialization with the LLM's existing embedding space rather than starting from nothing.

**Limitations.** Fixed-width digit formatting caps the maximum representable timestamp (4 whole-number digits ⇒ videos longer than 9999 seconds, i.e., ~2.7 hours, cannot be represented without a format change) and caps fractional precision at one decimal digit (0.1 s resolution).

**Future Improvements.** Variable-width or logarithmic time encoding for very long videos (out of current scope).

**Assumptions.** The "encoder" and "head" for time tokens are assumed to be lightweight (small MLP or single-layer projection over token embeddings), consistent with the general lightweight philosophy of the task modules, but this is currently unverified.

**TODO.**
- TODO (Verify from TRACE implementation) — exact architecture of the "Time Encoder" and "Time Head" (paper text does not specify beyond calling them an encoder/decoding head "sharing the same architecture" as the score encoder/head).
- TODO (Verify from TRACE implementation) — exact embedding-initialization procedure: which specific LLM token embeddings are copied for digit tokens `0-9`, `.`, `<sep>`, `<sync>`.
- TODO (Verify from TRACE implementation) — confirm the fixed-width formatting convention (4 whole + 1 dot + 1 fractional) via source code / tokenizer config, not solely the paper's single worked example.
- TODO (Verify from TRACE implementation) — precise mechanism for concatenating 8 visual tokens + 6 time tokens into the "visual input for each frame" (simple sequence concatenation is the current assumption, but interleaving or learned fusion is not ruled out).

## 3.7 Module: Prompt Builder

**Purpose.** Assemble the full input token sequence fed to the LCEM decoder, following the inter-event and intra-event ordering specified by Eq. 2 (§2.2–2.3): visual+time tokens for all sampled frames, followed by instruction tokens, followed by the growing sequence of per-event `[time][score][caption]` token groups.

**Inputs.**
- Per-frame fused visual+time tokens (from §3.5 and §3.6), for all `T` sampled frames.
- Tokenized textual instruction `I` (e.g., "Please locate a series of events in the video, output the start and end timestamps of each event, and describe each event in sentences...").
- (Training only) Ground-truth event tokens for teacher forcing.
- (Inference only) Growing sequence of previously-generated event tokens.

**Outputs.** A single flat token sequence (with associated per-token "modality/task" tags used by Adaptive Head Switching to know which head produced/should produce each token) ready for the LCEM decoder's forward pass.

**Pipeline.**
1. Concatenate all `T` frames' visual+time token blocks in temporal order → forms the `F` segment of Figure 3 in the TRACE paper.
2. Append the tokenized instruction `I`.
3. Append the event token stream, per event `k`: `t_k` tokens (formatted per §3.6) → `s_k` tokens (same 13-token-family scheme, but 3 tokens: 1 whole-number digit + dot + 1 fractional digit, per the TRACE paper's footnote 2 distinguishing score formatting from timestamp formatting) → `c_k` tokens (ordinary LLM subword tokenization) → repeat for `k+1, ..., K`.
4. Special-case handling for tasks without one or more of the three components (e.g., general captioning tasks have no timestamps/scores — Appendix A of the TRACE paper shows these are represented with a **single `<sync>` token as a placeholder** for the missing sub-task, rather than omitting the segment entirely).

**Mathematics.** No learned parameters in this module itself; it is a deterministic sequence-assembly procedure. Its correctness is what makes the LLM's plain next-token cross-entropy loss equivalent to the factorized objective in Eq. 2.

**Why TRACE uses this exact structure.** This *is* the concrete instantiation of Eq. 2 as a token sequence — see §2.2. Figure 3 of the TRACE paper is the authoritative reference diagram for this ordering.

**Why TinyTRACE keeps it.** This ordering is not a size-dependent detail — it is the architectural core of causal event modeling. It is frozen (Freeze 4/8) and must be reproduced exactly, only the underlying token embeddings/vocab sizes and total context length may shrink.

**Advantages.** Directly operationalizes the theoretical framework with no gap between theory and implementation; makes teacher forcing straightforward (single flat sequence, single next-token loss, with task-based masking to route each segment's loss to the correct head).

**Limitations.** Placeholder `<sync>`-only segments for missing sub-tasks (general captioning data) mean the model must learn to *sometimes* emit an immediate `<sync>` with no preceding digits — a delicate behavior that could be more fragile in a small model with less capacity to disambiguate "emit digits then sync" vs. "emit sync immediately" based on subtle instruction-text cues.

**Future Improvements.** N/A for Version 1 — this module is a faithful port, not a research contribution area.

**Assumptions.** The exact textual instruction templates (the natural-language prompts shown in Figures 6–10 of the TRACE paper's appendix) are assumed to be reusable near-verbatim for TinyTRACE's own instruction-tuning data, subject to verification.

**TODO.**
- TODO (Verify from TRACE implementation) — exact special/reserved token handling in the actual tokenizer configuration (how `<sync>`, `<sep>`, `<video>` etc. are registered — as true special tokens or as reused/extended vocabulary entries).
- TODO (Verify from TRACE implementation) — precisely how the visual segment `F` is delimited from the instruction segment `I` (is there a `<video>` boundary token, as suggested by the literal string `"<video>\n..."` appearing in the Appendix A JSON examples?).
- TODO (Verify from TRACE implementation) — full inventory of instruction templates per task type (dense captioning, moment retrieval, highlight detection, summarization, general QA) for faithful reproduction of Stage 1/Stage 2 data formatting.

## 3.8 Module: LCEM (Lightweight Causal Event Modeling Module)

**Purpose.** LCEM is the single original research contribution of this thesis: a decoder-only Transformer, trained from a substantially smaller parameter budget than Mistral-7B, that plays the same architectural role Mistral-7B plays in TRACE — consuming the interleaved token sequence produced by the Prompt Builder and producing, autoregressively, the hidden states from which the Time/Score/Text heads compute next-token logits.

**Inputs.** The full interleaved token sequence (visual+time tokens for `T` frames, instruction tokens, and — during generation — the growing event-token sequence), as embeddings.

**Outputs.** Per-position hidden states, consumed by the Adaptive Head Switching module (§3.9) to produce next-token logits from the currently-active head.

**Pipeline.** Standard decoder-only Transformer stack: token/positional embedding → `N` stacked decoder blocks (each: causal self-attention §2.5.1–2.5.2 → residual §2.5.5 → RMSNorm §2.5.6 → feed-forward §2.5.4 → residual → RMSNorm) → final norm → (hidden states handed to task-specific heads, not a single unified LM head, per §3.9).

**Mathematics.** See Part II, §2.5 in full — LCEM implements exactly the general Transformer-decoder mathematics documented there, at reduced scale (fewer layers, smaller hidden dimension, fewer attention heads, smaller feed-forward intermediate dimension, and a much smaller/simpler text vocabulary than Mistral's ~32K-token vocabulary, since LCEM is not required to be a general-purpose language model — see Part VI for the parameter-budget analysis).

**Why TRACE uses Mistral-7B.** Mistral-7B-v0.2 provides strong pretrained linguistic and world knowledge, which the causal event modeling framework leverages for the *caption* generation sub-task in particular (captions are open-ended natural language, unlike the closed-vocabulary time/score sub-tasks) — this is explicitly why TRACE fine-tunes an existing strong LLM rather than training a decoder from scratch.

**Why TinyTRACE replaces it with LCEM.** The project's frozen research direction (Freeze 6/7) is to build a purpose-specific lightweight decoder rather than either (a) using another off-the-shelf small pretrained LLM (TinyLlama, SmolLM — explicitly rejected, see Part VI for rationale) or (b) using a non-Transformer sequence model (GRU/LSTM — explicitly rejected). The rejection of TinyLlama/SmolLM as drop-in replacements, despite them being reasonable "lightweight LLM" choices in isolation, is a deliberate research-framing decision: LCEM is meant to be an architecture *designed for* causal event modeling specifically (small closed-vocabulary time/score heads + a modest open-vocabulary text head, all sharing one small decoder backbone) rather than a general-purpose small chat model repurposed for this task. This keeps the research contribution centered on "a decoder architecture built for causal event modeling" rather than "which off-the-shelf small LLM works best here," which is a narrower and more defensible thesis-scale claim.

**Advantages.** Dramatically smaller memory/compute footprint than Mistral-7B; architecture purpose-built around the three-head structure rather than adapted from a general chat-model checkpoint; full control over vocabulary size (text head need not cover Mistral's full ~32K subword vocabulary if the target caption domain is narrower, further reducing parameter count).

**Limitations.** No pretrained linguistic/world knowledge (unless a Stage-0 pretraining step is added — see §2.8's open question); likely materially weaker caption fluency and factual grounding than Mistral-7B-based TRACE, especially early in training or with limited data; smaller context window likely necessitates reduced `T` (frame count) or more aggressive visual compression (§3.5) relative to TRACE.

**Future Improvements.** Explore initializing LCEM's text head/embedding from a small pretrained model's embedding table (a middle ground between "fully from scratch" and "adopt TinyLlama wholesale") without adopting the full TinyLlama/SmolLM architecture — this preserves the "purpose-built decoder" framing while borrowing useful embedding priors; flagged as a candidate ablation, not a Version 1 requirement.

**Assumptions.** LCEM will be trained substantially or fully from scratch (random initialization), pending the Stage-0 pretraining question in §2.8; exact layer count, hidden size, head count are hyperparameters to be fixed via the parameter-budget analysis in Part VI, not specified here as frozen numbers.

**TODO.**
- TODO (Verify from TRACE implementation) — Mistral-7B-v0.2's exact architectural hyperparameters (layer count, hidden size, head count, KV-head count for its grouped-query attention, intermediate FFN size, RoPE base) as a reference point for proportionally scaling down LCEM's own hyperparameters in a principled way (Part VI, §6.8).
- TODO — finalize LCEM's own hyperparameters once the parameter budget (Part VI) and MobileCLIP's output shape (§3.4) are fixed.
- TODO — decide on tokenizer/vocabulary source for LCEM's text head (train a new small tokenizer on the target caption corpus vs. reuse an existing small open tokenizer).

## 3.9 Module: Adaptive Head Switching

**Purpose.** At every generation step, determine which of the three decoding heads (Time Head, Score Head, Text Head) should compute the next-token logits from LCEM's current hidden state, based on the finite-state cycle described mathematically in §2.6.

**Inputs.** LCEM's hidden state at the current generation position; the running head-state (Time / Score / Text), updated based on previously generated tokens.

**Outputs.** Next-token logits over the *currently active* head's vocabulary (13-token time/score vocab, or the full text vocab, depending on state); the (possibly updated) head-state for the following step.

**Pipeline.**
1. Initialize state = TIME at the start of each new event's generation (i.e., right after the previous event's `<sync>`-terminated text segment, or at the very start of the first event).
2. At each step, compute logits using the head corresponding to the current state.
3. Sample/select the next token.
4. If the sampled token is `<sync>`, advance the state cyclically: TIME→SCORE, SCORE→TEXT, TEXT→TIME (beginning the next event).
5. Repeat until an end-of-sequence condition is met (e.g., a maximum event count, an explicit `<END>`-style signal, or exhausting the target sequence length — TODO, see below).

**Mathematics.** See §2.6 for the formal state-machine equations.

**Why TRACE uses it.** This is the mechanism that makes separated multi-task heads *composable* into a single coherent autoregressive generation loop — without it, the model would have no way of knowing, at inference time, which vocabulary/head applies to the next token, since the decoder itself only produces a generic hidden state.

**Why TinyTRACE keeps it.** Structurally identical necessity applies regardless of model size — this is core to the causal event modeling framework itself, not a scale-dependent detail (Freeze 9).

**Advantages.** Cheap (a small deterministic controller, negligible compute overhead relative to the decoder forward pass itself); directly reflects the theoretical factorization in Eq. 2; empirically validated by TRACE's own ablation (shared heads fail to follow instructions entirely — Table 3).

**Limitations.** Brittle to head-vocabulary decoding errors — if the Time Head fails to emit a `<sync>` token when it should (e.g., due to a malformed or too-long digit sequence), the state machine will not advance, potentially causing a generation loop or malformed output; requires careful handling of the "placeholder-only" case (general captioning data with a single `<sync>` standing in for an empty time/score segment, per §3.7).

**Future Improvements.** N/A for Version 1 — faithful reproduction is the goal, not enhancement.

**Assumptions.** The controller is currently assumed to be implemented as generation-loop-external logic (a Python-level wrapper), not baked into LCEM's own forward-pass graph, but this is explicitly unverified (see the corresponding TODO in §2.6).

**TODO.**
- TODO (Verify from TRACE implementation) — the three implementation-approach options enumerated in §2.6 must be disambiguated by inspecting TRACE's actual generation code.
- TODO (Verify from TRACE implementation) — exact end-of-generation condition (max events, explicit terminal token, or max-length truncation).
- TODO (Verify from TRACE implementation) — behavior on malformed generations (e.g., a Time Head that never emits `<sync>`) — does TRACE have any timeout/fallback logic?

## 3.10 Module: Event Parser

**Purpose.** Convert the raw, flat token stream produced by the autoregressive generation loop (interleaved time-digit tokens, score-digit tokens, and text subword tokens, delimited by `<sep>`/`<sync>`) back into the structured output format the end user or downstream evaluation code expects: a list of `{timestamp(s), score, caption}` dictionaries, matching the case-study output format shown in Figure 11 of the TRACE paper (`{'timestamps': [[...]], 'scores': [[...]], 'captions': [...]}`).

**Inputs.** Flat generated token stream (post-generation, entire sequence for one video+instruction pair).

**Outputs.** Structured event list, e.g. `[{timestamps: [0.0, 54.6], scores: [], captions: "a small kitten is seen laying..."}, {timestamps: [54.6, 116.8], scores: [], captions: "the kitten then lays on the chicken..."}]` — note that the score list can be legitimately empty for tasks (like the dense captioning example in Figure 11) that do not require salience scores; this must be handled as a valid, expected case, not an error.

**Pipeline.**
1. Split the flat token stream into event segments using `<sync>` boundaries (respecting the three-heads-per-event cycle from §3.9).
2. Within each event, split the time segment on `<sep>` to recover one or more timestamp values (a single timestamp for point events, a `[start, end]` pair for span events).
3. Parse fixed-width digit sequences back into floating-point numbers using the inverse of the formatting convention in §3.6 (4 whole-digit + 1 dot + 1 fractional digit for time; 1 whole-digit + 1 dot + 1 fractional digit for score, per the TRACE paper's footnote 2).
4. Detokenize the text segment using the standard LLM text detokenizer.
5. Handle placeholder-only (`<sync>`-only) segments as empty timestamp/score lists, per §3.7.
6. Assemble the final structured list, ordered by event index (already chronological per §2.3).

**Mathematics.** No learned parameters — purely deterministic parsing logic, the exact inverse of the Prompt Builder's formatting/serialization scheme.

**Why TRACE needs it.** The LLM only ever produces token IDs; converting these into numerically usable timestamps/scores and human-readable captions is essential for the model to be usable in any downstream application or for automated benchmark evaluation.

**Why TinyTRACE keeps it.** This is a pure serialization-format contract — it must be reproduced exactly (or, per Freeze 13, may be *adapted* alongside the rest of the token-serialization scheme if reverse engineering reveals TRACE's actual scheme differs from the paper's simplified example), since Event Parser correctness is what actually determines whether the whole pipeline's output is usable at all.

**Advantages.** Simple, deterministic, testable independently of the rest of the model (unit-testable given only a token stream).

**Limitations.** Fragile to any generation-time formatting errors (missing `<sep>`, malformed digit counts, unexpected token ordering) — a robust implementation needs defensive parsing (e.g., best-effort recovery, or explicit failure signaling) rather than assuming well-formed output, especially given LCEM's smaller capacity may produce more malformed sequences than Mistral-7B-based TRACE, particularly early in training.

**Future Improvements.** Add configurable strict vs. lenient parsing modes for evaluation robustness (Version 1 nice-to-have, not a frozen requirement).

**Assumptions.** The digit-formatting convention documented in §3.6 (from the paper's single worked example) is assumed accurate but not yet source-verified.

**TODO.**
- TODO (Verify from TRACE implementation) — exact parsing/detokenization code in the TRACE repository's inference/evaluation scripts, to ensure the Event Parser is a precise inverse of the true serialization scheme (not just the paper's illustrative example).
- TODO (Verify from TRACE implementation) — error-handling behavior for malformed generations in TRACE's own evaluation pipeline (useful reference even if TinyTRACE ultimately implements its own, more defensive, parser).

---

# PART IV — TRAINING PIPELINE

## 4.1 Overview: Two-Stage Training, Mirrored from TRACE

As established in §2.8, TRACE trains in two stages, and TinyTRACE mirrors this structure. This part specifies the complete forward pass, teacher forcing scheme, loss computation, optimization procedure, and pseudo-algorithms for both stages, along with the conceptual token/tensor flow through training.

```
Stage 0 (OPEN QUESTION — see §2.8, §6.9)
   Optional: pretrain LCEM's text head / embeddings on general text
   for baseline linguistic competence, since LCEM (unlike Mistral-7B)
   has no pretrained knowledge by default.
        │
        ▼
Stage 1 — Task Module Initialization
   Trainable: Visual Token Builder (compression), Time Encoder/Head,
              Score Encoder/Head
   Frozen:    MobileCLIP, LCEM backbone
   Data:      Image/video captioning data (general alignment) +
              VTG-IT-style data (task encoder/head alignment)
        │
        ▼
Stage 2 — Instruction Tuning
   Trainable: LCEM backbone + all task modules (Visual Token Builder,
              Time/Score Encoder/Head, Text Head)
   Frozen:    MobileCLIP only
   Data:      VTG instruction-tuning data + caption-quality-preserving
              data + VQA-style reasoning data
```

## 4.2 Complete Forward Pass (Training Mode)

Given a training example consisting of a video `V`, an instruction `I`, and ground-truth events `{e_1, ..., e_K}`:

1. **Frame sampling** produces `T` frames + timestamps (§3.3).
2. **MobileCLIP** encodes each frame → per-frame patch embeddings (§3.4).
3. **Visual Token Builder** compresses each frame's embeddings → 8 (TRACE value; TBD for TinyTRACE) visual tokens/frame (§3.5).
4. **Time Encoder** encodes each frame's timestamp → 6 time tokens/frame (§3.6), fused with the frame's visual tokens.
5. **Prompt Builder** assembles the full training sequence: `[F][I][t_1][s_1][c_1]...[t_K][s_K][c_K]`, using the **ground-truth** event tokens (not model-generated ones) for the event segment — this is teacher forcing (§4.3).
6. **LCEM** performs a single non-autoregressive (i.e., fully parallel, causally-masked) forward pass over the entire assembled sequence, producing per-position hidden states.
7. **Task heads** (Time Head, Score Head, Text Head) compute logits at every position, using the *ground-truth* task-segment membership of each position (not the Adaptive Head Switching *inference-time* state machine, which is unnecessary during training since the correct head for every position is already known from the ground-truth data layout) — TODO (Verify from TRACE implementation) whether training literally computes all three heads at every position and masks/select post-hoc, or whether it only ever computes the relevant head per position via indexed gather, for efficiency.
8. **Loss** is computed per §4.4 below.

## 4.3 Teacher Forcing

Teacher forcing means that during training, the input to the decoder at every position is the *ground-truth* previous token (from the training example), not a token the model itself generated. This is what allows step 6 above to be a single parallel forward pass rather than a slow autoregressive loop — the entire target sequence is already known, so all positions can be processed simultaneously with a causal attention mask ensuring position `i` only attends to positions `≤ i`.

**Consequence for the three-head design.** Because ground-truth event structure (which sub-segment each position belongs to: time/score/text) is known in advance during training, the *training-time* equivalent of Adaptive Head Switching is trivial index-based head selection rather than a `<sync>`-token-triggered runtime state machine. The state-machine logic in §2.6/§3.9 is specifically an **inference-time** mechanism, needed only because at inference time the model does not have ground-truth structure to consult — it must infer head-switch points from its own generated `<sync>` tokens.

## 4.4 Loss Computation

Per §2.7, the loss is a sum of per-task cross-entropy terms over teacher-forced positions, excluding prompt/visual/instruction positions (which are not generation targets):

```
L_total = L_time + L_score + L_text

L_time  = − Σ_{i ∈ time positions}  log P_TimeHead(x_i | x_<i)
L_score = − Σ_{i ∈ score positions} log P_ScoreHead(x_i | x_<i)
L_text  = − Σ_{i ∈ text positions}  log P_TextHead(x_i | x_<i)
```

TODO (Verify from TRACE implementation) — relative weighting between `L_time`, `L_score`, `L_text` (unweighted sum is the current default assumption, per §2.7).

Prompt-masking (excluding visual/instruction tokens, and possibly the `<sync>` placeholder tokens for missing sub-tasks, from the loss) follows standard instruction-tuning practice; TODO (Verify from TRACE implementation) — the precise masking rule, particularly whether `<sync>`-as-placeholder tokens (§3.7) contribute to the loss or are also masked out.

## 4.5 Optimization

| Hyperparameter | TRACE Stage 1 | TRACE Stage 2 | TinyTRACE (proposed default; TODO tune) |
|---|---|---|---|
| Learning rate | 1e-3 | 5e-6 | TODO — likely higher than TRACE Stage 2 given smaller model / from-scratch training; needs empirical tuning |
| LR Scheduler | Cosine | Cosine | Cosine (kept, standard choice) |
| Batch size | 128 | 128 | TODO — likely smaller given reduced compute budget |
| Epochs | 1 | 2 | TODO — likely more epochs needed given no pretrained backbone |
| Frame count (T) | 128 | 128 (128 clips, 1 frame/clip) | TODO — likely reduced, see §3.3 |
| Model max length | 4096 | 4096 | TODO — determined by LCEM's context window, itself set by the parameter budget (Part VI) |

TODO (Verify from TRACE implementation) — optimizer choice (AdamW is the standard default for this model family, but exact betas/epsilon/weight-decay are unconfirmed), gradient clipping value, and warmup schedule (a bare "Cosine" scheduler entry in Table 7 does not by itself specify whether a warmup phase precedes the cosine decay).

## 4.6 Training Algorithm (Pseudocode)

```
# Stage 1 — Task Module Initialization
freeze(MobileCLIP)
freeze(LCEM)  # backbone untouched in Stage 1
initialize(VisualTokenBuilder, TimeEncoderHead, ScoreEncoderHead)

for epoch in range(STAGE1_EPOCHS):
    for batch in Stage1Dataloader:
        frames, timestamps = sample_frames(batch.video, T)
        clip_tokens = MobileCLIP(frames)                       # frozen
        vis_tokens  = VisualTokenBuilder(clip_tokens)           # trainable
        time_tokens = TimeEncoder(timestamps)                   # trainable
        seq         = PromptBuilder(vis_tokens, time_tokens,
                                     batch.instruction, batch.gt_events)
        hidden      = LCEM(seq)                                 # frozen, forward only
        logits      = compute_head_logits(hidden, seq.task_mask)
        loss        = cross_entropy(logits, seq.targets, mask=seq.task_mask)
        loss.backward()                                          # gradients flow only into
        optimizer.step()                                         # trainable modules above
        optimizer.zero_grad()

# Stage 2 — Instruction Tuning
freeze(MobileCLIP)                                                # remains frozen
unfreeze(LCEM, VisualTokenBuilder, TimeEncoderHead,
         ScoreEncoderHead, TextHead)

for epoch in range(STAGE2_EPOCHS):
    for batch in Stage2Dataloader:
        frames, timestamps = sample_frames_random_within_clip(batch.video, T)
        clip_tokens = MobileCLIP(frames)                       # frozen
        vis_tokens  = VisualTokenBuilder(clip_tokens)           # trainable
        time_tokens = TimeEncoder(timestamps)                   # trainable
        seq         = PromptBuilder(vis_tokens, time_tokens,
                                     batch.instruction, batch.gt_events)
        hidden      = LCEM(seq)                                 # trainable now
        logits      = compute_head_logits(hidden, seq.task_mask)
        loss        = cross_entropy(logits, seq.targets, mask=seq.task_mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**TODO (Verify from TRACE implementation)** — this pseudocode reflects the paper's prose description of what is trainable/frozen at each stage; it has not been checked against the actual training script's parameter-freezing logic (`requires_grad` settings, optimizer parameter groups) and must be verified before implementation.

## 4.7 Data Flow / Token Flow / Tensor Flow (Conceptual)

```
Video ──► [T, H, W, 3] frames
      ──► MobileCLIP ──► [T, N_patches, d_vision]
      ──► Compression ──► [T, N_compressed, d_model]
      ──► + Time tokens [T, 6, d_model] (concatenated per-frame)
      ──► Flatten frames ──► [T × (N_compressed + 6), d_model]   ("F" segment)
      ──► + Instruction tokens [L_instr, d_model]                 ("I" segment)
      ──► + Event tokens (ground truth, teacher-forced)
            [Σ_k (len(t_k) + len(s_k) + len(c_k)), d_model]       ("e_1...e_K" segment)
      ══► Full sequence [L_total, d_model], L_total ≤ LCEM max length
      ──► LCEM (causal self-attention stack) ──► [L_total, d_model] hidden states
      ──► Task-specific heads (indexed by position's task-mask) ──► logits
      ──► Cross-entropy loss vs. ground-truth next-token targets
```

**TODO (Verify from TRACE implementation)** — exact handling of `L_total` when it exceeds the model's max length for very long videos or many-event examples (truncation strategy, if any).

---

# PART V — INFERENCE PIPELINE

## 5.1 Stage-by-Stage Walkthrough

```
Video
  │
  ▼
Frames (Frame Sampling, §3.3 — inference-time strategy TODO, likely uniform)
  │
  ▼
Embeddings (MobileCLIP §3.4 → Visual Token Builder §3.5 → Time Encoder §3.6,
            fused into per-frame visual+time tokens; Instruction tokenized
            and appended via Prompt Builder §3.7)
  │
  ▼
Decoder (LCEM, §3.8 — processes the fixed [F][I] prefix once; see §5.3 on
          KV-caching for the autoregressive portion)
  │
  ▼
Autoregressive Loop (§5.2 below — token-by-token generation, driven by
                       Adaptive Head Switching, §3.9)
  │
  ▼
Event Tokens (flat stream of digits / <sep> / <sync> / text subwords)
  │
  ▼
Parser (Event Parser, §3.10)
  │
  ▼
Final Event(s): [{timestamp(s), score, caption}, ...]
```

## 5.2 The Autoregressive Loop in Detail

Unlike training (§4.2, a single parallel forward pass over a fully-known target sequence), inference must generate the event-token segment one token at a time, since each new token depends on tokens the model itself has just produced.

```
state = TIME
generated = []
context = [F][I]                      # fixed prefix, computed once

while not done(generated):
    hidden = LCEM(context + generated)          # or: LCEM.step(...) if using KV-cache
    head   = select_head(state)                  # TimeHead / ScoreHead / TextHead
    logits = head(hidden[-1])                    # logits for the next token only
    next_token = sample_or_argmax(logits)
    generated.append(next_token)

    if next_token == SYNC:
        state = advance(state)                   # TIME→SCORE→TEXT→TIME...

    done = check_termination(generated)          # TODO: exact termination condition
```

**Explanation of every stage:**

1. **Video → Frames.** Identical mechanism to training's Frame Sampling module (§3.3), though TODO (Verify from TRACE implementation) confirms whether inference uses the *uniform* strategy (Stage 1 style) or the *random-within-clip* strategy (Stage 2 style) — intuitively uniform sampling is more natural for a deterministic inference pipeline, but this must be verified rather than assumed, since a mismatch between training-time and inference-time sampling distributions can measurably affect model behavior.
2. **Frames → Embeddings.** Identical mechanism to training (§3.4–§3.6), producing the fixed `[F][I]` prefix. Because this prefix does not change across generation steps, it can be computed exactly once per inference call rather than being recomputed at every autoregressive step (see KV-caching note, §5.3).
3. **Embeddings → Decoder.** LCEM processes the growing sequence causally. In a naive (non-cached) implementation, the entire `context + generated` sequence is reprocessed at every step; in an efficient implementation, only the newest token needs a fresh forward pass, reusing cached key/value tensors from all previous positions.
4. **Decoder → Autoregressive Loop.** As detailed in the pseudocode above — this is where Adaptive Head Switching (§3.9) actively participates in generation, unlike in training where it is not needed.
5. **Autoregressive Loop → Event Tokens.** The loop terminates according to some condition — TODO (Verify from TRACE implementation): candidates include a maximum total generated-token budget, a maximum event count, or a dedicated end-of-sequence signal distinct from the per-event `<sync>` tokens. The paper does not describe an explicit `<END>`-style token distinct from `<sync>`; the pipeline diagram supplied in the project's frozen design decisions document includes an `<END>` marker after the final event's `<sync>`, but this is currently an assumption inherited from the project brief, not a paper-verified fact.
6. **Event Tokens → Parser.** Identical mechanism to §3.10.

## 5.3 KV-Caching (Efficiency Note, Not a Frozen Architectural Requirement)

Standard practice for autoregressive Transformer decoders is to cache the key/value projections computed at each layer for all previously-processed positions, so that each new generation step only requires computing query/key/value for the single newest token and attending it against the cached keys/values of all prior positions, rather than recomputing the entire sequence's attention from scratch. This is a pure inference-efficiency technique with no effect on the model's mathematical behavior (assuming numerically identical computation), and is strongly recommended for LCEM given the project's lightweight-deployment motivation (§1.1), even though it is not something the TRACE paper itself needs to discuss (it is a standard, implementation-level optimization applicable to any Transformer decoder, TRACE's or LCEM's).

## 5.4 Batch Inference Considerations

TODO (Verify from TRACE implementation) — whether TRACE's released inference code supports batched multi-video inference during generation (complicated by the fact that different videos in a batch will reach `<sync>` tokens, and therefore head-switch points, at different generation steps, requiring per-sequence state tracking within a batch). This is an implementation detail relevant to TinyTRACE's own inference-serving code but does not affect the model's architecture.

---

# PART VI — DESIGN DECISIONS: RATIONALE AND TRADE-OFF ANALYSIS

## 6.1 Why MobileCLIP?

MobileCLIP was chosen over alternatives for one primary reason: **philosophical continuity with CLIP**. TRACE's vision tower is a contrastively-pretrained image-text encoder; its role in the pipeline is not to model temporal/motion information directly (that is the job of the Time Encoder and the LLM's sequence modeling over multiple frames) but to produce per-frame embeddings that are already meaningfully positioned relative to natural language. MobileCLIP is trained with the same contrastive image-text objective family as CLIP, at a fraction of the parameter count, making the CLIP→MobileCLIP swap a comparatively "clean" single-variable substitution (encoder capacity/efficiency) rather than a substitution that also changes the *nature* of what is being modeled.

The alternative seriously considered — **Tiny VideoMAE** — was rejected specifically because VideoMAE's self-supervised masked-autoencoding objective is not vision-language-aligned at all; adopting it would have required bolting on an entirely separate vision-language alignment mechanism (since VideoMAE embeddings are not natively positioned relative to text), which would have introduced a second, orthogonal research variable (the vision-language alignment scheme) into what is meant to be a controlled comparison against TRACE. It would also have changed the pipeline's conceptual framing from "per-frame image-language encoding, temporally fused downstream" (TRACE's actual design) to "native video encoding," which is a different, not-strictly-lighter-weight redesign of the vision stage rather than a like-for-like lightweight substitution.

## 6.2 Why a Decoder Transformer for LCEM?

TRACE uses a decoder-only Transformer (Mistral-7B) as its reasoning backbone, and the causal event modeling framework itself (Eq. 2, §2.2) is expressed as a factorization that maps naturally onto autoregressive next-token prediction over an interleaved sequence (§2.2, §3.7). Preserving the decoder-Transformer paradigm for LCEM is therefore not an arbitrary architectural preference — it is what makes LCEM a faithful "lightweight TRACE" rather than a different framework that happens to also output timestamps and captions. A decoder Transformer, at any scale, natively supports: (a) causal self-attention over an arbitrarily-interleaved multi-modal token sequence, (b) parallel teacher-forced training via a single masked forward pass, and (c) the `<sync>`-triggered adaptive head-switching mechanism, all without requiring any framework-level modification as the model is scaled down. This continuity is the central justification for Freeze 7.

## 6.3 Why Not TinyLlama or SmolLM?

TinyLlama and SmolLM are both attractive *in isolation* as "lightweight LLM" choices — they are small, pretrained, decoder-only Transformers with demonstrated general-purpose language competence. They were nonetheless explicitly rejected as the reasoning module for two related reasons:

1. **Architectural mismatch with the three-head design.** TinyLlama and SmolLM are general-purpose chat/completion models with a single unified language-modeling head over a large (~32K+) subword vocabulary. Adapting either to the causal event modeling framework would require bolting on two *additional* heads (time, score) with their own small closed vocabularies and, per §3.9, requires their embeddings to be initialized from the base model's own embedding table. This is achievable, but it means the "lightweight decoder" is not purpose-built for the task — it is a general chat model retrofitted with task-specific heads, which shifts the thesis's contribution away from "a decoder architecture built for causal event modeling" and toward "which pretrained small chat model adapts best to this task," a narrower and less architecturally interesting research question, and one that risks conflating LCEM's actual novel contribution with whatever idiosyncrasies the chosen off-the-shelf checkpoint happens to bring with it.
2. **Loss of experimental control.** Both TinyLlama and SmolLM carry substantial pretraining history (specific tokenizers, specific data mixtures, specific architectural micro-choices) that are outside this project's control and not documented in a way that permits a clean, reproducible mapping to "what changed relative to TRACE." Using a purpose-built LCEM, trained (largely or fully) from scratch under the project's own control, keeps the CLIP→MobileCLIP and Mistral→LCEM substitutions as the *only* two independent variables relative to TRACE, preserving the controlled-comparison framing central to the research objective (§1.2).

This is a deliberate scope decision, not a claim that TinyLlama/SmolLM-based approaches are inferior in absolute terms — a future TinyTRACE version could legitimately explore that direction as a distinct ablation, but it is out of scope for Version 1 (Freeze 7).

## 6.4 Why Not GRU or LSTM?

Recurrent architectures (GRU, LSTM) were considered and rejected for two reasons, one theoretical and one practical:

- **Theoretical mismatch.** TRACE's Eq. 2 factorization is expressed and implemented as a Transformer-style interleaved-token autoregressive sequence with full (causally-masked) self-attention over the *entire* preceding context — including all `T` frames' visual tokens simultaneously — at every generation step. A recurrent model instead compresses all preceding context into a fixed-size hidden state that is updated sequentially, token by token. For a sequence that begins with potentially hundreds of visual tokens (frame count × tokens/frame) *before* any event tokens even begin, forcing that entire visual context through a fixed-size recurrent hidden state — updated one token at a time across potentially over a thousand visual tokens before generation even starts — risks severe information bottlenecking, exactly the kind of long-range dependency problem attention mechanisms were introduced to solve. This is a much more severe version of the classical vanishing-context problem that motivated the shift from RNNs to Transformers in NLP generally, and is particularly acute here given the visual segment is typically the single largest component of the sequence.
- **Practical mismatch with the causal event modeling framework's task-switching structure.** The 3.9's adaptive head-switching mechanism assumes a Transformer-style architecture where different heads read from a common hidden-state representation and where attention allows any position to directly reference any earlier position (e.g., a caption token attending directly back to the specific visual tokens of the frame it is describing). A recurrent model's fixed-size state makes this kind of direct, content-addressable long-range reference structurally harder to achieve.

## 6.5 Why Preserve TRACE's Mathematics Exactly?

The mathematical formulation (Eq. 2) is the theoretical contribution TRACE makes to the field — it is the reason TRACE outperforms prior video-LLMs on VTG tasks (per the paper's own ablation studies, notably Table 3's "w/o causal event modeling" row, which shows a clear performance drop when the structured factorization is abandoned in favor of plain natural-language generation). If TinyTRACE changed this formulation, it would cease to be testing the hypothesis in §1.2 ("can *this* framework be implemented lightweight?") and would instead be testing an unrelated hypothesis about a different framework. Preserving the mathematics exactly is therefore not architectural conservatism for its own sake — it is what makes the entire project a valid, controlled scientific comparison rather than an unrelated video-captioning system that happens to share superficial vocabulary with TRACE.

## 6.6 Parameter Analysis

| Component | TRACE (approx., from public knowledge of the cited checkpoints) | TinyTRACE (target — TBD, budget-dependent) |
|---|---|---|
| Vision Encoder | CLIP ViT-L/14-336 — ~300M parameters | MobileCLIP (variant TBD) — typically single-digit to low tens of millions of parameters, depending on variant |
| Reasoning Backbone | Mistral-7B-v0.2 — ~7.3B parameters | LCEM — target TBD; Part VI proposes this be fixed as an explicit thesis deliverable (e.g., a stated target such as "under 50M parameters" or similar), chosen based on available compute, not copied from any external reference |
| Task Heads (time/score/text embeddings+heads) | Small relative to the 7B backbone — TODO (Verify from TRACE implementation) for exact parameter count | Proportionally small relative to LCEM, mirroring TRACE's design ratio |
| Compression Module | TODO (Verify from TRACE implementation) | TODO — depends on compression architecture once verified from the released TRACE codepath (§3.5) |

**Note on precision.** This table intentionally avoids stating a specific target parameter count for LCEM as a "frozen" number in this document, since that number is a function of available compute/training budget (a project-management decision) rather than a purely architectural one; the eventual target should be informed by the parameter counts recovered from the audited TRACE implementation.

## 6.7 Complexity Analysis

Self-attention's computational cost scales as `O(L² · d)` in sequence length `L` and hidden dimension `d`, per layer. Because the visual segment (`T` frames × tokens/frame) typically dominates total sequence length `L`, three levers directly control LCEM's inference/training cost, in descending order of usual impact:

1. **Frame count `T`.** Directly and linearly reduces the number of visual tokens, and thus quadratically reduces the dominant `O(L²)` attention cost. This is the most impactful lever available and is the first parameter this document recommends tuning downward relative to TRACE's `T=128` (§3.3).
2. **Tokens-per-frame after compression (§3.5).** A more aggressive compression ratio (fewer than TRACE's 8 tokens/frame) directly reduces `L` with the same quadratic effect, but at greater risk of losing fine-grained visual information — this is a genuine accuracy/efficiency trade-off requiring empirical tuning, unlike lever 1, which mostly trades off *temporal* granularity (missing short events) rather than *per-frame* visual detail.
3. **Hidden dimension `d` and layer count `N`.** Standard model-scaling levers; reducing these lowers both the per-layer attention cost and the feed-forward cost roughly linearly (per layer) and reduces the number of layers linearly, but does not address the underlying `O(L²)` scaling in sequence length the way levers 1–2 do — so for a video-heavy pipeline like this one, sequence-length reduction (levers 1–2) should generally be prioritized over indiscriminate width/depth reduction to avoid degrading the model's *capacity* more than necessary to hit a given compute budget.

## 6.8 Proportional Scaling Reference (Pending Full TRACE Verification)

Once Mistral-7B-v0.2's exact architectural hyperparameters are confirmed (TODO, §3.8), Codex should compute a proportional scaling table (e.g., LCEM hidden size as a fraction of Mistral's hidden size, LCEM layer count as a fraction of Mistral's layer count) as a *starting point* for LCEM's hyperparameter search — not a final answer, since naive linear down-scaling of a 7B model's hyperparameters does not, in general, produce a well-balanced small model (small models typically benefit from different depth/width ratios than simply dividing a large model's dimensions by a constant factor). This is flagged explicitly so Codex does not treat "Mistral's hidden size ÷ 100" as an adequate substitute for an actual small-model architecture search.

## 6.9 Expected Trade-offs

| Trade-off | Direction of Effect |
|---|---|
| Smaller vision encoder (MobileCLIP vs. CLIP ViT-L) | Reduced fine-grained visual discrimination; likely most noticeable on tasks requiring precise visual detail (e.g., distinguishing visually similar consecutive events) |
| Smaller reasoning backbone (LCEM vs. Mistral-7B) | Reduced caption fluency/world-knowledge (text head has no pretrained linguistic prior, absent a Stage 0 pretraining step, §2.8); likely the single largest source of performance gap vs. TRACE |
| Reduced frame count `T` | Coarser temporal resolution; risk of missing short-duration events entirely |
| Reduced context window | Caps maximum representable video length / event count per generation |
| Preserved event representation & mathematics | No degradation from this factor — these are exactly reproduced, not approximated |

The overarching expectation, consistent with the project's research objective (§1.2), is that TinyTRACE will underperform TRACE in absolute benchmark terms (this is expected and not a failure condition — TRACE has ~150× more parameters in its backbone alone) while still demonstrating that the *architectural pattern* of causal event modeling — separated task heads, task-interleaved sequencing, adaptive head-switching — remains coherent, trainable, and meaningfully better than a naive "shared head" baseline (mirroring TRACE's own internal ablation in Table 3) even at drastically reduced scale. That internal ablation — TinyTRACE-with-causal-event-modeling vs. TinyTRACE-without-it (i.e., a shared-head baseline at the *same* small scale) — is the most scientifically meaningful comparison this thesis can make, arguably more meaningful than a direct TinyTRACE-vs-TRACE benchmark comparison, since it isolates the framework's contribution from the confound of raw model scale.

---

# PART VII — REPOSITORY AUDIT AND OPEN IMPLEMENTATION ITEMS

## 7.1 Purpose and Ground Rules

This part records the current repository-audit status of this document against the actual TRACE reference implementation at `https://github.com/gyxxyg/TRACE`.

**Ground rules:**

1. This audit answers "how exactly does TRACE implement each module?" — it does **not** re-litigate "what architecture should TinyTRACE use?" The frozen decisions in §1.7 stand unless a specific verified TRACE fact *proves* one of them impossible or fundamentally wrong.
2. Where a detail cannot be verified from source, it must remain marked **TODO (Verify from TRACE implementation)** rather than being replaced with a guess.
3. Any resolved TODO should be reflected consistently in the corresponding module text, diagrams, tensor/token-flow descriptions, and checklist entries.

## 7.2 Consolidated TODO Inventory

The following is every TODO marker raised throughout this document, consolidated into one checklist, grouped by module/topic, for tracking purposes.

### 7.2.1 Frame Sampling
- [ ] Exact decoding backend/library used (decord / OpenCV / PyAV) and preprocessing (resize/crop/normalize) applied before the vision encoder.
- [ ] Whether frame sampling differs between the training code path and the inference/generation code path.

### 7.2.2 Visual Encoding & Compression
- [ ] Precise architecture of Slot-Based Compression (number of attention iterations, slot initialization scheme, per-frame vs. shared slots).
- [ ] Whether compression has any cross-frame/temporal component, or is strictly per-frame.
- [ ] Exact CLIP preprocessing pipeline (crop strategy, normalization constants) — as a reference for replicating with MobileCLIP.
- [ ] Confirm CLIP (and by extension MobileCLIP in TinyTRACE) is frozen throughout all training stages, via `requires_grad` inspection, not just Table 7's prose.

### 7.2.3 Time / Score Encoding
- [ ] Exact architecture of the Time Encoder / Time Head and Score Encoder / Head (beyond "share the same architecture").
- [ ] Exact embedding-initialization procedure from LLM token embeddings for the 13-token time/score vocabulary.
- [ ] Confirm fixed-width digit-formatting convention (4 whole + 1 dot + 1 fractional for time; 1 whole + 1 dot + 1 fractional for score) directly from tokenizer/config source, not solely the paper's single worked example.
- [ ] Precise mechanism for concatenating 8 visual tokens + 6 time tokens per frame (simple concatenation vs. learned fusion).

### 7.2.4 Prompt Construction
- [ ] Exact special/reserved token registration in the tokenizer configuration (`<sync>`, `<sep>`, `<video>`, etc.).
- [ ] Precise delimiter mechanism between the visual segment `F` and instruction segment `I` (is there a literal `<video>` boundary token?).
- [ ] Full inventory of instruction templates per task type (dense captioning, moment retrieval, highlight detection, summarization, general QA).

### 7.2.5 Decoder / LCEM Reference Architecture
- [ ] Mistral-7B-v0.2's exact architectural hyperparameters (layers, hidden size, attention heads, KV-heads, FFN intermediate size, RoPE base) as a scaling reference.
- [ ] Confirm Mistral-7B-v0.2's exact activation/gating function (assumed SwiGLU) and normalization scheme (assumed RMSNorm) — used unmodified or adapted by TRACE.
- [ ] Confirm no auxiliary cross-attention layers exist anywhere in the released code (backbone or compression module).

### 7.2.6 Adaptive Head Switching
- [ ] Disambiguate implementation approach: (a) generation-loop wrapper, (b) custom `LogitsProcessor`, or (c) forward-pass computing all heads with masking.
- [ ] Exact end-of-generation / end-of-event-sequence condition (max events, explicit terminal token distinct from `<sync>`, or max-length truncation).
- [ ] Behavior on malformed generations (e.g., a head that never emits `<sync>`) — timeout/fallback logic, if any.

### 7.2.7 Loss / Training
- [ ] Relative loss weighting between time/score/text task streams (unweighted sum is current default assumption).
- [ ] Precise prompt-masking rule, including whether `<sync>`-only placeholder segments contribute to the loss.
- [ ] Optimizer choice and exact hyperparameters (AdamW betas/epsilon/weight-decay), gradient clipping, warmup schedule details beyond "Cosine."
- [ ] Whether training computes all three heads at every position (masked post-hoc) or only the relevant head per position (indexed gather), for efficiency.
- [ ] Whether an LCEM-specific Stage 0 (general text pretraining) is warranted, given LCEM lacks Mistral's pretrained knowledge (research decision, not a pure fact-finding item — Codex should surface data/compute trade-offs for the thesis author to decide, not decide unilaterally).

### 7.2.8 Inference
- [ ] Confirm inference-time frame-sampling strategy (uniform vs. random-within-clip).
- [ ] Confirm existence/absence of a distinct end-of-sequence signal beyond per-event `<sync>`.
- [ ] Confirm whether released code supports batched multi-video inference given per-sequence head-switching state.

### 7.2.9 Event Parsing
- [ ] Exact parsing/detokenization code in TRACE's inference/evaluation scripts (precise inverse of the true serialization scheme).
- [ ] Error-handling behavior for malformed generations in TRACE's own evaluation pipeline.

### 7.2.10 Data / Ordering
- [ ] Whether any post-hoc sorting/validation enforces chronological inter-event ordering, or whether it is assumed already true of raw annotations.

## 7.3 Current Reverse-Engineering Audit (2026-07-12)

This subsection records verified facts recovered from the checked-out TRACE repository in this workspace. Each finding cites the local source file and exact line(s) that support it. Any item not listed here remains open and should continue to carry its TODO marker elsewhere in this document.

### 7.3.1 Repository Entry Points Identified

- **Model definition files.**
  - `trace/model/language_model/trace_mistral.py` defines the TRACE-specific Mistral wrapper, extra heads, loss computation, and generation-time head switching (lines 73-346).
  - `trace/model/trace_arch.py` defines multimodal input preparation, time/score token injection, visual-token construction, and temporal aggregation (lines 29-520).
  - `trace/model/multimodal_encoder/*.py` define the CLIP vision tower plus time/score/sync token encoders.
  - `trace/model/multimodal_projector/builder.py` defines the visual compression / projector variants, including slot-based variants.
- **Training scripts.**
  - `trace/train_mt.py` is the main training/data pipeline entry point, including sampling, prompt preprocessing, and parameter-freezing logic (notably lines 796-845 and 1014-1175).
  - `scripts/train/*.sh` provide stage/run launch examples.
- **Inference / generation scripts.**
  - `scripts/inference/inference.py` is the clearest released generation example, including prompt suffixing with `<sync>`, initialization of the active head state, and post-generation parsing (lines 15-128).
- **Data / prompt artifacts.**
  - Prompt templates live in `trace/prompts/*.txt`.

### 7.3.2 Verified Findings

- **Frame decoding backend and sampling are now partially verified.**
  - Standard videos use `decord.VideoReader`; `.webm` uses `moviepy.VideoFileClip`; `.gif` uses `imageio.get_reader` in `trace/mm_utils.py` lines 400-437.
  - The released `process_video` helper supports `uniform`, `fps`, and `rand` frame sampling, with `uniform` and `rand` implemented in the nested `frame_sample()` function at `trace/mm_utils.py` lines 379-398.
  - Training calls `process_video(..., sample_scheme=sample_scheme)` in `trace/train_mt.py` line 806, while the public inference example calls `process_video(..., num_frames=64)` and relies on the default `sample_scheme='uniform'` in `scripts/inference/inference.py` line 34 together with `trace/mm_utils.py` line 379.
- **Chronological event ordering is not post-sorted in the visible training path.**
  - The dataset path maps annotated target times to the nearest sampled frame timestamps via `min(... abs(x[0] - target) ...)` but does not perform any explicit re-sorting of events afterward (`trace/train_mt.py` lines 838-843).
- **The CLIP vision tower is explicitly frozen.**
  - `self.vision_tower.requires_grad_(False)` appears in `trace/model/multimodal_encoder/clip_encoder.py` lines 23-29.
- **Time and score vocabularies are 13-token custom vocabularies.**
  - Both tokenizers define `<sync>`, `<sep>`, digits `0-9`, and `.` in `trace/model/multimodal_encoder/time_encoder.py` lines 80-108 and `trace/model/multimodal_encoder/score_encoder.py` lines 83-116.
- **Digit serialization format is now verified from code.**
  - Time values are formatted with `format(t, '0>6.1f')`, i.e. fixed-width `0000.0`-style strings before tokenization, in `trace/model/multimodal_encoder/time_encoder.py` lines 52-65.
  - Score values are formatted with `format(s, '0>3.1f')`, i.e. fixed-width `0.0`-style strings, in `trace/model/multimodal_encoder/score_encoder.py` lines 52-68.
- **Embedding initialization from the base LLM is not active in this repo snapshot.**
  - The code path that would copy pretrained text embeddings into the time/score towers is commented out in both `trace/model/multimodal_encoder/time_encoder.py` lines 23-40 and `trace/model/multimodal_encoder/score_encoder.py` lines 23-39.
  - `initialize_time_modules()` / `initialize_score_modules()` construct fresh towers with `build_time_tower(None, None, 4096)` and `build_score_tower(None, None, 4096)` in `trace/model/trace_arch.py` lines 38-40, and the training script invokes those initializers without passing pretrained tokenizer/embedding weights in `trace/train_mt.py` lines 1145-1147.
  - This should be interpreted as a finding about the released repo snapshot, not as proof that the frozen TinyTRACE architecture should abandon the paper-level assumption of TRACE-style embedding reuse.
- **Per-frame visual tokens and per-frame time tokens are concatenated directly, not fused by a separate learned combiner.**
  - `frames_features = torch.cat([frames_features, time_features], dim=2)` appears in `trace/model/trace_arch.py` lines 253-258.
- **Prompt special tokens are registered as extended multimodal tokens rather than plain textual delimiters.**
  - The reserved multimodal token inventory is defined in `trace/constants.py` lines 47-56, including negative placeholder indices for `VIDEO`, `TIME`, `SCORE`, and `SYNC`.
  - Prompt tokenization replaces literal strings like `<video>`, `<time>`, `<score>`, and `<sync>` with those modal indices in `trace/mm_utils.py` lines 523-545.
- **The released inference path uses a literal `<video>` multimodal token as the visual/instruction delimiter.**
  - The inference prompt is built as `default_mm_token + "\n" + question`, where `default_mm_token` is `DEFAULT_MMODAL_TOKEN["VIDEO"]`, then tokenized by `tokenizer_MMODAL_token_all(...)` in `scripts/inference/inference.py` lines 48-56.
- **Released instruction templates are now partially inventoried.**
  - Dense video captioning templates: `trace/prompts/dvc.txt`, `trace/prompts/dvc-anet.txt`, `trace/prompts/dvc-anet-ft.txt`.
  - Moment retrieval template: `trace/prompts/mr.txt`.
  - Highlight detection template: `trace/prompts/vhd.txt`.
  - These strings are visible in `trace/prompts/*.txt` and summarized by `nl` output lines 1-5.
- **Adaptive head switching is implemented as a generation-loop wrapper around Hugging Face generation, not as a custom `LogitsProcessor`.**
  - The model tracks a per-sequence `heads` state passed through `prepare_inputs_for_generation()` in `trace/model/language_model/trace_mistral.py` lines 317-345.
  - When the latest generated token matches `self.swap_tokens`, the next active head is updated in Python (`trace/model/language_model/trace_mistral.py` lines 336-344).
  - During forward passes with `heads` present, logits from the inactive heads are masked to `-inf` after concatenation (`trace/model/language_model/trace_mistral.py` lines 244-252).
  - The released inference example bootstraps the state machine by appending a textual `<sync>` to the prompt and seeding generation with `heads = [1]` before normal cyclic switching takes over (`scripts/inference/inference.py` lines 45 and 53-79).
- **Training computes all three heads and sums three cross-entropy losses with equal weight.**
  - `logits_list = [logits, time_logits, score_logits]` and `loss = sum(loss)` appear in `trace/model/language_model/trace_mistral.py` lines 204-237.
- **Training uses masked labels instead of selecting only one head per position before forward computation.**
  - The code builds `new_labels`, `new_time_labels`, and `new_score_labels` with `IGNORE_INDEX` masks in `trace/model/trace_arch.py` lines 430-489.
  - Each head is then trained against its own masked label tensor in `trace/model/language_model/trace_mistral.py` lines 204-237.
- **Prompt masking is partially verified.**
  - Visual-token slots inserted in place of `<video>` are assigned `IGNORE_INDEX` in `trace/model/trace_arch.py` line 432.
  - Negative multimodal placeholders are masked out via `cur_new_labels[cur_new_input_ids < 0] = IGNORE_INDEX` in line 434.
  - `<sync>` positions in the text stream are not masked; they are trained against `self.vocab_size` as the sync class in line 435.
- **Trainable/frozen-module behavior is partially verified from the training script.**
  - The Mistral backbone is frozen when `freeze_backbone` is enabled (`trace/train_mt.py` lines 1014-1018).
  - The vision projector may be selectively unfrozen via `tune_mm_mlp_adapter` and re-frozen via `freeze_mm_mlp_adapter` (`trace/train_mt.py` lines 1102-1111).
  - Time/score/sync towers are toggled by `tune_mm_embed_head`, and LM embeddings / LM heads by `tune_lm_embed_head` (`trace/train_mt.py` lines 1153-1168).
- **The released inference parser does not use a distinct verified end token.**
  - The example generation call uses `max_new_tokens=1024` with no explicit TRACE-specific EOS token beyond the underlying tokenizer stop behavior (`scripts/inference/inference.py` lines 66-79).
  - The parser simply consumes the generated token stream and splits by sync/separator IDs (`scripts/inference/inference.py` lines 81-128).
- **The visual compression module is not a single fixed design in the released code.**
  - `build_vision_projector()` can instantiate `linear`, `mlp2x_gelu`, convolution/pooling connectors, and three slot-based variants (`trace/model/multimodal_projector/builder.py` lines 94-120).
  - Slot-based implementations visible in this repo are `SlotPool`, `SpatialSlotPool(num_slots=8)`, and `SpatialTimeSlotPool(num_spatial_slots=8, num_time_slots=1)` (`trace/model/multimodal_projector/builder.py` lines 361-549).
  - This does not change TinyTRACE's frozen architectural commitment to keep a compression stage; it only means the exact TRACE repo configuration still needs disambiguation before implementation details are treated as fully verified.

### 7.3.3 Items Still Open After This Audit

- The exact TRACE training configuration actually used in the paper remains unresolved from this checkout alone because the repo exposes multiple projector/compression variants rather than a single canonical one.
- The repo does not provide a verified separate Stage 1 / Stage 2 schedule description tightly mapped to the paper's prose inside this document.
- Optimizer hyperparameters, warmup details, and exact launch arguments are not fully recoverable from code inspection alone without tying them to a specific training shell script/checkpoint recipe.
- Batched generation support beyond the toy inference example remains open.
- Malformed-generation fallback behavior remains open; no defensive timeout or repair path was found in the released inference example.

## 7.4 Next Document Updates

The remaining work for this document is straightforward maintenance rather than a separate reverse-engineering phase:

1. Resolve additional items from §7.2 where the released TRACE code supports a clear answer.
2. Keep unresolved items explicitly marked TODO rather than normalizing them into assumptions.
3. Update module text, diagrams, tensor-flow descriptions, and the checklist whenever a currently-open implementation detail becomes verified.
4. Preserve the frozen architectural decisions in §1.7 unless a verified TRACE fact proves one of them impossible as stated.

---


---
