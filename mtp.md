# Lookahead Forcing: On-Policy Multi-Chunk Prediction for Self-Forcing (v2)

**Implementation spec for agent working in `github.com/guandeh17/Self-Forcing`**
Working name: *Lookahead Forcing*; the loss is *Lookahead Self-Consistency (LSC)*.

> v2 changelog: fixed the exit-step premise (shared exit per rollout is the default, and committed targets are exit-step predictions); corrected Mode-1 speedup math (skip, not overlap; 1.67× ceiling for K=1, ~3× for chained drafting); added the **dual-tap** architecture separating the regularizer channel from the drafter channel; upgraded Variant A with recorded inter-step noises; added Variant C grounding; replaced fixed λ with a gradient-norm-ratio controller; documented the checkpoint-boundary tap pattern, strict-load fix, and pre-slice pairing; extended collapse monitoring; repositioned related work around EAGLE-style self-speculative decoding.

---

## 1. One-paragraph summary

Self-Forcing trains a few-step causal video diffusion student (Wan2.1-T2V-1.3B, block-wise AR, 3 latent frames/block, 4-step denoising, KV cache) by rolling out its own generations and applying DMD. We add lightweight **lookahead heads** (feature-conditioned drafters à la EAGLE, with MTP-style trunk supervision à la Next Forcing arXiv 2606.11187 / DeepSeek-V3) that, at rollout step *b*, predict the latent content of block *b+k* from the backbone's intermediate features. Unlike Next Forcing, which regresses against **ground-truth** future chunks under teacher forcing, we have no GT futures: supervision is the model's **own committed generation** of block *b+k*, harvested later in the same rollout, with stop-grad on the target. This is fully on-policy and data-free (consistent with Self-Forcing's setup: no video data, only prompts + teacher).

**Primary claim (drafter):** the heads enable *self-speculative streaming* — drafted blocks skip the 4-step denoise chain, paying only a head pass + the context-encode pass every block already pays, with a ceiling of 1.67× (K=1 alternate drafting) and ~3× (chained drafting, §7). **Secondary claim (regularizer — an empirically-tested hypothesis, not an a-priori guarantee):** the LSC gradient through the backbone features may act as anti-myopic representation shaping and reduce long-rollout drift. The two claims are architecturally decoupled (dual-tap, §3) so each can succeed or fail independently; if the regularizer is a wash, the project re-scopes to the pure drafter (kill criterion in §11).

Contributions: (1) **bootstrapped multi-chunk supervision without GT futures**, with an explicit collapse analysis and grounding mechanisms (§4, Variant C); (2) **drafter-free self-speculative streaming** for causal video diffusion, with the DMD fake score as an optional verifier; (3) Variant A as **one-step self-distillation of the model's own block-sampling chain** (an amortized sampler, not merely a predictor).

---

## 2. Repo facts you must build on (verified against current main)

- `pipeline/self_forcing_training.py :: SelfForcingTrainingPipeline.inference_with_trajectory`
  - Blocks of `num_frame_per_block=3` latent frames; default `num_max_frames=21` → 7 blocks; `frame_seq_length=1560` tokens per latent frame → **4680 tokens per block** (derived, not a literal; code uses `current_start_frame * frame_seq_length`).
  - Per block, it iterates `denoising_step_list = [1000, 750, 500, 250]`. `exit_flags` is drawn on rank 0 and `dist.broadcast`-synced (lines ~41-58). **The effective default is `same_step_across_blocks: true`** (`configs/default_config.yaml:4`; `model/dmd.py:18` also defaults True; `self_forcing_dmd.yaml` does not override): ONE shared exit index *s* per rollout is used for **all** blocks (`exit_flags[0]`, line ~147). Per-block random exit happens only when the flag is false.
  - **The denoise loop `break`s at the exit step** (line ~194). Consequently the "committed block" written to `output` is the **x0 prediction at exit step *s***, not the output of the full 4-step chain — a mixture of 4 target regimes across rollouts, only *s=3* matching what the deployed 4-step sampler emits. This drives the exit-step conditioning requirement (§3) and the `last_step_only` ablation (§4/§10).
  - Non-exit iterations run under `no_grad` and re-noise with **fresh `randn_like` draws** via `scheduler.add_noise` at the next timestep (lines ~166-171). These draws are the residual stochasticity of the committed target given (context, seed) — Variant A records and conditions on them (§4).
  - The exit-step forward is the only one that can carry grad, further gated by `current_start_frame >= start_gradient_frame_index` where `start_gradient_frame_index = num_output_frames - 21` (line ~137). With the default 21-frame config it is 0, so every block is graded; do not assume this for long-video configs.
  - **Step 3.3** (lines ~199-216): every committed block is re-run at `context_noise` timestep (default `context_noise: 0` → clean re-encode) under `no_grad` to write the KV cache. This pass exists for *every* block, both here and in `pipeline/causal_inference.py:226-235` — it is the drafter's feature source (§3) and the reason drafted blocks pay no extra cache cost (§7).
  - The full `noise` tensor for all blocks exists up front and is sliced per block — future blocks' seeds are available before generation.
  - Returns `(output, denoised_timestep_from, denoised_timestep_to)`; **`denoised_timestep_from/to` are `None` unless `same_step_across_blocks=true`**, and DMD/critic timestep sampling consults them when `ts_schedule` is on (`model/dmd.py:154-155, 269-270`; code default `ts_schedule=True`, `self_forcing_dmd.yaml` sets it false). Any lookahead config that touches `same_step_across_blocks` must **pin `ts_schedule` explicitly** or it silently changes generator *and* critic timestep ranges.
- `wan/modules/causal_model.py :: CausalWanModel` — for the 1.3B checkpoint: 30 `CausalWanAttentionBlock`s, dim 1536, 12 heads × 128, **ffn_dim 8960**. These come from the pretrained `config.json` via `from_pretrained`, **not** the class defaults (2048/16/32/8192) — head modules must read dims from the instantiated model. Two forward paths dispatched on `kv_cache`: `_forward_inference` (block loop at line ~812; the path used during rollout) and `_forward_train` (loop at ~981). No hidden-state return mechanism exists; it must be added. The graded pass per-block-checkpoints each transformer block (`torch.utils.checkpoint`, `use_reentrant=False`, lines ~813-825) — see §5 for the mandatory tap pattern. Cross-attention has **no mask support** (hardcoded `(-1,-1)` window); custom masks go through the FlexAttention self-attn `block_mask` path (`_prepare_teacher_forcing_mask` at line ~563 is the template). RoPE temporal position = absolute frame index via `causal_rope_apply(..., start_frame)`; a future block's positions are obtained by passing that block's absolute start frame. Timestep embedder: `self.time_embedding` (output `e ∈ R^{B·F × 1536}`) and `self.time_projection` (AdaLN 6×1536) are plain reusable modules.
- `utils/wan_wrapper.py :: WanDiffusionWrapper.forward` (~line 218) — the model outputs flow-matching **velocity**; clean x0 is derived in the wrapper via `_convert_flow_pred_to_x0` (:169-193, applied at :280-284; the `# X0 prediction` comment at :239 is misleading). Returns `(flow_pred, pred_x0)`. The wrapper permutes the model output at :249 — any tuple return with features must unpack before that permute. Precedent for attaching extra trainable modules to the wrapper: `adding_cls_branch` (:147-167).
- `model/base.py :: SelfForcingModel._run_generator / _consistency_backward_simulation` — `_run_generator` (:139-177) samples a variable number of blocks for long-video configs, **slices output to the last 21 frames, VAE-re-encodes the first retained frame, and gradient-masks the first block**. LSC pairing must therefore happen on the raw pre-slice `output`, keyed by absolute block index (§5).
- `model/dmd.py :: DMD` — `generator_loss` (:196-235) currently returns only the DMD term; LSC adds there. DMD loss form: `0.5 · MSE(x, (x − grad).detach())` with `grad = pred_fake − pred_real`, per-sample normalized (:113-120, :189-193) — its backbone-gradient scale is specific, motivating the ratio controller (§4). The **fake score is a non-causal 1.3B WanModel applied to full 21-frame windows** — this matters for Mode-2 verifier cost (§7).
- `trainer/distillation.py :: fwdbwd_one_step` — alternating updates with `dfake_gen_update_ratio: 5`: **generator (hence heads) updates every 5th step; critic every step**. Critic-step rollouts run under `no_grad` — free drafter training data (§4). FSDP: size-based auto-wrap (`min_num_params=5e7`, `use_orig_params=True`); heads attached under the generator wrapper are picked up by FSDP, the AdamW-over-`requires_grad` optimizer, and `EMA_FSDP` automatically. **`load_state_dict(strict=True)` at :168-170 will crash on missing head keys** — see §5.
- Configs: `configs/self_forcing_dmd.yaml` (`num_frame_per_block: 3`, `denoising_step_list`, `warp_denoising_step: true`, `gradient_checkpointing: true`, `ts_schedule: false`); `configs/default_config.yaml` supplies `same_step_across_blocks: true`, `context_noise: 0`, `last_step_only: false`. Add a `lookahead:` section (§8).

---

## 3. Architecture: lookahead heads with dual-tap

The single most important structural decision: **two feature sources, one per claim.**

**Tap A — exit-step tap (regularizer channel, with grad).** During block *b*'s exit-step forward (the graded one), collect hidden states at the outputs of backbone blocks `{8, 16, 24, 30}` (1-indexed), restricted to the current block's 4680 tokens. Collection must happen **at checkpoint boundaries** — i.e., in `_forward_inference`'s block loop, capture the tensor *returned by* `torch.utils.checkpoint.checkpoint(...)` for the tapped indices. Never use hooks or capture inside block forwards: under `use_reentrant=False` checkpointing, tensors captured inside a checkpointed region are recomputed at backward and a naive list-append double-collects. Boundary activations are already retained by checkpointing, so referencing them costs ~zero extra memory. Keep the autograd graph — this is the only channel through which LSC shapes the backbone.

**Tap B — context-pass tap (drafter channel, detached).** Also collect the same 4 layers during the **Step 3.3 clean re-encode** of block *b*, detached (that pass is `no_grad` in training anyway). Rationale: at inference, block *b+1* is generated (or drafted) after block *b*'s context re-encode; Step 3.3 features are (i) a deterministic function of the committed clean content (context_noise=0), (ii) produced *identically* at train and deploy, and (iii) computed by the very pass that writes the KV entries the main model conditions on next. Training the drafter on Tap A alone creates a train/deploy feature mismatch (exit-step features at a random regime vs. final clean context) and, worse, makes drafting **unchainable** (a drafted block has no exit-step features to draft the next block from). With Tap B, every committed *or drafted* block yields drafter features from the context pass it must run anyway → chained drafting (§7).

**Fusion.** Per tap: concatenate the 4 tapped layers along channels (4×1536) → 2-layer MLP → `h_fuse ∈ R^{B × 4680 × 1536}`. Apply the fusion MLP **lazily at loss time** on the retained boundary tensors (not inside the rollout) to avoid holding fusion outputs across the whole rollout. Condition the fusion (concatenate or AdaLN, reusing `time_embedding` outputs) on:
- an **exit-step embedding** (Tap A only) — needed for *target-regime identification*: with the shared-exit default, the committed target is "x0-pred after *s+1* steps," a 4-regime mixture; the head must know which regime it is predicting. (The v1 rationale — per-block random exit — was wrong; the embedding is constant within a rollout and can be injected once.)
- a **tap-source embedding** (A vs. B) so one head serves both channels.

**Head module (per depth k).** Input projection over `[h_fuse tokens ; target tokens]`; then **2 `CausalWanAttentionBlock`-style transformer blocks** (ablate {1,2,3}), instantiated with dims read from the loaded model (`dim=1536, ffn_dim=8960, num_heads=12, cross_attn_norm=True, qk_norm=True, eps=1e-6`), initialized from the main model's last 2 blocks (Next Forcing: init matters, +2pts). Attention over the concatenated sequence uses the FlexAttention **self-attn** `block_mask` path (target→all, fuse→fuse), built once per (shape, k) and cached — `create_block_mask` is expensive per call; `_prepare_teacher_forcing_mask` is the template. Keep text cross-attention in the head blocks (reading the crossattn cache after `is_init` is side-effect-free). Heads never read or write the main **KV** cache. RoPE for head target tokens uses block *b+k*'s absolute frame index (`start_frame = (b+k)·num_frame_per_block`) — unit-test this (§9).

**Output layer.** Fresh output projection, **zero-initialized**, so LSC starts as an exact no-op and the λ warmup is honest. Reuse the backbone's patchify/unpatchify frozen. Output convention: Variant B predicts **velocity** (matches the repo's flow convention `pred = ε − x0`); Variant A predicts **x0** directly (do not repurpose the velocity-trained backbone output layer for an x0 target).

**Depths.** Phase 1: K=1 only. Phase 2: K=2 with causal chaining (`h_prev^{(2)}` = head-1's pre-output hidden states), loss weights `w1=0.5, w2=0.2` starting points.

Parameter count: 2 blocks ≈ 93M (per block: 2×4×1536² attn + 2×1536×8960 FFN ≈ 46.4M) + fusion ≈ 12M → **~105M per depth**, ~8% of backbone. FSDP note: with size-based wrap (5e7 floor) a single 46M block won't wrap alone but the head ModuleList will; verify wrapping doesn't split heads from fusion in a way that breaks the EMA/param-name logic (`distillation.py:141-152` strips wrapper prefixes — test the round-trip, §9).

---

## 4. The LSC loss — variants behind a config switch

Let `x̄_{b+k} = stopgrad(output[:, block b+k])` — noting this is the **exit-step x0 prediction** (§2). Compute LSC only for blocks whose Tap-A features were collected under grad; drafter (Tap-B) terms have no backbone coupling and are exempt.

**Variant B — bootstrapped flow matching (regularizer default; M2).**

```
t' ~ shifted schedule with shift s_head (default 10; generator uses 5 via warp)
ε' ~ N(0, I)
x_t' = (1 − t') · x̄_{b+k} + t' · ε'
v̂ = head_k(x_t', t', h_fuse^A(b))          # Tap A, grad flows to backbone
L_k = || v̂ − (ε' − x̄_{b+k}) ||²
```

Rationale for `s_head > s_main` (Next Forcing ablation, 85.8 vs 83.2): high noise starves the head input of target information, forcing gradient into `h_fuse` and hence the backbone. Caveat to monitor: at high t' the FM gradient pushes features toward the **conditional mean** of futures — mean-seeking pressure that the collapse metrics below must watch. B is mandatory for k≥2.

**Variant A — noise-conditioned one-step draft (drafter default; M3).**
The committed block is `x̄_{b+1} = f_θ(context_{≤b}, ε_{b+1}, {ζ_i}_{i<s})` where `ε_{b+1}` is the block's seed (available up front) and `{ζ_i}` are the *s* fresh `randn_like` draws injected by `scheduler.add_noise` between denoise steps. **Record the ζ draws during rollout and condition the head on them** (project + concatenate alongside the seed): the target becomes near-deterministic for *every* exit depth — for s=0 it is exactly deterministic given (context, seed). This upgrades v1's "accept target variance" to a well-posed regression:

```
d_b = head_k(ε_{b+k}, {ζ_i}, t=T_max, h_fuse^B(b))   # Tap B, detached — drafter channel
L_k = || d_b − x̄_{b+k} ||²    (Huber fallback if unstable)
```

Framing: **one-step self-distillation of the model's own block-sampling chain** — an amortized sampler conditioned on previous-block features. At inference, sample fresh ζ. For k≥2 use B regardless. An A-head may also be trained on Tap A if the regularizer arm wants a deterministic variant, but the drafter ships on Tap B.

**Variant C — grounded lookahead (optional arms; the rebuttal to "vacuous self-distillation").**
The honest information argument: for k=1 the target is a function of the same network being trained — LSC alone injects zero bits about the data distribution and admits a cheap solution (make the model's own transition operator *simpler*/lower-entropy) alongside the good one (expose predictive features). Two grounding options, config-gated:
- **C1 — DMD-on-drafts:** apply the DMD real-vs-fake KL gradient directly to the head's drafted block (both scores are resident during training; cost ≈ one extra scored block per rollout). This routes *teacher* information about futures through the head into `h_fuse`.
- **C2 — EMA targets:** regress against the **EMA generator's** committed blocks (`EMA_FSDP` already exists). BYOL-style slow target; kills the cross-step feedback loop where updated trunk weights move the next rollout's targets toward what the head predicted.

**Total generator loss:** `L = L_DMD + λ · Σ_k w_k L_k^{TapA}` (+ optional C terms). Drafter-channel (Tap B) losses are optimized separately (below).

**λ via gradient-norm-ratio controller (replaces fixed λ).** DMD's backbone gradient has a specific, per-sample-normalized scale; raw LSC MSE lives on a different scale that drifts as the head trains — any fixed λ is wrong at two training stages. Every N=100 generator steps: backward `L_DMD` alone, snapshot backbone grad norm, zero; backward `λ·L_LSC` alone, snapshot; multiplicatively adjust λ (clip ×/÷1.5 per adjustment) to hold `‖g_LSC‖/‖g_DMD‖ ∈ [0.10, 0.25]`. Keep a warmup (λ from 0 over the first 1–2k generator steps) on top. Diagnostic cost: one extra backward per 100 steps. Never let LSC be active before DMD is stable.

**Head updates on critic steps (fixes 5× data starvation).** With `dfake_gen_update_ratio=5`, heads would train on 1/5 of rollouts. On critic steps the rollout already runs under `no_grad`: tap features detached, compute drafter-channel losses, and take a **head-only optimizer step** (separate param group, no backbone coupling, no interference with the critic). Free 5× more drafter data.

**Stop-grad discipline (test it, §9):**
- Gradient flows: head params ✔, fusion MLP ✔, backbone via Tap-A `h_fuse` ✔.
- No gradient into the target: assert `x̄_{b+k}.grad_fn is None` after detach, and that `torch.autograd.grad(L_LSC, pre-target activations, allow_unused=True)` is None when the only path is through the target. (Note: perturbing the target of course changes grads *through the loss value* — that is not the failure mode; graph connectivity is.)
- Heads' Tap-A terms update only on generator steps; Tap-B terms may update every step.

**Collapse monitoring (mandatory logging, every eval).** Dynamic degree alone catches frozen frames but misses the likelier failures: *predictable-but-moving* collapse (constant-velocity pans, loss of secondary motion) and *seed-diversity* collapse (DMD is mode-seeking and applies no pressure to preserve conditional diversity; CFG at guidance 3.0 already dampens motion, so the collapse direction is aligned with an existing bias). Log:
1. Latent temporal variance `mean_b ‖x_{b+1} − x_b‖` and VBench dynamic degree (alarm: >15% relative drop vs. compute-matched baseline).
2. **Continuation diversity:** fixed prompt + fixed first block, K seeds → mean pairwise latent distance of blocks 2..7 vs. baseline (alarm: <95% of baseline).
3. Optical-flow magnitude **distribution** (not just mean) on decoded video.
4. **LSC-too-easy alarm:** held-out head loss dropping below an M1-calibrated floor while DMD is flat = the model is making itself predictable.
5. Held-out **draft latent-MSE stratified by exit step and block index** (later blocks are harder under drift — this stratification is the early-warning system and the M3 gate input).

Never train LSC with DMD detached; DMD is the (partial) grounding term.

---

## 5. Integration plan (where the code goes)

1. `wan/modules/causal_model.py`: add `return_hidden_layers: Optional[List[int]]` to **`_forward_inference`** specifically (the kv_cache path used in rollout; block loop at ~line 812) and plumb through `forward`'s dispatch; optionally mirror in `_forward_train`. Collect at **checkpoint boundaries** (capture the return value of `checkpoint(...)` for tapped indices — never hooks/inside-block; see §3). Mirror the loop's quirk that the checkpointed branch omits `crossattn_cache` from kwargs (~:814-825). Zero overhead when the flag is None.
2. `utils/wan_wrapper.py`: plumb the flag + returned features through `WanDiffusionWrapper.forward`; unpack the `(flow_pred, features)` tuple **before** the permute at :249. Attach head/fusion modules to the wrapper following the `adding_cls_branch` precedent (:147-167) so they live under the generator before FSDP wrapping.
3. `pipeline/self_forcing_training.py`:
   - Exit-step branch (graded): request features when `lookahead.enabled` and the block is in the grad window and in the **upfront-chosen tapped subset** (choose tapped blocks before the rollout, don't subsample post-hoc); append `(abs_block_idx, exit_step, feats_A)`.
   - Step 3.3 branch: request features detached → `(abs_block_idx, feats_B)`. (v1's "do NOT touch Step 3.3" is rescinded — tap it, detached; still never give it grad by default.)
   - Record the inter-step `add_noise` ζ draws per block when `lookahead.variant == seed_draft`.
   - Return all of this alongside `output`.
4. New file `model/lookahead.py`: fusion MLP (lazy application at loss time) + head modules + block-mask cache + all loss variants (B / A / C1 / C2), config-driven.
5. `model/dmd.py` / `model/base.py`: compute LSC **on the raw pre-slice pipeline `output`, keyed by absolute block index** — i.e., inside or immediately after `_consistency_backward_simulation`, honoring `start_gradient_frame_index` — NOT after `_run_generator`'s last-21 slice + first-frame VAE re-encode (that would silently corrupt pairing on long-video configs; with the default 21-frame config the bug would be invisible until M5). Add the Tap-A term to `generator_loss`; log per-depth, per-channel losses.
6. `trainer/distillation.py`:
   - Heads/fusion registered under the generator wrapper before FSDP wrapping (sharding + optimizer + EMA pick them up automatically; `use_orig_params=True`).
   - **Checkpoint load:** switch to `strict=False`, assert `set(missing_keys) == expected_head_and_fusion_keys`, hard-fail on any unexpected key. Handle optimizer-state resume with the enlarged param set; verify EMA + `fsdp_state_dict` round-trip with heads (§9).
   - Separate `head_lr` param group (precedent: `trainer/gan.py:103-118` groups by name substring).
   - Head-only optimizer step on critic steps (Tap-B losses, detached features).
   - Gradient-norm-ratio controller for λ (every 100 generator steps).
7. `configs/self_forcing_dmd.yaml`: add the `lookahead:` block (§8). All behavior off by default. **Pin `ts_schedule` explicitly** in any lookahead config.

**Memory (honest accounting).** One tap = 4680×1536×2B ≈ 14.4 MB; 4 taps ≈ 57.5 MB/block/sample — but that's not the marginal cost. Tap-A features for block *b* must be retained until block *b+k* commits (K=1 all-pairs: 6×57.5 ≈ 345 MB) — however, tapped at checkpoint boundaries these tensors are **already resident** (the graded pass retains all 30 boundary activations ≈ 431 MB/block/sample regardless). The true marginal is fusion outputs + head forward graphs at loss time ≈ **~0.5 GB/sample** with per-block head checkpointing. Knobs: upfront tapped-block subset (`max_pairs_per_rollout`), lazy fusion, gradient checkpointing on head blocks.

---

## 6. Phases / milestones

- **M0 (plumbing + tests):** feature taps (both), head module, loss variants B and A, unit tests in §9 green, forward/backward on 1 GPU with tiny config, memory profile logged.
- **M1 (smoke train):** continue-train from the released Self-Forcing DMD checkpoint for ~500 generator steps with λ warmup + ratio controller; verify: DMD loss unchanged in scale, LSC decreasing, no NaN, collapse metrics (§4) stable, strict-load fix works, checkpoint save/load round-trips.
- **M2 (main result — go/no-go gate):** three arms from the released checkpoint, identical data/steps/seeds, ~3k generator updates (≈15k trainer steps at ratio 5), **2 seeds each**:
  - (A) baseline continue-train;
  - (B) +LSC Variant B, K=1, dual-tap, ratio-controlled λ;
  - (C) heads on **detached** features (isolates the drafter from the regularizer — makes the causal claim clean).
  Numeric gates: B's VBench total within 0.3 of A **and** dynamic degree ≥95% of A (safety); drift slope ≥15% better than A to *claim* the regularizer; continuation diversity ≥95% of A; drafter gate for M3 = held-out draft quality such that a 20-prompt Mode-1 pilot loses ≤1.0 VBench total at measured ≥1.4× wall-clock on one A100. **Go to M3 if the drafter gate passes regardless of the regularizer gates**; the regularizer gates decide only whether Mode 0 is a claimed contribution or a dropped hypothesis (kill criterion, §11).
- **M3 (self-speculative streaming):** Variant-A drafter (recorded-ζ, Tap B), Mode 1 including **chained drafting**; report throughput/latency/quality.
- **M4 (verified speculation):** Mode 2 with windowed fake-score verifier; Pareto sweep over threshold τ; validate the verifier proxy against actual draft error.
- **M5 (extensions, only after M2):** K=2 chaining; Variant C arms; long-rollout training à la Self-Forcing++ if drift numbers motivate it.

---

## 7. Inference modes (M3/M4)

Cost unit: P = one 30-layer pass over one block (4680 tokens). Head cost ≈ (2/30 layers) × (9360/4680 concatenated tokens) ≈ 0.13P per step. Baseline per block: 4 denoise + 1 context = 5P; 7 blocks = 35P.

- **Mode 0 — regularizer only.** Discard heads. Zero overhead. All M2 quality/drift gains report here.
- **Mode 1 — self-speculative block skipping.** The mechanism is **skipping**, not overlap — on one GPU two compute-bound passes contend for the same SMs; there is nothing to overlap. A drafted block skips its 4 denoise passes and pays only head + the context pass it needed anyway.
  - *Alternate drafting (K=1, conservative):* blocks 1,3,5,7 full (16P) + drafts for 2,4,6 (3×0.13P) + 7 context passes (7P) ≈ 23.4P → **1.50×** at 7 blocks; asymptotic ceiling **1.67×**. Claim 1.4–1.55× measured; the v1 claim of 1.9× is arithmetically unreachable for K=1.
  - *Chained drafting (the headline, enabled by Tap B):* each drafted block's mandatory context pass produces the features to draft the next block. Block 1 full (5P) + 6 × (1P context + 0.13P head) ≈ 11.8P → **~3.0×** ceiling, gated by draft quality (fall back to full denoising on rejection or every Nth block).
  - True overlap exists only with pipeline parallelism across 2 GPUs (main denoises block *t* on GPU0 while GPU1 context-encodes + drafts) — a separate system contribution; either commit to it or don't claim it.
  - Report actual FPS + latency-to-first-block on a single A100 at 832×480 (latency unchanged — a selling point), plus **seam consistency** at drafted/full block boundaries (§10).
- **Mode 2 — critic-verified speculation.** Cost accounting is mandatory: the fake score is a *bidirectional* model over 21-frame windows — a full-window pass ≈ 7P, which **exceeds the ~3.9P saved per drafted block** (net slowdown if done naively). Use: (i) **windowed verification** (last 2 committed blocks + draft ≈ 3P), (ii) verify every Nth draft, or (iii) a distilled verifier head. The verifier signal is a **fake-score-only proxy** (e.g., fake-score denoising error on the draft at a mid noise level) — the 14B real score is not deployable, so "real-vs-fake discrepancy" is out. Validate the proxy correlates with actual draft error before sweeping τ. Sweep τ for the quality–speed Pareto; SDVG's 1.59×@98.1% (arXiv 2604.17397) is the external reference on efficiency-per-added-parameter.

---

## 8. Config schema

```yaml
lookahead:
  enabled: false
  variant: fm_selftarget        # fm_selftarget (B) | seed_draft (A)
  grounding: none               # none | dmd_on_drafts (C1) | ema_targets (C2)
  depths: 1                     # K
  loss_weights: [0.5]           # w_k, length K
  # lambda: gradient-norm-ratio controller, not a fixed value
  grad_ratio_target: [0.10, 0.25]
  grad_ratio_interval: 100      # generator steps between controller adjustments
  lambda_warmup_steps: 1500
  tap_sources: [exit, context]  # Tap A (grad) and/or Tap B (detached)
  head_num_blocks: 2
  head_init_from_backbone: true
  head_lr: null                 # null = generator lr; else separate group
  head_updates_on_critic_steps: true
  fusion_layers: [8, 16, 24, 30]
  head_timestep_shift: 10.0     # variant B only
  exit_step_conditioning: true  # target-regime identification
  record_interstep_noise: true  # variant A conditioning (ζ draws)
  max_pairs_per_rollout: 6      # upfront tapped-block subset; 6 = all pairs at 7 blocks

# In any lookahead config, pin these explicitly (do not inherit silently):
# ts_schedule: false
# same_step_across_blocks: true   # flipping it changes DMD/critic ts sampling if ts_schedule on
```

---

## 9. Unit tests (write before training)

1. **RoPE offset:** head tokens for block b+1 receive positional encodings identical to what the main model uses when actually processing block b+1.
2. **No cache pollution:** KV cache tensors bit-identical with `lookahead.enabled` on vs. off for the same seed.
3. **Stop-grad (corrected spec):** `x̄_{b+k}.grad_fn is None`; `torch.autograd.grad(L_LSC, pre-target activations, allow_unused=True)` is None when the only path is through the target.
4. **Grad reaches backbone:** nonzero grad norm on tapped backbone layers from Tap-A LSC alone (DMD detached, test only); zero backbone grad from Tap-B losses.
5. **No future leakage:** `h_fuse(b)` invariant to the content of `noise[:, blocks > b]` (Variant B) — perturb future noise, assert equal features.
6. **Determinism harness:** fixed-seed rollout reproducible with feature taps enabled.
7. **Checkpointing parity:** grads bit-comparable with `gradient_checkpointing` on/off with taps enabled (protects against the double-collect failure mode).
8. **DMD invariance:** pipeline returns identical `denoised_timestep_from/to` with lookahead on/off (protects DMD/critic under `ts_schedule` configs).
9. **Pairing under slicing:** feature↔target pairing correct on a variable-block long-video config through `_run_generator`'s last-21 slice + first-frame VAE re-encode.
10. **Train/inference head parity:** head output at the inference tap point equals training-mode output given identical inputs.
11. **FSDP round-trip:** save→load→save idempotent with heads registered, EMA included; strict-load assertion (missing keys == expected head/fusion keys) fires correctly.

---

## 10. Evaluation protocol

- **Quality (5s):** VBench on the repo's standard prompt set — total plus dynamic degree, subject/background consistency, motion smoothness, imaging quality. **≥2 seeds or bootstrap CIs** — single-run VBench deltas <1 point are noise. LSC should improve consistency/smoothness without hurting dynamic degree.
- **Collapse (the failure-mode suite):** continuation diversity across seeds, optical-flow distribution, LSC-too-easy alarm (§4) — these catch what dynamic degree misses.
- **Drafter quality (independent of end quality):** held-out draft-vs-committed latent MSE / PSNR, **stratified by exit step and by block index**; this is the M2→M3 gate input and the early-warning system for drift-induced draft failure.
- **Drift:** extended rollouts (30–60s via the repo's extrapolation path). Per 5s bucket: CLIP similarity to first-bucket frames, per-bucket imaging quality, qualitative grids. Report degradation *slope* vs. baseline. Reference: Self-Forcing++ (2509). Note the confound: self-consistent videos can drift less *because* diversity collapsed — always report drift jointly with the collapse suite.
- **Efficiency:** FPS + latency-to-first-frame, single A100, modes 0/1/2; Mode 1 alternate vs. chained; **seam consistency** at drafted/full block boundaries (temporal metrics computed specifically across seams); Mode 2 acceptance-rate-vs-τ curves with the verifier proxy validated against actual draft error.
- **Baselines:** (i) released SF checkpoint, (ii) compute-matched continued SF (mandatory), (iii) heads-on-detached-features arm (isolates regularizer), (iv) SDVG reported numbers for speculation context, (v) if video data is available, teacher-forced Next-Forcing-style MCP as oracle upper bound — else state the data-free setting explicitly.
- **Ablations (priority order):** grad-ratio target sweep → variant B vs A (drafter) → Tap A vs B vs dual → grounding C1/C2 on/off → `last_step_only: true` (all targets become full-chain outputs — the cleanest drafter regime, at the cost of changing what DMD sees; this replaces v1's `same_step_across_blocks` ablation, which is the default, not an ablation) → fusion multi-layer vs last-only → head shift {5,10} → head blocks {1,2,3} → exit-step conditioning on/off → K=1 vs 2.

---

## 11. Known risks & mitigations

- **Predictability collapse (top risk).** The failure is not (only) frozen frames: LSC alone is a compressibility constraint on the model's own transition operator, satisfiable by making futures *simpler* — smoother, lower-entropy, still individually realistic motion that mode-seeking DMD will not resist (and CFG already biases toward). Mitigations: ratio-controlled λ + warmup, DMD always on, full collapse suite (§4), Variant C grounding. **Kill criterion (evaluated at M2):** if arm B shows no drift-slope improvement ≥15% AND (continuation diversity or dynamic degree down >5%) vs. arm A → detach `h_fuse` permanently (heads on frozen features) and re-scope to the pure self-speculative drafter, which needs no backbone gradient to be publishable.
- **Cross-step bootstrap loop.** Stop-grad blocks within-step gradient, but the trunk weights that generate targets are the weights being updated — next rollout's targets drift toward head predictions (BYOL/TD-target dynamics). Watch the LSC-too-easy alarm; C2 (EMA targets) is the standing fix.
- **Drafter train/deploy mismatch** — solved structurally by Tap B; test 10 guards it.
- **FSDP / checkpoint** — heads under the generator module before wrapping; `strict=False` load with exact-key assertion; save/load/EMA round-trip test (§9.11).
- **Long-video configs** — pre-slice pairing by absolute block index (§5); taps respect `start_gradient_frame_index`; Tap A never under no_grad.
- **Config coupling** — pin `ts_schedule` in lookahead configs; flipping `same_step_across_blocks` silently changes DMD *and* critic timestep sampling when `ts_schedule` is on.
- **Scooping / positioning.** The closest neighbor is **EAGLE** (feature-conditioned drafting from the target model's own hidden states), not Gloeckle-style MTP — differentiate: EAGLE's drafter trains offline against a frozen target with no backbone gradient; ours co-trains and doubles as a regularizer (which is also the risk surface). Must cite: EAGLE/-2/-3, Medusa, self-speculative decoding without auxiliary models (Draft & Verify, LayerSkip, Kangaroo), parallel diffusion sampling (ParaDiGMS, CLLMs/Jacobi), and the causal-video line (CausVid — explicitly, since Self-Forcing builds on it; Diffusion Forcing, Rolling Diffusion, FIFO-Diffusion, Self-Forcing++), plus the anchors Next Forcing (2606.11187) and SDVG (2604.17397). **Run a fresh 2025–26 literature sweep on speculative/parallel video generation before claiming novelty** — block-level drafting for causal video diffusion is obvious enough that parallel work is likely. Framing rank: (1) self-speculative streaming with a co-trained feature drafter; (2) Variant A as amortized one-step self-distillation of the rollout; (3) drift regularization as an empirical finding, not the headline. M2 go/no-go within ~3–4 weeks of wall time.
