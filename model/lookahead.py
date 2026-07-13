"""
Lookahead Forcing modules (see mtp.md).

A lookahead head predicts the latent content of a future block b+k from the
backbone's intermediate features at block b. Supervision is the model's own
committed generation of block b+k (stop-grad), harvested later in the same
rollout — no ground-truth futures.

Two feature channels (dual-tap):
  - Tap A ("exit"):    features from the graded exit-step forward; gradient
                       flows into the backbone through h_fuse (regularizer).
  - Tap B ("context"): features from the Step-3.3 clean context re-encode,
                       detached (drafter; matches inference-time features).

Loss variants:
  - fm_selftarget (B): bootstrapped flow matching against the committed block.
  - seed_draft (A):    one-step x0 draft conditioned on the future block's
                       seed noise and the recorded inter-step re-noising draws.

FSDP note: LookaheadModule owns private copies of the patch/time/text
embeddings (initialized from the backbone) instead of referencing backbone
submodules, and is invoked through WanDiffusionWrapper.forward so parameters
are gathered correctly under FSDP.
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask

from wan.modules.causal_model import (
    CausalWanAttentionBlock,
    CausalWanSelfAttention,
    causal_rope_apply,
    flex_attention,
)
from wan.modules.model import sinusoidal_embedding_1d


def _pad_to_128(x: torch.Tensor) -> torch.Tensor:
    padded_length = math.ceil(x.shape[1] / 128) * 128 - x.shape[1]
    if padded_length == 0:
        return x
    return torch.cat(
        [x, torch.zeros([x.shape[0], padded_length, *x.shape[2:]], device=x.device, dtype=x.dtype)],
        dim=1
    )


class LookaheadSelfAttention(CausalWanSelfAttention):
    """
    Self-attention over the concatenated [h_fuse tokens ; target tokens]
    sequence. Same parameters as CausalWanSelfAttention (so backbone-block
    state_dicts load directly); only the forward differs:
      - RoPE is applied per segment at each segment's absolute start frame.
      - Attention uses a fuse->fuse / target->all block mask.
    """

    def forward(self, x, grid_sizes, freqs, block_mask, fuse_len, fuse_start_frame, target_start_frame):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        # RoPE per segment at absolute frame positions
        roped_query = torch.cat([
            causal_rope_apply(q[:, :fuse_len], grid_sizes, freqs, start_frame=fuse_start_frame),
            causal_rope_apply(q[:, fuse_len:], grid_sizes, freqs, start_frame=target_start_frame),
        ], dim=1).type_as(v)
        roped_key = torch.cat([
            causal_rope_apply(k[:, :fuse_len], grid_sizes, freqs, start_frame=fuse_start_frame),
            causal_rope_apply(k[:, fuse_len:], grid_sizes, freqs, start_frame=target_start_frame),
        ], dim=1).type_as(v)

        padded_length = math.ceil(s / 128) * 128 - s
        x = flex_attention(
            query=_pad_to_128(roped_query).transpose(2, 1),
            key=_pad_to_128(roped_key).transpose(2, 1),
            value=_pad_to_128(v).transpose(2, 1),
            block_mask=block_mask
        )
        if padded_length > 0:
            x = x[:, :, :-padded_length]
        x = x.transpose(2, 1)

        x = x.flatten(2)
        x = self.o(x)
        return x


class LookaheadBlock(CausalWanAttentionBlock):
    """
    Transformer block over the concatenated [h_fuse ; target] sequence.
    Parameter layout is identical to CausalWanAttentionBlock (init from
    backbone blocks via load_state_dict); the self-attention forward uses
    segment-wise RoPE + the lookahead block mask instead of the KV cache.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Same submodule name and parameters, different forward
        self.self_attn = LookaheadSelfAttention(
            self.dim, self.num_heads, self.local_attn_size,
            0, self.qk_norm, self.eps)

    def forward(self, x, e, grid_sizes, freqs, context, context_lens, block_mask,
                fuse_len, fuse_start_frame, target_start_frame):
        """
        x: [B, L, C] concatenated sequence; e: [B, F_total, 6, C] per-frame AdaLN
        (fuse frames carry the tap timestep, target frames the head timestep).
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid_sizes, freqs, block_mask, fuse_len, fuse_start_frame, target_start_frame)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


class LookaheadOutputHead(nn.Module):
    """
    Fresh output projection (mirrors CausalHead), zero-initialized so the
    lookahead loss starts as an exact no-op at init.
    """

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, e):
        # x: [B, L, C]; e: [B, F, 1, C]
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x.flatten(1, 2)


class LookaheadHead(nn.Module):
    """One lookahead depth (predicts block b+k)."""

    def __init__(self, dim, ffn_dim, num_heads, num_blocks, out_dim, patch_size,
                 max_zeta_draws=3, eps=1e-6):
        super().__init__()
        self.blocks = nn.ModuleList([
            LookaheadBlock('t2v_cross_attn', dim, ffn_dim, num_heads,
                           local_attn_size=-1, sink_size=0, qk_norm=True,
                           cross_attn_norm=True, eps=eps)
            for _ in range(num_blocks)
        ])
        self.out = LookaheadOutputHead(dim, out_dim, patch_size, eps)
        # Variant A: zero-init projections for the recorded inter-step
        # re-noising draws (one per denoising-step gap); no-op at init.
        self.zeta_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in range(max_zeta_draws)])
        for proj in self.zeta_proj:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def init_from_backbone(self, backbone_blocks: nn.ModuleList):
        """Copy the last len(self.blocks) backbone blocks' weights."""
        src = backbone_blocks[-len(self.blocks):]
        for dst_block, src_block in zip(self.blocks, src):
            missing, unexpected = dst_block.load_state_dict(src_block.state_dict(), strict=False)
            assert not unexpected, f"unexpected keys initializing lookahead block: {unexpected}"

    def forward(self, target_tokens, h_fuse, e0, e_out, context, grid_sizes, freqs,
                block_mask, fuse_start_frame, target_start_frame):
        """
        target_tokens: [B, Lt, C] patch-embedded head input for block b+k
        h_fuse:        [B, Lf, C] fused backbone features of block b
        e0:            [B, F_total, 6, C] per-frame AdaLN for the concat sequence
        e_out:         [B, F_target, 1, C] modulation for the output head
        Returns patch-space prediction [B, Lt, prod(patch)*out_dim].
        """
        fuse_len = h_fuse.shape[1]
        x = torch.cat([h_fuse, target_tokens], dim=1)
        for block in self.blocks:
            x = block(x, e0, grid_sizes, freqs, context, None, block_mask,
                      fuse_len, fuse_start_frame, target_start_frame)
        return self.out(x[:, fuse_len:], e_out)


class LookaheadModule(nn.Module):
    """
    Fusion MLP + heads + private embedding copies + losses.

    Attach as a submodule of WanDiffusionWrapper BEFORE FSDP wrapping and
    invoke through the wrapper's forward (`lookahead_inputs=...`) so FSDP
    gathers parameters. Holds no reference to backbone modules; embeddings
    are copied at construction time.
    """

    def __init__(self, lookahead_cfg, backbone, num_denoising_steps,
                 num_frame_per_block=3):
        super().__init__()
        la = lookahead_cfg
        self.variant = la.get("variant", "fm_selftarget")
        self.grounding = la.get("grounding", "none")
        self.depths = int(la.get("depths", 1))
        self.loss_weights = list(la.get("loss_weights", [0.5]))
        self.fusion_layers = list(la.get("fusion_layers", [8, 16, 24, 30]))
        self.head_timestep_shift = float(la.get("head_timestep_shift", 10.0))
        self.tap_sources = list(la.get("tap_sources", ["exit", "context"]))
        self.num_frame_per_block = num_frame_per_block

        # dims read from the instantiated backbone (NOT class defaults)
        dim = backbone.dim
        self.dim = dim
        self.out_dim = backbone.out_dim
        self.patch_size = backbone.patch_size
        self.freq_dim = backbone.freq_dim
        self.text_len = backbone.text_len

        # private embedding copies (initialized from backbone; gradients from
        # the lookahead loss reach the backbone only through Tap-A h_fuse)
        self.patch_embedding = nn.Conv3d(
            backbone.in_dim, dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(backbone.text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.patch_embedding.load_state_dict(backbone.patch_embedding.state_dict())
        self.text_embedding.load_state_dict(backbone.text_embedding.state_dict())
        self.time_embedding.load_state_dict(backbone.time_embedding.state_dict())
        self.time_projection.load_state_dict(backbone.time_projection.state_dict())
        # plain attribute, like CausalWanModel.freqs (keeps float64 under .to())
        self.freqs = backbone.freqs.clone()

        self.fusion = nn.Sequential(
            nn.Linear(dim * len(self.fusion_layers), dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )
        self.heads = nn.ModuleList([
            LookaheadHead(dim, backbone.ffn_dim, backbone.num_heads,
                          num_blocks=int(la.get("head_num_blocks", 2)),
                          out_dim=self.out_dim, patch_size=self.patch_size,
                          max_zeta_draws=max(num_denoising_steps - 1, 1))
            for _ in range(self.depths)
        ])
        if la.get("head_init_from_backbone", True):
            for head in self.heads:
                head.init_from_backbone(backbone.blocks)
        self._mask_cache = {}

    # ---------------------------------------------------------------- masks
    def _get_block_mask(self, fuse_len, total_len, device):
        padded_len = math.ceil(total_len / 128) * 128
        key = (fuse_len, total_len, padded_len, str(device))
        if key not in self._mask_cache:
            def mask_mod(b, h, q_idx, kv_idx):
                fuse_q = q_idx < fuse_len
                target_q = (q_idx >= fuse_len) & (q_idx < total_len)
                return (target_q & (kv_idx < total_len)) | \
                       (fuse_q & (kv_idx < fuse_len)) | (q_idx == kv_idx)

            self._mask_cache[key] = create_block_mask(
                mask_mod, B=None, H=None, Q_LEN=padded_len, KV_LEN=padded_len,
                device=device)
        return self._mask_cache[key]

    # ------------------------------------------------------------- helpers
    def _patchify(self, latent):
        # latent: [B, F, C, H, W] -> tokens [B, F*H'*W', dim]
        x = self.patch_embedding(latent.permute(0, 2, 1, 3, 4))
        return x.flatten(2).transpose(1, 2)

    def _text_context(self, prompt_embeds):
        return self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in prompt_embeds
            ]))

    def _time_embed(self, t_per_frame, ref_dtype):
        """
        t_per_frame: [B, F] integer timesteps.
        Returns (e0 [B, F, 6, C], e_out [B, F, 1, C]).
        """
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t_per_frame.flatten()).to(ref_dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(0, t_per_frame.shape)
        e_out = e.unflatten(0, t_per_frame.shape).unsqueeze(2)
        return e0, e_out

    def _sample_head_sigma(self, batch_size, device):
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        shift = self.head_timestep_shift
        return shift * u / (1 + (shift - 1) * u)

    def fuse(self, feats: Dict[int, torch.Tensor]) -> torch.Tensor:
        fused_in = torch.cat([feats[layer] for layer in self.fusion_layers], dim=-1)
        return self.fusion(fused_in)

    # --------------------------------------------------------------- losses
    def forward(
        self,
        taps: List[dict],            # per-block tap records from the pipeline
        block_meta: Dict[int, dict],  # {block_index: {start_frame, seed, zetas}}
        output: torch.Tensor,        # pre-slice rollout output [B, F_total, C, H, W]
        conditional_dict: dict,
        latent_grid: tuple,          # (frames_per_block, H_patches, W_patches)
    ) -> Dict[str, torch.Tensor]:
        """
        taps entries:
          {block_index, start_frame, source: 'exit'|'context',
           tap_timestep: int, feats: {layer: [B, L, C]}}
        block_meta holds each block's seed noise and recorded inter-step
        re-noising draws (variant A conditioning), independent of the tap
        subset. Records are keyed by absolute block index; targets are read
        from the raw pre-slice `output` (mtp.md section 5).
        """
        device = output.device
        dtype = output.dtype
        batch_size = output.shape[0]
        fpb, hp, wp = latent_grid
        # one row per sample: causal_rope_apply iterates grid_sizes rows
        grid_sizes = torch.tensor([[fpb, hp, wp]], dtype=torch.long).repeat(batch_size, 1)
        context = self._text_context(conditional_dict["prompt_embeds"])
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        by_block = {}
        for rec in taps:
            by_block.setdefault(rec["block_index"], {})[rec["source"]] = rec

        losses = {}
        counts = {}
        drafts = []
        for k in range(1, self.depths + 1):
            head = self.heads[k - 1]
            w_k = self.loss_weights[k - 1]
            for b, sources in sorted(by_block.items()):
                # absolute frame index of the target block (robust to
                # initial-latent offsets, unlike (b+k)*fpb)
                target_start_frame = sources[next(iter(sources))]["start_frame"] + k * fpb
                if target_start_frame + fpb > output.shape[1]:
                    continue
                x_bar = output[:, target_start_frame:target_start_frame + fpb].detach()

                for source, rec in sources.items():
                    if source not in self.tap_sources:
                        continue
                    h_fuse = self.fuse(rec["feats"])
                    use_seed_draft = (self.variant == "seed_draft" and k == 1)

                    if use_seed_draft:
                        # Variant A: one-step x0 draft from the future block's
                        # seed + recorded inter-step draws
                        target_rec = block_meta.get(b + k)
                        if target_rec is None or target_rec.get("seed") is None:
                            continue
                        target_tokens = self._patchify(target_rec["seed"].to(dtype))
                        for i, zeta in enumerate(target_rec.get("zetas", [])):
                            target_tokens = target_tokens + head.zeta_proj[i](
                                self._patchify(zeta.to(dtype)))
                        t_target = torch.full((batch_size, fpb), 1000, device=device, dtype=torch.long)
                        regression_target = x_bar
                    else:
                        # Variant B: bootstrapped flow matching on the committed block
                        sigma = self._sample_head_sigma(batch_size, device)
                        eps = torch.randn_like(x_bar)
                        sigma_ = sigma.view(-1, 1, 1, 1, 1).to(dtype)
                        x_t = (1 - sigma_) * x_bar + sigma_ * eps
                        target_tokens = self._patchify(x_t)
                        t_target = (sigma * 1000).long().unsqueeze(1).expand(-1, fpb)
                        regression_target = eps - x_bar

                    t_fuse = torch.full((batch_size, fpb), rec["tap_timestep"],
                                        device=device, dtype=torch.long)
                    e0, e_out = self._time_embed(
                        torch.cat([t_fuse, t_target.to(device)], dim=1), dtype)
                    e_out = e_out[:, fpb:]

                    total_len = h_fuse.shape[1] + target_tokens.shape[1]
                    block_mask = self._get_block_mask(h_fuse.shape[1], total_len, device)

                    pred_tokens = head(
                        target_tokens, h_fuse, e0, e_out, context, grid_sizes,
                        self.freqs, block_mask,
                        fuse_start_frame=rec["start_frame"],
                        target_start_frame=target_start_frame,
                    )
                    # unpatchify: [B, L, prod(patch)*C] -> [B, F, C, H, W]
                    pred = self._unpatchify(pred_tokens, fpb, hp, wp)
                    loss = torch.nn.functional.mse_loss(
                        pred.float(), regression_target.float())

                    key = f"lookahead_k{k}_{source}"
                    losses[key] = losses.get(key, 0.0) + w_k * loss
                    counts[key] = counts.get(key, 0) + 1

                    # C1 grounding (mtp.md section 4): expose a clean draft of
                    # the future block, differentiable through h_fuse, so the
                    # caller can apply the DMD teacher gradient to it. Only
                    # useful when the features carry grad (generator steps,
                    # exit tap).
                    if (self.grounding == "dmd_on_drafts" and k == 1
                            and source == "exit" and h_fuse.requires_grad):
                        if use_seed_draft:
                            draft = pred
                        else:
                            # flow param: x0 = x_t - sigma * v
                            draft = x_t.float() - sigma_.float() * pred.float()
                        drafts.append({
                            "start_frame": target_start_frame,
                            "draft": draft,
                        })

        for key in losses:
            losses[key] = losses[key] / counts[key]
        if drafts:
            losses["_drafts"] = drafts
        return losses

    def _unpatchify(self, x, f, hp, wp):
        # x: [B, f*hp*wp, prod(patch)*out_dim] -> [B, f*pt, out_dim, hp*ph, wp*pw]
        pt, ph, pw = self.patch_size
        c = self.out_dim
        b = x.shape[0]
        x = x.view(b, f, hp, wp, pt, ph, pw, c)
        x = torch.einsum('bfhwpqrc->bfpchqwr', x)
        x = x.reshape(b, f * pt, c, hp * ph, wp * pw)
        return x
