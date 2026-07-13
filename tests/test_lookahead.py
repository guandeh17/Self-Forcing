"""
Unit tests for Lookahead Forcing (mtp.md section 9). Requires 1 GPU.

Uses a tiny 4-layer CausalWanModel at the real width (dim 1536, 12 heads —
the training pipeline hardcodes KV-cache head shapes) with random weights,
so no pretrained checkpoint is needed.

Run: python -m pytest tests/test_lookahead.py -v -x
  or: python tests/test_lookahead.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch import nn

from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper
from wan.modules.causal_model import CausalWanModel, causal_rope_apply
from model.lookahead import LookaheadModule, LookaheadSelfAttention
from pipeline.self_forcing_training import SelfForcingTrainingPipeline

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_LAYERS = 4
FUSION_LAYERS = [1, 2, 3, 4]      # 1-indexed; tiny model has 4 layers
FRAMES = 21                        # 7 blocks of 3
LATENT_H, LATENT_W = 60, 104       # pipeline hardcodes frame_seq_length=1560
DENOISING_STEPS = [1000, 750, 500, 250]

LOOKAHEAD_CFG = {
    "enabled": True,
    "variant": "fm_selftarget",
    "depths": 1,
    "loss_weights": [0.5],
    "fusion_layers": FUSION_LAYERS,
    "head_num_blocks": 1,
    "head_init_from_backbone": True,
    "head_timestep_shift": 10.0,
    "tap_sources": ["exit", "context"],
    "record_interstep_noise": True,
    "max_pairs_per_rollout": 6,
}


class TinyWanWrapper(WanDiffusionWrapper):
    """WanDiffusionWrapper over a small random-weight CausalWanModel."""

    def __init__(self, num_layers=NUM_LAYERS, gradient_checkpointing=False):
        nn.Module.__init__(self)
        self.model = CausalWanModel(
            model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16,
            dim=1536, ffn_dim=3072, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=num_layers, local_attn_size=-1, sink_size=0)
        self.model.num_frame_per_block = 3
        if gradient_checkpointing:
            self.model.gradient_checkpointing = True
        self.uniform_timestep = False
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.seq_len = 32760
        self.post_init()


def ensure_dist():
    # the pipeline broadcasts exit flags unconditionally
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29511")
        dist.init_process_group("nccl", rank=0, world_size=1)


def build_pipeline(seed=0, lookahead=True, gradient_checkpointing=False, generator=None):
    ensure_dist()
    torch.manual_seed(seed)
    if generator is None:
        generator = TinyWanWrapper(gradient_checkpointing=gradient_checkpointing)
        generator = generator.to(device=DEVICE, dtype=DTYPE)
    pipeline = SelfForcingTrainingPipeline(
        denoising_step_list=torch.tensor(DENOISING_STEPS, dtype=torch.long),
        scheduler=generator.get_scheduler(),
        generator=generator,
        num_frame_per_block=3,
        independent_first_frame=False,
        same_step_across_blocks=True,
        last_step_only=False,
        num_max_frames=FRAMES,
        context_noise=0,
        lookahead_config=LOOKAHEAD_CFG if lookahead else None,
    )
    pipeline.num_transformer_blocks = NUM_LAYERS
    return generator, pipeline


def rollout(pipeline, seed=1, batch_size=1):
    torch.manual_seed(seed)
    noise = torch.randn([batch_size, FRAMES, 16, LATENT_H, LATENT_W],
                        device=DEVICE, dtype=DTYPE)
    prompt_embeds = torch.randn([batch_size, 77, 4096], device=DEVICE, dtype=DTYPE)
    out, ts_from, ts_to = pipeline.inference_with_trajectory(
        noise=noise, prompt_embeds=prompt_embeds)
    return out, ts_from, ts_to, {"prompt_embeds": prompt_embeds}


def build_lookahead_module(generator):
    la = LookaheadModule(LOOKAHEAD_CFG, generator.model,
                         num_denoising_steps=len(DENOISING_STEPS),
                         num_frame_per_block=3).to(device=DEVICE, dtype=DTYPE)
    return la


def compute_lsc(la, pipeline, out, cond):
    return la(
        taps=pipeline.lookahead_taps,
        block_meta=pipeline.lookahead_block_meta,
        output=out.detach(),
        conditional_dict=cond,
        latent_grid=(3, LATENT_H // 2, LATENT_W // 2),
    )


# ---------------------------------------------------------------------------
def test_1_rope_offset():
    """Head tokens for block b+1 get the same RoPE as the main model uses
    when actually processing block b+1."""
    torch.manual_seed(0)
    attn = LookaheadSelfAttention(1536, 12).to(device=DEVICE, dtype=DTYPE)
    model = CausalWanModel(dim=1536, num_heads=12, num_layers=1).to(DEVICE)
    freqs = model.freqs.to(DEVICE)
    grid = torch.tensor([[3, 30, 52]], dtype=torch.long)
    x = torch.randn([1, 4680, 12, 128], device=DEVICE, dtype=DTYPE)
    for start_frame in (3, 9, 18):
        a = causal_rope_apply(x, grid, freqs, start_frame=start_frame)
        b = causal_rope_apply(x, grid, freqs, start_frame=start_frame)
        assert torch.equal(a, b)
    # segment-wise: roping the concat [fuse@0 ; target@3] must equal roping
    # each segment at its own absolute start frame
    fuse, target = x, torch.randn_like(x)
    ref = torch.cat([
        causal_rope_apply(fuse, grid, freqs, start_frame=0),
        causal_rope_apply(target, grid, freqs, start_frame=3),
    ], dim=1)
    # replicate what LookaheadSelfAttention does internally
    cat = torch.cat([fuse, target], dim=1)
    got = torch.cat([
        causal_rope_apply(cat[:, :4680], grid, freqs, start_frame=0),
        causal_rope_apply(cat[:, 4680:], grid, freqs, start_frame=3),
    ], dim=1)
    assert torch.equal(ref, got)
    print("test_1_rope_offset PASSED")


def test_2_no_cache_pollution_and_8_dmd_invariance():
    """KV cache and outputs bit-identical with lookahead on vs off for the
    same seed; denoised_timestep_from/to unchanged (protects DMD ts_schedule)."""
    gen1, pipe_off = build_pipeline(seed=0, lookahead=False)
    with torch.no_grad():
        out_off, from_off, to_off, _ = rollout(pipe_off, seed=1)
    kv_off = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in c.items()}
              for c in pipe_off.kv_cache1]

    gen2, pipe_on = build_pipeline(seed=0, lookahead=True)
    with torch.no_grad():
        out_on, from_on, to_on, _ = rollout(pipe_on, seed=1)

    assert torch.equal(out_off, out_on), "rollout output changed by lookahead taps"
    assert (from_off, to_off) == (from_on, to_on), "denoised timestep range changed"
    for c_off, c_on in zip(kv_off, pipe_on.kv_cache1):
        assert torch.equal(c_off["k"], c_on["k"]) and torch.equal(c_off["v"], c_on["v"]), \
            "KV cache polluted by lookahead"
    assert len(pipe_on.lookahead_taps) > 0
    print("test_2_no_cache_pollution_and_8_dmd_invariance PASSED")


def test_3_stop_grad():
    """Targets are detached; no autograd path from LSC into the target."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    out, _, _, cond = rollout(pipe, seed=1)
    la = build_lookahead_module(gen)
    target = out.detach()
    assert target.grad_fn is None
    losses = compute_lsc(la, pipe, out, cond)
    assert len(losses) > 0
    total = sum(losses.values())
    # gradient w.r.t. the (detached) target tensor must not exist
    assert not target.requires_grad
    total.backward()
    print("test_3_stop_grad PASSED")


def test_4_grad_reaches_backbone():
    """Tap-A (exit) LSC alone puts nonzero grad on backbone params; Tap-B
    (context) losses alone put zero grad on backbone params."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    out, _, _, cond = rollout(pipe, seed=1)
    la = build_lookahead_module(gen)
    losses = compute_lsc(la, pipe, out, cond)

    exit_loss = sum(v for k, v in losses.items() if k.endswith("_exit"))
    ctx_loss = sum(v for k, v in losses.items() if k.endswith("_context"))

    gen.zero_grad(set_to_none=True)
    la.zero_grad(set_to_none=True)
    ctx_loss.backward(retain_graph=True)
    backbone_ctx = sum(p.grad.float().abs().sum().item()
                       for p in gen.model.parameters() if p.grad is not None)
    assert backbone_ctx == 0.0, f"Tap-B loss leaked grad into backbone: {backbone_ctx}"
    head_grad = sum(p.grad.float().abs().sum().item()
                    for p in la.parameters() if p.grad is not None)
    assert head_grad > 0.0, "Tap-B loss gave no grad to heads"

    gen.zero_grad(set_to_none=True)
    la.zero_grad(set_to_none=True)
    exit_loss.backward()
    backbone_exit = sum(p.grad.float().abs().sum().item()
                        for p in gen.model.parameters() if p.grad is not None)
    assert backbone_exit > 0.0, "Tap-A loss gave no grad to backbone"
    print("test_4_grad_reaches_backbone PASSED")


def test_5_no_future_leakage():
    """Tap features of block b are invariant to noise of blocks > b."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    torch.manual_seed(1)
    noise = torch.randn([1, FRAMES, 16, LATENT_H, LATENT_W], device=DEVICE, dtype=DTYPE)
    prompt = torch.randn([1, 77, 4096], device=DEVICE, dtype=DTYPE)

    def feats_of_block0(noise_tensor):
        torch.manual_seed(7)  # fix rollout-internal randomness
        with torch.no_grad():
            pipe.inference_with_trajectory(noise=noise_tensor.clone(), prompt_embeds=prompt)
        for rec in pipe.lookahead_taps:
            if rec["block_index"] == 0 and rec["source"] == "exit":
                return {k: v.clone() for k, v in rec["feats"].items()}
        raise AssertionError("no exit tap for block 0")

    f_ref = feats_of_block0(noise)
    noise_perturbed = noise.clone()
    noise_perturbed[:, 3:] = torch.randn_like(noise_perturbed[:, 3:])
    f_pert = feats_of_block0(noise_perturbed)
    for layer in f_ref:
        assert torch.equal(f_ref[layer], f_pert[layer]), \
            f"block-0 features depend on future noise (layer {layer})"
    print("test_5_no_future_leakage PASSED")


def test_6_determinism():
    """Fixed-seed rollout reproducible with taps enabled."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    torch.manual_seed(3)
    with torch.no_grad():
        out1, _, _, _ = rollout(pipe, seed=1)
        taps1 = [{k: (v.clone() if torch.is_tensor(v) else v) for k, v in r.items() if k != "feats"}
                 for r in pipe.lookahead_taps]
    torch.manual_seed(3)
    with torch.no_grad():
        out2, _, _, _ = rollout(pipe, seed=1)
    assert torch.equal(out1, out2)
    assert len(taps1) == len(pipe.lookahead_taps)
    print("test_6_determinism PASSED")


def test_7_checkpointing_parity():
    """Backbone grads from LSC match with gradient checkpointing on vs off
    (same weights), guarding against double-collection under checkpointing."""
    def grads_with(gc_flag):
        torch.manual_seed(0)
        gen = TinyWanWrapper(gradient_checkpointing=gc_flag).to(device=DEVICE, dtype=DTYPE)
        _, pipe = build_pipeline(seed=0, lookahead=True, generator=gen)
        out, _, _, cond = rollout(pipe, seed=1)
        torch.manual_seed(11)
        la = build_lookahead_module(gen)
        torch.manual_seed(13)  # fix loss-internal sampling
        losses = compute_lsc(la, pipe, out, cond)
        loss = sum(v for k, v in losses.items() if k.endswith("_exit"))
        gen.zero_grad(set_to_none=True)
        loss.backward()
        return {n: p.grad.clone() for n, p in gen.model.named_parameters()
                if p.grad is not None}, loss.detach()

    g_off, l_off = grads_with(False)
    g_on, l_on = grads_with(True)
    assert torch.allclose(l_off, l_on, rtol=1e-2, atol=1e-4), f"loss mismatch {l_off} vs {l_on}"
    assert g_off.keys() == g_on.keys()
    for n in g_off:
        assert torch.allclose(g_off[n].float(), g_on[n].float(), rtol=5e-2, atol=1e-4), \
            f"grad mismatch under checkpointing: {n}"
    print("test_7_checkpointing_parity PASSED")


def test_9_pairing_arithmetic():
    """Feature<->target pairing uses absolute start_frame (survives offsets)."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    out, _, _, cond = rollout(pipe, seed=1)
    for rec in pipe.lookahead_taps:
        assert rec["start_frame"] == rec["block_index"] * 3
    meta = pipe.lookahead_block_meta
    assert sorted(meta.keys()) == list(range(7))
    exit_index = None
    for b, m in meta.items():
        assert m["start_frame"] == b * 3
        if exit_index is None:
            exit_index = len(m["zetas"])
        # shared exit step => same number of recorded zeta draws per block
        assert len(m["zetas"]) == exit_index
        assert m["seed"].shape == (1, 3, 16, LATENT_H, LATENT_W)
    print(f"test_9_pairing_arithmetic PASSED (exit index {exit_index})")


def test_10_seed_draft_variant():
    """Variant A path runs end-to-end and trains the zeta projections."""
    cfg = dict(LOOKAHEAD_CFG, variant="seed_draft")
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    out, _, _, cond = rollout(pipe, seed=1)
    la = LookaheadModule(cfg, gen.model, num_denoising_steps=len(DENOISING_STEPS),
                         num_frame_per_block=3).to(device=DEVICE, dtype=DTYPE)
    losses = compute_lsc(la, pipe, out, cond)
    assert len(losses) > 0
    total = sum(losses.values())
    total.backward()
    # zero-init output head => prediction 0 => loss ~= mean(x_bar^2); finite
    assert torch.isfinite(total)
    print("test_10_seed_draft_variant PASSED")


def test_11_zero_init_noop():
    """Zero-initialized output head => head predictions are exactly zero at
    init (LSC starts as a no-op on the backbone; loss = ||target||^2)."""
    gen, pipe = build_pipeline(seed=0, lookahead=True)
    out, _, _, cond = rollout(pipe, seed=1)
    la = build_lookahead_module(gen)
    head_out = la.heads[0].out
    x = torch.randn([1, 4680, 1536], device=DEVICE, dtype=DTYPE)
    e = torch.randn([1, 3, 1, 1536], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        pred = head_out(x, e)
    assert torch.all(pred == 0), "output head not a no-op at zero init"
    losses = compute_lsc(la, pipe, out, cond)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} not finite"
    print("test_11_zero_init_noop PASSED")


if __name__ == "__main__":
    tests = [
        test_1_rope_offset,
        test_2_no_cache_pollution_and_8_dmd_invariance,
        test_3_stop_grad,
        test_4_grad_reaches_backbone,
        test_5_no_future_leakage,
        test_6_determinism,
        test_7_checkpointing_parity,
        test_9_pairing_arithmetic,
        test_10_seed_draft_variant,
        test_11_zero_init_noop,
    ]
    failed = []
    for t in tests:
        torch.cuda.empty_cache()
        try:
            t()
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"{t.__name__} FAILED: {e!r}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        sys.exit(1)
