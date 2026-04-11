#!/usr/bin/env python3
"""
999-Frame Video Generation Script - Query-Conditioned Slot Gating (QCSG)

Generates 999-frame videos (~62.4 seconds at 16fps) using the QCSG
float token mechanism (Cycle 11 improvement).

Usage:
    python inference_999_frame_qcsg.py --prompt "your prompt here"
    python inference_999_frame_qcsg.py --mode qcsg --num_frames 999
"""

import argparse
import torch
import os
import sys
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# QCSG Configuration for 999-frame generation
QCSG_CONFIG = {
    "use_float_tokens": True,
    "use_kv_bank_v2": True,
    "use_hierarchical_float_tokens": True,
    "use_attention_guided_float_tokens": False,
    "float_token_num_slots_short": 4,
    "float_token_num_slots_mid": 4,
    "float_token_num_slots_long": 4,
    "float_token_alpha_short": 0.3,
    "float_token_alpha_mid": 0.15,
    "float_token_alpha_long": 0.05,
    "float_token_update_interval_short": 1,
    "float_token_update_interval_mid": 2,
    "float_token_update_interval_long": 4,
    "use_quality_scorer": True,
    "local_attn_size": 21,
    "timestep_shift": 5.0,
}

# Baseline (no float tokens)
BASELINE_CONFIG = {
    "use_float_tokens": False,
    "use_kv_bank_v2": False,
    "use_attention_guided_float_tokens": False,
    "local_attn_size": 21,
    "timestep_shift": 5.0,
}

parser = argparse.ArgumentParser(description="999-Frame Video Generation - QCSG")
parser.add_argument("--config_path", type=str,
                    default=os.path.join(SCRIPT_DIR, "configs/self_forcing_dmd.yaml"))
parser.add_argument("--checkpoint_path", type=str,
                    default=os.path.join(SCRIPT_DIR, "checkpoints/self_forcing_dmd.pt"))
parser.add_argument("--prompt", type=str,
                    default="A serene mountain landscape at sunrise, with mist rolling through valleys, "
                            "pine trees swaying gently, and a distant lake reflecting the golden light")
parser.add_argument("--prompt_file", type=str, default=None)
parser.add_argument("--output_folder", type=str, default="outputs_999_qcsg")
parser.add_argument("--num_frames", type=int, default=999,
                    help="Number of frames (must be divisible by 3)")
parser.add_argument("--fps", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_ema", action="store_true", default=True)
parser.add_argument("--mode", type=str, default="qcsg",
                    choices=["qcsg", "baseline"])
args = parser.parse_args()

# Ensure num_frames is divisible by 3
if args.num_frames % 3 != 0:
    args.num_frames = (args.num_frames // 3) * 3
    print(f"Adjusted num_frames to {args.num_frames} (must be divisible by 3)")


def setup_environment():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        set_seed(args.seed)

    print(f'Free VRAM: {get_cuda_free_memory_gb(gpu):.1f} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    torch.set_grad_enabled(False)
    return device, local_rank, low_memory


def load_config():
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load(os.path.join(SCRIPT_DIR, "configs/default_config.yaml"))
    config = OmegaConf.merge(default_config, config)

    if not hasattr(config, 'model_kwargs'):
        config.model_kwargs = {}

    selected_config = QCSG_CONFIG if args.mode == "qcsg" else BASELINE_CONFIG
    mode_name = "QCSG (Cycle 11)" if args.mode == "qcsg" else "Baseline (no float tokens)"

    print("=" * 80)
    print(f" Mode: {mode_name}")
    print(f" Target: {args.num_frames} frames @ {args.fps}fps = {args.num_frames/args.fps:.1f}s")
    print("=" * 80)

    for key, value in selected_config.items():
        config.model_kwargs[key] = value

    return config


def initialize_pipeline(config, device, low_memory):
    pipeline = CausalInferencePipeline(config, device=device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            missing, unexpected = pipeline.generator.load_state_dict(
                state_dict[key], strict=False)
            print(f"Loaded {key}: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            print(f"Warning: {key} not found in checkpoint")
    else:
        print("Warning: Checkpoint not found")

    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    return pipeline


def generate_video(pipeline, prompt, device, low_memory, output_path):
    total_frames = args.num_frames
    print(f"\nGenerating {total_frames} frames ({total_frames/args.fps:.1f}s @ {args.fps}fps)")

    text_prompts = [prompt]

    # Reset float banks
    if hasattr(pipeline.generator.model, 'blocks'):
        for block in pipeline.generator.model.blocks:
            if hasattr(block.self_attn, 'reset_float_bank'):
                block.self_attn.reset_float_bank()
        print("Float banks reset")

    # Generate noise for all frames at once
    sampled_noise = torch.randn(
        [1, total_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16
    )

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=text_prompts,
            return_latents=True,
            low_memory=low_memory,
        )

    video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    full_video = video[0]  # [T, H, W, C]

    print(f"Generated {full_video.shape[0]} frames")

    # Save video
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    video_uint8 = (full_video.float() * 255).clamp(0, 255).to(torch.uint8)
    write_video(output_path, video_uint8, fps=args.fps)
    print(f"Saved: {output_path}")

    # Clear cache
    if hasattr(pipeline.vae, 'model') and hasattr(pipeline.vae.model, 'clear_cache'):
        pipeline.vae.model.clear_cache()
    torch.cuda.empty_cache()

    return full_video


def main():
    device, local_rank, low_memory = setup_environment()
    config = load_config()
    pipeline = initialize_pipeline(config, device, low_memory)

    # Load prompts
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file) as f:
            prompts = [l.strip() for l in f if l.strip()]
    else:
        prompts = [args.prompt]

    os.makedirs(args.output_folder, exist_ok=True)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Prompt: {prompt[:80]}...")
        slug = prompt[:50].replace(' ', '_').replace(',', '').replace('.', '')
        slug = ''.join(c for c in slug if c.isalnum() or c == '_')
        filename = f"{i:03d}_{args.mode}_{args.num_frames}f_{slug}.mp4"
        output_path = os.path.join(args.output_folder, filename)

        generate_video(pipeline, prompt, device, low_memory, output_path)

    print(f"\nAll videos saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
