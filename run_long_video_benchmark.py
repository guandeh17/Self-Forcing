#!/usr/bin/env python
"""
Long Video Inference Benchmark Script

Generates 81-frame (8+ second) videos with local_attn_size=21 to trigger
eviction and engage float tokens for long-term consistency evaluation.

Usage:
    python run_long_video_benchmark.py --mode baseline --output_dir outputs/baseline
    python run_long_video_benchmark.py --mode float_v2 --output_dir outputs/float_v2

For quick testing with fewer prompts/frames:
    python run_long_video_benchmark.py --mode float_v2 --num_prompts 2 --num_frames 30

Note on num_frames:
    - Default 81 frames = 27 blocks, triggers many eviction windows
    - For fast iteration, use 30 frames (10 blocks) - still triggers eviction at frame 22+
"""

import argparse
import os
import re
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video
from einops import rearrange

from pipeline.causal_inference import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


def load_prompts_from_file(filepath):
    """Load prompts from a text file, one prompt per line."""
    with open(filepath, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


# Hardcoded prompts - nature scenes that are easy to evaluate visually
PROMPTS = [
    "A golden retriever runs joyfully through a sunlit meadow filled with wildflowers",
    "Ocean waves crashing against rocky cliffs at sunset, dramatic clouds overhead",
    "A timelapse of clouds moving over mountain peaks, sunlight casting long shadows",
    "A campfire burning in a forest clearing at night, stars visible through the trees",
    "Slow motion water droplets falling into a calm pond, creating ripple patterns",
]


def slugify_prompt(prompt: str) -> str:
    """Convert prompt to a safe filename slug."""
    # Take first 50 chars, replace spaces with underscores, remove non-alphanumeric
    short = prompt[:50].strip()
    slug = re.sub(r'[^\w\s-]', '', short)
    slug = re.sub(r'[-\s]+', '_', slug)
    return slug.lower()


def create_config(mode: str):
    """
    Create configuration for the specified mode.
    
    Args:
        mode: Either 'baseline' or 'float_v2'
        
    Returns:
        OmegaConf configuration object
    """
    # Base configuration matching self_forcing_dmd.yaml
    base_config = {
        # Denoising schedule for few-step inference
        "denoising_step_list": [1000, 750, 500, 250],
        "warp_denoising_step": True,
        "timestep_shift": 5.0,
        "num_frame_per_block": 3,
        "context_noise": 0.0,
        "independent_first_frame": False,
        "guidance_scale": 3.0,
    }
    
    # Model kwargs based on mode
    if mode == "baseline":
        # Local attention ON but no float tokens
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
        }
        print("=" * 80)
        print("Mode: BASELINE")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: DISABLED")
        print("=" * 80)
        
    elif mode == "float_v2":
        # Local attention ON with Float KV Bank V2
        # Note: update intervals are in terms of eviction count (not frames)
        # With local_attn_size=21 and 33 frames, we get ~4 evictions.
        # Set intervals to 1, 2, 4 so all tiers activate in short videos.
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,  # V2 uses hierarchical float tokens
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 4,
            "float_token_num_slots_long": 4,
            "float_token_update_interval_short": 1,   # Update every eviction
            "float_token_update_interval_mid": 2,     # Update every 2nd eviction
            "float_token_update_interval_long": 4,    # Update every 4th eviction
        }
        print("=" * 80)
        print("Mode: FLOAT_V2 (Float KV Bank V2)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED")
        print("  - Hierarchical float tokens: ENABLED")
        print("  - Short-term slots: 4")
        print("  - Mid-term slots: 4")
        print("  - Long-term slots: 4")
        print("=" * 80)
    elif mode == "float_v2_sink":
        # Float KV Bank V2 + Attention Sink (keeps first frame permanently)
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
            "sink_size": 1,  # Keep first frame's KV tokens as permanent anchor
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 4,
            "float_token_num_slots_long": 4,
        }
        print("=" * 80)
        print("Mode: FLOAT_V2_SINK (Float KV Bank V2 + Attention Sink)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED (V2)")
        print("  - Attention sink: ENABLED (sink_size=1, keeps first frame)")
        print("=" * 80)
    elif mode == "float_v2_shortonly":
        # Float KV Bank V2 with ONLY the short-term bank active (no mid/long)
        # Key finding: fewer active slots = better subject consistency
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 0,   # Disable mid-term bank
            "float_token_num_slots_long": 0,  # Disable long-term bank
            "float_token_update_interval_short": 1,
        }
        print("=" * 80)
        print("Mode: FLOAT_V2_SHORTONLY (Short-term bank only, 4 slots)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED (V2, short-term only)")
        print("  - 4 slots, update every eviction")
        print("=" * 80)
    elif mode == "float_v2_midlayers":
        # Float KV Bank V2 only in middle transformer layers (10-20 of 30)
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 4,
            "float_token_num_slots_long": 4,
            "float_token_layer_range": [10, 20],  # Only mid-layers 10-19
        }
        print("=" * 80)
        print("Mode: FLOAT_V2_MIDLAYERS (Float KV Bank V2, mid-layers only)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED (V2, layers 10-19 only)")
        print("=" * 80)
    elif mode == "aft":
        # Adaptive Float Tokens with all new features
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,
            "use_layer_adaptive_float_tokens": True,
            "layer_config_preset": "memory_efficient",
            "use_dynamic_intervals": True,
            "use_temporal_coherence": True,
            "use_progressive_activation": True,
            "progressive_warmup_frames": 300,
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 4,
            "float_token_num_slots_long": 4,
        }
        print("=" * 80)
        print("Mode: AFT (Adaptive Float Tokens)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED (V2)")
        print("  - Layer-adaptive config: ENABLED")
        print("  - Dynamic intervals: ENABLED")
        print("  - Temporal coherence: ENABLED")
        print("  - Progressive activation: ENABLED")
        print("=" * 80)
    elif mode == "agft":
        # Attention-Guided Float Tokens (Cycle 1 improvement)
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
            "use_float_tokens": True,
            "use_attention_guided_float_tokens": True,
            "agft_guidance_alpha": 0.1,
            "agft_temporal_weights": [0.5, 0.3, 0.2],
            "agft_num_slots_short": 4,
            "agft_num_slots_mid": 4,
            "agft_num_slots_long": 4,
            "agft_use_guidance_dropout": False,  # Disable for inference
            "agft_guidance_dropout_p": 0.0,
            "agft_update_interval_short": 1,
            "agft_update_interval_mid": 10,
            "agft_update_interval_long": 30,
        }
        print("=" * 80)
        print("Mode: AGFT (Attention-Guided Float Tokens) - Cycle 1")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Attention-guided float tokens: ENABLED")
        print("  - Guidance alpha: 0.1")
        print("  - Hierarchical slots: 4/4/4 (short/mid/long)")
        print("=" * 80)
    elif mode == "qcsg":
        # Query-Conditioned Slot Gating (Cycle 11 improvement)
        model_kwargs = {
            "timestep_shift": 5.0,
            "local_attn_size": 21,  # Sliding window of 21 frames
            "use_float_tokens": True,
            "use_kv_bank_v2": True,
            "use_hierarchical_float_tokens": True,
            "float_token_num_slots_short": 4,
            "float_token_num_slots_mid": 4,
            "float_token_num_slots_long": 4,
            "float_token_update_interval_short": 1,
            "float_token_update_interval_mid": 2,
            "float_token_update_interval_long": 4,
        }
        print("=" * 80)
        print("Mode: QCSG (Query-Conditioned Slot Gating - Cycle 11)")
        print("  - Local attention: ENABLED (local_attn_size=21)")
        print("  - Float tokens: ENABLED (V2 + QCSG)")
        print("  - Soft gating: ENABLED (temperature=0.5)")
        print("  - Temporal decay tau: 150.0")
        print("  - Magnitude-aware K scaling: ENABLED")
        print("=" * 80)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'baseline', 'float_v2', 'float_v2_sink', 'float_v2_midlayers', 'float_v2_shortonly', 'aft', 'agft', or 'qcsg'")
    
    base_config["model_kwargs"] = model_kwargs
    config = OmegaConf.create(base_config)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate long videos (81 frames) to benchmark float tokens"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["baseline", "float_v2", "float_v2_sink", "float_v2_midlayers", "float_v2_shortonly", "aft", "agft", "qcsg"],
        help="Inference mode: baseline, float_v2, float_v2_sink, float_v2_midlayers, float_v2_shortonly, aft, agft, or qcsg"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_long_benchmark",
        help="Directory to save output videos"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/self_forcing_dmd.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/self_forcing_dmd.yaml",
        help="Path to base config (for reference, model_kwargs is overridden)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames to generate (default: 81 = 27 blocks x 3 frames)"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Use EMA weights (default: True)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference"
    )
    parser.add_argument(
        "--low_memory",
        action="store_true",
        default=True,
        help="Enable low memory mode (swap models to CPU during inference, default: True)"
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=5,
        help="Number of prompts to use (default: 5, use all hardcoded prompts)"
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="Path to a text file containing prompts (one per line). Overrides built-in PROMPTS."
    )
    
    args = parser.parse_args()
    
    # Load prompts from file or use hardcoded list
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = PROMPTS
    
    # Slice to num_prompts
    prompts = prompts[:args.num_prompts]
    
    # Validate num_frames is divisible by num_frame_per_block
    num_frame_per_block = 3
    if args.num_frames % num_frame_per_block != 0:
        raise ValueError(
            f"num_frames ({args.num_frames}) must be divisible by "
            f"num_frame_per_block ({num_frame_per_block})"
        )
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create configuration for the specified mode
    config = create_config(args.mode)
    
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Checkpoint: {args.checkpoint_path}")
    print(f"  Seed: {args.seed}")
    print(f"  Number of frames: {args.num_frames}")
    print(f"  Number of blocks: {args.num_frames // num_frame_per_block}")
    print(f"  Use EMA: {args.use_ema}")
    print(f"  Number of prompts: {len(prompts)}")
    print()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = CausalInferencePipeline(config, device=device)
    
    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        
        # Determine which key to use
        key = "generator_ema" if args.use_ema else "generator"
        if key not in state_dict:
            print(f"Warning: {key} not found in checkpoint, trying 'generator'...")
            key = "generator"
        
        missing, unexpected = pipeline.generator.load_state_dict(state_dict[key], strict=False)
        print(f"Loaded weights from '{key}' (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Move to device and set dtype
    pipeline = pipeline.to(dtype=torch.bfloat16)

    # Use low_memory mode to avoid OOM during VAE decode
    free_vram = get_cuda_free_memory_gb(gpu)
    print(f"Free VRAM: {free_vram:.1f} GB")
    use_low_memory = args.low_memory or (free_vram < 20.0)
    print(f"Low memory mode: {use_low_memory}")

    if use_low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        DynamicSwapInstaller.install_model(pipeline.vae, device=gpu)
    else:
        pipeline.text_encoder.to(device=device)
        pipeline.vae.to(device=device)
    pipeline.generator.to(device=device)
    
    # Disable gradients for inference
    torch.set_grad_enabled(False)
    
    print("\n" + "=" * 80)
    print("Starting inference...")
    print("=" * 80 + "\n")
    
    generated_files = []
    
    # Generate video for each prompt
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt}")
        print("-" * 80)
        
        # Reset KV cache between prompts to ensure clean state
        pipeline.kv_cache1 = None
        
        # Create noise tensor
        # Shape: [batch_size=1, num_frames, channels=16, height=60, width=104]
        # This corresponds to 480x832 video (60*8=480, 104*8=832)
        noise = torch.randn(
            [1, args.num_frames, 16, 60, 104],
            device=device,
            dtype=torch.bfloat16
        )
        
        # Generate video - return latents for chunked VAE decode
        _, latents = pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
            return_latents=True,
            initial_latent=None,
            low_memory=use_low_memory,
        )
        # latents shape: [B, T, 16, 60, 104]
        # Decode in chunks of 21 frames to avoid OOM
        chunk_size = 21
        all_frames = []
        for chunk_start in range(0, latents.shape[1], chunk_size):
            chunk_end = min(chunk_start + chunk_size, latents.shape[1])
            chunk_latents = latents[:, chunk_start:chunk_end].clone()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                chunk_video = pipeline.vae.decode_to_pixel(chunk_latents, use_cache=False)
                chunk_video = (chunk_video * 0.5 + 0.5).clamp(0, 1)
            all_frames.append(chunk_video.cpu())
            torch.cuda.empty_cache()
        video = torch.cat(all_frames, dim=1)  # [B, T, C, H, W]

        # Rearrange to [batch, time, height, width, channels] for saving
        video = rearrange(video, 'b t c h w -> b t h w c')

        # Convert to uint8 range [0, 255]
        video_uint8 = (255.0 * video[0]).clamp(0, 255).to(torch.uint8)
        
        # Create filename
        prompt_slug = slugify_prompt(prompt)
        filename = f"{prompt_slug}_{args.mode}.mp4"
        output_path = os.path.join(args.output_dir, filename)
        
        # Save video at 16 fps (81 frames / 16 fps = ~5 seconds)
        # Actually: 81 frames / 16 fps = 5.06 seconds, but user requested ~8 seconds
        # Wait: 81 frames at 16fps = 5.06 seconds, not 8 seconds
        # Let me check: user said "81 frames at 16fps = enough to trigger eviction"
        # 81 frames / 16fps = 5.06 seconds
        # For 8 seconds at 16fps: 8 * 16 = 128 frames
        # But user specifically requested 81 frames, so I'll use 16 fps
        write_video(output_path, video_uint8, fps=16)
        
        generated_files.append(output_path)
        print(f"Saved: {output_path}")
        print(f"  Frames: {video_uint8.shape[0]}")
        print(f"  Resolution: {video_uint8.shape[1]}x{video_uint8.shape[2]}")
        print(f"  Duration: ~{video_uint8.shape[0] / 16:.1f} seconds @ 16fps")
        
        # Clear VAE cache if available
        if hasattr(pipeline.vae, 'model') and hasattr(pipeline.vae.model, 'clear_cache'):
            pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nMode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated {len(generated_files)} videos:")
    for i, path in enumerate(generated_files, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
