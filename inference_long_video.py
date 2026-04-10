#!/usr/bin/env python3
"""
长视频生成脚本 - 使用 Float Token 最优配置

支持生成120秒（1920帧）长视频，测试长期一致性
"""

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist

from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# Float Token 最优配置
FLOAT_TOKEN_OPTIMAL_CONFIG = {
    "use_float_tokens": True,
    "use_hierarchical_float_tokens": True,
    "float_token_num_slots_short": 4,
    "float_token_alpha_short": 0.3,
    "float_token_update_interval_short": 1,
    "float_token_num_slots_mid": 4,
    "float_token_alpha_mid": 0.15,
    "float_token_update_interval_mid": 30,
    "float_token_num_slots_long": 4,
    "float_token_alpha_long": 0.05,
    "float_token_update_interval_long": 90,
    "use_quality_scorer": True,
}

parser = argparse.ArgumentParser(description="长视频生成 - Float Token 最优配置")
parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
parser.add_argument("--prompt", type=str,
                    default="A serene lake surrounded by mountains at sunrise, gentle ripples on the water surface, birds flying in the distance")
parser.add_argument("--output_folder", type=str, default="outputs_long_video")
parser.add_argument("--duration", type=int, default=120, help="Video duration in seconds")
parser.add_argument("--fps", type=int, default=16, help="Frames per second")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--disable_float_tokens", action="store_true")
parser.add_argument("--num_frames_per_batch", type=int, default=81,
                    help="Number of frames to generate in each batch")
args = parser.parse_args()


def setup_environment():
    """设置环境"""
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

    print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40
    torch.set_grad_enabled(False)

    return device, local_rank, low_memory


def load_config():
    """加载配置"""
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    if not hasattr(config, 'model_kwargs'):
        config.model_kwargs = {}

    if not args.disable_float_tokens:
        print("=" * 80)
        print("启用 Float Token 最优配置 (长视频)")
        print("=" * 80)
        for key, value in FLOAT_TOKEN_OPTIMAL_CONFIG.items():
            config.model_kwargs[key] = value
            print(f"  {key}: {value}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Float Tokens 已禁用（用于对比）")
        print("=" * 80)
        config.model_kwargs['use_float_tokens'] = False

    return config


def initialize_pipeline(config, device, low_memory):
    """初始化 pipeline"""
    pipeline = CausalInferencePipeline(config, device=device)

    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        key = 'generator' if not args.use_ema else 'generator_ema'
        if key in state_dict:
            missing, unexpected = pipeline.generator.load_state_dict(state_dict[key], strict=False)
            print(f"Loaded {key} from checkpoint")
            print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        else:
            print(f"Warning: {key} not found in checkpoint")
    else:
        print(f"Warning: Checkpoint not found")

    pipeline = pipeline.to(dtype=torch.bfloat16)

    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    return pipeline


def generate_long_video(pipeline, prompt, device, low_memory, output_path):
    """生成长视频 - 使用 sliding window 方式"""
    total_frames = args.duration * args.fps
    num_frames_per_batch = args.num_frames_per_batch

    print(f"\nGenerating {args.duration}s video ({total_frames} frames @ {args.fps}fps)")
    print(f"Batch size: {num_frames_per_batch} frames per generation")

    # 创建 conditional dict
    text_prompts = [prompt]
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)

    # 分段生成
    all_frames = []
    num_batches = (total_frames + num_frames_per_batch - 1) // num_frames_per_batch

    for batch_idx in range(num_batches):
        # 只在第一个 batch 重置 Float Banks
        if batch_idx == 0 and hasattr(pipeline.generator.model, 'reset_float_banks'):
            pipeline.generator.model.reset_float_banks()
            print("Float banks reset for new video generation")

        start_frame = batch_idx * num_frames_per_batch
        end_frame = min(start_frame + num_frames_per_batch, total_frames)
        current_batch_size = end_frame - start_frame

        print(f"\n[Batch {batch_idx + 1}/{num_batches}] Generating frames {start_frame}-{end_frame-1}")

        # 生成 noise
        sampled_noise = torch.randn(
            [1, current_batch_size, 16, 60, 104],
            device=device, dtype=torch.bfloat16
        )

        # 生成视频段
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            video, latents = pipeline.inference(
                noise=sampled_noise,
                text_prompts=text_prompts,
                return_latents=True,
                low_memory=low_memory,
            )

        # 转换为像素格式
        video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_frames.append(video[0])  # [T, H, W, C]

        print(f"  Generated {current_batch_size} frames, total: {len(all_frames) * num_frames_per_batch}")

        # 清理缓存
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()

    # 合并所有帧
    full_video = torch.cat(all_frames, dim=0)  # [T, H, W, C]
    print(f"\nTotal frames generated: {full_video.shape[0]}")

    # 保存视频
    full_video = 255.0 * full_video
    full_video = full_video.clamp(0, 255).to(torch.uint8)

    print(f"Saving video to: {output_path}")
    write_video(output_path, full_video, fps=args.fps)

    return full_video


def main():
    print("\n" + "=" * 80)
    print(" 长视频生成 - Float Token 最优配置 ")
    print(f" 目标: {args.duration}s @ {args.fps}fps = {args.duration * args.fps} frames")
    print("=" * 80 + "\n")

    device, local_rank, low_memory = setup_environment()
    config = load_config()
    pipeline = initialize_pipeline(config, device, low_memory)

    os.makedirs(args.output_folder, exist_ok=True)

    # 生成文件名
    safe_prompt = "".join(c if c.isalnum() else "_" for c in args.prompt[:50])
    if args.disable_float_tokens:
        output_path = os.path.join(args.output_folder, f"{args.duration}s_no_float_{safe_prompt}.mp4")
    else:
        output_path = os.path.join(args.output_folder, f"{args.duration}s_float_{safe_prompt}.mp4")

    try:
        generate_long_video(pipeline, args.prompt, device, low_memory, output_path)
        print("\n" + "=" * 80)
        print(f" 视频生成成功！")
        print(f" 输出: {output_path}")
        print("=" * 80)
    except Exception as e:
        print(f"\nError generating video: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
