#!/usr/bin/env python3
"""
使用最优 Float Token 配置的推理脚本

这是基于改进的 Float Token 算法的推理脚本，配置为最佳质量。
"""

import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)
from utils.dataset import TextDataset, TextImagePairDataset
from utils.misc import set_seed

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller

# Float Token 最优配置
FLOAT_TOKEN_OPTIMAL_CONFIG = {
    # 启用 Float Tokens
    "use_float_tokens": True,
    "use_hierarchical_float_tokens": True,

    # 分层配置
    # 短期：快速响应，捕捉动态变化
    "float_token_num_slots_short": 4,
    "float_token_alpha_short": 0.3,
    "float_token_update_interval_short": 1,

    # 中期：稳定场景
    "float_token_num_slots_mid": 4,
    "float_token_alpha_mid": 0.15,
    "float_token_update_interval_mid": 30,

    # 长期：锁定布局
    "float_token_num_slots_long": 4,
    "float_token_alpha_long": 0.05,
    "float_token_update_interval_long": 90,

    # 质量评分
    "use_quality_scorer": True,
}

parser = argparse.ArgumentParser(description="推理脚本 - 使用最优 Float Token 配置")
parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml",
                    help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt",
                    help="Path to the checkpoint folder")
parser.add_argument("--prompt", type=str, default="A beautiful sunset over the ocean, with waves gently crashing on the shore.",
                    help="Text prompt for video generation")
parser.add_argument("--prompt_file", type=str, default=None,
                    help="Path to a file containing prompts (one per line)")
parser.add_argument("--output_folder", type=str, default="outputs_float_token",
                    help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=81,
                    help="Number of frames to generate")
parser.add_argument("--num_samples", type=int, default=1,
                    help="Number of samples to generate per prompt")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--use_ema", action="store_true",
                    help="Whether to use EMA parameters")
parser.add_argument("--disable_float_tokens", action="store_true",
                    help="Disable float tokens for comparison")
parser.add_argument("--debug", action="store_true",
                    help="Enable debug mode with verbose logging")
args = parser.parse_args()


def setup_environment():
    """设置环境"""
    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        set_seed(args.seed + local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0
        world_size = 1
        set_seed(args.seed)

    print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
    low_memory = get_cuda_free_memory_gb(gpu) < 40

    torch.set_grad_enabled(False)

    return device, local_rank, world_size, low_memory


def load_config():
    """加载配置"""
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    # 更新模型配置，添加 Float Token 参数
    if not hasattr(config, 'model_kwargs'):
        config.model_kwargs = {}

    # 如果未禁用，添加 Float Token 配置
    if not args.disable_float_tokens:
        print("=" * 80)
        print("启用 Float Token 最优配置:")
        print("=" * 80)
        for key, value in FLOAT_TOKEN_OPTIMAL_CONFIG.items():
            config.model_kwargs[key] = value
            print(f"  {key}: {value}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("Float Tokens 已禁用（用于对比）")
        print("=" * 80)
        # 明确禁用
        config.model_kwargs['use_float_tokens'] = False

    return config


def initialize_pipeline(config, device, low_memory):
    """初始化 pipeline"""
    # Initialize pipeline
    if hasattr(config, 'denoising_step_list'):
        # Few-step inference
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        # Multi-step diffusion inference
        pipeline = CausalDiffusionInferencePipeline(config, device=device)

    # Load checkpoint
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
        print(f"Warning: Checkpoint not found at {args.checkpoint_path}")

    pipeline = pipeline.to(dtype=torch.bfloat16)

    # Move models to device
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
    else:
        pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)

    return pipeline


def get_prompts():
    """获取 prompt 列表"""
    if args.prompt_file and os.path.exists(args.prompt_file):
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    else:
        prompts = [args.prompt]
        print(f"Using single prompt: {args.prompt}")

    return prompts


def generate_video(pipeline, prompt, device, low_memory, output_path):
    """生成单个视频"""
    batch_size = 1
    num_frames = args.num_output_frames

    # 创建 conditional dict
    text_prompts = [prompt] * args.num_samples
    conditional_dict = pipeline.text_encoder(text_prompts=text_prompts)

    # 初始化 noise
    sampled_noise = torch.randn(
        [args.num_samples, num_frames, 16, 60, 104],
        device=device, dtype=torch.bfloat16
    )

    # 重置 KV cache 和 Float Bank
    if hasattr(pipeline, 'kv_cache1') and pipeline.kv_cache1 is not None:
        for block_index in range(len(pipeline.kv_cache1)):
            pipeline.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)
            pipeline.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=device)

    # 重置 cross-attention cache
    if hasattr(pipeline, 'crossattn_cache'):
        for block_index in range(len(pipeline.crossattn_cache)):
            pipeline.crossattn_cache[block_index]["is_init"] = False

    # 重置 Float Banks（如果启用）
    if hasattr(pipeline.generator.model, 'reset_float_banks'):
        pipeline.generator.model.reset_float_banks()
        print("Float banks reset for new video generation")

    print(f"Generating {num_frames} frames...")

    # 生成视频
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=text_prompts,
            return_latents=True,
            low_memory=low_memory,
        )

    # 转换为视频格式
    video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    video = 255.0 * video

    # 保存视频
    for seed_idx in range(args.num_samples):
        if args.num_samples > 1:
            output_file = output_path.replace('.mp4', f'_{seed_idx}.mp4')
        else:
            output_file = output_path

        write_video(output_file, video[seed_idx], fps=16)
        print(f"Video saved to: {output_file}")

    # 清理 VAE cache
    pipeline.vae.model.clear_cache()

    return video


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print(" Float Token 最优配置推理 ")
    print("=" * 80 + "\n")

    # 设置环境
    device, local_rank, world_size, low_memory = setup_environment()

    # 加载配置
    config = load_config()

    # 初始化 pipeline
    pipeline = initialize_pipeline(config, device, low_memory)

    # 获取 prompts
    prompts = get_prompts()

    # 创建输出目录
    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    # 生成视频
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: {prompt[:80]}...")

        # 生成文件名
        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:50])
        if args.disable_float_tokens:
            output_path = os.path.join(args.output_folder, f"{i:04d}_no_float_{safe_prompt}.mp4")
        else:
            output_path = os.path.join(args.output_folder, f"{i:04d}_float_{safe_prompt}.mp4")

        try:
            generate_video(pipeline, prompt, device, low_memory, output_path)
        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(" 推理完成！")
    print(f" 输出目录: {args.output_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()
