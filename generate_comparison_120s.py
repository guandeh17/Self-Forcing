#!/usr/bin/env python3
"""
生成两段120秒对比视频：
1. 启用 Float Token
2. 不启用 Float Token

使用相同的 seed 和 prompt 确保公平对比
"""

import subprocess
import os
import sys

# 配置
PROMPT = "A beautiful cinematic scene of a peaceful forest with sunlight filtering through the trees, birds flying, and a gentle stream flowing. The camera slowly pans across the landscape, revealing the natural beauty."
SEED = 42
DURATION = 120  # 秒
FPS = 16
OUTPUT_FOLDER = "outputs_120s_comparison"

# 创建输出目录
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("=" * 80)
print(" 生成两段120秒对比视频 ")
print("=" * 80)
print(f"Prompt: {PROMPT}")
print(f"Duration: {DURATION}s ({DURATION * FPS} frames @ {FPS}fps)")
print(f"Seed: {SEED}")
print(f"Output folder: {OUTPUT_FOLDER}")
print("=" * 80)

# 1. 生成启用 Float Token 的视频
print("\n" + "=" * 80)
print(" [1/2] 生成启用 Float Token 的视频 ")
print("=" * 80)

cmd_float = [
    "python", "inference_long_video.py",
    "--config_path", "configs/self_forcing_dmd.yaml",
    "--checkpoint_path", "checkpoints/self_forcing_dmd.pt",
    "--prompt", PROMPT,
    "--output_folder", OUTPUT_FOLDER,
    "--duration", str(DURATION),
    "--fps", str(FPS),
    "--seed", str(SEED),
    "--num_frames_per_batch", "81",
]

print(f"Running: {' '.join(cmd_float)}")
result1 = subprocess.run(cmd_float, cwd="/content/Self-Forcing")

if result1.returncode != 0:
    print(f"Error generating video with float tokens: {result1.returncode}")
    sys.exit(1)

print("\n✓ 启用 Float Token 的视频生成完成！")

# 2. 生成不启用 Float Token 的视频
print("\n" + "=" * 80)
print(" [2/2] 生成不启用 Float Token 的视频 ")
print("=" * 80)

cmd_no_float = [
    "python", "inference_long_video.py",
    "--config_path", "configs/self_forcing_dmd.yaml",
    "--checkpoint_path", "checkpoints/self_forcing_dmd.pt",
    "--prompt", PROMPT,
    "--output_folder", OUTPUT_FOLDER,
    "--duration", str(DURATION),
    "--fps", str(FPS),
    "--seed", str(SEED),
    "--num_frames_per_batch", "81",
    "--disable_float_tokens",
]

print(f"Running: {' '.join(cmd_no_float)}")
result2 = subprocess.run(cmd_no_float, cwd="/content/Self-Forcing")

if result2.returncode != 0:
    print(f"Error generating video without float tokens: {result2.returncode}")
    sys.exit(1)

print("\n✓ 不启用 Float Token 的视频生成完成！")

# 总结
print("\n" + "=" * 80)
print(" 对比视频生成完成！")
print("=" * 80)
print(f"输出目录: {OUTPUT_FOLDER}")
print(f"\n生成文件:")
print(f"  1. {DURATION}s_float_*.mp4     - 启用 Float Token")
print(f"  2. {DURATION}s_no_float_*.mp4  - 不启用 Float Token")
print("\n对比维度:")
print("  - 长期一致性（角色/场景保持）")
print("  - 画面稳定性")
print("  - 时间连贯性")
print("=" * 80)
