# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self Forcing is a research codebase for training autoregressive video diffusion models that simulate the inference process during training (to eliminate train-test distribution mismatch). It enables real-time, streaming video generation with KV caching on a single RTX 4090. Built on top of [CausVid](https://github.com/tianweiy/CausVid) and [Wan2.1](https://github.com/Wan-Video/Wan2.1).

## Setup

```bash
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

Models are expected at `wan_models/Wan2.1-T2V-1.3B/`. Download with:
```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

## Common Commands

**GUI demo:**
```bash
python demo.py
```

**CLI inference:**
```bash
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```

**Distributed training (8 nodes × 8 GPUs):**
```bash
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```

## Architecture

### Config System
All training and inference is driven by YAML configs loaded with `OmegaConf`. `configs/default_config.yaml` contains shared defaults; task-specific configs (e.g. `configs/self_forcing_dmd.yaml`) are merged on top. Key fields: `trainer` (selects trainer class), `denoising_step_list` (timesteps for few-step inference), `num_frame_per_block`, `independent_first_frame`.

### Core Components

**`wan/`** — Wan2.1 backbone (mostly upstream):
- `wan/modules/model.py` — base `WanModel` transformer
- `wan/modules/causal_model.py` — `CausalWanModel`, extends `WanModel` with KV caching and causal block-sparse attention via `flex_attention` (compiled with `max-autotune-no-cudagraphs`)
- `wan/modules/vae.py` — video VAE
- `wan/modules/t5.py` — UMT5-XXL text encoder

**`utils/wan_wrapper.py`** — Three wrappers used throughout training and inference:
- `WanTextEncoder` — wraps UMT5-XXL, loads weights from `wan_models/`
- `WanVAEWrapper` — wraps video VAE with fixed normalization constants
- `WanDiffusionWrapper` — wraps either `WanModel` (non-causal) or `CausalWanModel` (causal), manages schedulers

**`model/`** — Training-time model logic. `BaseModel` (in `model/base.py`) initializes generator, real/fake score networks, text encoder, and VAE. Subclasses implement specific training objectives:
- `DMD` — Distribution Matching Distillation (main method)
- `SiD` — Score identity Distillation
- `CausVid` — CausVid baseline
- `GAN` — GAN variant
- `ODERegression` — ODE pair regression for initialization

**`trainer/`** — Distributed training loops using FSDP:
- `ScoreDistillationTrainer` (distillation.py) — used for DMD/SiD training
- `GANTrainer`, `ODETrainer`, `DiffusionTrainer`
- All trainers call `launch_distributed_job()` from `utils/distributed.py` for FSDP setup and EMA

**`pipeline/`** — Inference pipelines:
- `SelfForcingTrainingPipeline` — used during training for backward simulation (simulates autoregressive rollout to generate training samples)
- `CausalInferencePipeline` — few-step inference using `denoising_step_list`
- `CausalDiffusionInferencePipeline` — multi-step diffusion inference
- `BidirectionalInferencePipeline` / `BidirectionalDiffusionInferencePipeline` — non-causal variants

### Key Concepts

**KV Caching:** `CausalWanModel` uses per-transformer-block KV caches (`kv_cache1`, `kv_cache2`) allocated as fixed-size tensors (`num_max_frames * frame_seq_length`). Each frame block attends only to itself and prior context via causal block masks.

**Backward Simulation:** During training (`SelfForcingModel._consistency_backward_simulation`), the model first performs autoregressive rollout from noise to get realistic intermediate states, then trains on these states — this is the core Self Forcing innovation to bridge train-test gap.

**Frame Blocks:** Video is processed in blocks of `num_frame_per_block` (typically 3) latent frames. The optional `independent_first_frame` mode uses a `[1, 3, 3, 3, ...]` block structure to support image-to-video.

**Latent space:** Video latents have shape `[B, T, 16, H/8, W/8]`. For 480×832 video: `[B, T, 16, 60, 104]`. The 1.3B model uses 30 transformer blocks with `frame_seq_length=1560` tokens per frame.
