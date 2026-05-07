"""
Offline demo for Self-Forcing.

Runs the same autoregressive block-by-block generation as `demo.py`, but
without the Flask / SocketIO / threading machinery.

Usage:
    python demo_offline.py --enable_torch_compile --use_taehv

"""

import os
import argparse
import urllib.request

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalInferencePipeline
from demo_utils.constant import ZERO_VAE_CACHE
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

DEFAULT_PROMPT = "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides."

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default='./checkpoints/self_forcing_dmd.pt')
parser.add_argument("--config_path", type=str, default='./configs/self_forcing_dmd.yaml')
parser.add_argument('--trt', action='store_true')
parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--output', type=str, default='./videos/offline.mp4')
parser.add_argument('--fps', type=int, default=16)
parser.add_argument('--enable_torch_compile', action='store_true')
parser.add_argument('--enable_fp8', action='store_true')
parser.add_argument('--use_taehv', action='store_true')
args = parser.parse_args()

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

# Load models
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)

text_encoder = WanTextEncoder()

# Global variables for dynamic model switching
current_vae_decoder = None
current_use_taehv = False
fp8_applied = False
torch_compile_applied = False
models_compiled = False


def initialize_vae_decoder(use_taehv=False, use_trt=False):
    """Initialize VAE decoder based on the selected option"""
    global current_vae_decoder, current_use_taehv

    if use_trt:
        from demo_utils.vae import VAETRTWrapper
        current_vae_decoder = VAETRTWrapper()
        return current_vae_decoder

    if use_taehv:
        from demo_utils.taehv import TAEHV
        # Check if taew2_1.pth exists in checkpoints folder, download if missing
        taehv_checkpoint_path = "checkpoints/taew2_1.pth"
        if not os.path.exists(taehv_checkpoint_path):
            print(f"taew2_1.pth not found in checkpoints folder {taehv_checkpoint_path}. Downloading...")
            os.makedirs("checkpoints", exist_ok=True)
            download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            try:
                urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
                print(f"Successfully downloaded taew2_1.pth to {taehv_checkpoint_path}")
            except Exception as e:
                print(f"Failed to download taew2_1.pth: {e}")
                raise

        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        class TAEHVDiffusersWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dtype = torch.float16
                self.taehv = TAEHV(checkpoint_path=taehv_checkpoint_path).to(self.dtype)
                self.config = DotDict(scaling_factor=1.0)

            def decode(self, latents, return_dict=None):
                # n, c, t, h, w = latents.shape
                # low-memory, set parallel=True for faster + higher memory
                return self.taehv.decode_video(latents, parallel=False).mul_(2).sub_(1)

        current_vae_decoder = TAEHVDiffusersWrapper()
    else:
        current_vae_decoder = VAEDecoderWrapper()
        vae_state_dict = torch.load('wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth', map_location="cpu")
        decoder_state_dict = {}
        for key, value in vae_state_dict.items():
            if 'decoder.' in key or 'conv2' in key:
                decoder_state_dict[key] = value
        current_vae_decoder.load_state_dict(decoder_state_dict)

    current_vae_decoder.eval()
    current_vae_decoder.to(dtype=torch.float16)
    current_vae_decoder.requires_grad_(False)
    current_vae_decoder.to(gpu)
    current_use_taehv = use_taehv

    print(f"✅ VAE decoder initialized with {'TAEHV' if use_taehv else 'default VAE'}")
    return current_vae_decoder


# Initialize with default VAE
vae_decoder = initialize_vae_decoder(use_taehv=False, use_trt=args.trt)

transformer = WanDiffusionWrapper(is_causal=True)
state_dict = torch.load(args.checkpoint_path, map_location="cpu")
transformer.load_state_dict(state_dict['generator_ema'])

text_encoder.eval()
transformer.eval()

transformer.to(dtype=torch.float16)
text_encoder.to(dtype=torch.bfloat16)

text_encoder.requires_grad_(False)
transformer.requires_grad_(False)

pipeline = CausalInferencePipeline(
    config,
    device=gpu,
    generator=transformer,
    text_encoder=text_encoder,
    vae=vae_decoder
)

if low_memory:
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
transformer.to(gpu)


def tensor_to_uint8_frame(frame_tensor):
    """Convert a single CHW frame tensor in [-1, 1] to a uint8 HWC numpy array."""
    frame = torch.clamp(frame_tensor.float(), -1., 1.) * 127.5 + 127.5
    frame = frame.to(torch.uint8).cpu().numpy()

    # CHW -> HWC
    if len(frame.shape) == 3:
        frame = np.transpose(frame, (1, 2, 0))

    return frame


@torch.no_grad()
def generate_video(prompt, seed, enable_torch_compile=False, enable_fp8=False, use_taehv=False):
    """Generate video autoregressively, block by block, and return all frames.

    Returns:
        torch.Tensor of shape ``[T, H, W, C]`` with dtype ``uint8`` on CPU.
    """
    global models_compiled, torch_compile_applied, fp8_applied, current_vae_decoder, current_use_taehv

    # Handle VAE decoder switching
    if use_taehv != current_use_taehv:
        print(f"🔄 Switching VAE decoder to {'TAEHV' if use_taehv else 'default VAE'}")
        current_vae_decoder = initialize_vae_decoder(use_taehv=use_taehv)
        # Update pipeline with new VAE decoder
        pipeline.vae = current_vae_decoder

    # Handle FP8 quantization
    if enable_fp8 and not fp8_applied:
        print("🔧 Applying FP8 quantization to transformer")
        from torchao.quantization.quant_api import quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor
        quantize_(transformer, Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))
        fp8_applied = True

    # Text encoding
    print("📝 Encoding text prompt...")
    conditional_dict = text_encoder(text_prompts=[prompt])
    for key, value in conditional_dict.items():
        conditional_dict[key] = value.to(dtype=torch.float16)
    if low_memory:
        gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
        move_model_to_device_with_memory_preservation(
            text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

    # Handle torch.compile if enabled
    torch_compile_applied = enable_torch_compile
    if enable_torch_compile and not models_compiled:
        # Compile transformer and decoder
        transformer.compile(mode="max-autotune-no-cudagraphs")
        if not current_use_taehv and not low_memory and not args.trt:
            current_vae_decoder.compile(mode="max-autotune-no-cudagraphs")

    # Initialize generation
    print("⚙️  Initializing generation...")

    rnd = torch.Generator(gpu).manual_seed(seed)
    # all_latents = torch.zeros([1, 21, 16, 60, 104], device=gpu, dtype=torch.bfloat16)

    pipeline._initialize_kv_cache(batch_size=1, dtype=torch.float16, device=gpu)
    pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.float16, device=gpu)

    # Generation parameters
    num_blocks = 7
    current_start_frame = 0
    num_input_frames = 0
    all_num_frames = [pipeline.num_frame_per_block] * num_blocks
    if current_use_taehv:
        vae_cache = None
    else:
        vae_cache = ZERO_VAE_CACHE
        for i in range(len(vae_cache)):
            vae_cache[i] = vae_cache[i].to(device=gpu, dtype=torch.float16)

    all_frames = []  # list of HWC uint8 numpy arrays
    total_frames = 0

    gen_start_evt = torch.cuda.Event(enable_timing=True)
    gen_end_evt = torch.cuda.Event(enable_timing=True)
    gen_start_evt.record()

    for idx, current_num_frames in enumerate(all_num_frames):
        # Special message for first block with torch.compile
        if idx == 0 and torch_compile_applied and not models_compiled:
            print(f"🔥 Processing block {idx+1}/{len(all_num_frames)} - "
                  f"compiling models (may take 5-10 minutes)...")
            models_compiled = True
        else:
            print(f"🔄 Processing block {idx+1}/{len(all_num_frames)}")

        block_start_evt = torch.cuda.Event(enable_timing=True)
        block_end_evt = torch.cuda.Event(enable_timing=True)
        denoise_start_evt = torch.cuda.Event(enable_timing=True)
        denoise_end_evt = torch.cuda.Event(enable_timing=True)
        decode_start_evt = torch.cuda.Event(enable_timing=True)
        decode_end_evt = torch.cuda.Event(enable_timing=True)

        block_start_evt.record()

        noisy_input = torch.randn([1, current_num_frames, 16, 60, 104], device=gpu, dtype=torch.float16, generator=rnd)

        # Denoising loop
        denoise_start_evt.record()
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones([1, current_num_frames], device=noisy_input.device,
                                  dtype=torch.int64) * current_timestep

            if index < len(pipeline.denoising_step_list) - 1:
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([1 * current_num_frames], device=noisy_input.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = transformer(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length
                )

        denoise_end_evt.record()
        denoise_end_evt.synchronize()
        denoising_time_ms = denoise_start_evt.elapsed_time(denoise_end_evt)
        print(f"⚡ Block {idx+1} denoising completed in {denoising_time_ms:.2f}ms")

        # Record output
        # all_latents[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

        # Update KV cache for next block
        if idx != len(all_num_frames) - 1:
            transformer(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=torch.zeros_like(timestep),
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )

        # Decode to pixels
        print(f"🎨 Decoding block {idx+1} to pixels...")
        decode_start_evt.record()
        if args.trt:
            all_current_pixels = []
            for i in range(denoised_pred.shape[1]):
                is_first_frame = torch.tensor(1.0).cuda().half() if idx == 0 and i == 0 else \
                    torch.tensor(0.0).cuda().half()
                outputs = vae_decoder.forward(denoised_pred[:, i:i + 1, :, :, :].half(), is_first_frame, *vae_cache)
                # outputs = vae_decoder.forward(denoised_pred.float(), *vae_cache)
                current_pixels, vae_cache = outputs[0], outputs[1:]
                all_current_pixels.append(current_pixels.clone())
            pixels = torch.cat(all_current_pixels, dim=1)
            if idx == 0:
                pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
        else:
            if current_use_taehv:
                if vae_cache is None:
                    vae_cache = denoised_pred
                else:
                    denoised_pred = torch.cat([vae_cache, denoised_pred], dim=1)
                    vae_cache = denoised_pred[:, -3:, :, :, :]
                pixels = current_vae_decoder.decode(denoised_pred)
                print(f"denoised_pred shape: {denoised_pred.shape}")
                print(f"pixels shape: {pixels.shape}")
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block
                else:
                    pixels = pixels[:, 12:, :, :, :]

            else:
                pixels, vae_cache = current_vae_decoder(denoised_pred.half(), *vae_cache)
                if idx == 0:
                    pixels = pixels[:, 3:, :, :, :]  # Skip first 3 frames of first block

        decode_end_evt.record()
        decode_end_evt.synchronize()
        decode_time_ms = decode_start_evt.elapsed_time(decode_end_evt)
        print(f"🎨 Block {idx+1} VAE decoding completed in {decode_time_ms:.2f}ms")

        block_end_evt.record()
        block_end_evt.synchronize()
        block_time_ms = block_start_evt.elapsed_time(block_end_evt)
        block_frames = pixels.shape[1]
        print(f"✅ Block {idx+1} completed in {block_time_ms:.2f}ms ({block_frames} frames)")

        current_start_frame += current_num_frames

        # Collect frames from this block
        for frame_idx in range(block_frames):
            all_frames.append(tensor_to_uint8_frame(pixels[0, frame_idx]))
            total_frames += 1

    gen_end_evt.record()
    gen_end_evt.synchronize()
    generation_time_ms = gen_start_evt.elapsed_time(gen_end_evt)
    print(f"🎉 Generation completed in {generation_time_ms:.2f}ms! {total_frames} frames generated")

    # Stack frames into a single [T, H, W, C] uint8 tensor
    video = torch.from_numpy(np.stack(all_frames, axis=0))
    return video

if __name__ == '__main__':
    video = generate_video(
        prompt=args.prompt,
        seed=args.seed,
        enable_torch_compile=args.enable_torch_compile,
        enable_fp8=args.enable_fp8,
        use_taehv=args.use_taehv,
    )

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    write_video(args.output, video, fps=args.fps)
    print(f"📼 Video saved to {args.output}")
