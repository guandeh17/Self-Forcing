<p align="center">
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (HuggingFace)</a></h3>
</p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.

---


https://github.com/user-attachments/assets/7548c2db-fe03-4ba8-8dd3-52d2c6160739


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### GUI demo
```
python demo.py
```
Note:
* **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.
* You may want to adjust FPS so it plays smoothly on your device.
* The speed can be improved by enabling `torch.compile`, [TAEHV-VAE](https://github.com/madebyollin/taehv/), or using FP8 Linear layers, although the latter two options may sacrifice quality. It is recommended to use `torch.compile` if possible and enable TAEHV-VAE if further speedup is needed.

### CLI Inference
Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder and our [huggingface repo](https://huggingface.co/gdhe17/Self-Forcing/tree/main/checkpoints).

## Docker

To build a container capable of running Self-Forcing, use the following code.
```
docker build . --tag self-forcing
```

Then run it with:
```
docker run -it --rm -v $PWD:/self-forcing -p 5001:5001 --gpus all self-forcing python3 demo.py
```

### Full image

There is also a version of the image that contains the repository and model files necessary, meaning you can run it OOTB.

```
docker run -it -p 5001:5001 --gpus all atmaranto/self-forcing:full python3 demo.py
```

## Training
### Download text prompts and ODE initialized checkpoint
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
Note: Our training algorithm (except for the GAN version) is data-free (**no video data is needed**). For now, we directly provide the ODE initialization checkpoint and will add more instructions on how to perform ODE initialization in the future (which is identical to the process described in the [CausVid](https://github.com/tianweiy/CausVid) repo).

### Self Forcing Training with DMD
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```
Our training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. By implementing gradient accumulation, it should be possible to reproduce the results in less than 16 hours using 8 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
