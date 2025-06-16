
import torch
//editado


ZERO_VAE_CACHE = [
    torch.zeros(1, 16, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 384, 2, 60, 104),
    torch.zeros(1, 192, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 384, 2, 120, 208),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 192, 2, 240, 416),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832),
    torch.zeros(1, 96, 2, 480, 832)
]

feat_names = [f"vae_cache_{i}" for i in range(len(ZERO_VAE_CACHE))]
ALL_INPUTS_NAMES = ["z", "use_cache"] + feat_names
