import torch
import random
import numpy as np
import fvdb.nn as fvnn


# redproducibility
def set_reproducibility(is_reproducible=True, seed=42):
    if is_reproducible:
        print(f"Setting reproducibility with seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass
    else:
        print("Reproducibility is not set. Random seeds will not be fixed.")

# position encoding dimension
def positional_encoding(small_vdb, dim):
    '''helps the learning'''
    feat = small_vdb.jdata
    half_dim = dim // 2
    emb = torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=feat.device)
    emb = 2**emb * torch.pi
    emb = feat.float() * emb[None, :]
    new_feat = torch.cat([feat, emb.sin(), emb.cos()], dim=-1)
    return fvnn.VDBTensor(small_vdb.grid, small_vdb.grid.jagged_like(new_feat))


# model summary and parameters
def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return trainable_params