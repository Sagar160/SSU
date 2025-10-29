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
def positional_encoding(small_vdb, dim, is_t=False, is_sdf=False):
    '''helps the learning'''
    if is_t==False and is_sdf==False:
        return small_vdb
    if is_sdf:
        feat_s = small_vdb.jdata[:, 0].unsqueeze(1)
    if is_t:
        feat_t = small_vdb.jdata[:, -1].unsqueeze(1)  # time channel
    
    # frequencies
    half_dim = dim // 2
    emb = torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=small_vdb.grid.device)
    emb = 2**emb * torch.pi
    
    # phase shift
    if is_sdf and not is_t:
        emb_s = feat_s.float() * emb[None, :]
        input_feat = small_vdb.jdata[:, 1:]  # exclude sdf channel
        new_feat = torch.cat([feat_s, emb_s.sin(), emb_s.cos(), input_feat], dim=-1)
    if is_t and not is_sdf:
        emb_t = feat_t.float() * emb[None, :]
        input_feat = small_vdb.jdata[:, :-1]  # exclude time channel
        new_feat = torch.cat([input_feat, emb_t.sin(), emb_t.cos(), feat_t], dim=-1)
    if is_t and is_sdf:
        emb_s = feat_s.float() * emb[None, :]
        emb_t = feat_t.float() * emb[None, :]
        input_feat = small_vdb.jdata[:, 1:-1]  # exclude sdf and time channel
        new_feat = torch.cat([feat_s, emb_s.sin(), emb_s.cos(), input_feat, emb_t.sin(), emb_t.cos(), feat_t], dim=-1)
    
    return fvnn.VDBTensor(small_vdb.grid, small_vdb.grid.jagged_like(new_feat))


# model summary and parameters
def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return trainable_params