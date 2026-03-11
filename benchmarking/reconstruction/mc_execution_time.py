# Imports
import os
import sys
import gc
if True:
    sys.path.append('src/utils')
    sys.path.append('src')
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import joblib
import torch
import fvdb.nn as fvnn
import mesh_tools as mt
from models import unet as unetModels
import matplotlib.pyplot as plt
import math
from collections import defaultdict

import igl
import fvdb
from meshplot import plot
import time

with open('run/thingi30.txt', 'r') as f:
    filenames = f.read().splitlines()
filenames = [f'{name}.obj' for name in filenames]


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def vdb_marching_cubes(out: fvnn.VDBTensor):
    '''computes marching cubes for a VDBTensor'''
    nv, nf, _ = out.grid.marching_cubes(out.data)
    return nv.jdata.cpu().detach().numpy(), nf.jdata.cpu().detach().numpy()


def fetch_numpy_values(grid: fvdb.GridBatch, arr: np.array, size: int):
    '''fetches values from a numpy array based on the ijk indices in the grid'''
    # ijk = grid.ijk.jdata.cpu().detach().numpy()+(size-1)//2
    ijk = grid.ijk.jdata.cpu().detach().numpy()
    if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
        # If indices are out of bounds, we can add the maximum value to the indices
        ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
        # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
    values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return torch.tensor(values, dtype=torch.float32, device=grid.device)


def scaled_sdf(sdf_arr: np.array, sdf_scaling_value: int):
    '''scales the SDF array by the threshold value'''
    return (sdf_scaling_value-1)*sdf_arr[:, None]


def sdf_to_vdb(sdf_arr: np.array,
               mask: np.array,
               size=33):
    '''Converts a SDF array to a VDBTensor with a given size and mask.'''
    sdf_scaling_value = (size-1)*2 + 1

    #  create a grid of the size without nomalize actual shape
    ijk_mesh_grid = mt.mesh_grid(size)
    ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)

    ijk = torch.tensor(ijk_mesh_grid[mask],
                       dtype=torch.int,
                       device=device)
    grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk),
                                   voxel_sizes=(1/(size-1)),
                                   origins=torch.tensor([0, 0, 0],
                                                        device=device))

    sdf_values = fetch_numpy_values(grid, sdf_arr, size)
    sdf_values = scaled_sdf(sdf_values, sdf_scaling_value)
    return fvnn.VDBTensor(grid, grid.jagged_like(sdf_values))


def sdf_from_mesh(mesh_path: str, grid_n: int):
    '''Generates SDF from a mesh file using igl signed distance.'''
    v, f = igl.read_triangle_mesh(mesh_path)
    v = 2*mt.NDCnormalize(v)
    points = mt.mesh_grid(grid_n, True)
    sdf = igl.signed_distance(points, v, f)[
        0].reshape(grid_n, grid_n, grid_n)/2
    return sdf


def get_all_shifted_positions(vdb_tensor, size, upsample_factor):
    mfg = torch.tensor(mt.mesh_grid(upsample_factor+1),
                       device=vdb_tensor.device) - (upsample_factor//2)
    new_features = []
    for mg in mfg:
        org_ijk = vdb_tensor.grid.ijk.jdata
        ijk = (upsample_factor * org_ijk + mg).view(-1, 3)
        ijk = np.clip(ijk.cpu().detach().numpy(), 0, (size-1)*upsample_factor)
        ijk_vector = ijk - (org_ijk.cpu().detach().numpy() * upsample_factor)
        # Normalize to values between -1 and 1
        ijk_vector = ijk_vector / (upsample_factor // 2)
        ijk_vector = torch.tensor(
            ijk_vector, dtype=torch.float32, device=vdb_tensor.device)
        new_features.append(
            torch.cat([vdb_tensor.data.jdata, ijk_vector], axis=-1))
    return new_features


def prepare_all_inputs(sdf_numpy, grid_size):
    '''prepare voxel 4D input: SDF+displacement'''
    mask_threshold = grid_size*2+1
    mask = mt.make_mask_close(sdf_numpy, 3/5*mask_threshold)
    input_vdb = sdf_to_vdb(sdf_numpy, mask, grid_size)
    return input_vdb


def describe_exe_results(results):
    summary = defaultdict(list)
    for filename, grid_size, exec_time in results:
        summary[grid_size].append(exec_time)

    print("Grid Size | Mean | Std | Min | Max")
    for grid_size, times in summary.items():
        mean = sum(times) / len(times)
        std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
        min_time = min(times)
        max_time = max(times)
        print(
            f"{grid_size:9} | {mean:.4f} | {std:.4f} | {min_time:.4f} | {max_time:.4f}")


def mc_execution_time():

    execution_time_results = []
    for res in [32, 64, 128]:
        for filename in tqdm(filenames):
            # print('Processing file:', filename, 'at resolution:', res)

            start_time = time.time()
            # j = res
            # gx, gy, gz = np.meshgrid(
            #     np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
            # U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            # U_int = (U*(res/2) + (res/2)).astype(np.int32)

            with h5py.File(f'/home/nmaruani/data/gt_Thingi32_NDC_norm/{filename.split(".")[0]}.hdf5', 'r') as f:
                sdf_numpy = f[f'{res}_sdf'][:]
            all_inputs = prepare_all_inputs(sdf_numpy, res+1)
            Vr, Fr = vdb_marching_cubes(-all_inputs)
            mt.export_obj(
                2*Vr-1, Fr, 'mc_meshes/{}_{}.obj'.format(res, filename.split(".")[0]))
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_results.append((filename, res+1, execution_time))
        describe_exe_results(execution_time_results)
    describe_exe_results(execution_time_results)


if __name__ == "__main__":
    mc_execution_time()
