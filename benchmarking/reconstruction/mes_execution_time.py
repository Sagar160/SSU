import sys
sys.path.append("/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/reconstruction/mes_utils")

import time
import os
import h5py
import gpytoolbox as gpy
import numpy as np
import joblib
from collections import defaultdict
import math
from lie_cone import LieConeSDFReconstruction

with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
    filenames = f.read().splitlines()
filenames = [f'{name}.obj' for name in filenames]

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
        print(f"{grid_size:9} | {mean:.4f} | {std:.4f} | {min_time:.4f} | {max_time:.4f}")

def run_mes(filename, res):
    j = res
    gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    
    U_int = (U*(res/2) + (res/2)).astype(np.int32)
    with h5py.File(f'/data/workspaces/spanwar/dataset/thingi/thingi_all/gt_Thingi32_NDC_norm/{filename.split(".")[0]}.hdf5', 'r') as f:
        S = f[f'{res}_sdf'][:][U_int[:,0], U_int[:,1], U_int[:,2]]
        S = S*2

    psr_screening_weight = 1.
    cone = LieConeSDFReconstruction(np.concatenate([U,S[:,None]],axis=1),
                                        filter_type=3,cut_bbx_factor=1.,filter_results=False,
                                    psr_screening_weight=psr_screening_weight)
    v, f = cone.V, cone.F

def execution_wrapper(filename, res):
    start_time = time.time()
    run_mes(filename, res)
    end_time = time.time()  
    execution_time = end_time - start_time
    return filename, res, execution_time

if __name__ == "__main__":
    execution_time_results = []
    for res in [32]:
        out = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(execution_wrapper)(filename, res) for filename in filenames
        )
        execution_time_results.extend(out)
    describe_exe_results(execution_time_results)