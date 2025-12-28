import sys
sys.path.append("/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/reconstruction/mes_utils")

import time
import os
import h5py
import gpytoolbox as gpy
import numpy as np
import joblib
from lie_cone import LieConeSDFReconstruction

with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
    filenames = f.read().splitlines()

filenames = [f'{name}.obj' for name in filenames]

def run_mes(filename, res):
    output_dir = '/data/workspaces/spanwar/results/ssu/mes_and_rfta_objs/mes_rfta_objs/mes'
    print('Processing file:', filename, 'at resolution:', res)
    if os.path.exists(f"{output_dir}/mes_{res}_{filename.split('.')[0]}.obj"):
        return [filename, res, 'exists']
    
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
    gpy.write_mesh(f"{output_dir}/mes_{res}_{filename.split('.')[0]}.obj", v, f)
    return [filename, res, 'done']

for res in [32, 64, 128]:
    # for filename in filenames:
    #     run_mes(filename, res)
    out = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_mes)(filename, res) for filename in filenames
    )