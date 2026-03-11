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
    output_dir = '/data/workspaces/spanwar/results/ssu/rebuttal/mes'
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
    print('Processing file:', filename, 'at resolution:', res)
    if os.path.exists(f"{output_dir}/mes_{res}_{filename.split('.')[0]}.obj"):
        return [filename, res, 'exists']
    
    j = res
    # Get a signed distance function
    V,F = gpy.read_mesh(os.path.join(gt_dir, filename))
    V = gpy.normalize_points(V)
    
    # Create an SDF for the mesh
    sdf = lambda x: gpy.signed_distance(x, V, F)[0]
    gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    psr_screening_weight = 1.
    cone = LieConeSDFReconstruction(np.concatenate([U,S[:,None]],axis=1),
                                        filter_type=3,cut_bbx_factor=1.,filter_results=False,
                                    psr_screening_weight=psr_screening_weight)

    v, f = cone.V, cone.F
    gpy.write_mesh(f"{output_dir}/mes_{res}_{filename.split('.')[0]}.obj", v, f)
    return [filename, res, 'done']

for res in [32]:
    # for filename in filenames:
    #     run_mes(filename, res)
    out = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_mes)(filename, res) for filename in filenames
    )