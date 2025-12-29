import os
import gpytoolbox as gpy
import numpy as np
from meshplot import plot
from skimage.measure import marching_cubes
import h5py

# Get a signed distance function
# V,F = gpy.read_mesh("/data/workspaces/spanwar/dataset/preprocessing_nmc_data/abc_dataset_objs/00000020/model.obj")
# V = gpy.normalize_points(V)*2
# print('minimum and maximum of vertices:', V.min(), V.max())
# Create an SDF for the mesh

with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/thingi30.txt', 'r') as f:
    filenames = f.read().splitlines()

filenames = [f'{name}.obj' for name in filenames]
output_dir = '/data/workspaces/spanwar/results/ssu/mes_and_rfta_objs/mes_rfta_objs'

for res in [32, 64, 128]:
    for filename in filenames:
        print('Processing file:', filename, 'at resolution:', res)
        if os.path.exists(f"{output_dir}/rfta_{res}_{filename.split('.')[0]}.obj"):
            continue
        
        j = res
        gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        
        U_int = (U*(res/2) + (res/2)).astype(np.int32)
        with h5py.File(f'/data/workspaces/spanwar/dataset/thingi/thingi_all/gt_Thingi32_NDC_norm/{filename.split(".")[0]}.hdf5', 'r') as f:
            S = f[f'{res}_sdf'][:][U_int[:,0], U_int[:,1], U_int[:,2]]
            S = S*2

        # Reconstruct triangle mesh
        # Vr, Fr, P, N = gpy.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, fine_tune_iters=10)
        Vr, Fr = gpy.reach_for_the_arcs(U, S, verbose=True, parallel=True, return_point_cloud=False, max_points_per_sphere=3, fine_tune_iters=3)
        # print('filename:', filename, Vr.min(), Vr.max())
        gpy.write_mesh(f"{output_dir}/rfta_{res}_{filename.split('.')[0]}.obj", Vr, Fr)
