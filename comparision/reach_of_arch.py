import gpytoolbox as gpy
import numpy as np
from meshplot import plot
from skimage.measure import marching_cubes
import h5py

# def mesh_grid(grid_size: int, normalize=False):
#     """create mesh grid with default indexing"""
#     xx, yy, zz = np.mgrid[:grid_size, :grid_size, :grid_size]
#     grid_3d = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
#     if normalize:
#         return 2 * (grid_3d / (grid_size - 1)) - 1
#     return grid_3d

# # read hdf5 file
# with h5py.File("/data/workspaces/spanwar/dataset/preprocessing_nmc_data/data_preprocessing/get_groundtruth_NMC/gt_large/00000020.hdf5", "r") as h5f:
#     sdf_32 = h5f["32_sdf"][:]

# Get a signed distance function
V,F = gpy.read_mesh("/data/workspaces/spanwar/dataset/preprocessing_nmc_data/abc_dataset_objs/00000020/model.obj")
V = gpy.normalize_points(V)*2
print('minimum and maximum of vertices:', V.min(), V.max())
# Create an SDF for the mesh
j = 64
sdf = lambda x: gpy.signed_distance(x, V, F)[0]
gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)
# Choose an initial marching cube for reach_for_the_arcs
V_mc, F_mc = gpy.marching_cubes(S, U, j+1, j+1, j+1)

# Reconstruct triangle mesh
# Vr, Fr, P, N = gpy.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, fine_tune_iters=10)
Vr, Fr = gpy.reach_for_the_arcs(U, S, verbose=True, parallel=True, return_point_cloud=False, max_points_per_sphere=3, fine_tune_iters=50)
gpy.write_mesh("reach_for_the_arcs.ply", Vr, Fr)
