import os
import gpytoolbox as gpy
import numpy as np
from meshplot import plot
from skimage.measure import marching_cubes
import h5py


def get_file_name():
    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
        filenames = f.read().splitlines()
    return filenames    

def run_rfta():
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
    output_dir = '/data/workspaces/spanwar/results/ssu/rebuttal/rfta_v3'
    filenames = get_file_name()

    for res in [32, 64]:
        for name in filenames:
            print('Processing file:', name, 'at resolution:', res)
            j = res
            file_path = os.path.join(gt_dir, f'{name}.obj')
            if os.path.exists(f"{output_dir}/rfta_{res}_{name}.obj"):
                continue
            
            # Get a signed distance function
            V,F = gpy.read_mesh(file_path)
            V = gpy.normalize_points(V)
            
            # Create an SDF for the mesh
            sdf = lambda x: gpy.signed_distance(x, V, F)[0]
            gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
            U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            S = sdf(U)

            # Reconstruct triangle mesh
            # Vr, Fr, P, N = gpy.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, fine_tune_iters=10)
            # Vr, Fr = gpy.reach_for_the_arcs(U, S, verbose=True, parallel=True, return_point_cloud=False, max_points_per_sphere=3, fine_tune_iters=3)
            psr_screening_weight = 1.
            Vr, Fr = gpy.reach_for_the_arcs(U, S, verbose=True, parallel=True, return_point_cloud=False, screening_weight=psr_screening_weight)
            gpy.write_mesh(os.path.join(output_dir, f"rfta_{res}_{name}.obj"), Vr, Fr)


if __name__ == "__main__":
    run_rfta()