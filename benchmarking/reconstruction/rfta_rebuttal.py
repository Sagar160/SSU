import os
import gpytoolbox as gpy
import numpy as np
from skimage.measure import marching_cubes
import h5py


## Logic ##

def load_mesh(dir, filename, scale):
    # Get a signed distance function
    V,F = gpy.read_mesh(os.path.join(dir, filename))
    V = gpy.normalize_points(V)*scale
    print('minimum and maximum of vertices:', V.min(), V.max(), 'at scale:', scale)
    # Create an SDF for the mesh
    return V, F

def sample_positions(n, random_sampling, sdf):
    if random_sampling:
        U = (np.random.rand(n*n*n, 3)-0.5)*2
    else:
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    U_sdfvals = sdf(U)
    return U, U_sdfvals

def main(filenames, input_obj_dir, output_dir, scale):
    for res in [32, 64, 128]:
        for filename in filenames:
            print('Processing file:', filename, 'at resolution:', res)
            if os.path.exists(f"{output_dir}/rfta_{res}_{filename.split('.')[0]}.obj"):
                continue

            V, F = load_mesh(input_obj_dir, filename, scale)
            j = res
            # gx, gy, gz = np.meshgrid(np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
            # U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            
            sdf = lambda x: gpy.signed_distance(x, V, F)[0]
            # S = sdf(U)
            U, S = sample_positions(j, False, sdf)

            # Reconstruct triangle mesh
            # Vr, Fr, P, N = gpy.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, fine_tune_iters=10)
            Vr, Fr = gpy.reach_for_the_arcs(U, S, verbose=True, parallel=True, return_point_cloud=False, max_points_per_sphere=3, fine_tune_iters=3)
            gpy.write_mesh(f"{output_dir}/rfta_{res}_{filename.split('.')[0]}.obj", Vr, Fr)


if __name__ == "__main__":
    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/thingi30.txt', 'r') as f:
        filenames = f.read().splitlines()
    filenames = [f'{name}.obj' for name in filenames]
    max_retries = 10
    for scale in [0.5, 1.0, 1.5, 2.0]:
        input_obj_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
        output_dir = f'/data/workspaces/spanwar/results/ssu/rebuttal/rfta_a_{scale/2}'
        os.makedirs(output_dir, exist_ok=True)
        for attempt in range(max_retries):
            try:
                main(filenames, input_obj_dir, output_dir, scale)
                break  # If successful, exit the retry loop
            except Exception as e:
                print(f"Attempt {attempt+1} failed with error: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Moving on to the next scale.")
        # main(filenames, input_obj_dir, output_dir, scale)