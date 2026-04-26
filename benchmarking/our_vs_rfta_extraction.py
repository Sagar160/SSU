import os
import gpytoolbox as gpy
import numpy as np
from skimage.measure import marching_cubes
import h5py


## Logic ##

def load_mesh(dir, filename, s):
    # Get a signed distance function
    mesh_path = os.path.join(dir, filename)
    V_gt, F_gt = gpy.read_mesh(mesh_path)
    V_gt = gpy.normalize_points(V_gt)

    # s = 0.9 # 0.75
    # V_gt *= s/0.5
    V_gt *= s
    print('minimum and maximum of vertices:', V_gt.min(), V_gt.max(), 'at scale:', s)
    return V_gt, F_gt

def sample_positions(n, random_sampling, sdf):
    if random_sampling:
        U = (np.random.rand(n*n*n, 3)-0.5)*2
    else:
        gx, gy, gz = np.meshgrid(np.linspace(-0.5, 0.5, n+1), np.linspace(-0.5, 0.5, n+1), np.linspace(-0.5, 0.5, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    U_sdfvals = sdf(U)
    return U, U_sdfvals

def main(filenames, input_obj_dir, output_dir, s):
    for filename in filenames:
        if not filename.endswith('.obj'):
            continue

        # save sdf as hdf5 include [32, 64, 128]
        with h5py.File(f"{output_dir}/{filename.split('.')[0]}.hdf5", 'w') as f:
            for res in [32, 64, 128]:
                print('Processing file:', filename, 'at resolution:', res, 'with scale:', s)
                # if os.path.exists(f"{output_dir}/{filename.split('.')[0]}.hdf5"):
                #     continue

                V, F = load_mesh(input_obj_dir, filename, s)
                j = res
                
                sdf = lambda x: gpy.signed_distance(x, V, F)[0]
                U, U_sdfvals = sample_positions(j, False, sdf)
                
                # extract sdfs
                distances = gpy.signed_distance(U, V, F)[0]
                sdf_numpy = distances.reshape((j+1, j+1, j+1))
                sdf_numpy = sdf_numpy

                f.create_dataset(f"{res}_sdf", data=sdf_numpy, compression="gzip")
                print(f"Saved SDF for {filename} at resolution {res} to {output_dir}/{filename.split('.')[0]}.hdf5")


if __name__ == "__main__":
    s_value = [0.25]
    input_obj_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
    file_names = os.listdir(input_obj_dir)
    for s in s_value:
        output_dir = f'/data/workspaces/spanwar/results/ssu/rebuttal/our_vs_rfta/mesh/s_{s}'
        os.makedirs(output_dir, exist_ok=True)
        main(file_names, input_obj_dir, output_dir, s)