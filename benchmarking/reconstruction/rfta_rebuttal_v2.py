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
    V_gt *= s/0.5
    print('minimum and maximum of vertices:', V_gt.min(), V_gt.max(), 'at scale:', s)
    return V_gt, F_gt

def sample_positions(n, random_sampling, sdf):
    if random_sampling:
        U = (np.random.rand(n*n*n, 3)-0.5)*2
    else:
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    U_sdfvals = sdf(U)
    return U, U_sdfvals

def main(filenames, input_obj_dir, output_dir, s):
    for res in [32, 64, 128]:
        for filename in filenames:
            print('Processing file:', filename, 'at resolution:', res, 'with scale:', s)
            # if os.path.exists(f"{output_dir}/rfta_{res}_{filename.split('.')[0]}.obj"):
            #     continue

            V, F = load_mesh(input_obj_dir, filename, s)
            j = res
            
            sdf = lambda x: gpy.signed_distance(x, V, F)[0]
            U, U_sdfvals = sample_positions(j, False, sdf)

            # Reconstruct triangle mesh
            psr_screening_weight = 1.
            Vr, Fr = gpy.reach_for_the_arcs(U, U_sdfvals,parallel=True,screening_weight=psr_screening_weight, return_point_cloud=False, verbose=True)
            s_str = str(s).replace('.', 'p')
            gpy.write_mesh(f"{output_dir}/rfta_{res}_{s_str}_{filename.split('.')[0]}.obj", Vr, Fr)


if __name__ == "__main__":
    s_value = [0.9,0.75,0.5,1.5]
    file_name = '441708.obj'
    for s in s_value:
        input_obj_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
        output_dir = '/data/workspaces/spanwar/results/ssu/rebuttal/rfta_s'
        os.makedirs(output_dir, exist_ok=True)
        main([file_name], input_obj_dir, output_dir, s)