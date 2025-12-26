import os
import h5py
import joblib
import numpy as np
import trimesh
from skimage import measure
from sklearn.neighbors import KDTree
import fvdb
import sys
sys.path.append('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/prediction_utils')
from ponq_eval import get_cd_f1_nc

def run_eval(filename, filename_obj, input_dir, gt_dir):
    # load pred mesh
    input_path = os.path.join(input_dir, filename)
    pred_mesh = trimesh.load(input_path, force='mesh')
    v = pred_mesh.vertices/2 + 0.5
    f = pred_mesh.faces
    # v = NDCnormalize(v, return_scale=False)
    pred_mesh = trimesh.Trimesh(vertices=v, faces=f)

    # load gt mesh
    gt_obj_name = filename_obj + '.obj'
    gt_obj_path = os.path.join(gt_dir, gt_obj_name)
    gt_mesh = trimesh.load(gt_obj_path, force='mesh')


    data = (filename_obj, gt_mesh, pred_mesh)
    result = get_cd_f1_nc(data, scale_gt=1.0, eval_normalization=None)
    # print(f'Finished evaluation for {filename_obj}')
    return result
    

if __name__ == "__main__":
    input_dir = 'data'
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'

    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/thingi30.txt', 'r') as f:
        water_filenames = f.read().splitlines()

    for size in [32, 64, 128]:
        # filenames = os.listdir(input_dir)
        filenames = [
            f'rfta_{size}_{f}.obj' for f in water_filenames]
        filenames_obj = [f for f in water_filenames]

        out = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(run_eval)(file, file_obj, input_dir, gt_dir)for file, file_obj in zip(filenames, filenames_obj)
            )

        print(f'number of samples for size: {size}, method: rfta:', len(out))
        out = np.array(out)
        cd1 = out[:, 1].astype(float).mean(axis=0)
        cd2 = out[:, 2].astype(float).mean(axis=0)
        f1 = out[:, 3].astype(float).mean(axis=0)
        nc = out[:, 4].astype(float).mean(axis=0)
        ecd2 = out[:, 5].astype(float).mean(axis=0)
        ef1 = out[:, 6].astype(float).mean(axis=0)
        print('size:', size, 'method:', 'rfta', 'CD1  (x 1e-5):', cd1*1e5,
                'CD2  (x 1e-5):', cd2*1e5, 'F1:', f1, 'NC:', nc, 'ECD2:', ecd2, 'EF1:', ef1)
        