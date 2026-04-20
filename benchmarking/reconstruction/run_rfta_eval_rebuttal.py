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

# Normalize vertices to fit in a half unit ball
def NDCnormalize(vertices, return_scale=False):
    """normalization in half unit ball"""
    vM = vertices.max(0)
    vm = vertices.min(0)
    scale = np.sqrt(((vM - vm) ** 2).sum(-1))
    mean = (vM + vm) / 2.0
    nverts = (vertices - mean) / scale
    if return_scale:
        return nverts, mean, scale
    return nverts

def run_eval(filename, filename_obj, input_dir, gt_dir, a):
    # load pred mesh
    input_path = os.path.join(input_dir, filename)
    pred_mesh = trimesh.load(input_path, force='mesh')
    v = pred_mesh.vertices
    f = pred_mesh.faces
    # v = NDCnormalize(v, return_scale=False)
    # v = v+0.5
    # print(v.min(), v.max())

    # load gt mesh
    gt_obj_name = filename_obj + '.obj'
    gt_obj_path = os.path.join(gt_dir, gt_obj_name)
    gt_mesh = trimesh.load(gt_obj_path, force='mesh')
    gt_v = gt_mesh.vertices
    gt_f = gt_mesh.faces
    # gt_v = NDCnormalize(gt_v, return_scale=False)
    # gt_v = gt_v*2
    # print(gt_v.min(), gt_v.max())
    gt_mesh = trimesh.Trimesh(vertices=gt_v, faces=gt_f)
    # print(gt_mesh.vertices.min(), gt_mesh.vertices.max())

    v = v*((gt_v.max() - gt_v.min())/2)*(0.25/a)
    v = v/0.5
    v = v+0.5
    pred_mesh = trimesh.Trimesh(vertices=v, faces=f)

    data = (filename_obj, gt_mesh, pred_mesh)
    result = get_cd_f1_nc(data, scale_gt=1.0, eval_normalization=None)
    # print(f'Finished evaluation for {filename_obj}')
    return result
    

if __name__ == "__main__":
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'

    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
        water_filenames = f.read().splitlines()

    for a in [0.25, 0.5, 0.75]:
        print(f"Evaluating for scale: {a}")
        input_dir = f'/data/workspaces/spanwar/results/ssu/rebuttal/rfta_a_{a}'
        for size in [32, 64, 128]:
            # filenames = os.listdir(input_dir)
            filenames = [
                f'rfta_{size}_{f}.obj' for f in water_filenames]
            filenames_obj = [f for f in water_filenames]
            
            out = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(run_eval)(file, file_obj, input_dir, gt_dir, a)for file, file_obj in zip(filenames, filenames_obj)
                )

            print(f'number of samples for size: {size}, method: rfta:', len(out))
            out = np.array(out)
            cd1 = out[:, 1].astype(float).mean(axis=0)
            cd2 = out[:, 2].astype(float).mean(axis=0)
            f1 = out[:, 3].astype(float).mean(axis=0)
            nc = out[:, 4].astype(float).mean(axis=0)
            ecd2 = out[:, 5].astype(float).mean(axis=0)
            ef1 = out[:, 6].astype(float).mean(axis=0)
            # print('size:', size, 'method:', 'rfta', 'CD1  (x 1e-5):', cd1*1e5,
            #         'CD2  (x 1e-5):', cd2*1e5, 'F1:', f1, 'NC:', nc, 'ECD2:', ecd2, 'EF1:', ef1)
            print(f"size: {size} method: rfta CD1 (x 1e-5): {cd1*1e5:.3f} CD2 (x 1e-5): {cd2*1e5:.3f} F1: {f1:.3f} NC: {nc:.3f} ECD2: {ecd2*1e2:.3f} EF1: {ef1:.3f}")