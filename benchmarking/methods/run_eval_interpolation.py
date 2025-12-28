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

def load_3d_array(filename, input_dir, size):
    """
    Load a 3D array from a given file path.
    """
    # load h5py file
    file_path = os.path.join(input_dir, filename)
    with h5py.File(file_path, 'r') as f:
        data = f[f'{size-1}_sdf'][:]  
    return data

def load_prediction(filename, input_dir, size, device='cpu'):
    filename = filename.split('.')[0]
    grid, data, _ = fvdb.load(os.path.join(input_dir, f'{size}_{filename}.nvdb'), device=device)
    return grid, data

def compare_pred_with_gt_sdf(file_name, inter_pred, size):
    file_name = file_name.split('_')[1] + '.hdf5'
    input_dir='/data/workspaces/spanwar/dataset/thingi/thingi_large_all/gt_thingi_large'
    ssu_pred_dir = '/data/workspaces/spanwar/results/ssu/test_predictions/77_eval_thingi32'  
    up_size = (size)*4 + 1  
    gt_arr = load_3d_array(file_name, input_dir, up_size)
    ssu_pred = load_prediction(file_name, ssu_pred_dir, size, device='cpu')

    (grid_p_32, data_p_32) = ssu_pred
    ijk_p = grid_p_32.ijk.jdata.cpu().detach().numpy()
    data_p = data_p_32.jdata.cpu().detach().numpy()/(size*2)
    
    real_data = gt_arr[ijk_p[:,0], ijk_p[:,1], ijk_p[:,2]]
    inter_data = inter_pred[ijk_p[:,0], ijk_p[:,1], ijk_p[:,2]]

    # calculate mse, l1 error and mse/l1 per voxel and number of voxels
    mse_p = np.mean((data_p.squeeze() - real_data)**2)
    l1_p = np.mean(np.abs(data_p.squeeze() - real_data))
    
    mse_m = np.mean((inter_data - real_data)**2)
    l1_m = np.mean(np.abs(inter_data - real_data))

    num_voxels = len(ijk_p)
    # print(num_voxels)
    return [mse_p, l1_p, mse_m, l1_m, num_voxels]



def run_eval(filename, filename_obj, input_dir, gt_dir, method, size):
    # load pred mesh
    input_path = os.path.join(input_dir, filename)
    with h5py.File(input_path, 'r') as f:
        sdf_data = f[f'zoom_{4}_{method}_sdf'][:]
    v, f, _, _ = measure.marching_cubes(sdf_data, level=0.0)
    v = v / (sdf_data.shape[0]-1) 
    if method == 'bspline':
        sdf_comparison = compare_pred_with_gt_sdf(filename, sdf_data, size)
    # v = NDCnormalize(v, return_scale=False)
    pred_mesh = trimesh.Trimesh(vertices=v, faces=f)

    # export mesh
    # pred_mesh_v2 = trimesh.Trimesh(vertices=(v-0.5)*2, faces=f)
    # pred_mesh_v2.export(f'/data/workspaces/spanwar/results/ssu/bispline_objs/{size}_{filename_obj}.obj')

    # load gt mesh
    gt_obj_name = filename_obj + '.obj'
    gt_obj_path = os.path.join(gt_dir, gt_obj_name)
    gt_mesh = trimesh.load(gt_obj_path, force='mesh')


    data = (filename_obj, gt_mesh, pred_mesh)
    result = get_cd_f1_nc(data, scale_gt=1.0, eval_normalization=None)
    # print(f'Finished evaluation for {filename_obj}')
    return sdf_comparison, result
    

if __name__ == "__main__":
    input_dir = '/data/workspaces/spanwar/results/ssu/interpolation'
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'

    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
        water_filenames = f.read().splitlines()

    for size in [32, 64, 128]:
        for method in ['bspline']:
            # filenames = os.listdir(input_dir)
            filenames = [
                f'{size}_{f}_interpolation.hdf5' for f in water_filenames]
            filenames_obj = [f.split('_')[1] for f in filenames]
            out = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(run_eval)(file, file_obj, input_dir, gt_dir, method, size)for file, file_obj in zip(filenames, filenames_obj)
                )
            sdf_comparison = [o[0] for o in out if o[0] is not None]
            if len(sdf_comparison) != 0:
                print('number of samples for sdf comparison:', len(sdf_comparison))
                sdf_comparison = np.array(sdf_comparison)
                mse = sdf_comparison[:,0].mean()
                l1 = sdf_comparison[:,1].mean()

                mse_m = sdf_comparison[:,2].mean()
                l1_m = sdf_comparison[:,3].mean()
                num_voxels = sdf_comparison[:,4].sum()
                print('#'*20)
                print(f'Size: {size}, Method: {method} SSU 32: mse: {mse}, l1: {l1}')
                print(f'Size: {size}, Method: {method} Interpolation 32: mse: {mse_m}, l1: {l1_m}')
                print(f'Number of voxels 32: {num_voxels}')
                print('#'*20)

            out = [o[1] for o in out]
            print(f'number of samples for size: {size}, method: {method}:', len(out))
            out = np.array(out)
            cd1 = out[:, 1].astype(float).mean(axis=0)
            cd2 = out[:, 2].astype(float).mean(axis=0)
            f1 = out[:, 3].astype(float).mean(axis=0)
            nc = out[:, 4].astype(float).mean(axis=0)
            ecd2 = out[:, 5].astype(float).mean(axis=0)
            ef1 = out[:, 6].astype(float).mean(axis=0)
            # print('size:', size, 'method:', method, 'CD1  (x 1e-5):', cd1*1e5,
            #       'CD2  (x 1e-5):', cd2*1e5, 'F1:', f1, 'NC:', nc, 'ECD2:', ecd2, 'EF1:', ef1)
            print(f"size: {size} method: {method} CD1 (x 1e-5): {cd1*1e5:.3f} CD2 (x 1e-5): {cd2*1e5:.3f} F1: {f1:.3f} NC: {nc:.3f} ECD2: {ecd2*1e2:.3f} EF1: {ef1:.3f}")
        