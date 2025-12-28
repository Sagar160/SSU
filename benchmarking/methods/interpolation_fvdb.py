import os
import h5py
import numpy as np
import fvdb
import torch
import fvdb.nn as fvnn
import trimesh
from sklearn.neighbors import KDTree

import sys
sys.path.append('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/benchmarking/prediction_utils')
from ponq_eval import get_cd_f1_nc

def mesh_grid(grid_size: int, normalize=False):
    """create mesh grid with default indexing"""
    xx, yy, zz = np.mgrid[:grid_size, :grid_size, :grid_size]
    grid_3d = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    if normalize:
        return 2 * (grid_3d / (grid_size - 1)) - 1
    return grid_3d

def trilinear_upsample(small_tensor: fvnn.VDBTensor, large_grid: fvdb.GridBatch):
    new_centers = large_grid.grid_to_world(large_grid.ijk.float())
    new_features = small_tensor.grid.sample_trilinear(
        new_centers, small_tensor.data)
    return fvnn.VDBTensor(large_grid, new_features)

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

def compare_pred_with_gt_sdf(ssu_pred, tri_pred, gt_arr, size):
    (grid_p, data_p) = ssu_pred
    grid_t = tri_pred.grid
    data_t = tri_pred.data
    ijk_p = grid_p.ijk.jdata.cpu().detach().numpy()
    ijk_t = grid_t.ijk.jdata.cpu().detach().numpy()
    assert np.array_equal(ijk_p, ijk_t)
    
    scale = (size-1)*2 
    data_p_common = data_p.jdata.cpu().detach().numpy()/scale
    data_t_common = data_t.jdata.cpu().detach().numpy()
    real_data = gt_arr[ijk_t[:,0], ijk_t[:,1], ijk_t[:,2]]

    # calculate mse, l1 error and mse/l1 per voxel and number of voxels
    mse_p = np.mean((data_p_common.squeeze() - real_data)**2)
    l1_p = np.mean(np.abs(data_p_common.squeeze() - real_data))
    
    mse_t = np.mean((data_t_common.squeeze() - real_data)**2)
    l1_t = np.mean(np.abs(data_t_common.squeeze() - real_data))

    num_voxels = len(ijk_t)
    # print(num_voxels)
    return [mse_p, l1_p, mse_t, l1_t, num_voxels]

def run(filenames, input_dir, gt_dir=None, ssu_pred_dir=None):
    """
    Load a 3D array, apply B-spline interpolation, and save the result.
    """
    sdf_comparison = {33: [], 65: [], 129: []}
    results =  {33: [], 65: [], 129: []}
    
    for filename in filenames:
        print(f"Processing file: {filename}")
        for size in [33, 65, 129]:
            # Load the 3D array
            arr = load_3d_array(filename, input_dir, size)
            
            # create grid
            ijk = mesh_grid(size)
            mask = arr[ijk[:,0], ijk[:,1], ijk[:,2]] < 3/((size-1)*2)
            device = 'cpu'
            ijk_org = torch.tensor(ijk[mask], 
                                dtype=torch.int, 
                                device=device)
            grid = fvdb.gridbatch_from_ijk(ijk_org,
                                            voxel_sizes=(1/(size-1)),
                                            origins=torch.tensor([0, 0, 0], device=device))
            
            # create fvdb tensor
            ijk = grid.ijk.jdata.cpu().detach().numpy()
            values = torch.tensor(arr[ijk[:, 0].astype(int), ijk[:, 1].astype(int), ijk[:, 2].astype(int)], 
                                dtype=torch.float, 
                                device=device)
            small_vdb = fvnn.VDBTensor(grid,
                                    grid.jagged_like(values[:, None]))
            
            # large grid
            upsample_factor = 4
            m5g = torch.tensor(mesh_grid(upsample_factor+1), device=device) - (upsample_factor//2)
            up_ijk = (upsample_factor*grid.ijk.jdata[:, None, :]+ m5g[None, :, :]).view(-1, 3)
            up_ijk = torch.clamp(up_ijk, 0, upsample_factor*(size-1))
            large_grid = fvdb.gridbatch_from_ijk(up_ijk,
                                                voxel_sizes=(1/((size-1)*upsample_factor)),
                                                origins=torch.tensor([0, 0, 0], device=device))
            
            # trilinear
            up_vdb = trilinear_upsample(small_vdb, large_grid)

            # export sdf
            trilinear_sdf_dir = '/data/workspaces/spanwar/results/ssu/trilinear/trilinear_fvdb_sdfs'
            trilinear_sdf_path = os.path.join(trilinear_sdf_dir, f'{size-1}_{filename.split(".")[0]}.nvdb')
            fvdb.save(trilinear_sdf_path, up_vdb.grid, up_vdb.data, compressed=True)
            
            # compare sdf for size 33
            if size == 33 or size == 65 or size == 129:
                grid_p_size, data_p_size = load_prediction(filename, ssu_pred_dir, size-1, device=device)
                # trilinear on same grid
                up_vdb_size = trilinear_upsample(small_vdb, grid_p_size)
                arr_size = load_3d_array(filename, input_dir, (size-1)*4+1)
                sdf_comparison[size].append(compare_pred_with_gt_sdf((grid_p_size, data_p_size), up_vdb_size, arr_size, size)) 
            
            v, f,_ = up_vdb.grid.marching_cubes(up_vdb.data, level=0.0)
            v = v.jdata.cpu().detach().numpy()
            f = f.jdata.cpu().detach().numpy()
        
            pred_mesh = trimesh.Trimesh(vertices=v, faces=f)
            # pred_mesh_v2 = trimesh.Trimesh(vertices=(v-0.5)*2, faces=f)
            
            # export mesh
            # trilinear_dir = '/data/workspaces/spanwar/results/ssu/trilinear/trilinear_fvdb_objs'
            # trilinear_pred_path = os.path.join(trilinear_dir, f'{size}_{filename.split(".")[0]}.obj')
            # pred_mesh_v2.export(trilinear_pred_path)
            
            # load gt mesh
            filename_obj = filename.split('.')[0]
            gt_obj_name = filename_obj + '.obj'
            gt_obj_path = os.path.join(gt_dir, gt_obj_name)
            gt_mesh = trimesh.load(gt_obj_path, force='mesh')


            data = (filename_obj, gt_mesh, pred_mesh)
            result = get_cd_f1_nc(data, scale_gt=1.0, eval_normalization=None)
            results[size].append(result)

    return results, sdf_comparison

def run_parallel(n_jobs=-1):
    # input_dir='/data/workspaces/spanwar/dataset/thingi/thingi_all/gt_Thingi32_NDC_norm'
    input_dir='/data/workspaces/spanwar/dataset/thingi/thingi_large_all/gt_thingi_large'
    gt = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
    ssu_pred_dir = '/data/workspaces/spanwar/results/ssu/test_predictions/77_eval_thingi32'

    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
        filenames = f.read().splitlines()
    filenames = [f'{name}.hdf5' for name in filenames]
    results, sdf_comparison = run(filenames, input_dir, gt, ssu_pred_dir)

    print('number of samples for sdf 32 comparison:', len(sdf_comparison[33]))
    print('number of samples for sdf 64 comparison:', len(sdf_comparison[65]))
    print('number of samples for sdf 128 comparison:', len(sdf_comparison[129]))
    print('num samples for reconstruction sdf 32:', len(results[33]))
    print('num samples for reconstruction sdf 64:', len(results[65]))
    print('num samples for reconstruction sdf 128:', len(results[129]))

    # print(sdf_comparison_32)
    # print(results)

    # print results
    mse_32_p = np.array(sdf_comparison[33])[:,0].mean()
    l1_32_p = np.array(sdf_comparison[33])[:,1].mean()

    mse_32_t = np.array(sdf_comparison[33])[:,2].mean()
    l1_32_t = np.array(sdf_comparison[33])[:,3].mean()
    num_voxels_32 = np.array(sdf_comparison[33])[:,4].sum()
    print(f'SSU 32: mse: {mse_32_p}, l1: {l1_32_p}')
    print(f'Trilinear 32: mse: {mse_32_t}, l1: {l1_32_t}')
    print(f'Number of voxels 32: {num_voxels_32}')
    print('#'*20)

    mse_64_p = np.array(sdf_comparison[65])[:,0].mean()
    l1_64_p = np.array(sdf_comparison[65])[:,1].mean()
    mse_64_t = np.array(sdf_comparison[65])[:,2].mean()
    l1_64_t = np.array(sdf_comparison[65])[:,3].mean()
    num_voxels_64 = np.array(sdf_comparison[65])[:,4].sum()
    print(f'SSU 64: mse: {mse_64_p}, l1: {l1_64_p}')
    print(f'Trilinear 64: mse: {mse_64_t}, l1: {l1_64_t}')
    print(f'Number of voxels 64: {num_voxels_64}')
    print('#'*20)

    mse_128_p = np.array(sdf_comparison[129])[:,0].mean()
    l1_128_p = np.array(sdf_comparison[129])[:,1].mean()
    mse_128_t = np.array(sdf_comparison[129])[:,2].mean()
    l1_128_t = np.array(sdf_comparison[129])[:,3].mean()
    num_voxels_128 = np.array(sdf_comparison[129])[:,4].sum()
    print(f'SSU 128: mse: {mse_128_p}, l1: {l1_128_p}')
    print(f'Trilinear 128: mse: {mse_128_t}, l1: {l1_128_t}')
    print(f'Number of voxels 128: {num_voxels_128}')
    print('#'*20)

    # 33
    results_33 = np.array(results[33])
    cd1_33 = results_33[:, 1].astype(float).mean(axis=0)
    cd2_33 = results_33[:, 2].astype(float).mean(axis=0)
    f1_33 = results_33[:, 3].astype(float).mean(axis=0)
    nc_33 = results_33[:, 4].astype(float).mean(axis=0)
    ecd2_33 = results_33[:, 5].astype(float).mean(axis=0)
    ef1_33 = results_33[:, 6].astype(float).mean(axis=0)
    # print(f'33: cd1*1e-5: {cd1_33*1e5}, cd2*1e-5: {cd2_33*1e5}, f1: {f1_33}, nc: {nc_33}, ecd2: {ecd2_33}, ef1: {ef1_33}')
    print(f'33: method trilinear cd1*1e-5: {cd1_33*1e5:.3f}, cd2*1e-5: {cd2_33*1e5:.3f}, f1: {f1_33:.3f}, nc: {nc_33:.3f}, ecd2: {ecd2_33*1e2:.3f}, ef1: {ef1_33:.3f}')
    # 65
    results_65 = np.array(results[65])
    cd1_65 = results_65[:, 1].astype(float).mean(axis=0)
    cd2_65 = results_65[:, 2].astype(float).mean(axis=0)
    f1_65 = results_65[:, 3].astype(float).mean(axis=0)
    nc_65 = results_65[:, 4].astype(float).mean(axis=0)
    ecd2_65 = results_65[:, 5].astype(float).mean(axis=0)
    ef1_65 = results_65[:, 6].astype(float).mean(axis=0)
    # print(f'65: cd1*1e-5: {cd1_65*1e5}, cd2*1e-5: {cd2_65*1e5}, f1: {f1_65}, nc: {nc_65}, ecd2: {ecd2_65}, ef1: {ef1_65}')
    print(f'65: method trilinear cd1*1e-5: {cd1_65*1e5:.3f}, cd2*1e-5: {cd2_65*1e5:.3f}, f1: {f1_65:.3f}, nc: {nc_65:.3f}, ecd2: {ecd2_65*1e2:.3f}, ef1: {ef1_65:.3f}')
    # 129
    results_129 = np.array(results[129])
    cd1_129 = results_129[:, 1].astype(float).mean(axis=0)
    cd2_129 = results_129[:, 2].astype(float).mean(axis=0)
    f1_129 = results_129[:, 3].astype(float).mean(axis=0)
    nc_129 = results_129[:, 4].astype(float).mean(axis=0)
    ecd2_129 = results_129[:, 5].astype(float).mean(axis=0)
    ef1_129 = results_129[:, 6].astype(float).mean(axis=0)
    # print(f'129: cd1*1e-5: {cd1_129*1e5}, cd2*1e-5: {cd2_129*1e5}, f1: {f1_129}, nc: {nc_129}, ecd2: {ecd2_129}, ef1: {ef1_129}')
    print(f'129: method trilinear cd1*1e-5: {cd1_129*1e5:.3f}, cd2*1e-5: {cd2_129*1e5:.3f}, f1: {f1_129:.3f}, nc: {nc_129:.3f}, ecd2: {ecd2_129*1e2:.3f}, ef1: {ef1_129:.3f}')

if __name__ == "__main__":
    run_parallel()

