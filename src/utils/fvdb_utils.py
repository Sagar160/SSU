import numpy as np
import torch
import fvdb
import fvdb.nn as fvnn
import mesh_tools as mt
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Marching cubes algorithm to extract mesh from a VDBTensor
def vdb_marching_cubes(out: fvnn.VDBTensor):
    '''computes marching cubes for a VDBTensor'''
    nv, nf, _ = out.grid.marching_cubes(out.data)
    return nv.jdata.cpu().detach().numpy(), nf.jdata.cpu().detach().numpy()


# Fetches values from a numpy array based on the ijk indices in the grid
def fetch_numpy_values(grid: fvdb.GridBatch, arr: np.array, size:int):
    '''fetches values from a numpy array based on the ijk indices in the grid'''
    ijk = grid.ijk.jdata.cpu().detach().numpy()+(size-1)//2
    
    if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
        # If indices are out of bounds, we can add the maximum value to the indices
        ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
        # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
    
    values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    return torch.tensor(values, dtype=torch.float32, device=grid.device)


# Custom subdivision of a grid to create a finer grid
def custom_subdivide_grid(grid: fvdb.GridBatch):
    '''custom subdivision of a grid to create a finer grid:
        [0,    1,    2] -->
        [0, 1, 2, 3, 4]'''
    ijk = grid.ijk.jdata
    m3g = torch.tensor(mt.mesh_grid(3),device=device)-1
    new_ijk = (2*ijk[:, None, :]+ m3g[None, :, :]).view(-1, 3)
    return fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(new_ijk), origins=grid.origins, voxel_sizes=grid.voxel_sizes/2)


# Converts signed distance function (SDF) grids to VDBTensors   
def sdf_to_vdb(sdf_grid: fvnn.VDBTensor, 
               large_sdf_grid: fvnn.VDBTensor, 
               mask: np.array, 
               size=33,
               mode='train'):
    
    '''
    Takes SDF pair grids and a mask, returns small and large VDBTensors. 
    SDF values scaled by size
    '''

    if mode not in ['test']:
        #  create a grid of the size with out nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(size)
        
        # grid contains the (x, y, z) coordinates
        ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)
        
        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], dtype=torch.int, device=device)-(size-1)//2
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), voxel_sizes=(1/(size-1)), origins=torch.tensor([0, 0, 0], device=device))
        sdf_values = fetch_numpy_values(grid, sdf_grid, size)
        small_vdb = fvnn.VDBTensor(grid, grid.jagged_like((size-1)*sdf_values[:, None]))
        
        # extract large sdf grid
        big_vdb_grid = custom_subdivide_grid(small_vdb.grid)
        sdf_values = fetch_numpy_values(big_vdb_grid, large_sdf_grid, 2*size-1)
        large_vdb = fvnn.VDBTensor(big_vdb_grid, big_vdb_grid.jagged_like((size-1)*sdf_values[:, None]))

        return small_vdb, large_vdb

    else:
        #  create a grid of the size with out nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(size)
        
        # grid contains the (x, y, z) coordinates
        ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)
        
        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], dtype=torch.int, device=device)-(size-1)//2
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), voxel_sizes=(1/(size-1)), origins=torch.tensor([0, 0, 0], device=device))
        sdf_values = fetch_numpy_values(grid, sdf_grid, size)
        small_vdb = fvnn.VDBTensor(grid, grid.jagged_like((size-1)*sdf_values[:, None]))
        
        # extract large sdf grid
        big_vdb_grid = custom_subdivide_grid(small_vdb.grid)
        
        return small_vdb, big_vdb_grid