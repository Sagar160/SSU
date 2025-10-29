import numpy as np
import torch
import fvdb
import fvdb.nn as fvnn
import mesh_tools as mt
from meshplot import plot

class sdfToVDB:
    def __init__(self, threshold = 33):
        self.threshold = threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def vdb_marching_cubes(self, out: fvnn.VDBTensor):
        '''computes marching cubes for a VDBTensor'''
        nv, nf, _ = out.grid.marching_cubes(out.data)
        return nv.jdata.cpu().detach().numpy(), nf.jdata.cpu().detach().numpy()
    
    def plot_vdb(self, 
                 out: fvnn.VDBTensor):
        '''plots a VDBTensor using mesh_tools'''
        nv, nf = self.vdb_marching_cubes(out)
        plot(nv, nf)

    def fetch_numpy_values(self, grid: fvdb.GridBatch, arr: np.array, size:int):
        '''fetches values from a numpy array based on the ijk indices in the grid'''
        ijk = grid.ijk.jdata.cpu().detach().numpy()+(size-1)//2
        
        if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
            # If indices are out of bounds, we can add the maximum value to the indices
            ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
            # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
        
        values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

        return torch.tensor(values, dtype=torch.float32, device=grid.device)

    def custom_subdivide_grid(self, grid: fvdb.GridBatch):
        '''custom subdivision of a grid to create a finer grid:
            [0,    1,    2] -->
            [0, 1, 2, 3, 4]'''
        ijk = grid.ijk.jdata
        m3g = torch.tensor(mt.mesh_grid(3),device=grid.device)-1
        new_ijk = (2*ijk[:, None, :]+ m3g[None, :, :]).view(-1, 3)
        return fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(new_ijk), origins=grid.origins, voxel_sizes=grid.voxel_sizes/2)

    def scaled_sdf(self, sdf_arr: np.array):
        '''scales the SDF array by the threshold value'''
        return (self.threshold-1)*sdf_arr[:, None]
    
    def sdf_to_vdb(self,
                   sdf_arr: np.array, 
                   large_sdf_arr: np.array, 
                   mask: np.array, 
                   size=33):
        '''Converts a SDF array to a VDBTensor with a given size and mask.'''

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)
        
        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device=self.device)-(size-1)//2
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device=self.device))
        
        sdf_values = self.fetch_numpy_values(grid, sdf_arr, size)
        sdf_values = self.scaled_sdf(sdf_values)
        
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(sdf_values))
        
        # convert large sdf to vdb
        big_vdb_grid = self.custom_subdivide_grid(small_vdb.grid)
        sdf_values = self.fetch_numpy_values(big_vdb_grid, large_sdf_arr, 2*size-1)
        

        sdf_values = self.scaled_sdf(sdf_values)

        large_vdb = fvnn.VDBTensor(big_vdb_grid, 
                                    big_vdb_grid.jagged_like(sdf_values))
        return small_vdb, large_vdb
        
    def upscale_sdf_to_vdb(self, small_vdb: fvnn.VDBTensor, 
                            large_sdf_arr: np.array,
                            size):
        '''Upscales a small VDBTensor to a larger one using the large SDF array'''
        big_vdb_grid = self.custom_subdivide_grid(small_vdb.grid)

        sdf_values = self.fetch_numpy_values(big_vdb_grid, large_sdf_arr, 2*size-1)
        sdf_values = self.scaled_sdf(sdf_values)

        large_vdb = fvnn.VDBTensor(big_vdb_grid, 
                                    big_vdb_grid.jagged_like(sdf_values))

        return large_vdb