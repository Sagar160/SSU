import numpy as np
import torch
import fvdb
import fvdb.nn as fvnn
import mesh_tools as mt
from meshplot import plot

class sdfToVDB:
    def __init__(self):
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
        ijk = grid.ijk.jdata.cpu().detach().numpy()
        
        if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
            # If indices are out of bounds, we can add the maximum value to the indices
            ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
            # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
        
        values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return torch.tensor(values, dtype=torch.float32, device=grid.device)

    def fetch_numpy_values_shifted(self, ijk, arr: np.array):
        '''fetches values from a numpy array based on the ijk indices in the grid'''
        ijk = ijk.cpu().detach().numpy()
        values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.tensor(values, dtype=torch.float32, device=device)

    def custom_subdivide_grid(self, grid: fvdb.GridBatch, scale, m3g, upshape):
        '''custom subdivision of a grid to create a finer grid:
            [0,    1,    2] -->
            [0, 1, 2, 3, 4]'''
        ijk = grid.ijk.jdata
        # m3g = torch.tensor(mt.mesh_grid(3),device=grid.device)-1
        new_ijk = (scale*ijk[:, None, :]+ m3g[None, :, :]).view(-1, 3)
        new_ijk = np.clip(new_ijk, 0, upshape-1)
        return fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(new_ijk), origins=grid.origins, voxel_sizes=grid.voxel_sizes/2)

    def scaled_sdf(self, sdf_arr: np.array):
        '''scales the SDF array by the threshold value'''
        return (self.sdf_scaling_value-1)*sdf_arr[:, None]
    
    def sdf_to_vdb(self,
                   sdf_scaling_value: int,
                   sdf_arr: np.array, 
                   large_sdf_arr: np.array, 
                   mask: np.array, 
                   upsample_factor: int,
                   unique_random_direction: bool,
                   size=33,
                   is_test=False):

        '''Converts a SDF array to a VDBTensor with a given size and mask.'''
        if size == 33:
            if sdf_scaling_value != 65:
                raise ValueError("sdf_scaling_value must be 65 when size is 33.")
        elif size == 65:
            if sdf_scaling_value != 129:
                raise ValueError("sdf_scaling_value must be 129 when size is 65.")
        else:
            print('Warning: using different scaling')
        self.sdf_scaling_value = sdf_scaling_value

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)
        
        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device=self.device)
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device=self.device))
        
        sdf_values = self.fetch_numpy_values(grid, sdf_arr, size)
        sdf_values = self.scaled_sdf(sdf_values)
        if is_test:
            return fvnn.VDBTensor(grid, grid.jagged_like(sdf_values))
        
        
        mfg=torch.tensor(mt.mesh_grid(upsample_factor+1), device=self.device)-(upsample_factor//2)

        # Randomly select one coordinate from mfg
        if unique_random_direction:
            num_elements = grid.ijk.jdata.shape[0]
            random_indices = torch.randint(0, mfg.shape[0], (num_elements,), device=self.device)
        else:
            random_indices = np.random.randint(0, mfg.shape[0])

        # new ijk coordinates
        selected_mfg = mfg[random_indices]  # Shape: (num_elements, 3)
        new_ijk = (upsample_factor * grid.ijk.jdata) + selected_mfg
        new_ijk_cpu = new_ijk.cpu().detach().numpy()
        new_ijk = np.clip(new_ijk_cpu, 0, large_sdf_arr.shape[0]-1)
        new_ijk = torch.tensor(new_ijk, dtype=torch.int, device=self.device)
    
        direction_vector = new_ijk - (grid.ijk.jdata) * upsample_factor
        normalized_difference = direction_vector/(upsample_factor//2) # values between -1 and 1
        
        shifted_sdf_values = self.fetch_numpy_values_shifted(new_ijk, large_sdf_arr)
        shifted_sdf_values = self.scaled_sdf(shifted_sdf_values)

        # create VDBTensor
        shifted_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(shifted_sdf_values))

        small_features = torch.cat([sdf_values, normalized_difference], dim=-1) 
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_features))

        return small_vdb, shifted_vdb