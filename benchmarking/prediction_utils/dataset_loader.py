# Imports
import os
import sys
sys.path.append('../src/utils')
sys.path.append('../src/data_utils')
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import joblib
import torch
import fvdb.nn as fvnn
import mesh_tools as mt

import fvdb
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
        # ijk = grid.ijk.jdata.cpu().detach().numpy()+(size-1)//2
        ijk = grid.ijk.jdata.cpu().detach().numpy()
        
        if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
            # If indices are out of bounds, we can add the maximum value to the indices
            ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
            # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
        
        values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return torch.tensor(values, dtype=torch.float32, device=grid.device)

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
                   sdf_arr: np.array, 
                   mask: np.array, 
                   size=33):

        '''Converts a SDF array to a VDBTensor with a given size and mask.'''
        sdf_scaling_value = (size-1)*2 + 1
        self.sdf_scaling_value = sdf_scaling_value

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(size, size, size, 3)
        
        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        # ijk = torch.tensor(ijk_mesh_grid[mask], 
        #                     dtype=torch.int, 
        #                     device=self.device)-(size-1)//2

        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device=self.device)
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device=self.device))
        
        sdf_values = self.fetch_numpy_values(grid, sdf_arr, size)
        sdf_values = self.scaled_sdf(sdf_values)
        return fvnn.VDBTensor(grid, grid.jagged_like(sdf_values))


class Dataset(Dataset):
    def __init__(self, src_dir,
                 names_set,
                 dataset_grids,
                 upsample_factors,
                 n_jobs=-1):
        
        self.input_dir = src_dir
        self.names_set = names_set
        self.dataset_grids = dataset_grids
        self.mask_threshold = {grid_size: (grid_size)*2+1 for grid_size in dataset_grids}
        self.upsample_factors = upsample_factors
        self.n_jobs = n_jobs

        # helping class
        self.sdfToVDB = sdfToVDB()

        # stepup to read the dataset
        self._read_dataset()  # This will run setup() and read the files in parallel

    def _get_all_shifted_positions(self, vdb_tensor, size, upsample_factor):
        m3g = torch.tensor(mt.mesh_grid(upsample_factor+1), device=vdb_tensor.device) - (upsample_factor//2)

        new_ijks = []
        new_features = []
        for mg in m3g:
            # org_ijk = vdb_tensor.grid.ijk.jdata + (size-1)//2  # shift to positive
            org_ijk = vdb_tensor.grid.ijk.jdata
            ijk = (upsample_factor * org_ijk + mg).view(-1, 3)
            ijk = np.clip(ijk.cpu().detach().numpy(), 0, (size-1)*upsample_factor)
            ijk_vector = ijk - (org_ijk.cpu().detach().numpy() * upsample_factor)
            ijk_vector = ijk_vector / (upsample_factor // 2)  # Normalize to values between -1 and 1
            ijk_vector = torch.tensor(ijk_vector, dtype=torch.float32, device=vdb_tensor.device)

            new_features.append(torch.cat([vdb_tensor.data.jdata, ijk_vector], axis=-1))
            new_ijks.append(torch.tensor(ijk, dtype=torch.int, device=vdb_tensor.device))
        return new_features, new_ijks
    
    def _get_item(self, obj_name):
        '''Read the SDF in h5 file.'''

        sdf_dict = {}
        path = os.path.join(self.input_dir, obj_name + '.hdf5')
        
        with h5py.File(path, 'r') as f:
            # fetch the SDF and output SDF
            sdf_dict['obj_name'] = obj_name
            for grid_size in self.dataset_grids:
                sdf_dict[grid_size] = f[f'{grid_size-1}_sdf'][:]
        
        return sdf_dict

    def _read_dataset(self):
        out = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self._get_item)
                                                (obj_name) for obj_name in tqdm(self.names_set))

        # check for empty set
        if len(out) == 0:
            raise ValueError("No valid SDF data found in the provided dataset.")
        
        # mask SDFs 
        self.masks = {}
        for grid_size in self.dataset_grids:
            self.masks[grid_size] = [mt.make_mask_close(_dict[grid_size], self.mask_threshold[grid_size])  for _dict in out]
        self.out = out

    def _get_vdb_from_sdf(self, index):
        for _input_size in self.dataset_grids:
            if self.upsample_factors[_input_size] not in [2, 4]:
                Warning("Model only trained for upsample factors of 2 and 4 ....")
        
        _dict = self.out[index]

        # create a set to hold the vdb tensors
        output_set = []
        output_set.append([_dict['obj_name'] for _ in range(len(self.dataset_grids))])
        output_set.append(self.dataset_grids)

        # create a mask for the test set
        vdb_tensors = []
        # new_ijkss = {}
        new_featuress = []
        for grid_size in self.dataset_grids:
            input_size = grid_size
            input_sdf = _dict[input_size]
            mask = self.masks[input_size][index]
            self.upsample_factor = self.upsample_factors[grid_size]
            vdb_tensor = self.sdfToVDB.sdf_to_vdb(
                                        sdf_arr=input_sdf,
                                        mask=mask,
                                        size=input_size, #33
                                    )

            new_features, _ = self._get_all_shifted_positions(vdb_tensor, 
                                            size=input_size, 
                                            upsample_factor=self.upsample_factor)
            vdb_tensors.append(vdb_tensor)
            new_featuress.append(new_features)
            # new_ijkss[input_size] = new_ijks

        output_set.append(vdb_tensors)
        # output_set.append(new_ijks)
        output_set.append(new_featuress)
        return tuple(output_set)

    def __len__(self):
        return len(self.out)
    
    def __getitem__(self, index):
        return self._get_vdb_from_sdf(index)


class DataLoader():
    def __init__(self, 
                 input_dir, 
                 dataset_grids,
                 upsample_factors,
                 n_samples=None):
        self.input_dir = input_dir
        self.dataset_grids = dataset_grids
        self.upsample_factors = upsample_factors
        self.n_samples = n_samples
        
    @staticmethod
    def custom_collate_fn_test(batch):
        # batch is a list of tuples: [(vdb_tensor, new_ijks, new_features), ...]
        obj_names, input_sizes, vdb_tensors, new_featuress = zip(*batch)
        obj_names = obj_names[0]
        input_sizes = input_sizes[0]
        vdb_tensors = vdb_tensors[0]
        new_featuress = new_featuress[0]
        
        output = []
        for index in range(len(obj_names)):
            output.append((obj_names[index],
                           input_sizes[index],
                           vdb_tensors[index],
                           new_featuress[index]))
        return output

    def get_vdb_data_loaders(self,
                             test_dataset, 
                             num_workers=0):

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            collate_fn=self.custom_collate_fn_test,
                                            batch_size=1,  # Test loader usually has batch size of 1
                                            shuffle=False, 
                                            num_workers=num_workers)
        return test_dataloader

    def get(self, names_set):
        test_set = names_set

        test_dataset = Dataset(
                        src_dir=self.input_dir,
                        names_set=test_set,
                        dataset_grids=self.dataset_grids,
                        upsample_factors=self.upsample_factors,
                        n_jobs=-1
                    )
        test_dataloader = self.get_vdb_data_loaders(
                                test_dataset=test_dataset
                            )
        return test_dataloader