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
import random
import fvdb
import fvdb.nn as fvnn
import mesh_tools as mt
import fvdb_utils as fu

class ABCDataset(Dataset):
    def __init__(self, src_dir,
                 names_set,
                 dataset_grids,
                 input_size,
                 mask_threshold, # in term grid size i,e grid size = 32 -> 3/32
                #  is_crop,
                #  crops_size,
                #  crops_size_probability,
                #  crops_threshold,
                 upsample_factor,
                #  unique_random_direction, 
                 fvdb_saved_dir,
                 max_tries=100, 
                 is_test=False,
                 n_jobs=-1):
        
        self.input_dir = src_dir
        self.names_set = names_set
        self.dataset_grids = dataset_grids
        self.input_size = input_size
        self.mask_threshold = mask_threshold

        self.upsample_factor = upsample_factor #4
        # self.unique_random_direction = unique_random_direction
        self.fvdb_saved_dir = fvdb_saved_dir

        # self.is_crop = is_crop
        # self.crops_size = crops_size
        # self.crops_size_probability = crops_size_probability
        # self.crops_threshold = crops_threshold

        self.max_tries = max_tries
        self.is_test = is_test
        self.n_jobs = n_jobs

        # helping class
        self.sdfToVDB = fu.sdfToVDB(threshold=self.mask_threshold)

        # stepup to read the dataset
        self._read_dataset()  # This will run setup() and read the files in parallel

    
    def _get_item(self, obj_name):
        '''Read the SDF in h5 file.'''

        sdf_dict = {}
        path = os.path.join(self.input_dir, obj_name)
        
        with h5py.File(path, 'r') as f:
            # check if the file has the required datasets
            if '32_sdf' not in f or '64_sdf' not in f or '128_sdf' not in f:
                raise ValueError(f"File {path} does not contain required datasets.")
            
            # fetch the SDF and output SDF
            sdf_dict['obj_name'] = obj_name
            for grid_size in self.dataset_grids:
                sdf_dict[grid_size+1] = f[f'{grid_size}_sdf'][:]
        
        return sdf_dict

    def _cropped_mask(self, mask):
        """
        Randomly crop a 3D array so that the crop contains at least n nonzero elements.
        crop_size: int or tuple (crop_x, crop_y, crop_z)
        threshold: minimum number of nonzero elements required in the crop
        max_tries: maximum number of attempts
        """

        # _crop_size = np.random.choice(self.crops_size, p=self.crops_size_probability)
        # _crop_size = int(_crop_size)
        _crop_size = self.crops_size
        if isinstance(_crop_size, int):
            crop_size = (_crop_size, _crop_size, _crop_size)
        sx, sy, sz = mask.shape
        cx, cy, cz = crop_size

        for _ in range(self.max_tries):
            x = np.random.randint(0, sx - cx + 1)
            y = np.random.randint(0, sy - cy + 1)
            z = np.random.randint(0, sz - cz + 1)
            
            crop = mask[x:x+cx, y:y+cy, z:z+cz]
            crop_threshold = 400

            if np.count_nonzero(crop) >= crop_threshold:
                mask_crop = np.zeros_like(mask, dtype=bool)
                mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop
                return mask_crop
            
        # Ignore threshold - just return the crop
        print(f"Warning: Could not find a valid crop after {self.max_tries} attempts. Returning the last attempt.")
        mask_crop = np.zeros_like(mask, dtype=bool)
        mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop
        return mask_crop

    def _read_dataset(self):
        # if self.is_test:
        out = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self._get_item)
                                                (obj_name) for obj_name in tqdm(self.names_set))

        # check for empty set
        if len(out) == 0:
            raise ValueError("No valid SDF data found in the provided dataset.")
        
        # mask SDFs of 32
        self.mask_32s = [mt.make_mask_close(_dict[33], self.mask_threshold)  for _dict in out]
        self.out = out
        # else:
        #     self.out = [{'obj_name':  name} for name in self.names_set]

    @staticmethod
    def _select_random_grid_pair():
        """
        Select two different values from [16, 32, 64] where one is lower and one is higher
        Returns: (lower_grid_size, higher_grid_size, upscale_factor)
        """
        available_sizes = [32, 64, 128]
        # available_sizes = [16, 32, 64]
        
        # Randomly select two consecutive indices
        idx = np.random.randint(0, len(available_sizes) - 1)
        size1 = available_sizes[idx]
        size2 = available_sizes[idx + 1]

        # Ensure lower and higher are correctly assigned
        lower_size = min(size1, size2)
        higher_size = max(size1, size2)
        
        # Calculate upscale factor
        upscale_factor = higher_size // lower_size
        return int(lower_size), int(higher_size), int(upscale_factor)
    
    def _prepare_vdbs(self, sdf_input, sdf_output, lower_size, upscale_factor):
        def scaled_sdf(threshold, sdf_arr: np.array):
            '''scales the SDF array by the threshold value'''
            # return (threshold-1)*sdf_arr[:, None]
            return (threshold-1)*sdf_arr[:, None]
        
        threshold =  (lower_size)*2+1
        mask = mt.make_mask_close(sdf_input, threshold)
        threshold = self.mask_threshold

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(lower_size+1)
        ijk_mesh_grid = ijk_mesh_grid.reshape(lower_size+1, lower_size+1, lower_size+1, 3)

        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(lower_size)), 
                                        origins=torch.tensor([0, 0, 0]))
        ijk = grid.ijk.jdata
        small_sdf_arr = sdf_input[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        small_sdf_arr = scaled_sdf(threshold, small_sdf_arr)
        small_sdf_arr = torch.tensor(small_sdf_arr, dtype=torch.float32, device='cpu')
        vdb_input = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_sdf_arr))
        # scale and mask
        m3g  = mt.mesh_grid(3)-1
        up_grid = self.sdfToVDB.custom_subdivide_grid(grid, upscale_factor, m3g, (lower_size*2)+1)
        up_ijk = up_grid.ijk.jdata
        out_mask = abs(sdf_output[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]]) < (1)
        up_ijk = up_ijk[out_mask]
        up_filtered_grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(up_ijk), 
                                        voxel_sizes=up_grid.voxel_sizes,
                                        origins=up_grid.origins)
        up_ijk_filtered = up_filtered_grid.ijk.jdata
        # mask = torch.all(up_ijk_filtered % 2 == 0, dim=1)

        large_sdf_arr = sdf_output[up_ijk_filtered[:, 0], up_ijk_filtered[:, 1], up_ijk_filtered[:, 2]]
        large_sdf_arr = scaled_sdf(threshold, large_sdf_arr)
        large_sdf_arr = torch.tensor(large_sdf_arr, dtype=torch.float32, device='cpu')
        
        # small_sdf_arr = large_sdf_arr.clone()
        # small_sdf_arr[~mask] = torch.randn((~mask).sum(), 1, dtype=torch.float32, device='cpu') 
        # small_sdf_arr[mask] = small_sdf_arr[mask] + 0.1 * torch.randn((mask).sum(), 1, dtype=torch.float32, device='cpu')
        # small_sdf_arr = torch.clamp(small_sdf_arr, -10, 10)

        # vector = up_ijk_filtered - (up_ijk_filtered//upscale_factor)*upscale_factor
        # small_sdf_arr = torch.cat([small_sdf_arr, vector], dim=-1) 
        
        # vdb_input = fvnn.VDBTensor(up_filtered_grid, 
        #                             up_filtered_grid.jagged_like(small_sdf_arr))
        vdb_output = fvnn.VDBTensor(up_filtered_grid, 
                                    up_filtered_grid.jagged_like(large_sdf_arr))
        # trilinear
        tri_feat = vdb_input.grid.sample_trilinear(up_filtered_grid.ijk.float(), vdb_input.jdata)
        t = random.uniform(0, 1)
        tri_feat.jdata = t * tri_feat.jdata + (1 - t) * torch.randn_like(tri_feat.jdata)
        tri_vdb = fvnn.VDBTensor(up_filtered_grid, tri_feat)
        return vdb_output, tri_vdb

    def _get_vdb_from_sdf(self, index):
        _dict = self.out[index]
        # lower_size, higher_size, upscale_factor = self._select_random_grid_pair()
        sdf_32 = _dict[33]
        sdf_64 = _dict[65]
        sdf_128 = _dict[129]

        # create a set to hold the vdb tensors
        vdb_set = []
        vdb_set.append(_dict['obj_name'])

        if not self.is_test:
            vdb_input_65_tri, vdb_output_65 = self._prepare_vdbs(sdf_32, sdf_64, lower_size=32, upscale_factor=2)
            _, vdb_output_129 = self._prepare_vdbs(sdf_64, sdf_128, lower_size=64, upscale_factor=2)
            vdb_set.append(vdb_input_65_tri)
            vdb_set.append(vdb_output_65)
            vdb_set.append(vdb_output_129)
            return tuple(vdb_set)

        # check it it a test set
        if self.is_test:
            # crop mask SDFs of 32
            mask_32 = self.mask_32s[index]
            # if self.is_crop:
            #     mask_32_index = self._cropped_mask(mask_32_index)
            
            # create a mask for the test set
            vdb_tensor = self.sdfToVDB.sdf_to_vdb(
                                        sdf_arr=_dict[33],
                                        large_sdf_arr=None,
                                        mask=mask_32,
                                        upsample_factor=None,
                                        unique_random_direction=None,
                                        size=self.input_size, #33
                                        is_test=True
                                    )

            # new_features, new_ijks = self._get_all_shifted_positions(vdb_tensor, 
            #                                 size=self.input_size, 
            #                                 upsample_factor=self.upsample_factor)

            vdb_set.append(vdb_tensor)
            # vdb_set.append(new_ijks)
            # vdb_set.append(new_features)
            vdb_set.append(_dict[129]*(self.mask_threshold-1)) # scale the SDF by the threshold value
            return tuple(vdb_set)

    def __len__(self):
        return len(self.out)
    
    def __getitem__(self, index):
        return self._get_vdb_from_sdf(index)


class ABCDataLoader():
    def __init__(self, 
                 input_dir, 
                 config,
                 n_samples=None):
        self.input_dir = input_dir
        self.config = config
        self.n_samples = n_samples
        self.names_set_for_processing = None

    @staticmethod
    def custom_collate_fn(batch):
        # batch is a list of tuples: [(vdb_32, vdb_64, vdb_128), ...]
        # level 2: two vdbs
        # level 3: three vdbs
        # level 4: four vdbs
        level = len(batch[0])-1 # -1 because first element is obj_name
        if level == 2:
            obj_names, vdb_1s, vdb_2s = zip(*batch)
            return list(obj_names), list(vdb_1s), list(vdb_2s)
        elif level == 3:
            obj_names, vdb_1s, vdb_2s, vdb_3s = zip(*batch)
            return list(obj_names), list(vdb_1s), list(vdb_2s), list(vdb_3s)
        elif level == 4:
            obj_names, vdb_1s, vdb_2s, vdb_3s, vdb_4s = zip(*batch)
            return list(obj_names), list(vdb_1s), list(vdb_2s), list(vdb_3s), list(vdb_4s)
        else:
            raise ValueError(f"Unsupported upscaling (too many objects): workable upscaling are 64, 128, 256, not above 256")
        
    @staticmethod
    def custom_collate_fn_test(batch):
        # batch is a list of tuples: [(vdb_tensor, new_ijks, new_features), ...]
        obj_names, vdb_tensors, actual_sdf = zip(*batch)
        return list(obj_names), list(vdb_tensors), list(actual_sdf)

    def get_vdb_data_loaders(self,
                             train_dataset,
                             val_dataset,
                             test_dataset, 
                             batch_size=1, 
                             shuffle=None, 
                             num_workers=0):
        
        is_eval = False  # This can be set based on your evaluation mode
        if not is_eval:
            train_dataloader =  torch.utils.data.DataLoader(train_dataset, 
                                                collate_fn=self.custom_collate_fn,
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                num_workers=num_workers)
            val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                collate_fn=self.custom_collate_fn,
                                                batch_size=batch_size,
                                                shuffle=True, 
                                                num_workers=num_workers)
        else:
            train_dataloader = None
            val_dataloader = None

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            collate_fn=self.custom_collate_fn_test,
                                            # batch_size=batch_size,
                                            batch_size=1,  # Test loader usually has batch size of 1
                                            shuffle=False, 
                                            num_workers=num_workers)
        return train_dataloader, val_dataloader, test_dataloader


    def split_dataset(self,
                      names_set, 
                      train_ratio=0.6, 
                      val_ratio=0.2):
        """
        Splits the dataset into train, validation, and test sets.
        """
        total_size = len(names_set)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        np.random.shuffle(names_set)
        train_set = names_set[:train_size]
        val_set = names_set[train_size:train_size + val_size]
        test_set = names_set[train_size + val_size:]

        # Ensure right test set, only avoid this in testing cases
        if self.n_samples is None:
            with open('test_names_file.txt', 'r') as f:
                test_set_from_file = f.read().splitlines()
            assert set(test_set) == set(test_set_from_file), "Test set does not match the expected test set from file."

        print(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
        self.names_set_for_processing = train_set + val_set

        return train_set, val_set, test_set

    
    def get(self, names_set):
        if self.n_samples is not None:
            if not isinstance(self.n_samples, int):
                raise ValueError("n_samples must be an integer or None")
            names_set = names_set[:self.n_samples]

        train_set, val_set, test_set = self.split_dataset(names_set, 
                                        train_ratio=0.6, 
                                        val_ratio=0.2)
        
        is_eval = False
        if not is_eval:
            train_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set=train_set,
                dataset_grids=self.config['data']['dataset_grids'],
                input_size=self.config['data']['input_size'],
                mask_threshold=self.config['data']['mask_threshold'],
                # is_crop=self.config['data']['is_crop']['train'],
                # crops_size=self.config['data']['crops_size'],
                # crops_size_probability=self.config['data']['crops_size_probability'],
                # crops_threshold=self.config['data']['crops_threshold'],
                upsample_factor=self.config['data']['upsample_factor'],
                # unique_random_direction=self.config['data']['unique_random_direction'],
                fvdb_saved_dir=self.config['data']['fvdb_saved_dir'],
                n_jobs=-1
            )
            val_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set=val_set,
                dataset_grids=self.config['data']['dataset_grids'],
                input_size=self.config['data']['input_size'],
                mask_threshold=self.config['data']['mask_threshold'],
                # is_crop=self.config['data']['is_crop']['val'],
                # crops_size=self.config['data']['crops_size'],
                # crops_size_probability=self.config['data']['crops_size_probability'],
                # crops_threshold=self.config['data']['crops_threshold'],
                upsample_factor=self.config['data']['upsample_factor'],
                # unique_random_direction=self.config['data']['unique_random_direction'],
                fvdb_saved_dir=self.config['data']['fvdb_saved_dir'],
                n_jobs=-1
            )
        else:
            train_dataset = None
            val_dataset = None
            
        test_dataset = ABCDataset(
            src_dir=self.input_dir,
            names_set=test_set,
            dataset_grids=self.config['data']['dataset_grids'],
            input_size=self.config['data']['input_size'],
            mask_threshold=self.config['data']['mask_threshold'],
            # is_crop=self.config['data']['is_crop']['test'],
            # crops_size=self.config['data']['crops_size'],
            # crops_size_probability=self.config['data']['crops_size_probability'],
            # crops_threshold=self.config['data']['crops_threshold'],
            upsample_factor=self.config['data']['upsample_factor'],
            # unique_random_direction=self.config['data']['unique_random_direction'],
            fvdb_saved_dir=self.config['data']['fvdb_saved_dir'],
            is_test=True,
            n_jobs=-1
        )
        
        train_dataloader, val_dataloader, test_dataloader = self.get_vdb_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )
        return train_dataloader, val_dataloader, test_dataloader

    def get_names_set_for_data_processing(self):
        if self.names_set_for_processing is None:
            raise ValueError("Names set for processing is not initialized. Load the dataset first.")
        return self.names_set_for_processing