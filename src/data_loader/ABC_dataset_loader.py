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
import fvdb_utils as fu

class ABCDataset(Dataset):
    def __init__(self, src_dir,
                 names_set,
                 dataset_grids,
                 input_size,
                 mask_threshold, # in term grid size i,e grid size = 32 -> 3/32
                 sdf_scaling_value,
                 is_crop,
                 crops_ratio,
                 crops_threshold,
                 unique_random_direction, 
                 max_tries=100, 
                 is_test=False,
                 n_jobs=-1):
        
        self.input_dir = src_dir
        self.names_set = names_set
        self.dataset_grids = dataset_grids
        self.input_size = input_size
        self.mask_threshold = mask_threshold
        self.sdf_scaling_value = sdf_scaling_value

        self.unique_random_direction = unique_random_direction

        self.is_crop = is_crop
        self.crops_ratio = crops_ratio
        self.crops_threshold = crops_threshold

        self.max_tries = max_tries
        self.is_test = is_test
        self.n_jobs = n_jobs

        # helping class
        self.sdfToVDB = fu.sdfToVDB()

        # stepup to read the dataset
        self._read_dataset()  # This will run setup() and read the files in parallel

    def _get_all_shifted_positions(self, vdb_tensor, size, upsample_factor):
        m3g = torch.tensor(mt.mesh_grid(upsample_factor+1), device=vdb_tensor.device) - (upsample_factor//2)

        new_ijks = []
        new_features = []
        for mg in m3g:
            ijk = vdb_tensor.grid.ijk.jdata
            ijk = (upsample_factor * ijk + mg).view(-1, 3)
            ijk = np.clip(ijk.cpu().detach().numpy(), 0, (size-1)*upsample_factor)
            ijk_vector = ijk - (vdb_tensor.grid.ijk.jdata.cpu().detach().numpy() * upsample_factor)
            ijk_vector = ijk_vector / (upsample_factor // 2)  # Normalize to values between -1 and 1
            ijk_vector = torch.tensor(ijk_vector, dtype=torch.float32, device=vdb_tensor.device)

            new_features.append(torch.cat([vdb_tensor.data.jdata, ijk_vector], axis=-1))
            new_ijks.append(torch.tensor(ijk, dtype=torch.int, device=vdb_tensor.device))
        return new_features, new_ijks
    
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

    def _cropped_mask(self, mask, sdf_size):
        """
        Randomly crop a 3D array so that the crop contains at least n nonzero elements.
        crop_size: int or tuple (crop_x, crop_y, crop_z)
        threshold: minimum number of nonzero elements required in the crop
        max_tries: maximum number of attempts
        """

        _crop_ratio = np.random.choice(self.crops_ratio)
        if _crop_ratio == 1:
            # print("No cropping applied.")
            return mask
        else:
            _crop_size = int(sdf_size * _crop_ratio) + 1
        if isinstance(_crop_size, int):
            crop_size = (_crop_size, _crop_size, _crop_size)
        sx, sy, sz = mask.shape
        cx, cy, cz = crop_size

        best_mask_crop = np.zeros_like(mask, dtype=bool)
        for _ in range(self.max_tries):
            x = np.random.randint(0, sx - cx + 1)
            y = np.random.randint(0, sy - cy + 1)
            z = np.random.randint(0, sz - cz + 1)
            
            crop = mask[x:x+cx, y:y+cy, z:z+cz]
            crop_threshold = self.crops_threshold[sdf_size][_crop_ratio]

            if np.count_nonzero(crop) > np.count_nonzero(best_mask_crop):
                best_mask_crop = np.zeros_like(mask, dtype=bool)
                best_mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop

            if np.count_nonzero(crop) >= crop_threshold:
                mask_crop = np.zeros_like(mask, dtype=bool)
                mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop
                return mask_crop
            
        # Ignore threshold - just return the crop
        # print(f"Warning: Could not find a valid crop after {self.max_tries} attempts. Returning the best attempt.")
        return best_mask_crop

    def _read_dataset(self):
        out = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self._get_item)
                                                (obj_name) for obj_name in tqdm(self.names_set))

        # check for empty set
        if len(out) == 0:
            raise ValueError("No valid SDF data found in the provided dataset.")
        
        # mask SDFs of 32 and 64
        self.mask_32s = [mt.make_mask_close(_dict[33], self.mask_threshold[32])  for _dict in out]
        self.mask_64s = [mt.make_mask_close(_dict[65], self.mask_threshold[64])  for _dict in out]
        self.out = out

    def _get_vdb_from_sdf(self, index):
        _dict = self.out[index]

        # create a set to hold the vdb tensors
        vdb_set = []
        vdb_set.append(_dict['obj_name'])

        # input sdf or output sdf
        input_sdf_size  = np.random.choice([32, 64], p=[1, 0.0])
        output_size = 128
        if input_sdf_size == 32 and not self.is_test:
            input_size = self.input_size[32]
            input_sdf = _dict[33]
            output_sdf = _dict[129]
            mask = self.mask_32s[index]
            self.upsample_factor = 4
        elif input_sdf_size == 64 and not self.is_test:
            input_size = self.input_size[64]
            input_sdf = _dict[65]
            output_sdf = _dict[129]
            mask = self.mask_64s[index]
            self.upsample_factor = 2
        elif self.is_test:
            input_sizes = [33, 65]
            masks = {33: self.mask_32s[index], 65: self.mask_64s[index]}
            upsample_factors = {33: 4, 65: 2}

        # crop mask SDFs
        if self.is_crop:
            mask = self._cropped_mask(mask, input_sdf_size)

        if not self.is_test:
            input_vdb, output_vdb = self.sdfToVDB.sdf_to_vdb(sdf_scaling_value=self.sdf_scaling_value[input_size-1],
                                                            sdf_arr=input_sdf,
                                                            large_sdf_arr=output_sdf,
                                                            mask=mask,
                                                            size=input_size, # 33
                                                            upsample_factor=self.upsample_factor,
                                                            unique_random_direction=self.unique_random_direction)

            # add the input vdb to the set
            vdb_set.append(input_size)
            vdb_set.append(input_vdb)
            vdb_set.append(output_vdb)                                 
            return tuple(vdb_set)
        
        elif self.is_test:
            # create a mask for the test set
            vdb_tensors = {}
            new_ijkss = {}
            new_featuress = {}
            for input_size in input_sizes:
                input_sdf = _dict[input_size]
                mask = masks[input_size]
                self.upsample_factor = upsample_factors[input_size]
                vdb_tensors[input_size] = self.sdfToVDB.sdf_to_vdb(
                                            sdf_scaling_value=self.sdf_scaling_value[input_size-1],
                                            sdf_arr=input_sdf,
                                            large_sdf_arr=None,
                                            mask=mask,
                                            upsample_factor=None,
                                            unique_random_direction=None,
                                            size=input_size, #33
                                            is_test=True
                                        )

                new_features, new_ijks = self._get_all_shifted_positions(vdb_tensors[input_size], 
                                                size=input_size, 
                                                upsample_factor=self.upsample_factor)
                new_featuress[input_size] = new_features
                new_ijkss[input_size] = new_ijks

            vdb_set.append(vdb_tensors)
            vdb_set.append(new_ijkss)
            vdb_set.append(new_featuress)
            vdb_set.append(_dict[129]) # scale during testing
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
        obj_names, vdb_tensors, new_ijkss, new_featuress, actual_sdf = zip(*batch)
        vdb_tensor_32 = [vdb_dict[33] for vdb_dict in vdb_tensors]
        vdb_tensor_64 = [vdb_dict[65] for vdb_dict in vdb_tensors]
        new_ijks_32 = [ijks_dict[33] for ijks_dict in new_ijkss]
        new_ijks_64 = [ijks_dict[65] for ijks_dict in new_ijkss]
        new_features_32 = [features_dict[33] for features_dict in new_featuress]
        new_features_64 = [features_dict[65] for features_dict in new_featuress]

        return list(obj_names), list(vdb_tensor_32), list(vdb_tensor_64), list(new_ijks_32), list(new_ijks_64), list(new_features_32), list(new_features_64), list(actual_sdf)

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
                      val_ratio=0.2,
                      testing=True):
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

        # Ensure right test set
        if testing:
            with open('test_names_file.txt', 'r') as f:
                test_set_from_file = f.read().splitlines()
            assert set(test_set) == set(test_set_from_file), "Test set does not match the expected test set from file."

        print(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

        return train_set, val_set, test_set

    
    def get(self, names_set):
        train_set, val_set, test_set = self.split_dataset(names_set, 
                                        train_ratio=0.6, 
                                        val_ratio=0.2)
        if self.n_samples is not None:
            if not isinstance(self.n_samples, int):
                raise ValueError("n_samples must be an integer or None")
            names_set = names_set[:self.n_samples]
            train_set, val_set, test_set = self.split_dataset(names_set, 
                                        train_ratio=0.6, 
                                        val_ratio=0.2,
                                        testing=True)
        
        is_eval = False
        if not is_eval:
            train_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set=train_set,
                dataset_grids=self.config['data']['dataset_grids'],
                input_size=self.config['data']['input_size'],
                mask_threshold=self.config['data']['mask_threshold'],
                sdf_scaling_value=self.config['data']['sdf_scaling_value'],
                is_crop=self.config['data']['is_crop']['train'],
                crops_ratio=self.config['data']['crops_ratio'],
                crops_threshold=self.config['data']['crops_threshold'],
                unique_random_direction=self.config['data']['unique_random_direction'],
                n_jobs=-1
            )
            val_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set= val_set,
                dataset_grids=self.config['data']['dataset_grids'],
                input_size=self.config['data']['input_size'],
                mask_threshold=self.config['data']['mask_threshold'],
                sdf_scaling_value=self.config['data']['sdf_scaling_value'],
                is_crop=self.config['data']['is_crop']['val'],
                crops_ratio=self.config['data']['crops_ratio'],
                crops_threshold=self.config['data']['crops_threshold'],
                unique_random_direction=self.config['data']['unique_random_direction'],
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
            sdf_scaling_value=self.config['data']['sdf_scaling_value'],
            is_crop=self.config['data']['is_crop']['test'],
            crops_ratio=self.config['data']['crops_ratio'],
            crops_threshold=self.config['data']['crops_threshold'],
            unique_random_direction=self.config['data']['unique_random_direction'],
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