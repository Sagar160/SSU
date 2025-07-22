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
import fvdb
import mesh_tools as mt
import fvdb_utils as fu

class ABCDataset(Dataset):
    def __init__(self, src_dir,
                 names_set,
                 mask_threshold, # in term grid size i,e grid size = 32 -> 3/32
                 is_crop,
                 crop_size,
                 crop_threshold, 
                 max_tries=100, 
                 n_jobs=-1):
        
        self.input_dir = src_dir
        self.names_set = names_set
        self.mask_threshold = mask_threshold
        self.is_crop = is_crop
        self.crop_size = crop_size
        self.crop_threshold = crop_threshold 
        self.max_tries = max_tries
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
            sdf_dict[33] = f['32_sdf'][:]
            sdf_dict[65] = f['64_sdf'][:]
            sdf_dict[129] = f['128_sdf'][:]

        return sdf_dict
    
    def _cropped_mask(self, mask):
        """
        Randomly crop a 3D array so that the crop contains at least n nonzero elements.
        crop_size: int or tuple (crop_x, crop_y, crop_z)
        threshold: minimum number of nonzero elements required in the crop
        max_tries: maximum number of attempts
        """

        if isinstance(self.crop_size, int):
            crop_size = (self.crop_size, self.crop_size, self.crop_size)
        sx, sy, sz = mask.shape
        cx, cy, cz = crop_size

        for _ in range(self.max_tries):
            x = np.random.randint(0, sx - cx + 1)
            y = np.random.randint(0, sy - cy + 1)
            z = np.random.randint(0, sz - cz + 1)
            crop = mask[x:x+cx, y:y+cy, z:z+cz]
            if np.count_nonzero(crop) >= self.crop_threshold:
                mask_crop = np.zeros_like(mask, dtype=bool)
                mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop
                return mask_crop
            
        # Ignore threshold - just return the crop
        print(f"Warning: Could not find a valid crop after {self.max_tries} attempts. Returning the last attempt.")
        mask_crop = np.zeros_like(mask, dtype=bool)
        mask_crop[x:x+cx, y:y+cy, z:z+cz] = crop
        return mask_crop

    def _read_dataset(self):
        out = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(self._get_item)
                                                (obj_name) for obj_name in tqdm(self.names_set))

        # check for empty set
        if len(out) == 0:
            raise ValueError("No valid SDF data found in the provided dataset.")
        
        # mask SDFs of 32
        self.mask_32 = [mt.make_mask_close(_dict[33], self.mask_threshold)  for _dict in out]
        self.out = out

    def _get_vdb_from_sdf(self, index):
        # crop mask SDFs of 32
        mask_32_index = self.mask_32[index]
        if self.is_crop:
            mask_32_index = self._cropped_mask(mask_32_index)

        _dict = self.out[index]
        vdb_32, vdb_64 = self.sdfToVDB.sdf_to_vdb(_dict[33],
                                                    _dict[65], 
                                                    mask_32_index, 
                                                    size=33)
        

        vdb_128 = self.sdfToVDB.upscale_sdf_to_vdb(vdb_64,
                                                    _dict[129], 
                                                    size=65)
        
        return (_dict['obj_name'], vdb_32, vdb_64, vdb_128)

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
        obj_names, vdb_32s, vdb_64s, vdb_128s = zip(*batch)
        return list(obj_names), list(vdb_32s), list(vdb_64s), list(vdb_128s)

    def get_vdb_data_loaders(self,
                             train_dataset,
                             val_dataset,
                             test_dataset, 
                             batch_size=1, 
                             shuffle=None, 
                             num_workers=0):
        
        if not self.config['eval']['only_eval']:
            train_dataloader =  torch.utils.data.DataLoader(train_dataset, 
                                                collate_fn=self.custom_collate_fn,
                                                batch_size=batch_size, 
                                                shuffle=shuffle['train'], 
                                                num_workers=num_workers)
            val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                collate_fn=self.custom_collate_fn,
                                                batch_size=batch_size,
                                                shuffle=shuffle['val'], 
                                                num_workers=num_workers)
        else:
            train_dataloader = None
            val_dataloader = None

        test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            collate_fn=self.custom_collate_fn,
                                            # batch_size=batch_size,
                                            batch_size=1,  # Test loader usually has batch size of 1
                                            shuffle=shuffle['test'], 
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

        print(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

        return train_set, val_set, test_set

    
    def get(self):
        names_set = os.listdir(self.input_dir)
        if self.n_samples is not None:
            if not isinstance(self.n_samples, int):
                raise ValueError("n_samples must be an integer or None")
            names_set = names_set[:self.n_samples]

        train_set, val_set, test_set = self.split_dataset(names_set, 
                                        train_ratio=self.config['data']['data_split']['train'], 
                                        val_ratio=self.config['data']['data_split']['val'])
        
        if not self.config['eval']['only_eval']:
            train_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set=train_set,
                mask_threshold=self.config['data']['mask_threshold'],
                is_crop=self.config['data']['is_crop']['train'],
                crop_size=self.config['data']['crop_size'],
                crop_threshold=self.config['data']['crop_threshold'],
                n_jobs=self.config['data']['n_jobs']
            )
            val_dataset = ABCDataset(
                src_dir=self.input_dir,
                names_set=val_set,
                mask_threshold=self.config['data']['mask_threshold'],
                is_crop=self.config['data']['is_crop']['val'],
                crop_size=self.config['data']['crop_size'],
                crop_threshold=self.config['data']['crop_threshold'],
                n_jobs=self.config['data']['n_jobs']
            )
        else:
            train_dataset = None
            val_dataset = None
            
        test_dataset = ABCDataset(
            src_dir=self.input_dir,
            names_set=test_set,
            mask_threshold=self.config['data']['mask_threshold'],
            is_crop=self.config['data']['is_crop']['test'],
            crop_size=self.config['data']['crop_size'],
            crop_threshold=self.config['data']['crop_threshold'],
            n_jobs=self.config['data']['n_jobs']
        )
        
        train_dataloader, val_dataloader, test_dataloader = self.get_vdb_data_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=self.config['data']['shuffle'],
            num_workers=self.config['data']['num_workers']
        )
        return train_dataloader, val_dataloader, test_dataloader