# From https://github.com/nissmar/PoNQ/blob/main/src/utils/ABC_dataset.py
import os
import sys
sys.path.append('../utils')
sys.path.append('../data_utils')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Fetches the SDF and output SDF from the h5 file
def get_item(src_dir, model_name):
    path = os.path.join(src_dir, model_name)
    
    # check if file path exists
    if not os.path.exists(path):
        print(f"File {path} does not exist.")
        return None, None
    
    file = h5py.File(path, 'r')
    
    # check if the file has the required datasets
    if '32_sdf' not in file or '64_sdf' not in file:
        print(f"File {path} does not contain required datasets.")
        return None, None
    
    # fetch the SDF and output SDF
    sdf = file['32_sdf'][:]
    out_sdf = file['64_sdf'][:]
    return sdf, out_sdf


# Dataset class for loading ABC dataset
class ABCdataset(Dataset):
    def __init__(
        self,
        src_dir,
        names_set,
        grid_n=33,
        n_jobs=-1,
        mode='train'
    ):
        out = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_item)
                                             (src_dir, model_name) for model_name in tqdm(names_set))
        
        # Filter out None values
        out = [(sdf, out_sdf) for sdf, out_sdf in out if sdf is not None and out_sdf is not None]
        if len(out) == 0:
            raise ValueError("No valid SDF data found in the provided dataset.")
        
        # extract SDFs
        self.sdfs = [e[0] for e in out]
        self.out_sdfs = [e[1] for e in out]
        self.grid_n = grid_n
        self.mode = mode
    
    def __len__(self):
        return len(self.sdfs)
    
    def __getitem__(self, index):
        if self.mode not in ['test']:
                
            sdf = self.sdfs[index]
            out_sdf = self.out_sdfs[index]
            # todo: automate the grid_n value
            mask = mt.make_mask_close(sdf, self.grid_n)
            sdf_vdb, out_sdf_vdb = fu.sdf_to_vdb(sdf, 
                                                 out_sdf, 
                                                 mask=mask, 
                                                 size=self.grid_n)
            return (
                sdf_vdb,
                out_sdf_vdb
            )
        else:
            # For test mode, we return the original SDFs without processing
            sdf = self.sdfs[index]
            out_sdf = self.out_sdfs[index]
            mask = mt.make_mask_close(sdf, self.grid_n)
            sdf_vdb, out_sdf_vdb = fu.sdf_to_vdb(sdf,
                                                 out_sdf, 
                                                 mask=mask, 
                                                 size=self.grid_n)
            return (
                sdf,
                out_sdf,
                mask,
                sdf_vdb,
                out_sdf_vdb
            )
  

# Custom collate function for the DataLoader
def custom_collate_fn(batch):
    # batch is a list of tuples: [(small_vdb, large_vdb), ...]
    small_vdbs, large_vdbs = zip(*batch)
    # Move each VDBTensor to device if needed
    small_vdbs = [vdb.to(device) for vdb in small_vdbs]
    large_vdbs = [vdb.to(device) for vdb in large_vdbs]
    return small_vdbs[0], large_vdbs[0]


# Custom collate function for the test DataLoader
def test_custom_collate_fn(batch):
    # batch is a list of tuples: [(sdf, out_sdf, mask, small_vdb, large_vdb), ...]
    sdfs, out_sdfs, masks, small_vdbs, large_vdbs = zip(*batch)
    # Move each VDBTensor to device if needed
    small_vdbs = [vdb.to(device) for vdb in small_vdbs]
    large_vdbs = [vdb.to(device) for vdb in large_vdbs]
    return sdfs[0], out_sdfs[0], masks[0], small_vdbs[0], large_vdbs[0]


# Function to get the DataLoader for the dataset
def get_vdb_data_loader(dataset, batch_size=1, shuffle=None, num_workers=0, mode='train'):
    if mode not in ['test']:
        return torch.utils.data.DataLoader(dataset, 
                                            collate_fn=custom_collate_fn, 
                                            batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            num_workers=num_workers)
    else:
        return torch.utils.data.DataLoader(dataset, 
                                            collate_fn=test_custom_collate_fn, 
                                            batch_size=batch_size, 
                                            shuffle=False, 
                                            num_workers=num_workers)
