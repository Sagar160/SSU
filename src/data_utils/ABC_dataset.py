# From https://github.com/nissmar/PoNQ/blob/main/src/utils/ABC_dataset.py
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import joblib
import torch
import fvdb.nn as fvnn
import fvdb
device = 'cuda'

def get_item(src_dir, model_name):
    file = h5py.File(src_dir + model_name)
    sdf = file['32_sdf'][:]
    out_sdf = file['64_sdf'][:]
    return sdf, out_sdf

class ABCDataset(Dataset):
    def __init__(
        self,
        src_dir,
        names_set,
        grid_n=33,
        n_jobs=-1,
    ):
        out = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(get_item)
                                             (src_dir, model_name) for model_name in tqdm(names_set))
        self.sdfs = [e[0] for e in out]
        self.out_sdfs = [e[1] for e in out]
        
        # TODO: add fvdb dataset
    
    def __getitem__(self, index):
        return (
            self.sdfs[index],
            self.out_sdfs[index]
        )

    def __len__(self):
        return len(self.sdfs)
    
    
def get_vdb_data_loader(dataset, batch_size=1, shuffle=None, num_workers=0):
    return torch.utils.data.DataLoader(dataset, collate_fn=lambda x: fvdb.jcat(x), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
