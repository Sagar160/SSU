import gc
import os
import sys
import random
import wandb
sys.path.append('../utils')
sys.path.append('../data_utils')
sys.path.append('../models')
sys.path.append('../eval')
import numpy as np
# import matplotlib.pyplot as plt
import mesh_tools as mt
# import model_tools as mtools
import flow_matching_tools as fmt
import get_basic_eval as beval
import ponq_eval as peval
import fvdb_utils as fu
# import model as fvdbModel
# import unet as fvdbUnet
# from meshplot import plot
import torch
# import torch.nn as nn
from tqdm import tqdm
from skimage import measure
import trimesh
import h5py
import fvdb
import joblib


def normalize_mesh(v, target_range=1.0):
    # v: (N, 3) array of vertices
    v = np.array(v)
    centroid = v.mean(axis=0)
    v_centered = v - centroid
    max_extent = np.max(np.abs(v_centered))
    v_normalized = v_centered / max_extent * (target_range / 2)
    return v_normalized


# extract the vertices and faces from the abc obj file
def mesh_from_abc(abc_dir, file_name):
    '''extracts the vertices and faces from the abc obj file'''
    
    model_dir = os.path.join(abc_dir, file_name)
    obj_files = [f for f in os.listdir(model_dir) if f.endswith('.obj')]
    file = os.path.join(model_dir, obj_files[0])
    
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    
    # read the mesh
    mesh = trimesh.load(file)
    return mesh

def load_ponq_mesh(ponq_data_dir, file_name, dim='64'):
    '''loads the ponq mesh from the sdf'''
    
    file = os.path.join(ponq_data_dir, f'{file_name}.hdf5')
    
    if not os.path.exists(file):
        raise FileNotFoundError(f"File {file} does not exist.")
    
    with h5py.File(file, 'r') as f:
        if f'{dim}_sdf' in f:
            sdf = f[f'{dim}_sdf'][:]
        else:
            raise ValueError(f"File {file} does not contain '{dim}_sdf' dataset.")
    
    # read the mesh
    try:
        v, f, _, _ = measure.marching_cubes(sdf, level=0)
    except Exception as e:
        print(f"Error in marching cubes for {file_name}: {e}")
        return None
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    return mesh

def get_prediction_mesh(pred_dir, file_name):
    # check if the file already exists
    if not os.path.exists(os.path.join(pred_dir, f'{file_name}.nvdb')):
        print(f"Prediction for {file_name} does not exist. Skipping evaluation.")
        raise FileNotFoundError(f"Prediction for {file_name} does not exist.")
    
    # load the prediction
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_file = os.path.join(pred_dir, f'{file_name}.nvdb')
    grid, feat, _ = fvdb.load(pred_file)

    # convert feature to N, F with marching cubes
    v, f, _ = grid.marching_cubes(feat)
    v = v.jdata.cpu().detach().numpy()
    f = f.jdata.cpu().detach().numpy()
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    return mesh

# get test names
def get_test_names(dir):
    '''returns the test names from the ponq data directory'''
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # test file names
    train_set_names = os.listdir(dir)
    random.shuffle(train_set_names)
    
    train_size = int(0.6 * len(train_set_names))
    test_size = int(0.2 * len(train_set_names))
    val_size = len(train_set_names) - train_size - test_size
    test_names = train_set_names[train_size + val_size:]
    
    return test_names

# main evaluation function
def get_eval(abc_dir, predictions_dir, file_name_h5):
    '''evaluates the model on the given dataset'''
    file_name = file_name_h5.split('.')[0]
    pred_mesh = get_prediction_mesh(predictions_dir, file_name) 
    # pred_mesh = load_ponq_mesh(predictions_dir, file_name, dim='64')  
    gt_mesh = mesh_from_abc(abc_dir, file_name)

    # get matrix for evaluation
    try:
        metrix = peval.get_cd_f1_nc((file_name, gt_mesh, pred_mesh),
                                    scale_gt=1.0,
                                    eval_normalization=mt.NDCnormalize
                                    )
    except Exception as e:
        print(f"Error evaluating {file_name}: {e}")
        metrix = (file_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return metrix
        

def main(model_name, diffusion=True):
    '''main function to run the evaluation'''
    print(f"Running evaluation for model: {model_name}")
    print("Comparing PONQ Predictions with ABC dataset...")
    print("Using diffusion model:", diffusion)

    # clear memory
    gc.collect()
    torch.cuda.empty_cache()

    # device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Dataset Dir
    abc_dir = '/data/workspaces/spanwar/dataset/abc_dataset'
    ponq_data_dir = '/data/workspaces/spanwar/dataset/ponq_dataset/gt_Quadrics'

    # check if for the saved model prediction is available
    predictions_dir = os.path.join('/data/workspaces/spanwar/results/ssu/test_predictions', model_name)
    if not os.path.exists(predictions_dir):
        print(f"Model {model_name} does not have predictions available.")
        print(f"Creating a new directory for predictions {predictions_dir}")
        os.makedirs(predictions_dir, exist_ok=True)
    
    # check if we have all the predictions
    
    # load model
    model = torch.load(f'../save_models/{model_name}.pth')
    model.eval()
    model = model.to(device)
    
    test_names = get_test_names(ponq_data_dir)
    for file_name_h5 in tqdm(test_names):
        # check if the file already exists
        if os.path.exists(os.path.join(predictions_dir, file_name_h5.split('.')[0] + '.nvdb')):
            continue
        else:
            print(f"Saving prediction for {file_name_h5}...")
            input_sdf_path = os.path.join(ponq_data_dir, file_name_h5)
            with h5py.File(input_sdf_path, 'r') as f:
                sdf = f['32_sdf'][:]
                out_sdf = f['64_sdf'][:]


            mask = mt.make_mask_close(sdf, 33)
            sdf_vdb, out_sdf_vdb = fu.sdf_to_vdb(sdf,
                                                 out_sdf, 
                                                 mask=mask, 
                                                 size=33)
            # no grad
            with torch.no_grad():
                if diffusion:
                    result = fmt.fm_prediction(model, sdf_vdb, out_sdf_vdb, device=device)
                else:
                    sdf_vdb = fmt.positional_encoding(sdf_vdb, 10).to(device)
                    result = model(sdf_vdb, out_sdf_vdb.grid)
            
            # save the prediction
            output_file = os.path.join(predictions_dir, f'{file_name_h5.split(".")[0]}.nvdb')
            fvdb.save(output_file, result.grid, result.data, compressed=True)
    
    # Load mesh and evaluate
    out = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_eval)
                                     (abc_dir, predictions_dir, name) for name in (test_names))
    
    out = np.array(out)
    cd1 = out[:, 1].astype(float).mean(axis=0)
    cd2 = out[:, 2].astype(float).mean(axis=0)
    f1 = out[:, 3].astype(float).mean(axis=0)
    nc = out[:, 4].astype(float).mean(axis=0)
    ecd2 = out[:, 5].astype(float).mean(axis=0)
    ef1 = out[:, 6].astype(float).mean(axis=0)
    
    # save results to wandb
    wandb.init(project='SSU_Eval', entity="sp_kumar", name=model_name, config={
        'eval_mothed': 'PONQ eval',
        'GT_dataset': 'ABC',
        'Prediction': '64 SDF from 32 SDF PONQ test dataset',
        'Normalization': 'NDC',
    })
    columns = ['CD1 (x 1e-5)', 'CD2 (x 1e-5)', 'F1', 'NC', 'ECD', 'EF1']
    data = [[cd1 * 1e5, cd2 * 1e5, f1, nc, ecd2, ef1]]
    wandb.log({f'Evaluation': wandb.Table(data=data, columns=columns)})
    wandb.finish()

    print(f"cd1: {cd1 * 1e5:.3f}, cd2: {cd2 * 1e5:.3f}, f1: {f1:.3f}, nc: {nc:.3f}, ecd2: {ecd2:.3f}, ef1: {ef1:.3f}")
    
    # clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    wandb_model_name = 'SSU_PONQ_DATA_UPSAMPLER_Unet'
    main(wandb_model_name, diffusion=False)