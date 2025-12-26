import gc
import os
import sys
import h5py
import fvdb
import wandb
import torch
from tqdm import tqdm
import joblib
import trimesh
import numpy as np
import pandas as pd
from skimage import measure
from .ponq_eval import get_cd_f1_nc

sys.path.append('../src/utils')
import mesh_tools as mt
from fvdb_utils import sdfToVDB
from ssu_tools import positional_encoding


class EvaluationProcessor:
    def __init__(self, 
                 abc_dir,
                 predictions_dir 
                 ):
        self.predictions_dir = predictions_dir
        self.abc_dir = abc_dir
    
    @staticmethod
    def normalize_mesh(v, target_range=1.0):
        # v: (N, 3) array of vertices
        v = np.array(v)
        centroid = v.mean(axis=0)
        v_centered = v - centroid
        max_extent = np.max(np.abs(v_centered))
        v_normalized = v_centered / max_extent * (target_range / 2)
        return v_normalized
    
    @staticmethod
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
    
    @staticmethod
    def get_prediction_mesh(pred_dir, file_name):
        # file = os.path.join(pred_dir, f'{file_name}.obj')
        # mesh = trimesh.load(file)
        # return mesh
        # check if the file already exists
        pred_file = os.path.join(pred_dir, f'{file_name}.nvdb')
        if not os.path.exists(pred_file):
            raise FileNotFoundError(f"Prediction for {pred_file} does not exist.")
        
        # load the prediction
        # device = 'cpu'
        grid, feat, _ = fvdb.load(pred_file, device='cpu')

        # convert feature to N, F with marching cubes
        v, f, _ = grid.marching_cubes(feat)
        v = v.jdata.numpy()
        f = f.jdata.numpy()
        mesh = trimesh.Trimesh(vertices=v, faces=f)
        return mesh
    
    @staticmethod
    def mesh_from_abc(abc_dir, file_name):
        '''extracts the vertices and faces from the abc obj file'''
        
        # model_dir = os.path.join(abc_dir, file_name)
        # obj_files = [f for f in os.listdir(model_dir) if f.endswith('.obj')]
        if 'obj' not in file_name:
            file_name += '.obj'
        file = os.path.join(abc_dir, file_name)
        
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} does not exist.")
        
        # read the mesh
        mesh = trimesh.load(file)
        return mesh
    
    def get_eval(self, file_name_h5):
        '''evaluates the model on the given dataset'''
        
        print(f"Evaluating {file_name_h5}...")
        file_name = file_name_h5.split('.')[0]
        pred_mesh = self.get_prediction_mesh(self.predictions_dir, file_name)
        # remove input type information from file name
        file_name = file_name.replace("32_", "")
        file_name = file_name.replace("64_", "") 
        file_name = file_name.replace("128_", "")  
        gt_mesh = self.mesh_from_abc(self.abc_dir, file_name)

        # get matrix for evaluation using NDCnormalize
        try:
            metrics = get_cd_f1_nc((file_name, gt_mesh, pred_mesh),
                                        scale_gt=1.0,
                                        eval_normalization=mt.NDCnormalize
                                        )
        except Exception as e:
            print(f"Error evaluating {file_name}: {e}")
            metrics = (file_name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return metrics

class Evaluator:
    def __init__(self,
                 model_name,
                 abc_dir,
                 save_predictions_dir, 
                 n_job,
                 eval_discription):
        
        
        
        self.model_name = model_name
        # self.upsampling_level = upsampling_level
        self.abc_dir = abc_dir
        self.save_predictions_dir = save_predictions_dir

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.n_job = n_job
        self.eval_discription = eval_discription

    def evaluate(self):
        print(f"Checking predictions in dir: {self.save_predictions_dir}/{self.model_name}/")
        predictions_dir = os.path.join(self.save_predictions_dir, self.model_name)
        # predictions_dir = '/data/workspaces/spanwar/dataset/thingi/out/Thingi_pretrained_PoNQ_ABC_32'
        test_names = os.listdir(predictions_dir)
        test_names_32 = [name for name in test_names if name.startswith('32_') and name.endswith('.nvdb')]
        # test_names_32 = [name for name in test_names if name.endswith('.obj')]
        test_names_64 = [name for name in test_names if name.startswith('64_') and name.endswith('.nvdb')]
        # test_names_64 = [name for name in test_names if name.endswith('.obj')]
        test_names_128 = [name for name in test_names if name.startswith('128_') and name.endswith('.nvdb')]
        
        if len(test_names) == 0:
            raise ValueError(f"No predictions found in {predictions_dir}. Please run the model first.")

        # clean memory
        torch.cuda.empty_cache()
        gc.collect()

        # evaluate the predictions
        print(f"Evaluating predictions 32...")
        def eval_wrapper(args):
            file_name_h5 = args
            processor = EvaluationProcessor(abc_dir=self.abc_dir, 
                                            predictions_dir=predictions_dir)
            return (file_name_h5,processor.get_eval(file_name_h5))
        out_32 = joblib.Parallel(n_jobs=self.n_job )(joblib.delayed(eval_wrapper)(name) for name in (test_names_32))
        
        # clean memory
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Evaluating predictions 64...")
        out_64 = joblib.Parallel(n_jobs=self.n_job )(joblib.delayed(eval_wrapper)(name) for name in (test_names_64))

        # clean memory
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Evaluating predictions 128...")
        out_128 = joblib.Parallel(n_jobs=self.n_job )(joblib.delayed(eval_wrapper)(name) for name in (test_names_128))

        names = [_item[0] for _item in out_32]
        out_32 = [_item[1] for _item in out_32]
        out_64 = [_item[1] for _item in out_64]
        out_128 = [_item[1] for _item in out_128]
        
        out_32 = np.array(out_32)
        out_64 = np.array(out_64)
        out_128 = np.array(out_128)

        cd1_32 = out_32[:, 1].astype(float).mean(axis=0)
        cd2_32 = out_32[:, 2].astype(float).mean(axis=0)
        f1_32 = out_32[:, 3].astype(float).mean(axis=0)
        nc_32 = out_32[:, 4].astype(float).mean(axis=0)
        ecd2_32 = out_32[:, 5].astype(float).mean(axis=0)
        ef1_32 = out_32[:, 6].astype(float).mean(axis=0)

        cd1_64 = out_64[:, 1].astype(float).mean(axis=0)
        cd2_64 = out_64[:, 2].astype(float).mean(axis=0)
        f1_64 = out_64[:, 3].astype(float).mean(axis=0)
        nc_64 = out_64[:, 4].astype(float).mean(axis=0)
        ecd2_64 = out_64[:, 5].astype(float).mean(axis=0)
        ef1_64 = out_64[:, 6].astype(float).mean(axis=0)

        cd1_128 = out_128[:, 1].astype(float).mean(axis=0)
        cd2_128 = out_128[:, 2].astype(float).mean(axis=0)
        f1_128 = out_128[:, 3].astype(float).mean(axis=0)
        nc_128 = out_128[:, 4].astype(float).mean(axis=0)
        ecd2_128 = out_128[:, 5].astype(float).mean(axis=0)
        ef1_128 = out_128[:, 6].astype(float).mean(axis=0)

        # save results stats
        df_results = pd.DataFrame({'names': names,
                                   'cd1 (x 1e-5)(32->128)': out_32[:, 1].astype(float)* 1e5,
                                #    'cd1 (x 1e-5)(64->128)': out_64[:, 1].astype(float)* 1e5,
                                   'cd2 (x 1e-5)(32->128)': out_32[:, 2].astype(float)* 1e5,
                                #    'cd2 (x 1e-5)(64->128)': out_64[:, 2].astype(float)* 1e5,
                                   'f1 (32->128)': out_32[:, 3].astype(float),
                                #    'f1 (64->128)': out_64[:, 3].astype(float),
                                   'nc (32->128)': out_32[:, 4].astype(float),
                                #    'nc (64->128)': out_64[:, 4].astype(float),
                                   'ecd2 (32->128)': out_32[:, 5].astype(float),
                                #    'ecd2 (64->128)': out_64[:, 5].astype(float),
                                   'ef1 (32->128)': out_32[:, 6].astype(float)
                                   })
        
        df_results_describe = df_results.describe().reset_index()

        # save results to logger
        columns = ['u_level', 'CD1 (x 1e-5)', 'CD2 (x 1e-5)', 'F1', 'NC', 'ECD', 'EF1']
        data = [
            [self.eval_discription[32], cd1_32 * 1e5, cd2_32 * 1e5, f1_32, nc_32, ecd2_32, ef1_32],
                [self.eval_discription[64], cd1_64 * 1e5, cd2_64 * 1e5, f1_64, nc_64, ecd2_64, ef1_64],
                [self.eval_discription[128], cd1_128 * 1e5, cd2_128 * 1e5, f1_128, nc_128, ecd2_128, ef1_128]
                ]
        df_log = pd.DataFrame(data, columns=columns)
        print(df_log)
        print(f"cd1_32: {cd1_32 * 1e5:.3f}, cd2_32: {cd2_32 * 1e5:.3f}, f1_32: {f1_32:.3f}, nc_32: {nc_32:.3f}, ecd2_32: {ecd2_32:.3f}, ef1_32: {ef1_32:.3f}")
        print(f"cd1_64: {cd1_64 * 1e5:.3f}, cd2_64: {cd2_64 * 1e5:.3f}, f1_64: {f1_64:.3f}, nc_64: {nc_64:.3f}, ecd2_64: {ecd2_64:.3f}, ef1_64: {ef1_64:.3f}")
        print(f"cd1_128: {cd1_128 * 1e5:.3f}, cd2_128: {cd2_128 * 1e5:.3f}, f1_128: {f1_128:.3f}, nc_128: {nc_128:.3f}, ecd2_128: {ecd2_128:.3f}, ef1_128: {ef1_128:.3f}")

