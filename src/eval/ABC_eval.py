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
        
        model_dir = os.path.join(abc_dir, file_name)
        obj_files = [f for f in os.listdir(model_dir) if f.endswith('.obj')]
        file = os.path.join(model_dir, obj_files[0])
        
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
                 pos_enc_dim,
                 test_loader,
                 upsampling_level,
                 abc_dir,
                 save_model_dir,
                 save_predictions_dir, 
                 n_job,
                 logger):
        
        self.model_name = model_name
        self.pos_enc_dim = pos_enc_dim
        
        # load the model
        model_path = os.path.join(save_model_dir, 
                                  f'{model_name}.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        else:
            print(f"Loading model from {model_path}")
            model = torch.load(model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
        self.model = model
        
        self.test_loader = test_loader
        self.upsampling_level = upsampling_level
        self.abc_dir = abc_dir
        self.save_predictions_dir = save_predictions_dir

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # helper class for SDF to VDB conversion
        self.sdf_to_vdb = sdfToVDB()

        self.n_job = n_job
        # logger
        self.logger = logger

    def get_target_grid(self, input): 
        return self.sdf_to_vdb.custom_subdivide_grid(input.grid)

    def eval_sub_step(self, input):
        inputs = positional_encoding(input, self.pos_enc_dim)
        tragets_grid = self.get_target_grid(input)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs, tragets_grid)
        return outputs
    
    def save_predictions(self):
        '''saving predictions one by one'''
        file_names = []
        for batch in tqdm(self.test_loader, desc='Evaluating'):
            obj_names, vdb_32s, _, _ = batch
            vdb_32 = vdb_32s[0]
            obj_name = obj_names[0]
            obj_name = obj_name.split('.')[0]
            
            save_dir = os.path.join(self.save_predictions_dir, 
                                    self.model_name, 
                                    str(self.upsampling_level))
            os.makedirs(save_dir, exist_ok=True)
            
            # check if file already exists the process it
            output_file = os.path.join(save_dir, f'{obj_name}.nvdb')
            if os.path.exists(output_file):
                print(f"Skipping prediction: {obj_name}, already exists.")
            else:
                # print(f"Processing {obj_name}...")
                output = vdb_32
                for _ in range(self.upsampling_level):
                    output = self.eval_sub_step(output)
                
                fvdb.save(output_file, output.grid, output.data, compressed=True)
            file_names.append(obj_name)
    
        return save_dir, file_names
    
    def evaluate(self):
        print(f"Checking predictions in dir: {self.save_predictions_dir}/{self.model_name}/{self.upsampling_level}")
        predictions_dir,  test_names = self.save_predictions()

        # clean memory
        torch.cuda.empty_cache()
        gc.collect()

        # evaluate the predictions
        print(f"Evaluating predictions")
        def eval_wrapper(args):
            file_name_h5 = args
            processor = EvaluationProcessor(abc_dir=self.abc_dir, 
                                            predictions_dir=predictions_dir)
            return (file_name_h5,processor.get_eval(file_name_h5))
        out = joblib.Parallel(n_jobs=self.n_job )(joblib.delayed(eval_wrapper)(name) for name in (test_names))
        
        names = [_item[0] for _item in out]
        out = [_item[1] for _item in out]
        out = np.array(out)
        cd1 = out[:, 1].astype(float).mean(axis=0)
        cd2 = out[:, 2].astype(float).mean(axis=0)
        f1 = out[:, 3].astype(float).mean(axis=0)
        nc = out[:, 4].astype(float).mean(axis=0)
        ecd2 = out[:, 5].astype(float).mean(axis=0)
        ef1 = out[:, 6].astype(float).mean(axis=0)

        # save results stats
        df_results = pd.DataFrame({'names': names,
                                   'cd1 (x 1e-5)': out[:, 1].astype(float)* 1e5,
                                   'cd2 (x 1e-5)': out[:, 2].astype(float)* 1e5,
                                   'f1': out[:, 3].astype(float),
                                   'nc': out[:, 4].astype(float),
                                   'ecd2': out[:, 5].astype(float),
                                   'ef1': out[:, 6].astype(float)})
        df_results_describe = df_results.describe().reset_index()
        self.logger.log({'data/eval': wandb.Table(dataframe=df_results)})
        self.logger.log({'stats/eval': wandb.Table(dataframe=df_results_describe)})

        # save results to logger
        columns = ['u_level', 'CD1 (x 1e-5)', 'CD2 (x 1e-5)', 'F1', 'NC', 'ECD', 'EF1']
        data = [[self.upsampling_level, cd1 * 1e5, cd2 * 1e5, f1, nc, ecd2, ef1]]
        self.logger.log({'Evaluation': wandb.Table(data=data, columns=columns)})

        print(f"cd1: {cd1 * 1e5:.3f}, cd2: {cd2 * 1e5:.3f}, f1: {f1:.3f}, nc: {nc:.3f}, ecd2: {ecd2:.3f}, ef1: {ef1:.3f}")

