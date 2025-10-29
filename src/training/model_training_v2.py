import os
import sys
import fvdb
import torch
import wandb
import fvdb.nn as fvnn
import numpy as np
from tqdm import tqdm
import pandas as pd
from .loss import LossFunctions
import torch.nn.functional as F

sys.path.append('../src/utils')
from ssu_tools import positional_encoding
import mesh_tools as mt

parent_path = os.path.abspath(os.path.join(__file__, "../../../flow_matching"))
# print(f"Parent path: {parent_path}")
sys.path.append(parent_path)
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
# from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

class ModelTrainer:
    def __init__(self,
                 model_name, 
                 model, 
                 num_epochs,
                 train_loader, 
                 val_loader,
                 test_loader,
                 sdf_scaling,
                #  pos_enc_dim, 
                 optimizer, 
                 scheduler,
                 loss_fn_name,
                 is_save_model,
                 is_save_predictions,
                 save_model_dir,
                 save_predictions_dir, 
                 logger
                 ):

        self.model_name = model_name
        self.model = model
        self.num_epochs = num_epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.sdf_scaling = sdf_scaling
        # self.pos_enc_dim = pos_enc_dim

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn_name
        self.loss_fn = LossFunctions(loss_fn_name).loss_fn
        
        self.is_save_model = is_save_model
        self.is_save_predictions = is_save_predictions
        self.save_model_dir = save_model_dir
        self.save_predictions_dir = save_predictions_dir
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.logger = logger
        # self.logger.log({'loss_weights': wandb.Table(data=[self.loss_weights], 
                                                    #  columns=['w1', 'w2', 'w3'])})
    def train_sub_step(self, inputs):
        # inputs = positional_encoding(inputs, self.pos_enc_dim)
        outputs = self.model(inputs)
        return outputs

    def append_feature(self, input_vdbs, feature):
        concat_features = torch.cat([input_vdbs.jdata, feature], dim=-1)
        return fvnn.VDBTensor(input_vdbs.grid, input_vdbs.grid.jagged_like(concat_features))

    def sample_fm(self, input_vdbs, output_vdbs):
        t = torch.rand_like(output_vdbs.jdata, device=self.device)
        features = t * input_vdbs.jdata[:, 0].unsqueeze(1) + (1 - t) * output_vdbs.jdata
        features = torch.cat([features, input_vdbs.jdata[:, 1:], t], dim=-1)
        sampled_vdbs = fvnn.VDBTensor(input_vdbs.grid,
                                      input_vdbs.grid.jagged_like(features))
        return sampled_vdbs

    def velocity_fm(self, input_vdbs, output_vdbs):
        velocity = output_vdbs.jdata[:, 0] - input_vdbs.jdata[:, 0]
        velocity_vdb = fvnn.VDBTensor(input_vdbs.grid,
                                  input_vdbs.grid.jagged_like(velocity[:, None]))
        return velocity_vdb
    
    @torch.no_grad()
    def eval_fm_steps(self, input_vdbs, model, n_steps=4):
        dt = 1 / n_steps
        t = torch.full_like(input_vdbs.jdata[:, 0], 0).to(self.device)
        t = t.unsqueeze(1)  # Ensure t is a column vector
        input_vdbs = self.append_feature(input_vdbs, t)
        for t in range(n_steps):
            t = torch.full_like(input_vdbs.jdata[:, 0], t/n_steps).to(self.device)
            # t = t.unsqueeze(1)  # Ensure t is a column vector
            input_vdbs.jdata[:, -1] = t
            # xt = positional_encoding(input_vdbs, dim=6, is_t=False, is_sdf=False)
            xt = input_vdbs
            updated_sdf = input_vdbs.jdata[:, 0].unsqueeze(1) + dt * model(xt).jdata
            input_vdbs.jdata[:, 0] = updated_sdf.squeeze(1)
            input_vdbs = fvnn.VDBTensor(input_vdbs.grid,
                                        input_vdbs.grid.jagged_like(
                                            input_vdbs.jdata
                                        ))
            
        predicted_vdbs = fvnn.VDBTensor(input_vdbs.grid,
                                        input_vdbs.grid.jagged_like(
                                            input_vdbs.jdata[:, 0].unsqueeze(1)
                                        ))
        return predicted_vdbs
    
    def meta_lib_fm(self, input_vdbs, output_vdbs):
        path = AffineProbPath(scheduler=CondOTScheduler())
        t = torch.rand(output_vdbs.jdata.shape[0]).to(self.device)

        path_sample = path.sample(t=t, 
                                  x_0=input_vdbs.jdata[:,0], 
                                  x_1=output_vdbs.jdata[:,0])
        xt = path_sample.x_t
        t = path_sample.t
        velocity = path_sample.dx_t

        xt_feature = torch.cat([xt.unsqueeze(1), 
                                input_vdbs.jdata[:, 1:], 
                                t.unsqueeze(1)], dim=-1)
        
        xt = fvnn.VDBTensor(input_vdbs.grid,
                            input_vdbs.grid.jagged_like(xt_feature))
        velocity = fvnn.VDBTensor(input_vdbs.grid,
                                    input_vdbs.grid.jagged_like(velocity[:, None]))
        return xt, velocity

    def train(self):
        min_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                obj_names, input_sizes, vdb_input, vdb_output = batch
                
                vdb_inputs = fvdb.jcat(vdb_input)
                vdb_outputs = fvdb.jcat(vdb_output)
                vdb_inputs = vdb_inputs.cuda()
                vdb_outputs = vdb_outputs.cuda()
                self.optimizer.zero_grad()

                # manual flow matching steps
                # xt = self.sample_fm(vdb_inputs, vdb_outputs)
                # velocity = self.velocity_fm(vdb_inputs, vdb_outputs)

                # wirghted FM
                # in_s = vdb_inputs.jdata[:,0]
                # out_s = vdb_outputs.jdata[:,0]
                # cost_loss = ((in_s - out_s)**2)        # squared L2 example -> [B]
                # # Gibbs weight
                # eps = 2
                # cost = torch.exp(-cost_loss / eps)
                
                xt, velocity = self.meta_lib_fm(vdb_inputs, vdb_outputs)
                preds = self.train_sub_step(xt)
                
                # Compute losses for each output and target
                loss = self.loss_fn(preds.jdata, velocity.jdata)
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # try 1.0–2.0
                self.optimizer.step()
                if self.scheduler!=None:
                    self.scheduler.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            if self.val_loader:
                with torch.no_grad():
                    (avg_val_loss_32, avg_val_loss_64, avg_val_loss) = self.validation()
            
            # Log the training loss
            self.logger.log({
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'val_loss per ele(32->128)': avg_val_loss_32,
                'val_loss per ele(64->128)': avg_val_loss_64,
                'epoch': epoch + 1
            })
            
            # Check if validation loss is lower than the minimum recorded loss
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                if self.is_save_model:
                    self.save_model()
        
        print(f"Training complete. Minimum validation loss: {min_val_loss:.4f}")

        if self.is_save_predictions:
            print(f'Saving the predictions to {self.save_predictions_dir}')
            self.save_predictions()
        
    def validation(self):
        self.model.eval()
        total_loss = 0
        total_loss_32 = 0
        total_loss_64 = 0
        ele_32 = 0
        ele_64 = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                obj_names, input_sizes, vdb_input, vdb_output = batch

                # group by input sizes vdb input and vdb output
                vdb_input_32 = [vdb_input[i] for i in range(len(vdb_input)) if input_sizes[i]==33]
                vdb_output_32 = [vdb_output[i] for i in range(len(vdb_output)) if input_sizes[i]==33]
                vdb_input_64 = [vdb_input[i] for i in range(len(vdb_input)) if input_sizes[i]==65]
                vdb_output_64 = [vdb_output[i] for i in range(len(vdb_output)) if input_sizes[i]==65]

                try:
                    vdb_inputs_32 = fvdb.jcat(vdb_input_32)
                    vdb_outputs_32 = fvdb.jcat(vdb_output_32)
                    vdb_inputs_32 = vdb_inputs_32.cuda()
                    vdb_outputs_32 = vdb_outputs_32.cuda()
                    
                    preds = self.eval_fm_steps(vdb_inputs_32, self.model, n_steps=4)
                    loss_32 = self.loss_fn(preds.jdata, vdb_outputs_32.jdata)
                except:
                    loss_32 = torch.tensor(0.0)

                try:
                    vdb_inputs_64 = fvdb.jcat(vdb_input_64)
                    vdb_outputs_64 = fvdb.jcat(vdb_output_64)
                    vdb_inputs_64 = vdb_inputs_64.cuda()
                    vdb_outputs_64 = vdb_outputs_64.cuda()

                    preds = self.eval_fm_steps(vdb_inputs_64, self.model, n_steps=4)
                    loss_64 = self.loss_fn(preds.jdata, vdb_outputs_64.jdata)
                except:
                    loss_64 = torch.tensor(0.0)
                    
                total_loss += loss_32.item() + loss_64.item()
                total_loss_32 += loss_32.item()
                total_loss_64 += loss_64.item()
                ele_32 += len(vdb_input_32)
                ele_64 += len(vdb_input_64)

        avg_loss_32 = total_loss_32 / (ele_32 if ele_32 > 0 else 1)
        avg_loss_64 = total_loss_64 / (ele_64 if ele_64 > 0 else 1)
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}, 32 per ele: {avg_loss_32:.4f}, 64 per ele: {avg_loss_64:.4f}")
        return avg_loss_32, avg_loss_64, avg_loss

    def save_model(self):
        path = os.path.join(self.save_model_dir, f"{self.model_name}.pth")
        torch.save(self.model, path)
        print(f"Model saved to {path}")

    
    @torch.no_grad()
    def test_fm_steps(self, input_vdbs, model, n_steps=4):
        """
        Midpoint (RK2) integrator over time t in [0, 1], updating ONLY the first channel (SDF).
        """
        def replace_jdata(vdb, new_jdata):
            # Re-wrap to keep jagged layout correct
            return fvnn.VDBTensor(vdb.grid, vdb.grid.jagged_like(new_jdata))
        
        def _pred(model, vdbs):
            xt = positional_encoding(vdbs, dim=6, is_t=False, is_sdf=False)
            return model(xt).jdata

        device = input_vdbs.jdata.device
        dtype  = input_vdbs.jdata.dtype

        # Time setup
        dt = 1.0 / n_steps
        # Create t column (start at 0). Make it 2D so concat works: [N,1]
        t_col = torch.zeros((input_vdbs.jdata.shape[0], 1), device=device, dtype=dtype)

        # Append time as LAST channel
        vdb = self.append_feature(input_vdbs, t_col)

        for step in range(n_steps):
            # Current time as scalar and as column
            t_scalar = step * dt
            t_curr = torch.full_like(t_col, t_scalar)

            # Set t in-place as LAST channel
            vdb_j = vdb.jdata
            vdb_j[:, -1] = t_curr.squeeze(-1)
            vdb = replace_jdata(vdb, vdb_j)

            # ---- k1 at (x, t) ----
            # Take current SDF values
            x_curr = vdb.jdata[:, 0:1]  # [N,1]
            # k1 = model(vdb).jdata       # [N,1]  -> dSDF/dt
            k1 = _pred(model, vdb)

            # ---- midpoint state ----
            x_mid = x_curr + 0.5 * dt * k1
            t_mid = t_scalar + 0.5 * dt

            # Build a temp VDBTensor with SDF replaced by x_mid and time set to t_mid
            vdb_mid_j = vdb.jdata.clone()
            vdb_mid_j[:, 0] = x_mid.squeeze(-1)            # replace SDF channel with midpoint guess
            vdb_mid_j[:, -1] = t_mid                       # set time to midpoint
            vdb_mid = replace_jdata(vdb, vdb_mid_j)

            # ---- k2 at (x_mid, t_mid) ----
            # k2 = model(vdb_mid).jdata  # [N,1]
            k2 = _pred(model, vdb_mid)

            # ---- final update ----
            x_next = x_curr + dt * k2

            # Write back updated SDF into vdb
            vdb_next_j = vdb.jdata.clone()
            vdb_next_j[:, 0] = x_next.squeeze(-1)
            vdb = replace_jdata(vdb, vdb_next_j)

        # Return only the predicted SDF channel as a VDBTensor with single channel
        out = fvnn.VDBTensor(
            vdb.grid,
            vdb.grid.jagged_like(vdb.jdata[:, 0:1])
        )
        return out


    def predictions_fm_steps(self, 
                             input_size,
                             input_vdb, 
                             new_features, 
                             new_ijks, 
                             model, 
                             n_steps,
                             actual_sdf=None):
        if input_size == 33:
            self.upsample_factor = 4
        elif input_size == 65:
            self.upsample_factor = 2
        self.input_size = input_size

        all_inputs = []
        for feature in new_features:
            all_inputs.append(fvnn.VDBTensor(input_vdb.grid,
                                            input_vdb.grid.jagged_like(feature)))
        all_inputs_vdb = fvdb.jcat(all_inputs)

        upsampled_sdf_size = ((self.input_size - 1) * self.upsample_factor) + 1
        sdf = np.full((upsampled_sdf_size, 
                       upsampled_sdf_size, 
                       upsampled_sdf_size), 100.0)
         
        pred = self.test_fm_steps(all_inputs_vdb, model, n_steps)
        pred_ijk = pred.grid.ijk.jdata.cpu().detach().numpy()
        vector = all_inputs_vdb.jdata[:, 1:4].cpu().detach().numpy()  
        pred_ijk = (pred_ijk)*self.upsample_factor + (vector*(self.upsample_factor//2)).astype(int)
        pred_values = pred.jdata.detach().cpu().numpy().squeeze()  # Remove extra dimension
        # sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]] = pred_values
        
        # means predictions
        D, H, W = sdf.shape
        flat_idx = np.ravel_multi_index(pred_ijk.T, sdf.shape)  # (N,)

        sum_arr = np.zeros(sdf.size, dtype=np.float32)
        cnt_arr = np.zeros(sdf.size, dtype=np.int64)

        np.add.at(sum_arr, flat_idx, pred_values)     # accumulate sums per voxel
        np.add.at(cnt_arr, flat_idx, 1)               # accumulate counts per voxel

        mask = cnt_arr > 0
        mean_arr = np.zeros_like(sum_arr, dtype=np.float32)
        mean_arr[mask] = sum_arr[mask] / cnt_arr[mask]

        sdf.flat[mask] = mean_arr[mask] 

        
        sdf_mask = np.abs(sdf) < 100

        # Error Calculation
        if actual_sdf is not None:
            # error between sdfs
            if input_size == 33:
                scale = self.sdf_scaling[input_size-1]
                assert (scale-1)==64
                actual_sdf = actual_sdf*(scale-1)
            elif input_size == 65:
                scale = self.sdf_scaling[input_size-1]
                assert (scale-1)==128
                actual_sdf = actual_sdf*(scale-1)
            else:
                raise ValueError("Input size must be either 33 or 65.")
            
            actual_values = actual_sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]]
            error = np.abs(actual_values - pred_values)
            l1_error = np.mean(error)
            mean_squared_error = np.mean(error**2)
            
        # create a fvdb tensor from the sdf
        up_grid = fvdb.gridbatch_from_ijk(
                fvdb.JaggedTensor(torch.tensor(np.array(np.where(sdf_mask)).T)),
                voxel_sizes=(1/(upsampled_sdf_size-1)),
                origins=torch.tensor([0, 0, 0])
            )
        up_ijk = up_grid.ijk.jdata.cpu().detach().numpy()
        up_values = sdf[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]]
        up_tensor = fvnn.VDBTensor(up_grid,
                                    up_grid.jagged_like(torch.tensor(up_values)))
        if actual_sdf is not None:
            # return up_tensor, error, mean_squared_error
            return up_tensor, l1_error, mean_squared_error
        else:
            # return up_tensor without error
            return up_tensor, None, None

    def save_predictions(self):
        # predictions_dir
        save_dir = os.path.join(self.save_predictions_dir, self.model_name)
        os.makedirs(save_dir, exist_ok=True)

        # load best model
        model_path = os.path.join(self.save_model_dir, f"{self.model_name}.pth")
        model = torch.load(model_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        model.eval()
        l1_errors_32 = []
        mean_squared_errors_32 = []
        l1_errors_64 = []
        mean_squared_errors_64 = []
        names = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                (obj_names, 
                 vdb_input_32, 
                 vdb_input_64, 
                 new_ijkss_32, 
                 new_ijkss_64, 
                 new_featuress_32, 
                 new_featuress_64, 
                 actual_sdfs) = batch
                vdb_inputs_32 = fvdb.jcat(vdb_input_32)
                vdb_inputs_32 = vdb_inputs_32.cuda()
                new_ijks_32 = new_ijkss_32[0]
                new_features_32 = new_featuress_32[0]

                vdb_inputs_64 = fvdb.jcat(vdb_input_64)
                vdb_inputs_64 = vdb_inputs_64.cuda()
                new_ijks_64 = new_ijkss_64[0]
                new_features_64 = new_featuress_64[0]

                actual_sdf = actual_sdfs[0]
                names.append(obj_names[0])

                # test 32, 64 separately
                (up_tensor_32, 
                 l1_error_32, 
                 mean_squared_error_32) = self.predictions_fm_steps(33,
                                                      vdb_inputs_32, 
                                                      new_features_32, 
                                                      new_ijks_32, 
                                                      model, 
                                                      n_steps=10,
                                                      actual_sdf=actual_sdf)

                l1_errors_32.append(l1_error_32)
                mean_squared_errors_32.append(mean_squared_error_32)

                (up_tensor_64, 
                 l1_error_64, 
                 mean_squared_error_64) = self.predictions_fm_steps(65,
                                                      vdb_inputs_64,
                                                      new_features_64,
                                                      new_ijks_64,
                                                      model,
                                                      n_steps=10,
                                                      actual_sdf=actual_sdf)

                l1_errors_64.append(l1_error_64)
                mean_squared_errors_64.append(mean_squared_error_64)

                # save the predictions 32 and 64
                file_names = [f"{32}_{name.split('.')[0]}" for name in obj_names]
                output_file = os.path.join(save_dir, f'{file_names[0]}.nvdb')
                fvdb.save(output_file, up_tensor_32.grid, up_tensor_32.data, compressed=True)
                print(f"Saved predictions for {file_names[0]} to {output_file}")

                file_names = [f"{64}_{name.split('.')[0]}" for name in obj_names]
                output_file = os.path.join(save_dir, f'{file_names[0]}.nvdb')
                fvdb.save(output_file, up_tensor_64.grid, up_tensor_64.data, compressed=True)
                print(f"Saved predictions for {file_names[0]} to {output_file}")

        # log the errors
        df_error = pd.DataFrame({
            'object_name': names,
            'l1_error_32': l1_errors_32,
            'l1_error_64': l1_errors_64,
            'mean_squared_error_32': mean_squared_errors_32,
            'mean_squared_error_64': mean_squared_errors_64
        })
        df_error_describe = df_error.describe().reset_index()
        self.logger.log({'data/sdf_eval': wandb.Table(dataframe=df_error)})
        self.logger.log({'stats/sdf_eval': wandb.Table(dataframe=df_error_describe)})
        print(df_error_describe)