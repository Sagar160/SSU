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
                 dataProcessor,
                 model_name, 
                 model, 
                 num_epochs,
                 train_loader, 
                 val_loader,
                 test_loader,
                 upsample_factor,
                 input_size,
                #  pos_enc_dim, 
                 optimizer, 
                 loss_fn_name,
                 is_save_model,
                 save_model_dir,
                 save_predictions_dir, 
                 logger
                 ):

        self.dataProcessor = dataProcessor

        self.model_name = model_name
        self.model = model
        self.num_epochs = num_epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.upsample_factor = upsample_factor
        self.input_size = input_size
        # self.pos_enc_dim = pos_enc_dim

        self.optimizer = optimizer
        self.loss_fn = loss_fn_name
        self.loss_fn = LossFunctions(loss_fn_name).loss_fn
        
        self.is_save_model = is_save_model
        self.save_model_dir = save_model_dir
        self.save_predictions_dir = save_predictions_dir
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
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
        # t = torch.full_like(input_vdbs.jdata[:, 0], 0).to(self.device)
        # print(t.shape, input_vdbs.jdata[:, 0].shape, )
        features = t * input_vdbs.jdata[:, 0].unsqueeze(1) + (1 - t) * output_vdbs.jdata
        features = torch.cat([features, input_vdbs.jdata[:, 1:], t], dim=-1)
        # print(features.shape)
        sampled_vdbs = fvnn.VDBTensor(input_vdbs.grid,
                                      input_vdbs.grid.jagged_like(features))
        return sampled_vdbs

    def velocity_fm(self, input_vdbs, output_vdbs):
        velocity = output_vdbs.jdata[:, 0] - input_vdbs.jdata[:, 0]
        velocity_vdb = fvnn.VDBTensor(input_vdbs.grid,
                                  input_vdbs.grid.jagged_like(velocity[:, None]))
        return velocity_vdb
    
    def eval_fm_steps(self, input_vdbs, model, n_steps=4):
        # mask = input_vdbs.jdata[:, 1] > 0.5
        # input_vdbs = fvnn.VDBTensor(input_vdbs.grid,
        #                             input_vdbs.grid.jagged_like(
        #                                 input_vdbs.jdata[:, 0].unsqueeze(1)
        #                             ))
        # mask = input_vdbs.jdata[:, 1] > 0.5
        dt = 1 / n_steps
        t = torch.full_like(input_vdbs.jdata[:, 0], 0).to(self.device)
        t = t.unsqueeze(1)  # Ensure t is a column vector
        # for mask put t=1
        # t[mask] = 1.0
        input_vdbs = self.append_feature(input_vdbs, t)
        model.eval()
        with torch.no_grad():
            for t in range(n_steps):
                t = torch.full_like(input_vdbs.jdata[:, 0], t/n_steps).to(self.device)
                # t[mask] = 1.0
                # t = t.unsqueeze(1)  # Ensure t is a column vector
                input_vdbs.jdata[:, -1] = t
                updated_sdf = input_vdbs.jdata[:, 0].unsqueeze(1) + dt * model(input_vdbs).jdata
                # updated_sdf = input_vdbs.jdata[:, 0].unsqueeze(1) + 1 * model(input_vdbs).jdata
                # updated_sdf[mask] = input_vdbs.jdata[mask, 0].unsqueeze(1)
                input_vdbs.jdata[:, 0] = updated_sdf.squeeze(1)
                # print(input_vdbs.jdata.shape)
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
        # t = torch.full_like(input_vdbs.jdata[:, 0], 0).to(self.device)
        # mask = input_vdbs.jdata[:, 1] > 0.5
        # t[mask] = 1.0
        # path_sample = path.sample(t=t, 
        #                           x_0=input_vdbs.jdata[:,0], 
        #                           x_1=output_vdbs.jdata[:,0])
        path_sample = path.sample(t=t, 
                                  x_0=input_vdbs.jdata[:,0], 
                                  x_1=output_vdbs.jdata[:,0])
        xt = path_sample.x_t
        t = path_sample.t
        # print(t)
        # return 1
        velocity = path_sample.dx_t

        xt_feature = torch.cat([xt.unsqueeze(1), 
                                input_vdbs.jdata[:, 1:], 
                                t.unsqueeze(1)], dim=-1)
        
        xt = fvnn.VDBTensor(output_vdbs.grid,
                            output_vdbs.grid.jagged_like(xt_feature))
        velocity = fvnn.VDBTensor(output_vdbs.grid,
                                    output_vdbs.grid.jagged_like(velocity[:, None]))
        return xt, velocity

    def train(self):
        min_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            # run data processor at equal intervals
            # if (epoch) == 0:
            #     self.dataProcessor.run_data_processing(epoch)

            # if self.dataProcessor is not None:
            #     self.dataProcessor.run_data_processing(epoch)

            self.model.train()
            total_loss = 0

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                obj_names, vdb_input_33_tri, vdb_output_65, vdb_output_129 = batch

                vdb_inputs_33_tri = fvdb.jcat(vdb_input_33_tri)
                vdb_outputs_65 = fvdb.jcat(vdb_output_65)
                vdb_outputs_129 = fvdb.jcat(vdb_output_129)
                vdb_inputs_33_tri = vdb_inputs_33_tri.cuda()
                vdb_outputs_65 = vdb_outputs_65.cuda()
                self.optimizer.zero_grad()

                xt, velocity = self.meta_lib_fm(vdb_inputs_33_tri, vdb_outputs_65)
                # preds = self.model(vdb_inputs)
                preds = self.train_sub_step(xt)

                # Compute losses for each output and target
                loss1 = self.loss_fn(preds.jdata, velocity.jdata)

                vdb_inputs_65 = vdb_inputs_33_tri.clone().detach()
                vdb_inputs_65.jdata[:, 0] = (vdb_inputs_65.jdata[:, 0] + preds.jdata[:, 0]).detach().cpu()
                vdb_inputs_65 = vdb_inputs_65.detach().cpu()
                tri_feat = vdb_inputs_65.grid.sample_trilinear(vdb_outputs_129.grid.ijk.float(), vdb_inputs_65.jdata)
                vdb_inputs_65_tri = fvnn.VDBTensor(vdb_outputs_129.grid, tri_feat)
                vdb_inputs_65_tri = vdb_inputs_65_tri.cuda()
                vdb_outputs_129 = vdb_outputs_129.cuda()

                xt, velocity = self.meta_lib_fm(vdb_inputs_65_tri, vdb_outputs_129)
                preds = self.train_sub_step(xt)
                loss2 = self.loss_fn(preds.jdata, vdb_outputs_129.jdata)

                loss = loss1*0.7 + loss2*0.3
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}")
            if self.val_loader:
                (avg_val_loss) = self.validation()
            
            # Log the training loss
            self.logger.log({
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch + 1
            })
            
            # Check if validation loss is lower than the minimum recorded loss
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                if self.is_save_model:
                    self.save_model()
        
        print(f"Training complete. Minimum validation loss: {min_val_loss:.4f}")

        if self.is_save_model:
            print(f'Saving the predictions to {self.save_predictions_dir}')
            self.save_predictions()
        
    def validation(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                obj_names, vdb_input_33_tri, vdb_output_65, vdb_output_129 = batch

                vdb_inputs_33_tri = fvdb.jcat(vdb_input_33_tri)
                vdb_outputs_65 = fvdb.jcat(vdb_output_65)
                vdb_outputs_129 = fvdb.jcat(vdb_output_129)
                vdb_inputs_33_tri = vdb_inputs_33_tri.cuda()
                vdb_outputs_65 = vdb_outputs_65.cuda()
                vdb_outputs_129 = vdb_outputs_129.cuda()
                self.optimizer.zero_grad()

                # preds = self.model(vdb_inputs)
                preds = self.eval_fm_steps(vdb_inputs_33_tri, self.model, n_steps=4)


                # Compute losses for each output and target
                loss1 = self.loss_fn(preds.jdata, vdb_outputs_65.jdata)

                tri_feat = preds.grid.sample_trilinear(vdb_outputs_129.grid.ijk.float(), preds.jdata)
                vdb_inputs_65_tri = fvnn.VDBTensor(vdb_outputs_129.grid, tri_feat)
                vdb_inputs_65_tri = vdb_inputs_65_tri.cuda()
                vdb_outputs_129 = vdb_outputs_129.cuda()

                preds = self.eval_fm_steps(vdb_inputs_65_tri, self.model, n_steps=4)
                loss2 = self.loss_fn(preds.jdata, vdb_outputs_129.jdata)

                # preds = self.eval_fm_steps(vdb_inputs, self.model, n_steps=4)
                # preds = self.model(vdb_inputs)
                # loss = self.loss_fn(preds.jdata, vdb_outputs.jdata)
                loss = loss1*0.7 + loss2*0.3
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self):
        path = os.path.join(self.save_model_dir, f"{self.model_name}.pth")
        torch.save(self.model, path)
        print(f"Model saved to {path}")
    
    def test_fm_steps(self, input_vdb, model, n_steps):
        dt = 1 / n_steps
        t = torch.full_like(input_vdb.jdata[:, 0], 0).to(input_vdb.device)
        t = t.unsqueeze(1)
        input_vdb = self.append_feature(input_vdb, t)
        model.eval()
        with torch.no_grad():
            for t in range(n_steps):
                t = torch.full_like(input_vdb.jdata[:, 0], float(t)/n_steps).to(input_vdb.device)
                input_vdb.jdata[:, -1] = t
                out_vdb = model(input_vdb)
                updated_sdf = input_vdb.jdata[:, 0].unsqueeze(1) + dt * out_vdb.jdata
                input_vdb.jdata[:, 0] = updated_sdf.squeeze(1)
                input_vdb = fvnn.VDBTensor(input_vdb.grid,
                                            input_vdb.grid.jagged_like(
                                                input_vdb.jdata
                                            ))

            predicted_vdbs = fvnn.VDBTensor(input_vdb.grid,
                                            input_vdb.grid.jagged_like(
                                                input_vdb.jdata[:, 0].unsqueeze(1)
                                            ))
            return predicted_vdbs
    
    def _get_all_shifted_positions(self, vdb_tensor, size, upsample_factor=2, is_positive_direction=True):
        m3g = torch.tensor(mt.mesh_grid(upsample_factor+1), device=vdb_tensor.device) - (upsample_factor//2)

        new_ijks = []
        new_features = []
        if is_positive_direction:
            m3g = m3g[(m3g >= 0).all(axis=1)]
        else:
            m3g = m3g[(m3g <= 0).all(axis=1)] 
        for mg in m3g:
            ijk = vdb_tensor.grid.ijk.jdata
            ijk = (upsample_factor * ijk + mg).view(-1, 3)
            ijk = np.clip(ijk.cpu().detach().numpy(), 0, (size-1)*upsample_factor)
            ijk_vector = ijk - (vdb_tensor.grid.ijk.jdata.cpu().detach().numpy() * upsample_factor)
            ijk_vector = ijk_vector / (upsample_factor // 2)  # Normalize to values between -1 and 1
            ijk_vector = torch.tensor(ijk_vector, dtype=torch.float32, device=vdb_tensor.device)
            print(ijk_vector.shape, vdb_tensor.data.jdata.shape)
            new_features.append(torch.cat([vdb_tensor.data.jdata, ijk_vector], axis=-1))
            new_ijks.append(torch.tensor(ijk, dtype=torch.int, device=vdb_tensor.device))
        return new_features, new_ijks

    def upsample_vdb(self, input_vdb, higher_size, scale=2, is_mask=True):
        def custom_subdivide_grid(grid: fvdb.GridBatch, scale, m3g, upshape):
            '''custom subdivision of a grid to create a finer grid:
                [0,    1,    2] -->
                [0, 1, 2, 3, 4]'''
            ijk = grid.ijk.jdata
            # m3g = torch.tensor(mt.mesh_grid(3),device=grid.device)-1
            new_ijk = (scale*ijk[:, None, :]+ m3g[None, :, :]).view(-1, 3)
            new_ijk = torch.clamp(new_ijk, 0, upshape-1)
            return fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(new_ijk), origins=grid.origins, voxel_sizes=grid.voxel_sizes/2)

        grid = input_vdb.grid
        ijk = grid.ijk.jdata
        up_ijk = scale*ijk

        up_feature_tensor = torch.randn((higher_size, higher_size, higher_size), device=input_vdb.device)
        up_feature_tensor[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]] = input_vdb.data.jdata.squeeze()

        # mask = input_vdb.jdata[:, 0] < (3/(higher_size))
        # mask = input_vdb.jdata[:, 0] < (3/(higher_size))*(higher_size-1)
        if is_mask:
            mask = (input_vdb.jdata[:, 0]) < (3/(higher_size))*(65-1)
            ijk_tensor = torch.tensor(ijk[mask], 
                                dtype=torch.int, 
                                device=self.device)
            mask_grid = fvdb.gridbatch_from_ijk(
                        fvdb.JaggedTensor(ijk_tensor),
                        voxel_sizes=grid.voxel_sizes,
                        origins=grid.origins)
        else:
            mask_grid = grid
        m3g = torch.tensor(mt.mesh_grid(scale+1),device=grid.device)-(scale//2)
        up_grid = custom_subdivide_grid(mask_grid, scale, m3g, upshape=higher_size)
        up_feature = up_feature_tensor[up_grid.ijk.jdata[:, 0], 
                                      up_grid.ijk.jdata[:, 1], 
                                      up_grid.ijk.jdata[:, 2]]
        up_feature = up_feature[:, None].to(torch.float32)
        up_tensor = fvnn.VDBTensor(up_grid,
                                    up_grid.jagged_like(up_feature))
        return up_tensor

    def predictions_fm_steps(self, 
                             input_vdb, 
                             upsample_factor, 
                             model, 
                             n_steps,
                             actual_sdf=None):

        lower_size = self.input_size
        while upsample_factor != 1:
            upsample_factor = upsample_factor / 2
            higher_size = int(((lower_size-1) * 2) + 1)
            up_vdb = self.upsample_vdb(input_vdb, higher_size, scale=2)
            pred = self.test_fm_steps(up_vdb, model, n_steps)
            lower_size = higher_size
            input_vdb = pred

        sdf = np.full((higher_size, higher_size, higher_size), 100.0)
        pred_ijk = pred.grid.ijk.jdata.cpu().detach().numpy()
        pred_values = pred.data.jdata.cpu().detach().numpy().squeeze()
        sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]] = pred_values
        sdf_mask = (np.abs(sdf)/(65-1)) < 3/((higher_size-1)*2 + 1)

        # create a fvdb tensor from the sdf
        up_grid = fvdb.gridbatch_from_ijk(
                fvdb.JaggedTensor(torch.tensor(np.array(np.where(sdf_mask)).T)),
                voxel_sizes=(1/(higher_size-1)),
                origins=torch.tensor([0, 0, 0])
            )
        up_ijk = up_grid.ijk.jdata.cpu().detach().numpy()
        up_values = sdf[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]]
        up_values = up_values[:, None].astype(np.float32)
        up_tensor = fvnn.VDBTensor(up_grid,
                                    up_grid.jagged_like(torch.tensor(up_values)))
            
        if actual_sdf is not None:
            # return up_tensor, error, mean_squared_error
            # error between sdfs
            actual_values = actual_sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]]
            error = np.abs(actual_values - pred_values)
            l1_error = np.mean(error)
            mean_squared_error = np.mean(error**2)
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
        l1_errors = []
        mean_squared_errors = []
        names = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                obj_names, vdb_input, actual_sdfs = batch
                vdb_inputs = fvdb.jcat(vdb_input)
                actual_sdf = actual_sdfs[0]
                # vdb_outputs = fvdb.jcat(vdb_output)

                up_tensor, l1_error, mean_squared_error = self.predictions_fm_steps(input_vdb=vdb_inputs, 
                                                      model=model, 
                                                      upsample_factor=4,
                                                      n_steps=10,
                                                      actual_sdf=actual_sdf)

                names.append(obj_names[0])
                l1_errors.append(l1_error)
                mean_squared_errors.append(mean_squared_error)

                # save the predictions
                file_names = [name.split('.')[0] for name in obj_names]
                output_file = os.path.join(save_dir, f'{file_names[0]}.nvdb')
                fvdb.save(output_file, up_tensor.grid, up_tensor.data, compressed=True)
                print(f"Saved predictions for {file_names[0]} to {output_file}")
                
        # log the errors
        df_error = pd.DataFrame({
            'object_name': names,
            'l1_error': l1_errors,
            'mean_squared_error': mean_squared_errors
        })
        df_error_describe = df_error.describe().reset_index()
        self.logger.log({'data/sdf_eval': wandb.Table(dataframe=df_error)})
        self.logger.log({'stats/sdf_eval': wandb.Table(dataframe=df_error_describe)})
        print(df_error_describe)