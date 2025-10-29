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
        dt = 1 / n_steps
        t = torch.full_like(input_vdbs.jdata[:, 0], 0).to(self.device)
        t = t.unsqueeze(1)  # Ensure t is a column vector
        input_vdbs = self.append_feature(input_vdbs, t)
        for t in range(n_steps):
            t = torch.full_like(input_vdbs.jdata[:, 0], t/n_steps).to(self.device)
            # t = t.unsqueeze(1)  # Ensure t is a column vector
            input_vdbs.jdata[:, -1] = t
            updated_sdf = input_vdbs.jdata[:, 0].unsqueeze(1) + dt * model(input_vdbs).jdata
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
            # run data processor at equal intervals
            # if (epoch) % 3 == 0:
            if self.dataProcessor is not None:
                self.dataProcessor.run_data_processing(epoch)

            self.model.train()
            total_loss = 0

            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                obj_names, vdb_input, vdb_output = batch

                vdb_inputs = fvdb.jcat(vdb_input)
                vdb_outputs = fvdb.jcat(vdb_output)
                vdb_inputs = vdb_inputs.cuda()
                vdb_outputs = vdb_outputs.cuda()
                self.optimizer.zero_grad()

                # manual flow matching steps
                # xt = self.sample_fm(vdb_inputs, vdb_outputs)
                # velocity = self.velocity_fm(vdb_inputs, vdb_outputs)

                # Add gaussian noise to input vdbs
                # noise = torch.randn_like(vdb_inputs.jdata) * 0.06 
                # vdb_inputs_noisy = fvnn.VDBTensor(vdb_inputs.grid,
                #                                  vdb_inputs.grid.jagged_like(vdb_inputs.jdata + noise)) 
                # flow matching steps using meta-lib
                xt, velocity = self.meta_lib_fm(vdb_inputs, vdb_outputs)

                preds = self.train_sub_step(xt)

                # Compute losses for each output and target
                loss = self.loss_fn(preds.jdata, velocity.jdata)

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
                obj_names, vdb_input, vdb_output = batch
                vdb_inputs = fvdb.jcat(vdb_input)
                vdb_outputs = fvdb.jcat(vdb_output)
                vdb_inputs = vdb_inputs.cuda()
                vdb_outputs = vdb_outputs.cuda()

                # preds = self.train_sub_step(vdb_inputs)
                preds = self.eval_fm_steps(vdb_inputs, self.model, n_steps=4)

                loss = self.loss_fn(preds.jdata, vdb_outputs.jdata)

                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_model(self):
        path = os.path.join(self.save_model_dir, f"{self.model_name}.pth")
        torch.save(self.model, path)
        print(f"Model saved to {path}")

    # def get_all_shifted_positions(self, vdb_tensor, size, upsample_factor):
    #     m3g = torch.tensor(mt.mesh_grid(upsample_factor+1), device=vdb_tensor.device) - (upsample_factor//2)

    #     new_ijks = []
    #     new_features = []
    #     for mg in m3g:
    #         ijk = vdb_tensor.grid.ijk.jdata
    #         ijk = (upsample_factor * ijk + mg).view(-1, 3)
    #         ijk = np.clip(ijk.cpu().detach().numpy(), 0, (size-1)*upsample_factor)
    #         ijk_vector = ijk - (vdb_tensor.grid.ijk.jdata.cpu().detach().numpy() * upsample_factor)
    #         ijk_vector = ijk_vector / (upsample_factor // 2)  # Normalize to values between -1 and 1
    #         ijk_vector = torch.tensor(ijk_vector, dtype=torch.float32, device=vdb_tensor.device)

    #         new_features.append(torch.cat([vdb_tensor.data.jdata[:, 0][:, None], ijk_vector], axis=-1))
    #         new_ijks.append(torch.tensor(ijk, dtype=torch.int, device=vdb_tensor.device))
    #     return new_features, new_ijks
    
    def test_fm_steps(self, input_vdb, model, n_steps):
        dt = 1 / n_steps
        t = torch.full_like(input_vdb.jdata[:, 0], 0).to(input_vdb.device)
        t = t.unsqueeze(1)
        input_vdb = self.append_feature(input_vdb, t)
        for t in range(n_steps):
            t = torch.full_like(input_vdb.jdata[:, 0], float(t)/n_steps).to(input_vdb.device)
            input_vdb.jdata[:, -1] = t
            updated_sdf = input_vdb.jdata[:, 0].unsqueeze(1) + dt * model(input_vdb).jdata
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

    def predictions_fm_steps(self, 
                             input_vdb, 
                             new_features, 
                             new_ijks, 
                             model, 
                             n_steps,
                             actual_sdf=None):
        # new_features, new_ijks = self.get_all_shifted_positions(input_vdb, 
        #                                                         size=self.input_size, 
        #                                                         upsample_factor=self.upsample_factor)
        all_inputs = []
        for feature in new_features:
            all_inputs.append(fvnn.VDBTensor(input_vdb.grid,
                                            input_vdb.grid.jagged_like(feature)))
        all_inputs_vdb = fvdb.jcat(all_inputs)

        upsampled_sdf_size = ((self.input_size - 1) * self.upsample_factor) + 1
        sdf = np.full((upsampled_sdf_size, 
                       upsampled_sdf_size, 
                       upsampled_sdf_size), 100.0)
        
        # for shifted_feature, shifted_ijk in zip(new_features, new_ijks):
        #     new_vdb_tensor = fvnn.VDBTensor(input_vdb.grid,
        #                                     input_vdb.grid.jagged_like(shifted_feature))
        #     pred = self.test_fm_steps(new_vdb_tensor, model, n_steps)

        #     ijk = shifted_ijk.cpu().detach().numpy()
        #     pred_values = pred.jdata.cpu().detach().numpy().squeeze()  # Remove extra dimension
        #     sdf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = pred_values  
        pred = self.test_fm_steps(all_inputs_vdb, model, n_steps)
        pred_ijk = pred.grid.ijk.jdata.cpu().detach().numpy()
        vector = all_inputs_vdb.jdata[:, 1:4].cpu().detach().numpy()  
        pred_ijk = (pred_ijk)*self.upsample_factor + (vector*(self.upsample_factor//2)).astype(int)
        pred_values = pred.jdata.detach().cpu().numpy().squeeze()  # Remove extra dimension
        sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]] = pred_values
        
        sdf_mask = np.abs(sdf) < 100
        if actual_sdf is not None:
            # error between sdfs
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
        l1_errors = []
        mean_squared_errors = []
        names = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                obj_names, vdb_input, new_ijkss, new_featuress, actual_sdfs = batch
                vdb_inputs = fvdb.jcat(vdb_input)
                new_ijks = new_ijkss[0]
                new_features = new_featuress[0]
                actual_sdf = actual_sdfs[0]
                # vdb_outputs = fvdb.jcat(vdb_output)

                up_tensor, l1_error, mean_squared_error = self.predictions_fm_steps(vdb_inputs, 
                                                      new_features, 
                                                      new_ijks, 
                                                      model, 
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