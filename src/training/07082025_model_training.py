import os
import sys
import fvdb
import torch
import wandb
from tqdm import tqdm
from .loss import LossFunctions

sys.path.append('../src/utils')
from ssu_tools import positional_encoding

class ModelTrainer:
    def __init__(self,
                 model_name, 
                 model, 
                 num_epochs,
                 train_loader, 
                 val_loader,
                #  test_loader,
                 pos_enc_dim, 
                 optimizer, 
                 loss_fn_name,
                 loss_weights,
                 is_save_model,
                 save_model_dir, 
                 logger):
        
        self.model_name = model_name
        self.model = model
        self.num_epochs = num_epochs

        self.train_loader = train_loader
        self.val_loader = val_loader
        # self.test_loader = test_loader
        self.pos_enc_dim = pos_enc_dim

        self.optimizer = optimizer
        self.loss_fn = loss_fn_name
        self.loss_fn = LossFunctions(loss_fn_name).loss_fn
        self.loss_weights = loss_weights
        
        self.is_save_model = is_save_model
        self.save_model_dir = save_model_dir
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.logger = logger
        self.logger.log({'loss_weights': wandb.Table(data=[self.loss_weights], 
                                                     columns=['w1', 'w2', 'w3'])})

    def train_sub_step(self, inputs, targets):
        inputs = positional_encoding(inputs, self.pos_enc_dim)

        outputs = self.model(inputs, targets.grid)
        return outputs

    def train(self):
        min_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            avg_loss_1 = 0
            avg_loss_2 = 0
            avg_loss_3 = 0
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.num_epochs}'):
                obj_names, vdb_32s, vdb_64s, vdb_128s = batch
                vdb_32s = fvdb.jcat(vdb_32s)
                vdb_64s = fvdb.jcat(vdb_64s)
                vdb_128s = fvdb.jcat(vdb_128s)
                self.optimizer.zero_grad()  

                output_64 = self.train_sub_step(vdb_32s, vdb_64s)
                output_128 = self.train_sub_step(vdb_64s, vdb_128s)
                output_64_128 = self.train_sub_step(output_64, vdb_128s)

                # Compute losses for each output and target
                loss_1 = self.loss_fn(output_64.jdata, vdb_64s.jdata)
                loss_2 = self.loss_fn(output_128.jdata, vdb_128s.jdata)
                loss_3 = self.loss_fn(output_64_128.jdata, vdb_128s.jdata) 

                # Combine losses (sum or weighted sum)
                [w1, w2, w3] = self.loss_weights
                loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                avg_loss_1 += loss_1.item()
                avg_loss_2 += loss_2.item()
                avg_loss_3 += loss_3.item()
            avg_loss = total_loss / len(self.train_loader)
            avg_loss_1 /= len(self.train_loader)
            avg_loss_2 /= len(self.train_loader)
            avg_loss_3 /= len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {avg_loss:.4f}, Avg Loss 1: {avg_loss_1:.4f}, Avg Loss 2: {avg_loss_2:.4f}, Avg Loss 3: {avg_loss_3:.4f}")
            if self.val_loader:
                (avg_val_loss, 
                 avg_val_loss_1, 
                 avg_val_loss_2, 
                 avg_val_loss_3) = self.validation()
            
            # Log the training loss
            self.logger.log({
                'train_loss': avg_loss,
                'train_loss_32->64': avg_loss_1,
                'train_loss_64->128': avg_loss_2,
                'train_loss_32->64->128': avg_val_loss_3,
                'val_loss': avg_val_loss,
                'val_loss_32->64': avg_val_loss_1,
                'val_loss_64->128': avg_val_loss_2,
                'val_loss_32->64->128': avg_val_loss_3,
                'epoch': epoch + 1
            })
            
            # Check if validation loss is lower than the minimum recorded loss
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                if self.is_save_model:
                    self.save_model()
        
        print(f"Training complete. Minimum validation loss: {min_val_loss:.4f}")
        
    def validation(self):
        self.model.eval()
        total_loss = 0
        avg_loss_1 = 0
        avg_loss_2 = 0
        avg_loss_3 = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                obj_names, vdb_32s, vdb_64s, vdb_128s = batch
                vdb_32s = fvdb.jcat(vdb_32s)
                vdb_64s = fvdb.jcat(vdb_64s)
                vdb_128s = fvdb.jcat(vdb_128s)  

                output_64 = self.train_sub_step(vdb_32s, vdb_64s)
                output_128 = self.train_sub_step(vdb_64s, vdb_128s)
                output_64_128 = self.train_sub_step(output_64, vdb_128s)

                loss1 = self.loss_fn(output_64.jdata, vdb_64s.jdata)
                loss2 = self.loss_fn(output_128.jdata, vdb_128s.jdata)
                loss3 = self.loss_fn(output_64_128.jdata, vdb_128s.jdata)

                [w1, w2, w3] = self.loss_weights
                loss = w1 * loss1 + w2 * loss2 + w3 * loss3
                
                total_loss += loss.item()
                avg_loss_1 += loss1.item()
                avg_loss_2 += loss2.item()
                avg_loss_3 += loss3.item()
                
        avg_loss = total_loss / len(self.val_loader)
        avg_loss_1 /= len(self.val_loader)
        avg_loss_2 /= len(self.val_loader)
        avg_loss_3 /= len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.4f}, Avg Loss 1: {avg_loss_1:.4f}, Avg Loss 2: {avg_loss_2:.4f}, Avg Loss 3: {avg_loss_3:.4f}")
        return avg_loss, avg_loss_1, avg_loss_2, avg_loss_3

    def save_model(self):
        path = os.path.join(self.save_model_dir, f"{self.model_name}.pth")
        torch.save(self.model, path)
        print(f"Model saved to {path}")

