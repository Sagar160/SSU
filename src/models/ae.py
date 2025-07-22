import torch
import fvdb.nn as fvnn

class FVDBAutoencoder(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=64, levels=3, base_channels=32):
        super().__init__()
        # Encoder
        encoder_layers = []
        in_ch = in_channels
        out_ch = base_channels
        for i in range(levels):
            encoder_layers.append(fvnn.SparseConv3d(in_ch, out_ch, kernel_size=3, stride=2))
            encoder_layers.append(fvnn.BatchNorm(out_ch))
            encoder_layers.append(fvnn.SiLU(inplace=True))
            in_ch = out_ch
            out_ch = min(out_ch * 2, latent_dim)
        encoder_layers.append(fvnn.SparseConv3d(in_ch, latent_dim, kernel_size=3, stride=2))
        encoder_layers.append(fvnn.BatchNorm(latent_dim))
        encoder_layers.append(fvnn.SiLU(inplace=True))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Decoder: store layers and flags for transposed layers
        decoder_layers = []
        decoder_transposed_flags = []
        in_ch = latent_dim
        out_ch = max(base_channels, latent_dim // (2 ** levels))
        for i in range(levels):
            decoder_layers.append(fvnn.SparseConv3d(in_ch, out_ch, kernel_size=3, stride=2, transposed=True))
            decoder_transposed_flags.append(True)
            decoder_layers.append(fvnn.BatchNorm(out_ch))
            decoder_transposed_flags.append(False)
            decoder_layers.append(fvnn.SiLU(inplace=True))
            decoder_transposed_flags.append(False)
            in_ch = out_ch
            out_ch = max(base_channels, out_ch // 2)
        # Last layer
        decoder_layers.append(fvnn.SparseConv3d(in_ch, out_channels, kernel_size=3, stride=2, transposed=True))
        decoder_transposed_flags.append(True)
        self.decoder_layers = torch.nn.ModuleList(decoder_layers)
        self.decoder_transposed_flags = decoder_transposed_flags

    def forward(self, x, out_grid):
        z = self.encoder(x)
        out = z
        # Pass out_grid to all transposed layers
        for layer, is_transposed in zip(self.decoder_layers, self.decoder_transposed_flags):
            if is_transposed:
                out = layer(out, out_grid=out_grid)
            else:
                out = layer(out)
        return out

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z, out_grid):
        out = z
        for layer, is_transposed in zip(self.decoder_layers, self.decoder_transposed_flags):
            if is_transposed:
                out = layer(out, out_grid=out_grid)
            else:
                out = layer(out)
        return out
    
import gc
import os
import sys
import random
import wandb
sys.path.append('../utils')
sys.path.append('../data_utils')
sys.path.append('../models')
import numpy as np
import matplotlib.pyplot as plt
from ABC_dataset import get_item, ABCdataset, get_vdb_data_loader
import mesh_tools as mt
import model_tools as mtools
import flow_matching_tools as fmt
import fvdb_utils as fu
import model as fvdbModel
import unet as fvdbUnet
import igl
from meshplot import plot
import fvdb
import fvdb.nn as fvnn
import torch
import torch.nn as nn
from tqdm import tqdm
from skimage import measure
import trimesh


def load_dataset(data_dir, n_samples=None):
    
    # raise ValueError if the n_samples is not None and Int
    if n_samples is not None and not isinstance(n_samples, int):
        raise ValueError("n_samples must be an integer or None")
    # raise ValueError if the dir is not a directory
    if not os.path.isdir(data_dir):
        raise ValueError(f"The provided path {dir} is not a directory")
    
    if n_samples is not None:
        train_set_names = os.listdir(data_dir)[:n_samples]
    elif n_samples is None:
        train_set_names = os.listdir(data_dir)
        
    random.shuffle(train_set_names)
    print(train_set_names[:5]) # Print first 5 names for debugging 
    
    train_size = int(0.6 * len(train_set_names))
    test_size = int(0.2 * len(train_set_names))
    val_size = len(train_set_names) - train_size - test_size
    train_names = train_set_names[:train_size]
    val_names = train_set_names[train_size:train_size + val_size]
    test_names = train_set_names[train_size + val_size:]

    train_dataset = ABCdataset(data_dir, train_names)
    val_dataset = ABCdataset(data_dir, val_names)
    test_dataset = ABCdataset(data_dir, test_names, mode='test')

    print(f'Number of samples in the dataset: {len(train_set_names)}')
    print(f'Number of samples in the train set: {len(train_dataset)}')
    print(f'Number of samples in the test set: {len(test_dataset)}')
    print(f'Number of samples in the validation set: {len(val_dataset)}')

    train_data_loader = get_vdb_data_loader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_data_loader = get_vdb_data_loader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_data_loader = get_vdb_data_loader(test_dataset, batch_size=1, shuffle=False, num_workers=0, mode='test')
    return train_data_loader, val_data_loader, test_data_loader


def train(model, model_name, 
          pos_enc_dim, train_data_loader, 
          val_data_loader, optimizer, epochs, 
          device, wandb_run=False, save_model=False):
    history = []
    saliency_labels = ['sdf'] + [f'pos_enc_{i}' for i in range(pos_enc_dim)]
    saliency_feature = []
    min_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        Loss = []
        saliency_feature_epoch = []
        
        model.train()
        for small_vdb, large_vdb in tqdm(train_data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            t = torch.rand_like(large_vdb.jdata).to(device)

            #transform vdb
            # xt, noise = fmt.transform_input(small_vdb, 
            #                             large_vdb, 
            #                             t, 
            #                             pos_enc_dim=pos_enc_dim,
            #                             scale_factor=2,
            #                             upsampler='trilinear', 
            #                             g_noise=True)

            # xt.jdata.requires_grad_()
            large_vdb = large_vdb.to(device)
            
            optimizer.zero_grad()
            pred = model(large_vdb, large_vdb.grid)
            loss = torch.sqrt(torch.mean((pred.jdata - large_vdb.jdata) ** 2))
            # loss = fmt.fm_loss(pred, large_vdb, noise)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # save saliency 
            # saliency_feature_epoch.append(xt.jdata.grad.abs().mean(dim=0).cpu().numpy())

        # average saliency
        # saliency_feature.append(np.stack(saliency_feature_epoch).mean(axis=0))

        # loss
        avg_loss = epoch_loss / len(train_data_loader)
        Loss.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')

        # Validation
        with torch.no_grad():
            
            model.eval()
            val_loss = 0
            
            for small_vdb, large_vdb in tqdm(val_data_loader, desc='Validation'):
                # t = torch.rand(batch_size, 1).to(device)
                # t = torch.rand_like(large_vdb.jdata).to(device)

                #transform vdb
                # xt, noise = fmt.transform_input(small_vdb, 
                #                             large_vdb, 
                #                             t,
                #                             pos_enc_dim=pos_enc_dim, 
                #                             scale_factor=2,
                #                             upsampler='trilinear', 
                #                             g_noise=False)

                large_vdb = large_vdb.to(device)


                pred = model(large_vdb, large_vdb.grid)
                loss = torch.sqrt(torch.mean((pred.jdata - large_vdb.jdata) ** 2))

                # pred = model(xt)
                # loss = fmt.fm_loss(pred, large_vdb, noise)
                
                val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_data_loader)
            print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # save best model
        if save_model:
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(model, f'../save_models/{model_name}.pth')
                print(f'Saved best model at epoch {epoch+1} with validation loss: {min_val_loss:.4f}')
        
        Loss.append(avg_val_loss)
        if wandb_run:
            wandb.log({
            "train_loss": avg_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch})

        history.append(Loss)

    # save saliency feature
    if wandb_run:
        epochs = np.arange(1, len(saliency_feature) + 1)
        saliency_feature = np.array(saliency_feature)
        # percentage each row
        def get_percentages(arr):
            arr = np.asarray(arr)
            if arr.ndim == 1:
                return arr / np.sum(arr)
            else:
                return arr / np.sum(arr, axis=1, keepdims=True)
        saliency_percent = get_percentages(saliency_feature)
        saliency_percent = np.array(saliency_percent).T
        
        plt.figure(figsize=(10, 6))
        for i, label in enumerate(saliency_labels):
            plt.plot(epochs, saliency_percent[i], label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Feature Importance in %")
        plt.title("Saliency Feature Importance Over Epochs of Last Train Example")
        plt.legend()
        plt.tight_layout()

        # Log the plot to wandb
        wandb.log({"Saliency feature importance over epochs of last train example": wandb.Image(plt)})
        plt.close()

    return history


def main(epochs, wandb_run=True):
    # args
    random.seed(42)
    
    pos_enc_dim=10
    wandb_model_name = '26_rerun_24_SSU_PONQ_DATA_UPSAMPLER_FM_trilinear_Unet_v2'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sdf_dir = '/data/workspaces/spanwar/dataset/ssu_data/ssu_processed_data/sdf_data_unit_circle_norm'
    sdf_nmc_dir = '/data/workspaces/spanwar/dataset/nmc_data/groundtruth/gt_NMC'
    ponq_data_dir = '/data/workspaces/spanwar/dataset/ponq_dataset/gt_Quadrics'

    # load dataset
    (train_data_loader, 
     val_data_loader, 
     test_data_loader) = load_dataset(ponq_data_dir, n_samples=150)
    
    # model
    model = FVDBAutoencoder(in_channels=1, out_channels=1)
    model = model.to(device)
    mtools.print_model_summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # criterion = nn.MSELoss()

    gc.collect()
    torch.cuda.empty_cache()

    # wandb init
    if wandb_run:
        print("Initializing wandb run...")
        print(f"Model name: {wandb_model_name}")
        wandb.init(project="SSU", entity="sp_kumar", name=wandb_model_name, config={
            "Discpription": "24 Rerun: Implementing Unet model and FM, train on PONQ Dataset. Improved sampling method, selection of t. Debug the code.",
            "batch_size": 1,
            "learning_rate": 1e-3,
            "epochs": 10,
            "positional_encoding_dim": 10,
            "model": "Unet",
            "dataset": "PONQ Dataset",
            "dataset_size": len(train_data_loader),
            "dataset_split": {
                "train": len(train_data_loader),
                "val": len(val_data_loader),
                "test": len(test_data_loader)
            },
            "model_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "loss_function": "MSE loss",
            "optimizer": "Adam",
            "device": device
        })

    history = train(model=model, 
                    model_name=wandb_model_name, 
                    pos_enc_dim=pos_enc_dim, 
                    train_data_loader=train_data_loader, 
                    val_data_loader=val_data_loader, 
                    optimizer=optimizer, 
                    epochs=epochs, 
                    device=device, 
                    wandb_run=wandb_run, 
                    save_model=False)



if __name__ == "__main__":
    # get the epochs and wandb_run from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train Flow Matching Upsampler Model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model')
    parser.add_argument('--wandb_run', type=str, default='no', help='Whether to run with wandb logging')
    args = parser.parse_args()  
    
    # Convert wandb_run to boolean
    if args.wandb_run.lower() not in ['yes', 'no']:
        raise ValueError("wandb_run must be 'yes' or 'no'")
    wandb_run = True if args.wandb_run.lower() == 'yes' else False
    
    # print the arguments
    print(f"epochs: {args.epochs}")
    print(f"wandb_run: {args.wandb_run}")

    main(epochs=args.epochs, wandb_run=wandb_run)
    
