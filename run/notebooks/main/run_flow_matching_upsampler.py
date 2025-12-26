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
from data_loader import load_dataset
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
            xt, noise = fmt.transform_input(small_vdb, 
                                        large_vdb, 
                                        t, 
                                        pos_enc_dim=pos_enc_dim,
                                        scale_factor=2,
                                        upsampler='trilinear', 
                                        g_noise=True)

            xt.jdata.requires_grad_()
            large_vdb = large_vdb.to(device)
            
            optimizer.zero_grad()
            pred = model(xt)
            loss = fmt.fm_loss(pred, large_vdb, noise)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

            # save saliency 
            saliency_feature_epoch.append(xt.jdata.grad.abs().mean(dim=0).cpu().numpy())

        # average saliency
        saliency_feature.append(np.stack(saliency_feature_epoch).mean(axis=0))

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
                t = torch.rand_like(large_vdb.jdata).to(device)

                #transform vdb
                xt, noise = fmt.transform_input(small_vdb, 
                                            large_vdb, 
                                            t,
                                            pos_enc_dim=pos_enc_dim, 
                                            scale_factor=2,
                                            upsampler='trilinear', 
                                            g_noise=False)

                large_vdb = large_vdb.to(device)

                pred = model(xt)
                loss = fmt.fm_loss(pred, large_vdb, noise)
                
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
     test_data_loader) = load_dataset(ponq_data_dir, n_samples=None)
    
    # model
    model = fvdbUnet.FVDBUNetBase(in_channels=pos_enc_dim+2, out_channels=1)
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
                    save_model=True)



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