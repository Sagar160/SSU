import os
import sys
import copy
import torch
import argparse
from pprint import pprint


# Import ssu packages
sys.path.append('../src')
sys.path.append('../config')
# config packages
import read_config
# src packages
from eval import ABC_eval
from models import unet
from models import model as simpleModels
from models import unet as unetModels
from training import model_training as model_training
from logger import wandb_logging
from data_loader import a20082025_ABC_dataset_loader as ABC_dataset_loader
from utils import fvdb_utils as fu
from utils import ssu_tools as st 

import torch
import gc

def clear_gpu_memory():
    """Clear all GPU memory allocated by PyTorch"""
    
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        # Clear all tensors from GPU
        torch.cuda.synchronize()
        
        # Get memory info
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
        
        # If memory is still allocated, try more aggressive clearing
        if allocated > 0:
            print("Attempting to clear more GPU memory...")
            print("Avoiding this step")
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            # gc.collect()
            
        print("✅ GPU memory cleared")
    else:
        print("❌ No CUDA GPU available")

# Call this function
clear_gpu_memory()

def main(config_file):

    # read config file
    config = read_config.read_yaml_config(f'{config_file}')
    
    print("Configuration loaded:")
    for key, value in config.items():
        pprint(f"{key}: {value}")

    # initialize logging
    logger = wandb_logging.WandbLogger(
                        logging=config['logging'],
                        project_name=config['wandb']['project_name'],
                        entity=config['wandb']['entity'],
                        name=config['wandb']['name'],
                        group=config['wandb']['group'],
                        tags=config['wandb']['tags'],
                        notes=config['wandb']['notes'],
                        config=config['wandb']['config'],
                        # resume="allow",
                        # id = 'z3zwr3y6'
                    )
    logger.update_config('config_file_name', config_file)
    
    # set reproducibility
    st.set_reproducibility(is_reproducible=config['reproducibility']['is_reproducible'],
                           seed=config['reproducibility']['seed'])
    
    # load data
    input_dir = config['data']['input_dir']
    names_set = os.listdir('/data/workspaces/spanwar/dataset/preprocessing_nmc_data/data_preprocessing/get_groundtruth_NMC/gt')
    

    ### if this work ###
    # 2. change the normalization
    # 3. Increae the steps 10 to 20
    # 4. use dataprocessing
    dataLoader = ABC_dataset_loader.ABCDataLoader(
                                        input_dir=input_dir,
                                        config=config,
                                        # n_samples=400
                                    )
    (train_dataloader, 
    val_dataloader, 
    test_dataloader) = dataLoader.get(names_set=names_set)
    logger.update_config('data_size', len(os.listdir(input_dir)))

    
    if not config['eval']['only_eval']:
        if False: #config['training']['use_pre_train_model']:
            print("Using a pre-trained model for training.")
            pretrained_model = torch.load(os.path.join(config['training']['save_model_dir'], f"{config['training']['pre_train_model_name']}.pth"))
            model = copy.deepcopy(pretrained_model)
            model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            print("Training a new model from scratch.")
            in_channels = 1
            out_channels = 1
            vector_dim = 3
            t_dim = 1
            t_pos = 0 #6
            s_pos = 0 #6
            
            model = unetModels.FVDBUNetBase(
                in_channels=in_channels + vector_dim + t_dim + t_pos + s_pos,
                out_channels=out_channels)

        trainable_params = st.print_model_summary(model)
        logger.update_config('model_parameters', trainable_params)
        
        ## optimizer
        # optimizer = torch.optim.SGD(model.parameters(),
        #                             lr=config['training']['lr'], # PyTorch ignores this once CyclicLR starts, but keep consistent
        #                             momentum=0.9,
        #                             weight_decay=1e-4,
        #                             nesterov=True,          # requires dampening=0 (default), momentum>0
        #                             )
        
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config['training']['lr'])
        # scheduler = torch.optim.lr_scheduler.CyclicLR(
        #                             optimizer,
        #                             base_lr=1e-5,
        #                             max_lr=config['training']['lr'],
        #                             step_size_up=len(train_dataloader) * 3,  # ~3 epochs warm ramp
        #                             mode='triangular2',
        #                             cycle_momentum=True,     # ties momentum to LR
        #                             base_momentum=0.85,
        #                             max_momentum=0.95,
        #                         )
        ## training
        trainer = model_training.ModelTrainer(
            dataProcessor=None,
            upsample_factor=4,
            input_size=33,

                                model_name=config['training']['model_name'],
                                model=model,
                                num_epochs=config['training']['epochs'],
                                train_loader=train_dataloader,
                                val_loader=val_dataloader,
                                test_loader=test_dataloader,
                                optimizer=optimizer,
                                # scheduler=scheduler,
                                loss_fn_name=config['training']['loss_function'],
                                is_save_model=config['training']['save_model'],
                                # is_save_predictions=config['training']['is_save_predictions'],
                                save_model_dir=config['training']['save_model_dir'],
                                save_predictions_dir=config['training']['save_predictions_dir'],
                                logger=logger
                            )
        
        print("Now initializing logger  :)")
        logger.init()
        trainer.train()
    else:
        print("Skipping training as only evaluation is requested.")
        logger.init()

    ## Evaluation
    evaluator = ABC_eval.Evaluator(
                            model_name=config['training']['model_name'],
                            # pos_enc_dim=config['training']['positional_encoding'],
                            test_loader=test_dataloader,
                            # upsampling_level=config['eval']['upsampling_level'],
                            abc_dir=config['eval']['eval_dir'],
                            save_model_dir=config['training']['save_model_dir'],
                            save_predictions_dir=config['training']['save_predictions_dir'],
                            n_job=config['eval']['eval_job'],
                            eval_discription=config['eval']['eval_discription'],
                            logger=logger
                        )
    if config['eval']['run_eval']:
        evaluator.evaluate()

    # finish logging
    logger.finish()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the SSU main script.')
    parser.add_argument('--config', type=str, default='None',
                        help='Path to the configuration file.')
    args = parser.parse_args()
    config_file = args.config

    print(f"Using config file: {config_file}")

    if config_file == 'None':
        raise ValueError("No config file provided. Please specify a config file using --config.")
    
    main(config_file)