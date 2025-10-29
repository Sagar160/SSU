import os
import sys
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
from training import model_training_test as model_training
from logger import wandb_logging
from data_processing import data_processing
# from data_loader import a20082025_ABC_dataset_loader as ABC_dataset_loader
from data_loader import ABC_dataset_loader
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
    
    dataLoader = ABC_dataset_loader.ABCDataLoader(
                                        input_dir=input_dir,
                                        config=config,
                                        # n_samples=10
                                    )
    (train_dataloader, 
    val_dataloader, 
    test_dataloader) = dataLoader.get(names_set=names_set)
    logger.update_config('data_size', len(os.listdir(input_dir)))

    # prepare for data processing
    processing_names_set = dataLoader.get_names_set_for_data_processing()
    dataProcessor = data_processing.DataProcessing(processing_names_set, 
                                                   input_size=config['data']['input_size'],
                                                   threshold=config['data']['mask_threshold'],
                                                   random_direction_type=config['data']['random_direction_type'],
                                                   fvdb_saved_dir=config['data']['fvdb_saved_dir'],
                                                   sdf_gt_large=config['data']['input_dir'],
                                                   gt_objs=config['eval']['eval_dir'])
    # dataProcessor = None
    
    if not config['eval']['only_eval']:
        ## Model Simple CNN
        vector_dim = 0
        t_dim = 1
        # CNN_vanilla_without_transpose, FVDBUNetBase
        # model = simpleModels.CNN_vanilla_without_transpose(
        #     in_channels=config['training']['in_channels'] + vector_dim + t_dim,
        #     out_channels=config['training']['out_channels'])

        model = unetModels.FVDBUNetBase(
            in_channels=config['training']['in_channels'] + vector_dim + t_dim,
            out_channels=config['training']['out_channels'])
        trainable_params = st.print_model_summary(model)
        logger.update_config('model_parameters', trainable_params)
        
        ## optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config['training']['lr'], 
                                    # weight_decay=1e-5
                                    )
        
        ## training
        trainer = model_training.ModelTrainer(
                                dataProcessor=dataProcessor,
                                model_name=config['training']['model_name'],
                                model=model,
                                num_epochs=config['training']['epochs'],
                                train_loader=train_dataloader,
                                val_loader=val_dataloader,
                                test_loader=test_dataloader,
                                upsample_factor=config['data']['upsample_factor'],
                                input_size=config['data']['input_size'],
                                # pos_enc_dim=config['training']['positional_encoding'],
                                optimizer=optimizer,
                                loss_fn_name=config['training']['loss_function'],
                                # loss_weights=config['training']['loss_weights'],
                                is_save_model=config['training']['save_model'],
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
    # evaluator.evaluate()

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