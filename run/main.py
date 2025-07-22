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
from training import model_training
from logger import wandb_logging
from data_loader import ABC_dataset_loader
from utils import fvdb_utils as fu
from utils import ssu_tools as st 

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
                        # id = '1e7bv85c'
                    )
    logger.update_config('config_file_name', config_file)
    
    # set reproducibility
    st.set_reproducibility(is_reproducible=config['reproducibility']['is_reproducible'],
                           seed=config['reproducibility']['seed'])
    
    # load data
    input_dir = config['data']['input_dir']
    (train_dataloader, 
    val_dataloader, 
    test_dataloader) = ABC_dataset_loader.ABCDataLoader(
                                        input_dir=input_dir,
                                        config=config,
                                        # n_samples=20
                                    ).get()
    logger.update_config('data_size', len(os.listdir(input_dir)))
    
    if not config['eval']['only_eval']:
        # Model training
        ## Model Unet
        # model = unet.FVDBUNetBaseUpsampler(
        #     in_channels=config['training']['in_channels'] + config['training']['positional_encoding'],
        #     out_channels=config['training']['out_channels'])
        ## Model Simple CNN
        model = simpleModels.CNN_vanilla(
            in_channels=config['training']['in_channels'] + config['training']['positional_encoding'],
            out_channels=config['training']['out_channels'])
        trainable_params = st.print_model_summary(model)
        logger.update_config('model_parameters', trainable_params)
        
        ## optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=config['training']['lr'], 
                                    weight_decay=1e-5)
        
        ## training
        trainer = model_training.ModelTrainer(
                                model_name=config['training']['model_name'],
                                model=model,
                                num_epochs=config['training']['epochs'],
                                train_loader=train_dataloader,
                                val_loader=val_dataloader,
                                pos_enc_dim=config['training']['positional_encoding'],
                                optimizer=optimizer,
                                loss_fn_name=config['training']['loss_function'],
                                loss_weights=config['training']['loss_weights'],
                                is_save_model=config['training']['save_model'],
                                save_model_dir=config['training']['save_model_dir'],
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
                            pos_enc_dim=config['training']['positional_encoding'],
                            test_loader=test_dataloader,
                            upsampling_level=config['eval']['upsampling_level'],
                            abc_dir=config['eval']['eval_dir'],
                            save_model_dir=config['training']['save_model_dir'],
                            save_predictions_dir=config['eval']['save_predictions_dir'],
                            n_job=config['eval']['eval_job'],
                            logger=logger
                        )
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