import torch
import copy
from prediction_utils.dataset_loader import DataLoader
from prediction_utils.prediction import save_predictions
from prediction_utils import eval
import sys
import gc
import os
import argparse

sys.path.append('../config')
sys.path.append('../src')
# config packages
import read_config
from models import unet as unetModels


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

def main(config_file):
    clear_gpu_memory()

    with open('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/run/thingi30.txt', 'r') as f:
        file_names = f.read().splitlines()

    config = read_config.read_yaml_config(f'{config_file}')

    # data_loader = DataLoader(input_dir=config['data']['input_dir'], 
    #                         dataset_grids=config['data']['dataset_grids'],
    #                         upsample_factors=config['data']['upsampling_factors'],)
    # test_dataloader = data_loader.get(names_set=file_names)  # Use all files in thingi30.txt

    # model = unetModels.FVDBUNetBase(
    #             in_channels=4,
    #             out_channels=1)
    # # load the trained model
    # model.load_state_dict(torch.load(config['model']['save_model_path']))
    # model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # save_predictions(test_loader=test_dataloader,
    #                 upsample_factor_dict=config['data']['upsampling_factors'],
    #                 save_predictions_dir=config['prediction']['save_predictions_dir'],
    #                 prediction_folder_name=config['prediction']['prediction_folder_name'],
    #                 model=model)
    
    evaluator = eval.Evaluator(
                        model_name=config['prediction']['prediction_folder_name'],
                        abc_dir=config['eval']['eval_dir'],
                        save_predictions_dir=config['prediction']['save_predictions_dir'],
                        n_job=config['eval']['eval_job'],
                        eval_discription=config['eval']['eval_discription']
                    )
    evaluator.evaluate()
    
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