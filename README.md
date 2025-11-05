# SSU: Sparse SDF Upsampler

## Downloading data

```shell
mkdir data 
cd data
pip install gdown
gdown 1YyYOgn8uxGH6Nz_gGk8OR7IuLKUF89Ze
```

untar 
```shell
pip install py7zr
py7zr x groundtruth.7z 
```

Remove unecessary file
```shell
rm groundtruth.7z 
```

## Getting Started

1. Start by cloning the repository and *fVDB* submodule:

```shell
git clone --recursive https://github.com/Sagar160/SSU/tree/sp-core-implementation
```

2. Create the `ssu` conda environment (tested with CUDA 12.1):
````shell
conda env create -f dev_env.yml
conda activate ssu
````


3. Our code requires building *fVDB*, which can take a while (please refer to the original [README](openvdb/fvdb/README.md) for more details). Run: 
```shell
cd openvdb/fvdb
export MAX_JOBS=$(free -g | awk "/^Mem:/{jobs=int($4/2.5); if(jobs<1) jobs=1; print jobs}")
pip install .
cd ../..
```

## Implementation

Please consider this config file: config_68_29102025_1200.yaml 
```shell
conda activate ssu
cd ssu/run
python main.py --config config_68_29102025_1200.yaml
```
Now you can play with the parameters and run different experiements.

if you want to change model, can be done in main.py file.

## Config File (key variable)
logging: if you want to log to wandb


data:

  dataset_grids: &dataset_grids : `load specific grid data` \
  mask_threshold: &mask_threshold : `masking`\
  sdf_scaling_value: &sdf_scaling_value `scaling`\
  unique_random_direction: &unique_random_direction `is random direction each voxel`\
  
training:

  use_pre_train_model: `do want to use pretrained model`\
  pre_train_model_name: `pretrained model`\
  
eval:

  only_eval: &only_eval `only wanted to run evaluation, used when training completed but eval failed`\
  run_eval: &run_eval `do you wnat to run evaluation`\
  normalize: &eval_normalize `normalization method`\
  