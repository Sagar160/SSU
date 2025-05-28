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
git clone --recursive https://github.com/nissmar/SSU.git
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
