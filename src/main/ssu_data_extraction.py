import sys
sys.path.append('../utils')
import os
import io
import igl
import h5py
import trimesh
import numpy as np
import mesh_tools as mt

data_dir = '/data/workspaces/spanwar/dataset/abc_dataset'
processed_dir = '/data/workspaces/spanwar/dataset/ssu_data/ssu_processed_data'


def mesh_to_sdf(path, resolutions=[32, 64, 128, 256], save_dir=None):
    # extract file name
    sdf_file_name = os.path.basename(os.path.dirname(path))

    # create a dictionary to store the sdf
    if save_dir is None:
        raise ValueError("save_dir must be specified to save the SDF files.")
    os.makedirs(save_dir, exist_ok=True)

    # save path
    sdf_path = os.path.join(save_dir, f'{sdf_file_name}.hdf5')
    print(f'Saving SDF to path: {sdf_path}')

    # check if the file already exists
    new_resolutions = []
    
    if os.path.exists(sdf_path):
        with h5py.File(sdf_path, 'r') as f:
            existing_keys = set(f.keys())
    
    for res in resolutions:
        if os.path.exists(sdf_path) and f'{res}_sdf' in existing_keys:
            print(f'SDF for resolution {res} already exists in {sdf_path}. Skipping...')
        else:
            new_resolutions.append(res)
            print(f'Processing SDF for resolution {resolutions}...')
    
    if len(new_resolutions) == 0:
        print(f"All resolutions already processed. Exiting...")
        return
    
    # load mesh
    v, f = igl.read_triangle_mesh(path)
    # normalize object inisde unit cube [-0.5 to 0.5] and multiply by 2 to fit [-1 to 1]
    v = 2*mt.NDCnormalize(v)
    
    for res in new_resolutions:
        
        # create the grid
        points = mt.mesh_grid(res, True)
        
        # compute the signed distance field for the resolution
        sdf = igl.signed_distance(points, v, f)[0].reshape(res, res, res)

        # save the sdf as hdf5 file
        with h5py.File(sdf_path, 'a') as file:
            file.create_dataset(f'{res}_sdf', data=sdf)


def extract_sdf_from_dataset(dataset_dir, processed_dir):
    
    # get all mesh files in the dataset directory
    folder_paths = [os.path.join(dataset_dir, folder) for folder in os.listdir(dataset_dir)]
    
    ## extract first 100 meshes for testing
    # folder_paths = folder_paths[:100]
    mesh_files = []

    for index, folder_path in enumerate(folder_paths):
        mesh_files = mesh_files + [os.path.join(folder_path, filename) 
                                    for filename in os.listdir(folder_path) 
                                    if filename.endswith('.obj')]
    
    # save directory for processed SDF files
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f'Processed directory {processed_dir} does not exist.')
    save_dir = os.path.join(processed_dir, 'sdf_data_unit_circle_norm')
    print(f'Saving SDF files to dir: {save_dir}')
    
    # process each mesh file
    for _, mesh_file in enumerate(mesh_files):
        # check the file exists
        if not os.path.exists(mesh_file):
            raise FileNotFoundError(f'File {mesh_file} does not exist.')

        print(f'Processing {mesh_file}...')
        mesh_to_sdf(mesh_file, save_dir=save_dir)

if __name__ == '__main__':
    # extract SDF from the dataset
    extract_sdf_from_dataset(data_dir, processed_dir)
    print('SDF extraction completed.')