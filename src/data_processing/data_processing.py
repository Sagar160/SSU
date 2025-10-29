import os
import sys
import igl
import time
import h5py
import torch
import joblib
import random
import trimesh
import numpy as np
import fvdb
from tqdm import tqdm
import fvdb.nn as fvnn
import open3d as o3d
import subprocess
import math

sys.path.append('../src')
from utils import mesh_tools as mt


def compute_sdf_open3d(vertices, faces, query_points):
    """
    Open3D is 3-5x faster than IGL for SDF computation
    """
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Create scene for ray casting
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)
    
    if type(query_points) is list:
        sdf_list = []
        for q_points in query_points:
            q_points_t = o3d.core.Tensor(q_points, dtype=o3d.core.Dtype.Float32)
            sdf = scene.compute_signed_distance(q_points_t)
            sdf_list.append(torch.tensor(sdf.numpy(), dtype=torch.float32, device='cpu'))

    # Query points
    query_points_t = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
    
    # Compute signed distances
    sdf = scene.compute_signed_distance(query_points_t)

    return torch.tensor(sdf.numpy(), dtype=torch.float32, device='cpu')


def gt_normalize_vertices(vertices):
    """
    Normalize vertices and optionally return normalization parameters
    """
    #normalize diagonal=1
    x_max = np.max(vertices[:,0])
    y_max = np.max(vertices[:,1])
    z_max = np.max(vertices[:,2])
    x_min = np.min(vertices[:,0])
    y_min = np.min(vertices[:,1])
    z_min = np.min(vertices[:,2])
    x_mid = (x_max+x_min)/2
    y_mid = (y_max+y_min)/2
    z_mid = (z_max+z_min)/2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale*x_scale + y_scale*y_scale + z_scale*z_scale)
    
    vertices[:,0] = (vertices[:,0]-x_mid)/scale
    vertices[:,1] = (vertices[:,1]-y_mid)/scale
    vertices[:,2] = (vertices[:,2]-z_mid)/scale

    return vertices

def random_rotation_matrix(gt_mesh):
    # Generate random rotation matrix
    angles = np.random.uniform(0, 2*np.pi, 3)  # Random angles for x, y, z
    
    # Rotation matrices for each axis
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    gt_mesh.vertices = gt_mesh.vertices @ R.T
    return gt_mesh

def scaled_sdf(threshold, sdf_arr: np.array):
    '''scales the SDF array by the threshold value'''
    # return (threshold-1)*sdf_arr[:, None]
    return (threshold-1)*sdf_arr[:, None]

def fetch_numpy_values(grid: fvdb.GridBatch, arr: np.array, size:int):
        '''fetches values from a numpy array based on the ijk indices in the grid'''
        ijk = grid.ijk.jdata.cpu().detach().numpy()
        
        if max(ijk[:, 0]) >= arr.shape[0] or max(ijk[:, 1]) >= arr.shape[1] or max(ijk[:, 2]) >= arr.shape[2]:
            # If indices are out of bounds, we can add the maximum value to the indices
            ijk = np.clip(ijk, 0, np.array(arr.shape) - 1)
            # print(f"Indices out of bounds. Clipping to max shape: {arr.shape}")
        
        values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        return torch.tensor(values, dtype=torch.float32, device=grid.device)

def fetch_numpy_values_shifted(ijk, arr: np.array):
    '''fetches values from a numpy array based on the ijk indices in the grid'''
    ijk = ijk.cpu().detach().numpy()
    values = arr[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    device = 'cpu'
    return torch.tensor(values, dtype=torch.float32, device=device)

def path_information(path_name):
    paths = {
        'fvdb_saved_dir': '/data/workspaces/spanwar/preprocessed_data/ssu/51_complete_random',
        'sdf_gt_large': '/data/workspaces/spanwar/dataset/preprocessing_nmc_data/data_preprocessing/get_groundtruth_NMC/gt_large',
        'gt_objs': '/data/workspaces/spanwar/dataset/preprocessing_nmc_data/abc_dataset_objs'
    }
    return paths.get(path_name, None)

class DataProcessing:
    def __init__(self,
                 names_set, 
                 input_size, 
                 threshold, 
                 random_direction_type,
                 fvdb_saved_dir,
                 sdf_gt_large,
                 gt_objs):
        self.names_set = names_set
        self.input_size = input_size
        self.threshold = threshold
        self.random_direction_type = random_direction_type
        self.fvdb_saved_dir = fvdb_saved_dir
        self.sdf_gt_large = sdf_gt_large
        self.gt_objs = gt_objs
        # self.out_dict = self.read_dataset()

    def read_dataset(self):
        out = joblib.Parallel(n_jobs=-1)(joblib.delayed(self.read_sdf_file)
                                   (name) for name in tqdm(self.names_set))
        out_dict = {item[0]: {32: item[1], 64: item[2], 128: item[3]} for item in out}
        return out_dict

    def read_sdf_file(self, obj_name):
        obj_name = obj_name.split('.')[0]
        h5 = h5py.File('/data/workspaces/spanwar/dataset/preprocessing_nmc_data/data_preprocessing/get_groundtruth_NMC/gt_large/{}.hdf5'.format(obj_name), 'r')
        sdf_32 = h5['{}_sdf'.format(32)][:]
        sdf_64 = h5['{}_sdf'.format(64)][:]
        sdf_128 = h5['{}_sdf'.format(128)][:]
        h5.close()
        return obj_name,sdf_32, sdf_64, sdf_128

    @staticmethod
    def read_sdf_file_with_offsets(name):
        """
        Read SDF file containing [phi(float), phi_org(float), offset_i(int), offset_j(int), offset_k(int)] per grid point
        Returns: (sdf_data, sdf_org_data, offset_data)
        """
        with open(name, 'rb') as fp:
            # Read header
            line = fp.readline().strip()
            if not line.startswith(b'#sdf'):
                raise IOError('Not a sdf file')
            
            # Read dimensions
            dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
            
            # Skip "data" line
            fp.readline()
            
            # Read data point by point (2 floats + 3 ints per grid point)
            total_elements = dims[0] * dims[1] * dims[2]
            sdf_data = np.zeros(dims, dtype=np.float32)
            sdf_org_data = np.zeros(dims, dtype=np.float32)  # Add original SDF
            offset_data = np.zeros(dims + [3], dtype=np.int32)
            
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        # Read phi value (float) - modified SDF
                        phi_bytes = fp.read(4)
                        phi_value = np.frombuffer(phi_bytes, dtype=np.float32)[0]
                        sdf_data[i, j, k] = phi_value
                        
                        # Read phi_org value (float) - original SDF
                        phi_org_bytes = fp.read(4)
                        phi_org_value = np.frombuffer(phi_org_bytes, dtype=np.float32)[0]
                        sdf_org_data[i, j, k] = phi_org_value
                        
                        # Read 3 offset values (ints)
                        offset_bytes = fp.read(12)  # 3 ints * 4 bytes
                        offsets = np.frombuffer(offset_bytes, dtype=np.int32)
                        offset_data[i, j, k] = offsets
            
            return sdf_data, sdf_org_data, offset_data

    @staticmethod
    def read_sdf_file_with_offsets_x(name):
        with open(name, 'rb') as fp:
            # Read header
            line = fp.readline().strip()
            if not line.startswith(b'#sdf'):
                raise IOError('Not a sdf file')
            
            # Read dimensions
            dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
            
            # Skip "data" line
            fp.readline()
            
            # Read data point by point (1 float + 3 ints per grid point)
            total_elements = dims[0] * dims[1] * dims[2]
            sdf_data = np.zeros(dims, dtype=np.float32)
            offset_data = np.zeros(dims + [3], dtype=np.int32)
            
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        # Read phi value (float)
                        phi_bytes = fp.read(4)
                        phi_value = np.frombuffer(phi_bytes, dtype=np.float32)[0]
                        sdf_data[i, j, k] = phi_value
                        
                        # Read 3 offset values (ints)
                        offset_bytes = fp.read(12)  # 3 ints * 4 bytes
                        offsets = np.frombuffer(offset_bytes, dtype=np.int32)
                        offset_data[i, j, k] = offsets
            
            return sdf_data, offset_data

    def get_processed_sdf_data(self, obj_name,
                               r,
                               threshold=65,
                               grid_size=33,
                               random_direction_type='nonUniform'):
        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        _dir_gt = self.gt_objs  
        obj_file = os.path.join(_dir_gt, file_name, 'model.obj')
        # file_dir_gt = os.path.join(_dir_gt, obj_name.split('.')[0])
        # obj_files = [f for f in os.listdir(file_dir_gt) if f.endswith('.obj')]
        
        # all paths
        # file_gt = os.path.join(file_dir_gt, obj_files[0])
        file_gt = obj_file
        sdf_file_path = os.path.join(output_path, f'{file_name}.sdf')
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')

        # factor = random.randint(1, 5)*2
        factor = 8

        # run c++ binary file
        # run_cmd = f"/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/src/data_processing/run {file_gt} {sdf_file_path} {factor}"
        # os.system(run_cmd)
        if np.random.rand() < r:
            run_cmd = [
                    "/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/src/data_processing/run",
                    file_gt,
                    sdf_file_path,
                    str(factor)
                ]
            # print("Running command:", ' '.join(run_cmd))
            subprocess.run(run_cmd, 
                            check=True,  # Raises exception on non-zero exit
                            capture_output=True,  # Capture stdout/stderr
                            text=True,  # Return strings instead of bytes
                            # timeout=300
                            )
        # read sdf file
        # gt_large = self.sdf_gt_large
        # with h5py.File(os.path.join(gt_large, f'{file_name}.hdf5'), 'r') as h5_file:
        #     sdf_32 = h5_file['32_sdf'][:]
        sdf_data, sdf_32, offset_data = self.read_sdf_file_with_offsets(sdf_file_path)
        offset_data = offset_data/(factor//2)
        # print(factor, file_name, (sdf_data<0).sum(), sdf_32.shape, offset_data.shape)
        # input and output sdfs preparation
        # print((np.abs(sdf_data) < 3/(2*32)).sum())
        mask = mt.make_mask_close(sdf_data, threshold)
        # print(mask.sum())

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(grid_size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(grid_size, grid_size, grid_size, 3)

        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(grid_size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device='cpu'))
        
        # scale and mask
        ijk = grid.ijk.jdata
        sdf_data = sdf_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        sdf_32 = sdf_32[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        sdf_data = scaled_sdf(threshold, sdf_data)
        sdf_32 = scaled_sdf(threshold, sdf_32)
        sdf_data = torch.tensor(sdf_data, dtype=torch.float32, device='cpu')
        sdf_32 = torch.tensor(sdf_32, dtype=torch.float32, device='cpu')
        offset_data = offset_data[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        offset_data = torch.tensor(offset_data, dtype=torch.float32, device='cpu')

        # print(mask.shape, ijk.shape, sdf_data.shape, sdf_32.shape, offset_data.shape)
        # create VDBTensor
        shifted_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(sdf_data))

        small_features = torch.cat([sdf_32, offset_data], dim=-1) 
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_features))
        fvdb.save(input_vdb_path, small_vdb.grid, small_vdb.data, compressed=True)
        fvdb.save(output_vdb_path, shifted_vdb.grid, shifted_vdb.data, compressed=True)

    @staticmethod
    def read_simple_sdf_file(name):

        with open(name, 'rb') as fp:
            # Read header
            line = fp.readline().strip()
            if not line.startswith(b'#sdf'):
                raise IOError('Not a sdf file')
            
            # Read dimensions
            dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
            
            # Skip "data" line
            fp.readline()
            
            # Read all float data at once (more efficient)
            total_elements = dims[0] * dims[1] * dims[2]
            float_data = fp.read(total_elements * 4)  # 4 bytes per float
            
            # Convert to numpy array and reshape
            sdf_data = np.frombuffer(float_data, dtype=np.float32)
            sdf_data = sdf_data.reshape(dims)
            
            return sdf_data

    def create_multi_resolution_sdf(self, sdf_file_path, grid_size_list=[16, 32, 64]):
        # Load the original SDF
        sdf_data = self.read_simple_sdf_file(sdf_file_path)
        original_size = sdf_data.shape[0]  # Assuming cubic grid
        
        # Dictionary to store different resolution SDFs
        LOD_sdf = {}
        
        for grid_size in grid_size_list:
            
            # Calculate downscale factor
            downscale = original_size // grid_size
            
            if downscale < 1:
                print('original size:', original_size)
                print(f"Warning: Grid size {grid_size} is larger than original {original_size}, skipping")
                continue
              
            # Create downsampled SDF
            # Using the same method as in your reference code
            i, j, k = 0, 0, 0
            tmp_sdf = sdf_data[i::downscale, j::downscale, k::downscale]
            
            # Create output array with target size
            grid_size_1 = grid_size + 1
            output_sdf = np.full((grid_size_1, grid_size_1, grid_size_1), 1.0, dtype=np.float32)
            
            # Copy the downsampled data
            output_sdf[:tmp_sdf.shape[0], :tmp_sdf.shape[1], :tmp_sdf.shape[2]] = tmp_sdf
            
            LOD_sdf[grid_size] = output_sdf
        
        return LOD_sdf

    def get_processed_sdf_data_v2(self, obj_name,
                               r,
                               threshold=65,
                               grid_size=17,
                               random_direction_type='nonUniform'):
        
        def custom_subdivide_grid(grid: fvdb.GridBatch, scale, m3g, upshape):
            '''custom subdivision of a grid to create a finer grid:
                [0,    1,    2] -->
                [0, 1, 2, 3, 4]'''
            ijk = grid.ijk.jdata
            # m3g = torch.tensor(mt.mesh_grid(3),device=grid.device)-1
            new_ijk = (scale*ijk[:, None, :]+ m3g[None, :, :]).view(-1, 3)
            new_ijk = np.clip(new_ijk, 0, upshape-1)
            return fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(new_ijk), origins=grid.origins, voxel_sizes=grid.voxel_sizes/2)
    
        def select_random_grid_pair():
            """
            Select two different values from [16, 32, 64] where one is lower and one is higher
            Returns: (lower_grid_size, higher_grid_size, upscale_factor)
            """
            available_sizes = [32, 64, 128]
            # available_sizes = [16, 32, 64]
            
            # Randomly select two consecutive indices
            idx = np.random.randint(0, len(available_sizes) - 1)
            size1 = available_sizes[idx]
            size2 = available_sizes[idx + 1]

            # Ensure lower and higher are correctly assigned
            lower_size = min(size1, size2)
            higher_size = max(size1, size2)
            
            # Calculate upscale factor
            upscale_factor = higher_size // lower_size
            return int(lower_size), int(higher_size), int(upscale_factor)
        
        def get_neighbors_low_res(high_res_ijk_coords, low_res_sdf_data, low_res_size, upscale_factor, neighborhood_size=3):
            """
            Extracts a local 3D neighborhood from the LOW-RESOLUTION SDF for each voxel.

            Args:
                high_res_ijk_coords (torch.Tensor): Coordinates of the high-res voxels.
                low_res_sdf_data (np.ndarray): The full LOW-RESOLUTION SDF array.
                low_res_size (int): The dimension of the low-res SDF array.
                upscale_factor (int): The factor by which the low-res grid was upsampled.
                neighborhood_size (int): Size of the cube neighborhood (e.g., 3 for a 3x3x3 cube).

            Returns:
                torch.Tensor: A tensor of flattened low-res neighborhood features for each voxel.
            """
            half_neighborhood = neighborhood_size // 2

            num_voxels = high_res_ijk_coords.shape[0]
            num_features = neighborhood_size ** 3
            neighborhood_features = torch.zeros(num_voxels, num_features, dtype=torch.float32)

            # Map high-res coordinates to low-res coordinates
            low_res_ijk_np = (high_res_ijk_coords.cpu().numpy() / upscale_factor).astype(int)

            for i in range(num_voxels):
                coords = low_res_ijk_np[i]

                # Define the neighborhood bounds on the low-res grid
                min_x, max_x = coords[0] - half_neighborhood, coords[0] + half_neighborhood + 1
                min_y, max_y = coords[1] - half_neighborhood, coords[1] + half_neighborhood + 1
                min_z, max_z = coords[2] - half_neighborhood, coords[2] + half_neighborhood + 1
                
                # Create an empty neighborhood with a padding value (e.g., 100)
                patch = np.full((neighborhood_size, neighborhood_size, neighborhood_size), 100.0)
                
                # Define the valid slice in the source data
                src_x_min = max(0, min_x)
                src_x_max = min(low_res_size, max_x)
                src_y_min = max(0, min_y)
                src_y_max = min(low_res_size, max_y)
                src_z_min = max(0, min_z)
                src_z_max = min(low_res_size, max_z)
                
                # Define the target slice in the patch
                tgt_x_min = src_x_min - min_x
                tgt_x_max = src_x_max - min_x
                tgt_y_min = src_y_min - min_y
                tgt_y_max = src_y_max - min_y
                tgt_z_min = src_z_min - min_z
                tgt_z_max = src_z_max - min_z
                
                # Copy the data from the source to the target, ensuring a consistent patch size
                patch[tgt_x_min:tgt_x_max, tgt_y_min:tgt_y_max, tgt_z_min:tgt_z_max] = \
                    low_res_sdf_data[src_x_min:src_x_max, src_y_min:src_y_max, src_z_min:src_z_max]

                # Flatten and store
                neighborhood_features[i] = torch.from_numpy(patch.flatten())

            return neighborhood_features
        
        def get_single_low_res_value(high_res_ijk_coords, low_res_sdf_data, low_res_size, upscale_factor):
            """
            Extracts the single corresponding SDF value from the LOW-RESOLUTION SDF for each high-res voxel.
            
            Args:
                high_res_ijk_coords (torch.Tensor): Coordinates of the high-res voxels.
                low_res_sdf_data (np.ndarray): The full LOW-RESOLUTION SDF array.
                low_res_size (int): The dimension of the low-res SDF array.
                upscale_factor (int): The factor by which the low-res grid was upsampled.

            Returns:
                torch.Tensor: A tensor of single low-res SDF values for each voxel.
            """
            num_voxels = high_res_ijk_coords.shape[0]
            low_res_values = torch.zeros(num_voxels, 1, dtype=torch.float32)

            # Map high-res coordinates to low-res coordinates
            low_res_ijk_np = (high_res_ijk_coords.cpu().numpy() / upscale_factor).astype(int)

            # Use a vectorized operation for efficiency
            low_res_values[:, 0] = torch.from_numpy(low_res_sdf_data[low_res_ijk_np[:, 0], 
                                                                    low_res_ijk_np[:, 1], 
                                                                    low_res_ijk_np[:, 2]])

            return low_res_values
        
        def get_sparse_low_res_values(high_res_ijk_coords, low_res_sdf_data, low_res_size, upscale_factor):
            """
            Creates a sparse tensor of low-res SDF values corresponding to high-res voxels.
            Only voxels at the original low-res grid locations will have data.

            Args:
                high_res_ijk_coords (torch.Tensor): Coordinates of the high-res voxels.
                low_res_sdf_data (np.ndarray): The full LOW-RESOLUTION SDF array.
                upscale_factor (int): The factor by which the low-res grid was upsampled.

            Returns:
                torch.Tensor: A tensor where only voxels at original low-res locations have their
                            corresponding SDF value. All others are padded.
            """
            num_voxels = high_res_ijk_coords.shape[0]
            
            # Initialize a tensor with a padding value (e.g., a large float)
            padded_values = torch.full((num_voxels, 1), 100.0, dtype=torch.float32)

            # Check which high-res voxels align with the low-res grid
            is_low_res_voxel = torch.all(high_res_ijk_coords % upscale_factor == 0, dim=1)

            # Get the coordinates for the low-res voxels
            low_res_coords = high_res_ijk_coords[is_low_res_voxel]
            
            # Map high-res coordinates to low-res coordinates
            low_res_coords_np = (low_res_coords.cpu().numpy() // upscale_factor).astype(int)

            # Fetch the corresponding SDF values from the low-res grid
            low_res_values = low_res_sdf_data[low_res_coords_np[:, 0], low_res_coords_np[:, 1], low_res_coords_np[:, 2]]

            # Place the fetched low-res values into the padded tensor
            padded_values[is_low_res_voxel] = torch.from_numpy(low_res_values).unsqueeze(-1).float()
            
            return padded_values
        
        def get_positional_encoding(ijk_coords, pe_dim=16):
            """
            Applies sinusoidal positional encoding to 3D integer coordinates.

            Args:
                ijk_coords (torch.Tensor): The integer coordinates (N, 3).
                pe_dim (int): The dimensionality of the positional encoding vector.
            
            Returns:
                torch.Tensor: The positional encoding vector (N, pe_dim).
            """
            pe = torch.zeros(ijk_coords.shape[0], pe_dim, device=ijk_coords.device)
            
            for i in range(pe_dim // 2):
                div_term = torch.exp(torch.arange(0, pe_dim, 2).float() * (-math.log(10000.0) / pe_dim))
                
                # Calculate sine and cosine components for each dimension
                sin_x = torch.sin(ijk_coords[:, 0].float().unsqueeze(1) * div_term[:pe_dim // 6])
                cos_x = torch.cos(ijk_coords[:, 0].float().unsqueeze(1) * div_term[:pe_dim // 6])
                sin_y = torch.sin(ijk_coords[:, 1].float().unsqueeze(1) * div_term[:pe_dim // 6])
                cos_y = torch.cos(ijk_coords[:, 1].float().unsqueeze(1) * div_term[:pe_dim // 6])
                sin_z = torch.sin(ijk_coords[:, 2].float().unsqueeze(1) * div_term[:pe_dim // 6])
                cos_z = torch.cos(ijk_coords[:, 2].float().unsqueeze(1) * div_term[:pe_dim // 6])
                
                pe[:, 0::6] = sin_x
                pe[:, 1::6] = cos_x
                pe[:, 2::6] = sin_y
                pe[:, 3::6] = cos_y
                pe[:, 4::6] = sin_z
                pe[:, 5::6] = cos_z

            return pe


        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        _dir_gt = self.gt_objs  
        obj_file = os.path.join(_dir_gt, file_name, 'model.obj')
        
        # all paths
        file_gt = obj_file
        sdf_file_path = os.path.join(output_path, f'{file_name}.sdf')
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')


        # run c++ binary file
        # if np.random.rand() < r:
        #     run_cmd = [
        #             "/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/src/data_processing/run_v2",
        #             file_gt,
        #             sdf_file_path
        #         ]
        #     # print("Running command:", ' '.join(run_cmd))
        #     subprocess.run(run_cmd, 
        #                     check=True,  # Raises exception on non-zero exit
        #                     capture_output=True,  # Capture stdout/stderr
        #                     text=True,  # Return strings instead of bytes
        #                     # timeout=300
        #                     )
        # read sdf file
        # LOD_sdf = self.create_multi_resolution_sdf(sdf_file_path)
        lower_size, higher_size, upscale_factor = select_random_grid_pair()
        # sdf_input = LOD_sdf[lower_size]
        # sdf_output = LOD_sdf[higher_size]

        h5 = h5py.File('/data/workspaces/spanwar/dataset/preprocessing_nmc_data/data_preprocessing/get_groundtruth_NMC/gt_large/{}.hdf5'.format(file_name), 'r')
        sdf_input = h5['{}_sdf'.format(lower_size)][:]
        sdf_output = h5['{}_sdf'.format(higher_size)][:]
        h5.close()

        threshold =  (lower_size)*2+1
        mask = mt.make_mask_close(sdf_input, threshold)
        # print(mask.shape, upscale_factor, lower_size, higher_size)

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(lower_size+1)
        ijk_mesh_grid = ijk_mesh_grid.reshape(lower_size+1, lower_size+1, lower_size+1, 3)

        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(lower_size)), 
                                        origins=torch.tensor([0, 0, 0]))

        # scale and mask
        m3g  = mt.mesh_grid(3)-1
        up_grid = custom_subdivide_grid(grid, upscale_factor, m3g, (lower_size*2)+1)
        up_ijk = up_grid.ijk.jdata
        out_mask = abs(sdf_output[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]]) < (1)
        up_ijk = up_ijk[out_mask]
        up_filtered_grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(up_ijk), 
                                        voxel_sizes=up_grid.voxel_sizes,
                                        origins=up_grid.origins)
        up_ijk_filtered = up_filtered_grid.ijk.jdata
        mask = torch.all(up_ijk_filtered % 2 == 0, dim=1)

        large_sdf_arr = sdf_output[up_ijk_filtered[:, 0], up_ijk_filtered[:, 1], up_ijk_filtered[:, 2]]
        large_sdf_arr = scaled_sdf(threshold, large_sdf_arr)
        large_sdf_arr = torch.tensor(large_sdf_arr, dtype=torch.float32, device='cpu')
        
        # ijk = grid.ijk.jdata
        # input_sdf_arr = sdf_input[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        # input_sdf_arr = scaled_sdf(threshold, input_sdf_arr)
        # input_sdf_arr = torch.tensor(input_sdf_arr, dtype=torch.float32, device='cpu')
        # small_sdf_arr = grid.sample_trilinear(up_filtered_grid.ijk.jdata.float(), grid.jagged_like(input_sdf_arr))
        # small_sdf_arr = small_sdf_arr.jdata + (torch.randn_like(small_sdf_arr.jdata) * 0.01)
        small_sdf_arr = large_sdf_arr.clone()
        small_sdf_arr[~mask] = torch.randn((~mask).sum(), 1, dtype=torch.float32, device='cpu') 
        # noise = torch.randn_like(small_sdf_arr) * 0.1
        # small_sdf_arr[~mask] = torch.full(((~mask).sum().item(), 1), -10, dtype=torch.float32, device='cpu')
        small_sdf_arr[mask] = small_sdf_arr[mask] + 0.1 * torch.randn((mask).sum(), 1, dtype=torch.float32, device='cpu')
        vector = up_ijk_filtered - (up_ijk_filtered//upscale_factor)*upscale_factor
        # small_sdf_arr = torch.randn_like(large_sdf_arr) 
        small_sdf_arr = torch.clamp(small_sdf_arr, -10, 10)
        # neighborhood_features = get_sparse_low_res_values(up_ijk_filtered, sdf_input, lower_size, upscale_factor)
        # pe = get_positional_encoding(up_ijk_filtered, pe_dim=10)

        # small_sdf_arr = scaled_sdf(threshold, small_sdf_arr)
        # small_sdf_arr = torch.tensor(small_sdf_arr, dtype=torch.float32, device='cpu')
        # , mask.float().unsqueeze(-1)
        # small_sdf_arr = torch.cat([small_sdf_arr, vector], dim=-1) 
        vdb_input = fvnn.VDBTensor(up_filtered_grid, 
                                    up_filtered_grid.jagged_like(small_sdf_arr))
        vdb_output = fvnn.VDBTensor(up_filtered_grid, 
                                    up_filtered_grid.jagged_like(large_sdf_arr))

        # merge ijk
        # up_new_ijk =  torch.concat([new_ijk, ijk], dim=0)
        # up_grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(up_new_ijk), 
        #                                 voxel_sizes=(1/(((grid_size-1)*2)-1)),
        #                                 origins=torch.tensor([0, 0, 0], device='cpu'))
        # up_sdf_32 = fetch_numpy_values(up_grid, sdf_32, (grid_size-1)*2+1)

        fvdb.save(input_vdb_path, vdb_input.grid, vdb_input.data, compressed=True)
        fvdb.save(output_vdb_path, vdb_output.grid, vdb_output.data, compressed=True)


    def get_processed_sdf_data_sdf_ex(self, obj_name,
                             threshold=65,
                             grid_size=33,
                             random_direction_type='nonUniform'):
        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        # _dir_gt = self.gt_objs
        # # find the .obj file in the directory by walking through the directory
        # file_dir_gt = os.path.join(_dir_gt, obj_name.split('.')[0])
        # obj_files = [f for f in os.listdir(file_dir_gt) if f.endswith('.obj')]
        # file_gt = os.path.join(file_dir_gt, obj_files[0])
        
        # get sdf from the point
        # gt_mesh = trimesh.load(file_gt, process=False)
        # gt_mesh.vertices = gt_normalize_vertices(gt_mesh.vertices)

        gt_large = self.sdf_gt_large
        # read hdf5 file
        filename = obj_name.split('.')[0]
        h5_file = h5py.File(os.path.join(gt_large, f'{filename}.hdf5'), 'r')
        sdf_32 = h5_file['32_sdf'][:]
        mask = mt.make_mask_close(sdf_32, threshold)

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(grid_size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(grid_size, grid_size, grid_size, 3)

        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(grid_size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device='cpu'))

        sdf_values_a = fetch_numpy_values(grid, sdf_32, grid_size)
        sdf_values_a = scaled_sdf(threshold, sdf_values_a)

        # upsample_factor = random.randint(1, 5)
        upsample_factor = 2
        upsample_factor = upsample_factor*4
        ########
        m3g=torch.tensor(mt.mesh_grid(upsample_factor+1), device='cpu')-(upsample_factor//2)
        
        # Randomly select one coordinate from m3g 
        unique_random_direction = True
        if unique_random_direction:
            num_elements = grid.ijk.jdata.shape[0]
            random_indices = torch.randint(0, m3g.shape[0], (num_elements,), device='cpu')
        else:
            random_indices = np.random.randint(0, m3g.shape[0])

        # new ijk coordinates
        selected_m3g = m3g[random_indices]  # Shape: (num_elements, 3)
        new_ijk = (upsample_factor * grid.ijk.jdata) + selected_m3g
        new_ijk_cpu = new_ijk.cpu().detach().numpy()
        new_ijk = np.clip(new_ijk_cpu, 0, (grid_size-1)*upsample_factor)
        new_ijk = torch.tensor(new_ijk, dtype=torch.int, device='cpu')
    
        direction_vector = new_ijk - (grid.ijk.jdata) * upsample_factor
        normalized_difference = direction_vector/(upsample_factor//2)

        # shifted_sdf_values = torch.tensor(shifted_sdf_values, dtype=torch.float32, device='cpu')
        shifted_sdf_values_a = fetch_numpy_values_shifted(new_ijk, h5_file['256_sdf'][:])
        shifted_sdf_values_a = scaled_sdf(threshold, shifted_sdf_values_a)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-shifted_sdf_values_a.numpy()))*65)
        # shifted_sdf_values = scaled_sdf(threshold, shifted_sdf_values)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-sdf_values.numpy()))*65)

        # create VDBTensor
        shifted_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(shifted_sdf_values_a))

        small_features = torch.cat([sdf_values_a, normalized_difference], dim=-1) 
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_features))
        
        # return small_vdb, shifted_vdb, shifted_vdb_a, sdf_values_a
        # save the input and output VDB tensors
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')

        fvdb.save(input_vdb_path, small_vdb.grid, small_vdb.data, compressed=True)
        fvdb.save(output_vdb_path, shifted_vdb.grid, shifted_vdb.data, compressed=True)


    def get_processed_sdf_data____red(self, obj_name,
                             threshold=65,
                             grid_size=33,
                             random_direction_type='nonUniform'):
        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        _dir_gt = self.gt_objs
        # find the .obj file in the directory by walking through the directory
        file_dir_gt = os.path.join(_dir_gt, obj_name.split('.')[0])
        obj_files = [f for f in os.listdir(file_dir_gt) if f.endswith('.obj')]
        file_gt = os.path.join(file_dir_gt, obj_files[0])
        
        # get sdf from the point
        gt_mesh = trimesh.load(file_gt, process=False)
        gt_mesh.vertices = gt_normalize_vertices(gt_mesh.vertices)

        gt_large = self.sdf_gt_large
        # read hdf5 file
        filename = obj_name.split('.')[0]
        # h5_file = h5py.File(os.path.join(gt_large, f'{filename}.hdf5'), 'r')
        # sdf_32 = h5_file['32_sdf'][:]
        grid_o = mt.mesh_grid(grid_size, normalize=False)
        grid_o_n = grid_o/(grid_size-1)-0.5
        sdf_32 = compute_sdf_open3d(gt_mesh.vertices,
                                      gt_mesh.faces,
                                      grid_o_n)
        sdf_32 = sdf_32.reshape(33, 33, 33)
        mask = mt.make_mask_close(sdf_32, threshold)

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(grid_size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(grid_size, grid_size, grid_size, 3)

        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(grid_size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device='cpu'))

        ijk = grid.ijk.jdata
        ijk_norm = ijk/(grid_size-1)-0.5
        # sdf_values = compute_sdf_open3d(gt_mesh.vertices,
        #                                   gt_mesh.faces,
        #                                   ijk_norm.numpy())
        # sdf_values_a = fetch_numpy_values(grid, sdf_32, grid_size)
        # sdf_values_a = scaled_sdf(threshold, sdf_values_a)

        # upsample_factor = random.randint(1, 5)
        upsample_factor = 2
        upsample_factor = upsample_factor*2
        ########
        m3g=torch.tensor(mt.mesh_grid(upsample_factor+1), device='cpu')-(upsample_factor//2)
        
        # Randomly select one coordinate from m3g 
        unique_random_direction = True
        if unique_random_direction:
            num_elements = grid.ijk.jdata.shape[0]
            random_indices = torch.randint(0, m3g.shape[0], (num_elements,), device='cpu')
        else:
            random_indices = np.random.randint(0, m3g.shape[0])

        # new ijk coordinates
        selected_m3g = m3g[random_indices]  # Shape: (num_elements, 3)
        new_ijk = (upsample_factor * grid.ijk.jdata) + selected_m3g
        new_ijk_cpu = new_ijk.cpu().detach().numpy()
        new_ijk = np.clip(new_ijk_cpu, 0, (grid_size-1)*upsample_factor)
        new_ijk = torch.tensor(new_ijk, dtype=torch.int, device='cpu')
    
        direction_vector = new_ijk - (grid.ijk.jdata) * upsample_factor
        normalized_difference = direction_vector/(upsample_factor//2)

        new_ijk_norm = new_ijk / (32*upsample_factor) - 0.5
        # shifted_sdf_values = igl.signed_distance(new_ijk_norm, 
        #                                             gt_mesh.vertices, 
        #                                             gt_mesh.faces)[0]
        [shifted_sdf_values, sdf_values] = compute_sdf_open3d(gt_mesh.vertices, 
                                               gt_mesh.faces, 
                                               [new_ijk_norm.numpy(), ijk_norm.numpy()])
        sdf_values = scaled_sdf(threshold, sdf_values)
        # shifted_sdf_values = torch.tensor(shifted_sdf_values, dtype=torch.float32, device='cpu')
        # shifted_sdf_values_a = fetch_numpy_values_shifted(new_ijk, h5_file['128_sdf'][:])
        # shifted_sdf_values_a = scaled_sdf(threshold, shifted_sdf_values_a)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-shifted_sdf_values_a.numpy()))*65)
        shifted_sdf_values = scaled_sdf(threshold, shifted_sdf_values)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-sdf_values.numpy()))*65)

        # create VDBTensor
        shifted_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(shifted_sdf_values))

        small_features = torch.cat([sdf_values, normalized_difference], dim=-1) 
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_features))
        
        # return small_vdb, shifted_vdb, shifted_vdb_a, sdf_values_a
        # save the input and output VDB tensors
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')

        fvdb.save(input_vdb_path, small_vdb.grid, small_vdb.data, compressed=True)
        fvdb.save(output_vdb_path, shifted_vdb.grid, shifted_vdb.data, compressed=True)

    def get_processed_sdf_data_exp(self, obj_name, 
                             threshold=65, 
                             grid_size=33, 
                             random_direction_type='nonUniform'):
        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        _dir_gt = self.gt_objs
        # find the .obj file in the directory by walking through the directory
        file_dir_gt = os.path.join(_dir_gt, obj_name.split('.')[0])
        obj_files = [f for f in os.listdir(file_dir_gt) if f.endswith('.obj')]
        file_gt = os.path.join(file_dir_gt, obj_files[0])
        
        # get sdf from the point
        gt_mesh = trimesh.load(file_gt, process=False)
        gt_mesh.vertices = gt_normalize_vertices(gt_mesh.vertices)

        gt_large = self.sdf_gt_large
        # read hdf5 file
        filename = obj_name.split('.')[0]
        h5_file = h5py.File(os.path.join(gt_large, f'{filename}.hdf5'), 'r')
        # sdf_32 = h5_file['32_sdf'][:]
        grid_o = mt.mesh_grid(grid_size, normalize=False)
        grid_o_n = grid_o/(grid_size-1)-0.5
        sdf_32 = compute_sdf_open3d(gt_mesh.vertices,
                                      gt_mesh.faces,
                                      grid_o_n)
        sdf_32 = sdf_32.reshape(33, 33, 33)
        mask = mt.make_mask_close(sdf_32, threshold)
        sdf_32 = h5_file['32_sdf'][:]

        #  create a grid of the size without nomalize actual shape
        ijk_mesh_grid = mt.mesh_grid(grid_size)
        ijk_mesh_grid = ijk_mesh_grid.reshape(grid_size, grid_size, grid_size, 3)

        # consider only the points where the mask is True
        # normalize the ijk coordinates to be centered around (0, 0, 0)
        ijk = torch.tensor(ijk_mesh_grid[mask], 
                            dtype=torch.int, 
                            device='cpu')
        grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(ijk), 
                                        voxel_sizes=(1/(grid_size-1)), 
                                        origins=torch.tensor([0, 0, 0], 
                                        device='cpu'))

        ijk = grid.ijk.jdata
        ijk_norm = ijk/(grid_size-1)-0.5
        # sdf_values = compute_sdf_open3d(gt_mesh.vertices,
        #                                   gt_mesh.faces,
        #                                   ijk_norm.numpy())
        sdf_values_a = fetch_numpy_values(grid, sdf_32, grid_size)
        sdf_values_a = scaled_sdf(threshold, sdf_values_a)

        # upsample_factor = random.randint(1, 5)
        upsample_factor = 10
        upsample_factor = upsample_factor*2
        ########
        m3g=torch.tensor(mt.mesh_grid(upsample_factor+1), device='cpu')-(upsample_factor//2)
        
        # Randomly select one coordinate from m3g 
        unique_random_direction = True
        if unique_random_direction:
            num_elements = grid.ijk.jdata.shape[0]
            random_indices = torch.randint(0, m3g.shape[0], (num_elements,), device='cpu')
        else:
            random_indices = np.random.randint(0, m3g.shape[0])

        # new ijk coordinates
        selected_m3g = m3g[random_indices]  # Shape: (num_elements, 3)
        new_ijk = (upsample_factor * grid.ijk.jdata) + selected_m3g
        new_ijk_cpu = new_ijk.cpu().detach().numpy()
        new_ijk = np.clip(new_ijk_cpu, 0, (grid_size-1)*upsample_factor)
        new_ijk = torch.tensor(new_ijk, dtype=torch.int, device='cpu')
    
        direction_vector = new_ijk - (grid.ijk.jdata) * upsample_factor
        normalized_difference = direction_vector/(upsample_factor//2)

        new_ijk_norm = new_ijk / (32*upsample_factor) - 0.5
        # shifted_sdf_values = igl.signed_distance(new_ijk_norm, 
        #                                             gt_mesh.vertices, 
        #                                             gt_mesh.faces)[0]
        [shifted_sdf_values, sdf_values] = compute_sdf_open3d(gt_mesh.vertices, 
                                               gt_mesh.faces, 
                                               [new_ijk_norm.numpy(), ijk_norm.numpy()])
        sdf_values = scaled_sdf(threshold, sdf_values)
        # shifted_sdf_values = torch.tensor(shifted_sdf_values, dtype=torch.float32, device='cpu')
        # shifted_sdf_values_a = fetch_numpy_values_shifted(new_ijk, h5_file['128_sdf'][:])
        # shifted_sdf_values_a = scaled_sdf(threshold, shifted_sdf_values_a)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-shifted_sdf_values_a.numpy()))*65)
        shifted_sdf_values = scaled_sdf(threshold, shifted_sdf_values)
        # print(np.mean(np.abs(shifted_sdf_values.numpy()-sdf_values.numpy()))*65)

        # create VDBTensor
        shifted_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(shifted_sdf_values))

        small_features = torch.cat([sdf_values, normalized_difference], dim=-1) 
        small_vdb = fvnn.VDBTensor(grid, 
                                    grid.jagged_like(small_features))
        # shifted_vdb_a = fvnn.VDBTensor(grid,
        #                                grid.jagged_like(shifted_sdf_values_a))
        shifted_vdb_a = None
        return small_vdb, shifted_vdb, shifted_vdb_a, sdf_values_a
        # save the input and output VDB tensors
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')

        fvdb.save(input_vdb_path, small_vdb.grid, small_vdb.data, compressed=True)
        fvdb.save(output_vdb_path, shifted_vdb.grid, shifted_vdb.data, compressed=True)

    def get_processed_sdf_data_old(self, 
                             obj_name, 
                             threshold=65, 
                             grid_size=33, 
                             random_direction_type='nonUniform'):
        output_path = self.fvdb_saved_dir
        file_name = obj_name.split('.')[0]

        _dir_gt = self.gt_objs
        # find the .obj file in the directory by walking through the directory
        file_dir_gt = os.path.join(_dir_gt, obj_name.split('.')[0])
        obj_files = [f for f in os.listdir(file_dir_gt) if f.endswith('.obj')]
        file_gt = os.path.join(file_dir_gt, obj_files[0])
        
        # get sdf from the point
        gt_mesh = trimesh.load(file_gt, process=False)
        gt_mesh.vertices = gt_normalize_vertices(gt_mesh.vertices)

        gt_large = self.sdf_gt_large
        # read hdf5 file
        filename = obj_name.split('.')[0]
        h5_file = h5py.File(os.path.join(gt_large, f'{filename}.hdf5'), 'r')

        
        sdf_32 = h5_file['32_sdf'][:]
        mask_sdf_32 = mt.make_mask_close(sdf_32, threshold)
        grid_original = mt.mesh_grid(grid_size, normalize=False)
        mask_sdf_32 = mask_sdf_32.flatten()
        grid_original = grid_original[mask_sdf_32]
        m3g = mt.mesh_grid(2)-1

        # random select a random direction for each voxel
        num_elements = grid_original.shape[0]
        random_indices = torch.randint(0, m3g.shape[0], (num_elements,))
        
        # random number between 0 to 1
        if random_direction_type=='randomNonUniform':
            rand_values = np.random.rand(num_elements, 3)
        elif random_direction_type=='randomUniform':
            rand_values = np.random.rand(num_elements, 1)
        elif random_direction_type=='randomChoice':
            my_list = [-1, -.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
            rand_values = random.choice(my_list)
        
        # create new grid    
        grid_new = grid_original + m3g[random_indices] * (rand_values/2)
        grid_new = np.clip(grid_new, 0, grid_size-1)
        
        # voxel vector
        voxel_vector = (grid_new - grid_original)*2
        # print(rand_values, voxel_vector, m3g[random_indices])

        # normalize grid
        grid_norm = grid_new/(grid_size-1)-0.5

        # sdf calculation
        sdf_val = igl.signed_distance(grid_norm, gt_mesh.vertices, gt_mesh.faces)[0]
        sdf_val_array = np.full((grid_size, grid_size, grid_size), 100.0)
        sdf_val_array[grid_original[:, 0], grid_original[:, 1], grid_original[:, 2]] = sdf_val

        grid_original = torch.tensor(grid_original, dtype=torch.int, device='cpu')

        # fvdb tensors
        fvdb_grid = fvdb.gridbatch_from_ijk(fvdb.JaggedTensor(grid_original),
                                            voxel_sizes=(1/(grid_size-1)),
                                    origins=torch.tensor([0, 0, 0], device='cpu')
                                    )
        ijk = fvdb_grid.ijk.jdata
        sdf_val_output = sdf_val_array[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        sdf_val_output = torch.tensor(sdf_val_output, dtype=torch.float32, device='cpu')
        sdf_val_output = scaled_sdf(threshold=threshold, sdf_arr=sdf_val_output)

        sdf_val_input = sdf_32[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        sdf_val_input = torch.tensor(sdf_val_input, dtype=torch.float32, device='cpu')
        sdf_val_input = scaled_sdf(threshold=threshold, sdf_arr=sdf_val_input)

        # input feature
        voxel_vector = torch.tensor(voxel_vector, dtype=torch.float32, device='cpu')
        sdf_val_cat = torch.cat([sdf_val_input, voxel_vector], dim=-1)

        input_tensor = fvnn.VDBTensor(fvdb_grid, fvdb_grid.jagged_like(sdf_val_cat))
        output_tensor = fvnn.VDBTensor(fvdb_grid, fvdb_grid.jagged_like(sdf_val_output))

        # save the input and output VDB tensors
        input_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_input.nvdb')
        output_vdb_path = os.path.join(output_path, 
                                    f'{file_name}_output.nvdb')

        fvdb.save(input_vdb_path, input_tensor.grid, input_tensor.data, compressed=True)
        fvdb.save(output_vdb_path, output_tensor.grid, output_tensor.data, compressed=True)

    def run_data_processing(self, epoch):
        def get_r(epoch, T, r_start=1, r_end=0.3):
            return r_end + (r_start - r_end) * (1 - epoch / T)
        r = get_r(epoch, 100)
        print(f"Data Processing Epoch: {epoch}, annealed schedule r: {r}")
        joblib.Parallel(n_jobs=-1)(joblib.delayed(self.get_processed_sdf_data_v2)
                                   (name, 
                                    r=r,
                                    threshold=self.threshold,
                                    grid_size=self.input_size,
                                    random_direction_type=self.random_direction_type) 
                                    for name in tqdm(self.names_set))
        # for name in tqdm(self.names_set):
        #     self.get_processed_sdf_data_v2(name, 
        #                                    r=r,
        #     threshold=self.threshold,
        #     grid_size=self.input_size,
        #     random_direction_type=self.random_direction_type) 



        # def get_random_voxel_vector(upsample_factor, grid_size, grid, unique_random_direction=True):
        #     # random voxel vector
        #     m3g=torch.tensor(mt.mesh_grid(upsample_factor+1), device='cpu')-(upsample_factor//2)
        #     if unique_random_direction:
        #         num_elements = grid.ijk.jdata.shape[0]
        #         random_indices = torch.randint(0, m3g.shape[0], (num_elements,), device='cpu')
        #     else:
        #         random_indices = np.random.randint(0, m3g.shape[0])

        #     # new ijk coordinates
        #     selected_m3g = m3g[random_indices]  # Shape: (num_elements, 3)
        #     new_ijk = (upsample_factor * grid.ijk.jdata) + selected_m3g
        #     new_ijk_cpu = new_ijk.cpu().detach().numpy()
        #     new_ijk = np.clip(new_ijk_cpu, 0, (grid_size-1)*upsample_factor)
        #     new_ijk = torch.tensor(new_ijk, dtype=torch.int, device='cpu')
        
        #     direction_vector = new_ijk - (grid.ijk.jdata) * upsample_factor
        #     normalized_difference = direction_vector/(upsample_factor//2)
        #     return new_ijk, normalized_difference
                                    