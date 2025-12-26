import gc
import os
import numpy as np
import torch
import trimesh
import fvdb
import fvdb.nn as fvnn
from tqdm import tqdm

def predictions(sdf_scaling, 
                input_size,
                upsample_factor,
                input_vdb, 
                new_features, 
                model,
                actual_sdf=None):

    all_inputs = []
    for feature in new_features:
        all_inputs.append(fvnn.VDBTensor(input_vdb.grid,
                                        input_vdb.grid.jagged_like(feature)))
    # all_inputs_vdb = fvdb.jcat(all_inputs)

    upsampled_sdf_size = ((input_size - 1) * upsample_factor) + 1
    sdf = np.full((upsampled_sdf_size, 
                    upsampled_sdf_size, 
                    upsampled_sdf_size), 100.0)
        
    # pred = self.test_fm_steps(all_inputs_vdb, model, n_steps)
    # pred = model(all_inputs_vdb)
    BATCH_SIZE = 20
    num_samples = len(all_inputs)

    pred_values_list = []
    pred_ijk_list = []
    vector_list = []

    for start in range(0, num_samples, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_samples)
        batch_vdb = fvdb.jcat(all_inputs[start:end])
        vector_list.append(batch_vdb.jdata[:, 1:4].cpu().detach().numpy())
        batch_vdb = batch_vdb.cuda()
        batch_pred = model(batch_vdb)
        pred_values_list.append(batch_pred.jdata.detach().cpu().numpy().squeeze())
        pred_ijk_list.append(batch_pred.grid.ijk.jdata.cpu().detach().numpy())

    pred_values = np.concatenate(pred_values_list, axis=0)
    pred_ijk = np.concatenate(pred_ijk_list, axis=0)
    vector = np.concatenate(vector_list, axis=0)
    # pred_ijk = pred.grid.ijk.jdata.cpu().detach().numpy() + (input_size//2)
    # pred_ijk = pred.grid.ijk.jdata.cpu().detach().numpy()
    # vector = all_inputs_vdb.jdata[:, 1:4].cpu().detach().numpy()  
    pred_ijk = (pred_ijk)*upsample_factor + (vector*(upsample_factor//2)).astype(int)
    # pred_values = pred.jdata.detach().cpu().numpy().squeeze()  # Remove extra dimension
    # print(pred_ijk.max(), pred_ijk.min(), sdf.shape)
    # sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]] = pred_values
    
    # means predictions
    D, H, W = sdf.shape
    flat_idx = np.ravel_multi_index(pred_ijk.T, sdf.shape)  # (N,)

    sum_arr = np.zeros(sdf.size, dtype=np.float32)
    cnt_arr = np.zeros(sdf.size, dtype=np.int64)

    np.add.at(sum_arr, flat_idx, pred_values)     # accumulate sums per voxel
    np.add.at(cnt_arr, flat_idx, 1)               # accumulate counts per voxel

    mask = cnt_arr > 0
    mean_arr = np.zeros_like(sum_arr, dtype=np.float32)
    mean_arr[mask] = sum_arr[mask] / cnt_arr[mask]

    sdf.flat[mask] = mean_arr[mask] 

    # # median
    # D, H, W = sdf.shape
    # flat_idx = np.ravel_multi_index(pred_ijk.T, sdf.shape)  # (N,)

    # # Compute median per voxel
    # from collections import defaultdict

    # voxel_values = defaultdict(list)
    # for idx, flat in enumerate(flat_idx):
    #     voxel_values[flat].append(pred_values[idx])

    # median_arr = np.zeros(sdf.size, dtype=np.float32)
    # for flat, values in voxel_values.items():
    #     median_arr[flat] = np.median(values)

    # mask = np.zeros_like(median_arr, dtype=bool)
    # mask[list(voxel_values.keys())] = True

    # sdf.flat[mask] = median_arr[mask]
    
    ####
    sdf_mask = np.abs(sdf) < 100

    # Error Calculation
    if actual_sdf is not None:
        # error between sdfs
        if input_size == 33:
            scale = sdf_scaling[input_size-1]
            assert (scale-1)==64
            actual_sdf = actual_sdf*(scale-1)
        elif input_size == 65:
            scale = sdf_scaling[input_size-1]
            assert (scale-1)==128
            actual_sdf = actual_sdf*(scale-1)
        else:
            raise ValueError("Input size must be either 33 or 65.")
        
        actual_values = actual_sdf[pred_ijk[:, 0], pred_ijk[:, 1], pred_ijk[:, 2]]
        error = np.abs(actual_values - pred_values)
        l1_error = np.mean(error)
        mean_squared_error = np.mean(error**2)
        
    # create a fvdb tensor from the sdf
    # up_ijk = fvdb.JaggedTensor(torch.tensor(np.array(np.where(sdf_mask)).T)) - (upsampled_sdf_size//2)
    up_ijk = fvdb.JaggedTensor(torch.tensor(np.array(np.where(sdf_mask)).T))
    up_grid = fvdb.gridbatch_from_ijk(
            up_ijk,
            voxel_sizes=(1/(upsampled_sdf_size-1)),
            origins=torch.tensor([0, 0, 0])
        )
    # up_ijk = up_grid.ijk.jdata.cpu().detach().numpy() + (upsampled_sdf_size//2)
    up_ijk = up_grid.ijk.jdata.cpu().detach().numpy()
    up_values = sdf[up_ijk[:, 0], up_ijk[:, 1], up_ijk[:, 2]]
    up_tensor = fvnn.VDBTensor(up_grid,
                                up_grid.jagged_like(torch.tensor(up_values)))
    if actual_sdf is not None:
        # return up_tensor, error, mean_squared_error
        return up_tensor, l1_error, mean_squared_error
    else:
        # return up_tensor without error
        return up_tensor, None, None


def save_predictions(test_loader, 
                     upsample_factor_dict,
                     save_predictions_dir, 
                     prediction_folder_name, 
                     model):
    # predictions_dir
    save_dir = os.path.join(save_predictions_dir, prediction_folder_name)
    save_dir_obj = os.path.join(save_predictions_dir, prediction_folder_name+ '_objects')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_obj, exist_ok=True)

    # load best model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    names = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            for mini_batch in batch:
                (obj_name, input_size, vdb_inputs, new_features) = mini_batch 
                sdf_scaling_value = (input_size-1)*2+1
                upsample_factor = upsample_factor_dict[input_size]
                vdb_inputs = fvdb.jcat(vdb_inputs)
                vdb_inputs = vdb_inputs.cuda()

                # if input_size != 33:
                #     continue
                # clean memory
                torch.cuda.empty_cache()
                gc.collect()

                # save the predictions path
                file_names = f"{input_size-1}_{obj_name.split('.')[0]}"
                output_file = os.path.join(save_dir, f'{file_names}.nvdb')

                if os.path.exists(output_file):
                    print(f"Predictions for {file_names} already exist. Skipping...")
                    continue

                # testing
                (vdb_up_tensor, 
                    l1_error, 
                    mean_squared_error) = predictions(sdf_scaling_value,
                                                        input_size,
                                                        upsample_factor,
                                                        vdb_inputs, 
                                                        new_features,  
                                                        model)


                # saving predictions
                fvdb.save(output_file, vdb_up_tensor.grid, vdb_up_tensor.data, compressed=True)
                print(f"Saved predictions for {file_names} to {output_file}")

                # save the object files
                output_obj_file = os.path.join(save_dir_obj, f'{file_names}.obj')
                v, f, _ = vdb_up_tensor.grid.marching_cubes(-vdb_up_tensor.data)
                v = v.jdata.detach().cpu().numpy()
                f = f.jdata.detach().cpu().numpy()
                v = (v-0.5)*2.0  # Scale to [-1, 1]
                mesh = trimesh.Trimesh(vertices=v, faces=f)
                mesh.export(output_obj_file)

            