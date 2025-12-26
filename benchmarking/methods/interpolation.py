import os
import h5py
import joblib
import numpy as np
# import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.interpolate import RegularGridInterpolator

def load_3d_array(filename, input_dir, size):
    """
    Load a 3D array from a given file path.
    """
    # load h5py file
    file_path = os.path.join(input_dir, filename)
    with h5py.File(file_path, 'r') as f:
        data = f[f'{size-1}_sdf'][:]  
    return data

# def interpolate(numpy_array, up_size, method):
#     # Convert numpy -> SimpleITK (np: z,y,x -> SITK: x,y,z)
#     sitk_image = sitk.GetImageFromArray(numpy_array)

#     original_size = np.array(sitk_image.GetSize(), dtype=float)      # (x, y, z) or (x, y)
#     original_spacing = np.array(sitk_image.GetSpacing(), dtype=float)

#     dim = len(original_size)


#     # New size (in number of pixels/voxels)
#     up_size = np.array([up_size]*dim)
#     new_size = np.maximum(np.round(up_size), 1).astype(int)

#     # New spacing so that physical size stays consistent:
#     # spacing_new = spacing_old * (size_old / size_new)
#     new_spacing = original_spacing * (original_size / new_size.astype(float))
#     # print(new_size, new_spacing)

#     # Map method string -> SimpleITK interpolator enum
#     m = method.lower()
#     if m in ("nearest", "nn", "nearest_neighbor"):
#         interpolator = sitk.sitkNearestNeighbor
#     elif m in ("linear", "bilinear", "trilinear"):
#         interpolator = sitk.sitkLinear
#     elif m in ("bspline", "cubic", "cubic_bspline", "spline"):
#         interpolator = sitk.sitkBSpline          # cubic B-spline
#     elif m in ("lanczos", "lanczos_windowed_sinc"):
#         interpolator = sitk.sitkLanczosWindowedSinc
#     else:
#         raise ValueError(
#             f"Unknown method '{method}'. "
#             "Use one of: 'nearest', 'linear', 'bspline', 'lanczos'."
#         )

#     # Set up resampler
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetSize([int(s) for s in new_size.tolist()])
#     resampler.SetOutputSpacing([float(s) for s in new_spacing.tolist()])
#     resampler.SetOutputOrigin(sitk_image.GetOrigin())
#     resampler.SetOutputDirection(sitk_image.GetDirection())
#     resampler.SetInterpolator(interpolator)

#     # Reasonable default pixel value for regions outside original image
#     # if np.issubdtype(numpy_array.dtype, np.floating):
#     #     default_value = float(numpy_array.min())
#     # else:
#     default_value = 1.0
#     resampler.SetDefaultPixelValue(default_value)

#     # Execute resampling
#     resampled_sitk = resampler.Execute(sitk_image)

#     # Back to numpy (SITK x,y,z -> np z,y,x)
#     resampled_numpy = sitk.GetArrayFromImage(resampled_sitk)
#     return resampled_numpy

def create_sdf_interpolator(
    i_coords: np.ndarray,
    j_coords: np.ndarray,
    k_coords: np.ndarray,
    upijks: np.ndarray,
    sdf_values: np.ndarray
):
    # Check for minimum points for 'cubic' method
    if len(i_coords) < 4 or len(j_coords) < 4 or len(k_coords) < 4:
        method_to_use = 'linear'
        print(f"Warning: Using 'linear' method for shape {sdf_values.shape}, cubic needs >= 4 points.")
    else:
        method_to_use = 'cubic'

    interpolator_func = RegularGridInterpolator(
        points=(i_coords, j_coords, k_coords), # Pass tuple of 1D coordinate arrays
        values=sdf_values,                     # Pass the full 3D array here
        method=method_to_use,
        bounds_error=False,
        fill_value=None
    )
    
    # Return the evaluated points
    return interpolator_func(upijks)


def run(filename, input_dir, output_dir=None):
    """
    Load a 3D array, apply B-spline interpolation, and save the result.
    """

    print(f"Processing file: {filename}")
    # [33, 65, 129]
    for size in [33, 65, 129]:
        # Load the 3D array
        arr = load_3d_array(filename, input_dir, size)
        
        # methods = ['nearest', 'trilinear', 'bspline', 'lanczos']
        methods = ['bspline']
        for method in methods:
            # output_size = (size)*4   # e.g., 33 -> 129
            # Apply interpolation
            # zoomed_arr = zoom(arr, zoom=4, order=3)  # order=3 for bspline
            # zoomed_arr = interpolate(arr, output_size, method=method)
            # zoomed_arr = zoomed_arr[:-3, :-3, :-3]  # Remove last 3 slices to get desired size
            # print(zoomed_arr.shape)
            scale_factor=4
            # x = np.linspace(0, 1, size)
            # y = np.linspace(0, 1, size)
            # z = np.linspace(0, 1, size)
            
            # # # 2. Define the target 'upsampled' shape and coordinates
            up_shape_len = (size - 1) * scale_factor + 1 # Target length, e.g., 33 -> 129
            # up_shape_len = size * scale_factor
            # up_shape = (up_shape_len, up_shape_len, up_shape_len)
            
            # # Generate target query points (flattened N x 3 array)
            # x_new = np.linspace(0, 1, up_shape[0])
            # y_new = np.linspace(0, 1, up_shape[1])
            # z_new = np.linspace(0, 1, up_shape[2])
            
            # Xn, Yn, Zn = np.meshgrid(x_new, y_new, z_new, indexing="ij")
            # pts = np.stack([Xn, Yn, Zn], axis=-1).reshape(-1, 3)

            # # 3. Apply the B-spline interpolation
            # zoomed_arr_flat = create_sdf_interpolator(
            #     i_coords=x, 
            #     j_coords=y, 
            #     k_coords=z,
            #     upijks=pts,
            #     sdf_values=arr # Pass the full 3D array here
            # )
            
            # 4. Reshape the result back to 3D volume
            # zoomed_arr = zoomed_arr_flat.reshape(up_shape)
            zoom_factor=up_shape_len/size
            zoomed_arr = zoom(
                    arr,
                    zoom=(zoom_factor, zoom_factor, zoom_factor),
                    order=3,            # cubic B-spline
                    mode="nearest",     # boundary handling
                    prefilter=True      # MUST be True for accuracy
                )
            # zoomed_arr = bspline_upsample(
            #     volume=arr,
            #     out_size=(up_shape),
            #     spacing=(1.0, 1.0, 1.0),
            #     # order=3
            # )
            print(zoomed_arr.shape)
            # ijk = np.indices(arr.shape)
            # ijk = ijk.reshape(3, -1).T  # (N, 3)
            # print(ijk.shape)
            # print((arr[ijk[:,0], ijk[:,1], ijk[:,2]] == zoomed_arr[ijk[:,0]*4, ijk[:,1]*4, ijk[:,2]*4]).sum())
            # assert (arr[ijk[:,0], ijk[:,1], ijk[:,2]] == zoomed_arr[ijk[:,0]*4, ijk[:,1]*4, ijk[:,2]*4]).all()
            
            # Save the zoomed array to an h5py file
            _filename = filename.split('.')[0]  # Remove file extension
            output_path = os.path.join(output_dir, f'{size-1}_{_filename}_interpolation.hdf5')
            with h5py.File(output_path, 'a') as f:  # 'a' mode allows read/write
                key = f'zoom_{4}_{method}_sdf'
                if key in f:
                    del f[key]  # Remove existing dataset
                f.create_dataset(key, data=zoomed_arr)
            
            # print(f"Zoomed array saved to {output_path}")


def run_parallel(n_jobs=-1):
    """
    Run  interpolation on multiple files in parallel.
    """
    input_dir='/data/workspaces/spanwar/dataset/thingi/thingi_all/gt_Thingi32_NDC_norm'
    output_dir='/data/workspaces/spanwar/results/ssu/interpolation'
    filenames = os.listdir(input_dir)
    # print(filenames)
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run)(filename, input_dir, output_dir) 
        for filename in filenames
    )

if __name__ == "__main__":
    run_parallel(n_jobs=-1)