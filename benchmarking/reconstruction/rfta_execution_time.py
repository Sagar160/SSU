import time
import gpytoolbox as gpy
import numpy as np
from collections import defaultdict
import math
import h5py
from tqdm import tqdm

with open('run/thingi30.txt', 'r') as f:
    filenames = f.read().splitlines()
filenames = [f'{name}.obj' for name in filenames]


def describe_exe_results(results):
    summary = defaultdict(list)
    for filename, grid_size, exec_time in results:
        summary[grid_size].append(exec_time)

    print("Grid Size | Mean | Std | Min | Max")
    for grid_size, times in summary.items():
        mean = sum(times) / len(times)
        std = math.sqrt(sum((t - mean) ** 2 for t in times) / len(times))
        min_time = min(times)
        max_time = max(times)
        print(
            f"{grid_size:9} | {mean:.4f} | {std:.4f} | {min_time:.4f} | {max_time:.4f}")


def rfta_execution_time():
    execution_time_results = []
    for res in [128]:
        for filename in tqdm(filenames):
            print('Processing file:', filename, 'at resolution:', res)

            start_time = time.time()
            j = res
            gx, gy, gz = np.meshgrid(
                np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1), np.linspace(-1, 1, j+1))
            U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
            U_int = (U*(res/2) + (res/2)).astype(np.int32)

            with h5py.File(f'/home/nmaruani/data/gt_Thingi32_NDC_norm/{filename.split(".")[0]}.hdf5', 'r') as f:
                S = f[f'{res}_sdf'][:][U_int[:, 0], U_int[:, 1], U_int[:, 2]]
                S = S*2

            # Reconstruct triangle mesh
            try:
                Vr, Fr = gpy.reach_for_the_arcs(
                    U, S, verbose=True, parallel=True, return_point_cloud=False, max_points_per_sphere=3, fine_tune_iters=3)
            except:
                print("No mesh:", filename)
            end_time = time.time()
            execution_time = end_time - start_time
            execution_time_results.append((filename, res+1, execution_time))
        describe_exe_results(execution_time_results)
    describe_exe_results(execution_time_results)


if __name__ == "__main__":
    rfta_execution_time()
