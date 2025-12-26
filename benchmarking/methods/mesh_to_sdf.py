import os
import igl
import sys
import time
sys.path.append('/user/spanwar/home/Documents/learn-fvdb/ssu/SSU/src/utils')
import mesh_tools as mt

def get_sdf_from_mesh(filename, grid_size=32):
    grid_n = grid_size+1 # don’t forget the +1 =)
    gt_dir = '/data/workspaces/spanwar/dataset/thingi/GT_thingi'
    mesh_path = os.path.join(gt_dir, filename)

    v, f = igl.read_triangle_mesh(mesh_path)
    v = 2*mt.NDCnormalize(v)
    points = mt.mesh_grid(grid_n, True)
    sdf = igl.signed_distance(points, v, f)[0].reshape(grid_n, grid_n, grid_n)/2
    return sdf

if __name__ == "__main__":
    filename = '252119.obj'
    grid_size = 512
    time_start = time.time()
    sdf = get_sdf_from_mesh(filename, grid_size=grid_size)
    end_time = time.time()
    print(f'execution time {grid_size}:', end_time - time_start)