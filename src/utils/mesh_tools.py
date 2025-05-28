"""Various mesh utilities"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import ipywidgets


def count_parameters(model, print_result=True):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_result:
        if num > 1e6:
            print("The model has {:.1f}M parameters".format(num/1000000))
        elif num > 1000:
            print("The model has {:.1f}k parameters".format(num/1000))
        return
    return num

def mesh_grid(grid_size: int, normalize=False):
    """create mesh grid with default indexing"""
    xx, yy, zz = np.mgrid[:grid_size, :grid_size, :grid_size]
    grid_3d = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
    if normalize:
        return 2 * (grid_3d / (grid_size - 1)) - 1
    return grid_3d

def NDCnormalize(vertices, return_scale=False):
    """normalization in half unit ball"""
    vM = vertices.max(0)
    vm = vertices.min(0)
    scale = np.sqrt(((vM - vm) ** 2).sum(-1))
    mean = (vM + vm) / 2.0
    nverts = (vertices - mean) / scale
    if return_scale:
        return nverts, mean, scale
    return nverts


def plotSlice(sdf_array, vmax):
    def helper(xhi, slice, vmax,  cmap='seismic'):
        plt.imshow(xhi[slice], origin='lower',
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    slider = ipywidgets.IntSlider(
        min=0, max=sdf_array.shape[0]-1, step=1, value=sdf_array.shape[0]//2)
    return ipywidgets.interact(lambda s: helper(sdf_array, s, vmax), s=slider)

def marching_cubes(vox: np.ndarray, iso=0.0, ret=False):
    """marching cube from NxNxN array"""
    im_res = vox.shape[0]
    vox_v, vox_f, _, _ = measure.marching_cubes(
        vox, iso, spacing=[1.0 for i in range(3)]
    )
    vox_v = 2 * (vox_v / (im_res - 1)) - 1
    nf = vox_f.copy()
    if ret:
        vox_f[:, 0], vox_f[:, 1] = nf[:, 1], nf[:, 0]
    return vox_v.astype(np.float64), vox_f

def export_obj(nv: np.ndarray, nf: np.ndarray, name: str, export_lines=False):
    try:
        file = open(name, "x")
    except:
        file = open(name, "w")
    for e in nv:
        file.write("v {} {} {}\n".format(*e))
    file.write("\n")
    for face in nf:
        header = "l " if export_lines else "f "
        file.write(header + " ".join([str(fi + 1) for fi in face]) + "\n")
    file.write("\n")