import os
import sys
import torch
import fvdb.nn as fvnn
from ssu_tools import positional_encoding

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from flow_matching import flow_matching

class FMTools:
    def __init__(self):
        pass        


#### old functions ####

def fm_sampling(output, noise, t):
    ouput_j = output.jdata
    noise_j = noise.jdata
    
    xt = ouput_j * t + noise_j * (1 - t)
    xt = fvnn.VDBTensor(output.grid, output.grid.jagged_like(xt))
    return xt

def additional_feature(input, t):
    '''add additional feature to the input'''
    input_j = input.jdata
    
    # add time as a feature
    # t_col = torch.ones_like(input_j[:, :1]) * t
    new_features = torch.cat([input_j, t], dim=1)
    return fvnn.VDBTensor(input.grid, input.grid.jagged_like(new_features))

def add_g_noise(vdb, t, noise_level=0.1):
    '''add noise to the small vdb'''
    
    # stochastic preconditioning
    noise = torch.randn_like(vdb.jdata) * noise_level * (1 - t)
    # noise = torch.randn_like(vdb.jdata)
    noisy_small_vdb = vdb.jdata + noise
    return fvnn.VDBTensor(vdb.grid, vdb.grid.jagged_like(noisy_small_vdb))

def transform_input(input, 
                    output, t, 
                    pos_enc_dim=10,
                    scale_factor=2, 
                    upsampler=None, 
                    g_noise=True,
                    test=False):
    
    # check upsampler args is one of 'nearest', 'trilinear', or None
    if upsampler is not None and not (isinstance(upsampler, str) and upsampler in ['nearest', 'trilinear']):
        raise ValueError("Upsampler must be a string: 'nearest', 'trilinear', or None.")

    if upsampler == 'nearest':
        upsampler = fvnn.UpsamplingNearest(scale_factor=scale_factor)
        upsample_input = upsampler(input)
    elif upsampler == 'trilinear':
        new_centers = output.grid.grid_to_world(output.ijk.float())
        upsample_data = input.grid.sample_trilinear(new_centers, input.data)
        upsample_input = fvnn.VDBTensor(output.grid, upsample_data)
    elif upsampler is None:
        upsample_input = input
    else:
        raise ValueError("Invalid upsampler type. Use 'nearest', 'trilinear', or None.")

    if upsampler is not None:
        # match the grid to the output grid
        gridFiller = fvnn.FillFromGrid()
        upsample_input = gridFiller(upsample_input, output.grid)
    
    # if keep_gt:
    #     to_change_idx = upsample_input.grid.ijk_to_index(
    #                             input.grid.ijk).jdata
    #     upsample_input.data.jdata[to_change_idx] = input.jdata

    if test is False:
        xt = fm_sampling(output, upsample_input, t)
    else:
        # print("Test mode: using upsampled input directly")
        xt = upsample_input

    if g_noise:
        xt = add_g_noise(xt, t, noise_level=0.1)
        xt = positional_encoding(xt, pos_enc_dim)
        xt = additional_feature(xt, t)
        xt = xt
    elif not g_noise:
        xt = positional_encoding(xt, pos_enc_dim)
        xt = additional_feature(xt, t)
        xt = xt

    return xt, upsample_input

def fm_loss(pred_flow, target, noise):
    'calculate the flow matching loss'
    pred_flow = pred_flow.jdata
    target = target.jdata
    noise = noise.jdata
    # print(pred_flow.shape, target.shape, noise.shape)

    true_flow = target - noise
    return torch.mean((pred_flow - true_flow) ** 2)


def fm_prediction(model, input, output, device):
    steps = 10
    for i, t in enumerate(torch.linspace(0.0, 1, steps), start=1):
        # print(i,t)
        t = torch.full_like(output.jdata, t).to(device)
        if i==1:
            xt, noise = transform_input(input, output, t=t, 
                                        scale_factor=2, 
                                        upsampler='trilinear', 
                                        g_noise=False,
                                        # keep_gt=True,
                                        test=True)
            noise_i = noise
        else:
            xt, noise = transform_input(noise, output, t=t, 
                                        upsampler=None, 
                                        g_noise=False,
                                        test=True)
            
        pred = model(xt)
        noise = noise.jdata + (1 / steps) * pred.jdata
        noise = fvnn.VDBTensor(xt.grid, xt.grid.jagged_like(noise))
    return noise
