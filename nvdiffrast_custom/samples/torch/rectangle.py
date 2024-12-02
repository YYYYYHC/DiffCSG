# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import pdb
import imageio
import numpy as np
import torch
import nvdiffrast.torch as dr
import sys

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def rectangle_meshing(resolution, a, b,d):
    '''
    |
    |
    |___a__|
    |    - |
    |  -   b
    |-_____|_______
    '''
    meshed_rectangle = tensor(np.zeros([1,3*2*resolution**2, 4]),dtype=torch.float32)
    re = a/resolution
    ce = b/resolution
    for recid in range(resolution**2):
        tri1id = 2*recid
        tri2id = 2*recid + 1
        rn = recid // resolution
        cn = recid - rn * resolution
        
        x1 = rn*re
        y1 = cn*ce
    
        x2 = rn*re + re
        y2 = cn*ce
        
        x3 = rn*re + re
        y3 = cn*ce + ce
        
        x4 = rn*re
        y4 = cn*ce + ce
        
        meshed_rectangle[0][3*tri1id] = tensor([x1,y1,d,1],dtype=torch.float32)
        meshed_rectangle[0][3*tri1id+1] = tensor([x3,y3,d,1],dtype=torch.float32)
        meshed_rectangle[0][3*tri1id+2] = tensor([x4,y4,d,1],dtype=torch.float32)
        
        meshed_rectangle[0][3*tri2id] = tensor([x1,y1,d,1],dtype=torch.float32)
        meshed_rectangle[0][3*tri2id+1] = tensor([x2,y2,d,1],dtype=torch.float32)
        meshed_rectangle[0][3*tri2id+2] = tensor([x3,y3,d,1],dtype=torch.float32)
    return meshed_rectangle
if sys.argv[1:] == ['--cuda']:
    glctx = dr.RasterizeCudaContext()
elif sys.argv[1:] == ['--opengl']:
    glctx = dr.RasterizeGLContext()
else:
    print("Specify either --cuda or --opengl")
    exit(1)

pos_idx = tensor([[0,1,2],[0,2,3],[4,6,7],[4,5,6]], dtype = torch.int32)
vtx_pos = tensor([[-0.5,-0.5,0,1],[0.5,-0.5,0,1],[0.5,0.5,0,1],[-0.5,0.5,0,1],
                  [0,-1,0.1,1],[1,-1,0.1,1],[1,0,0.1,1],[0,0,0.1,1]],dtype=torch.float32)
vtx_col = tensor([[1.0000, 1.0000, 1.0000],
    [1.0000, 1.0000, 0.0000],
    [0.0000, 0.0000, 1.0000],
    [0.0000, 1.0000, 0.0000],
    [1.0000, 0.0000, 0.0000],
    [1.0000, 0.6470, 0.0000]],dtype=torch.float32)
col_idx = tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3]],dtype=torch.int32)
ranges = torch.tensor([[0,2],[2,2]],dtype = torch.int32)
rast, _ = dr.rasterize(glctx, pos = vtx_pos, tri = pos_idx, resolution=[256, 256],ranges=ranges)

out, _ = dr.interpolate(vtx_col[None,...], rast, col_idx)

out = dr.antialias(out, rast, vtx_pos, pos_idx)

#z-buffering:

img = out.cpu().numpy()[1, ::-1, :, :] # Flip vertically.
img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8) # Quantize to np.uint8

print("Saving to 'rec.png'.")
imageio.imsave('rec.png', img)
