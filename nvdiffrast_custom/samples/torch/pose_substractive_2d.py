# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import pdb
import argparse
import os
import pathlib
import sys
import numpy as np
import torch
import imageio

import util
import nvdiffrast
import nvdiffrast.torch as dr

#----------------------------------------------------------------------------
# Quaternion math.
#----------------------------------------------------------------------------

# Unit quaternion.
def q_unit():
    return np.asarray([1, 0, 0, 0], np.float32)

# Get a random normalized quaternion.
def q_rnd():
    u, v, w = np.random.uniform(0.0, 1.0, size=[3])
    v *= 2.0 * np.pi
    w *= 2.0 * np.pi
    return np.asarray([(1.0-u)**0.5 * np.sin(v), (1.0-u)**0.5 * np.cos(v), u**0.5 * np.sin(w), u**0.5 * np.cos(w)], np.float32)
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

def recPose_rnd():
    """
    random recPos translate, scale
    """
    tx1, ty1, sx1, sy1,tx2, ty2, sx2, sy2 = np.random.uniform(0.0,1.0, size=[8])
    return np.asarray([tx1,ty1,sx1,sy1,tx2, ty2, sx2, sy2], np.float32)
# Get a random quaternion from the octahedral symmetric group S_4.
_r2 = 0.5**0.5
_q_S4 = [[ 1.0, 0.0, 0.0, 0.0], [ 0.0, 1.0, 0.0, 0.0], [ 0.0, 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0, 1.0],
         [-0.5, 0.5, 0.5, 0.5], [-0.5,-0.5,-0.5, 0.5], [ 0.5,-0.5, 0.5, 0.5], [ 0.5, 0.5,-0.5, 0.5],
         [ 0.5, 0.5, 0.5, 0.5], [-0.5, 0.5,-0.5, 0.5], [ 0.5,-0.5,-0.5, 0.5], [-0.5,-0.5, 0.5, 0.5],
         [ _r2,-_r2, 0.0, 0.0], [ _r2, _r2, 0.0, 0.0], [ 0.0, 0.0, _r2, _r2], [ 0.0, 0.0,-_r2, _r2],
         [ 0.0, _r2, _r2, 0.0], [ _r2, 0.0, 0.0,-_r2], [ _r2, 0.0, 0.0, _r2], [ 0.0,-_r2, _r2, 0.0],
         [ _r2, 0.0, _r2, 0.0], [ 0.0, _r2, 0.0, _r2], [ _r2, 0.0,-_r2, 0.0], [ 0.0,-_r2, 0.0, _r2]]
def q_rnd_S4():
    return np.asarray(_q_S4[np.random.randint(24)], np.float32)

# Quaternion slerp.
def q_slerp(p, q, t):
    d = np.dot(p, q)
    if d < 0.0:
        q = -q
        d = -d
    if d > 0.999:
        a = p + t * (q-p)
        return a / np.linalg.norm(a)
    t0 = np.arccos(d)
    tt = t0 * t
    st = np.sin(tt)
    st0 = np.sin(t0)
    s1 = st / st0
    s0 = np.cos(tt) - d*s1
    return s0*p + s1*q

# Quaterion scale (slerp vs. identity quaternion).
def q_scale(q, scl):
    return q_slerp(q_unit(), q, scl)

# Quaternion product.
def q_mul(p, q):
    s1, V1 = p[0], p[1:]
    s2, V2 = q[0], q[1:]
    s = s1*s2 - np.dot(V1, V2)
    V = s1*V2 + s2*V1 + np.cross(V1, V2)
    return np.asarray([s, V[0], V[1], V[2]], np.float32)

# Angular difference between two quaternions in degrees.
def q_angle_deg(p, q):
    p = p.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    d = np.abs(np.dot(p, q))
    d = min(d, 1.0)
    return np.degrees(2.0 * np.arccos(d))

# Quaternion product
def q_mul_torch(p, q):
    a = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    b = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    c = p[0]*q[2] + p[2]*q[0] + p[3]*q[1] - p[1]*q[3]
    d = p[0]*q[3] + p[3]*q[0] + p[1]*q[2] - p[2]*q[1]
    return torch.stack([a, b, c, d])

# Convert quaternion to 4x4 rotation matrix.
def q_to_mtx(q):
    r0 = torch.stack([1.0-2.0*q[1]**2 - 2.0*q[2]**2, 2.0*q[0]*q[1] - 2.0*q[2]*q[3], 2.0*q[0]*q[2] + 2.0*q[1]*q[3]])
    r1 = torch.stack([2.0*q[0]*q[1] + 2.0*q[2]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[2]**2, 2.0*q[1]*q[2] - 2.0*q[0]*q[3]])
    r2 = torch.stack([2.0*q[0]*q[2] - 2.0*q[1]*q[3], 2.0*q[1]*q[2] + 2.0*q[0]*q[3], 1.0 - 2.0*q[0]**2 - 2.0*q[1]**2])
    rr = torch.transpose(torch.stack([r0, r1, r2]), 1, 0)
    rr = torch.cat([rr, torch.tensor([[0], [0], [0]], dtype=torch.float32).cuda()], dim=1) # Pad right column.
    rr = torch.cat([rr, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32).cuda()], dim=0)  # Pad bottom row.
    return rr

# Transform vertex positions to clip space
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, pos, pos_idx, col, col_idx, resolution: int):
    # Setup TF graph for reference.
    pos_clip    = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    
    color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
    
    return color

class CustomTransform(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, pose):
        # 存储输入以供反向传播使用
        ctx.save_for_backward(params, pose)

        # 提取参数
        tx1, ty1, sx1, sy1,tx2, ty2, sx2, sy2 = params

        # 计算变换矩阵
        transform_matrix1 = torch.tensor([[sx1, 0, 0, 0],
                                         [0, sy1, 0, 0],
                                         [0,  0, 1,  0],
                                         [tx1,  ty1, 0,  1]], dtype=params.dtype).cuda()
        transform_matrix2 = torch.tensor([[sx2, 0, 0, 0],
                                         [0, sy2, 0, 0],
                                         [0,  0, 1,  0],
                                         [tx2,  ty2, 0,  1]], dtype=params.dtype).cuda()

        # 乘在pose的右边
        result1 = torch.matmul(pose, transform_matrix1.unsqueeze(0))
        result2 = torch.matmul(pose, transform_matrix2.unsqueeze(0))
        top_half_tensor1 = result1[:, :4, :]
        bottom_half_tensor2 = result2[:, 4:, :]

        # 合并张量
        result = torch.cat((top_half_tensor1, bottom_half_tensor2), dim=1)

        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        params, pose = ctx.saved_tensors
        

        grad_params = None
        grad_pose = None
        pose1 = pose[:4,:]
        pose2 = pose[4:,:]
        grad_output1 = grad_output[:,:4,:]
        grad_output2 = grad_output[:,4:,:]
        
        if ctx.needs_input_grad[0]:
            # 计算params的梯度
            grad_tx1 = torch.sum(grad_output1[..., 0])
            grad_ty1 = torch.sum(grad_output1[..., 1])
            grad_sx1 = torch.sum(grad_output1[..., 0] * pose1[...,0])
            grad_sy1 = torch.sum(grad_output1[..., 1] * pose1[...,1])
            
            grad_tx2 = torch.sum(grad_output2[..., 0])
            grad_ty2 = torch.sum(grad_output2[..., 1])
            grad_sx2 = torch.sum(grad_output2[..., 0] * pose2[...,0])
            grad_sy2 = torch.sum(grad_output2[..., 1] * pose2[...,1])
            
            grad_params = torch.tensor([grad_tx1, grad_ty1, grad_sx1, grad_sy1,grad_tx2, grad_ty2, grad_sx2, grad_sy2]).cuda()
        # if(torch.sum(torch.abs(grad_params))>0):
        #     pdb.set_trace()
        
        return grad_params, None


def render_rec(glctx, pose, pos, pos_idx, col,col_idx, resolution: int):
    
    pos_transformed = CustomTransform.apply(pose, pos)
    number_layers = 1
    with nvdiffrast.torch.DepthPeeler(glctx, pos_transformed, pos_idx, [resolution,resolution]) as peeler:
        for i in range(number_layers):
            rast_out, rast_db = peeler.rasterize_next_layer()
    # rast_out, _ = dr.rasterize(glctx, pos_transformed, pos_idx, resolution=[resolution, resolution])
    color   , _ = dr.interpolate(col[None, ...], rast_out, col_idx)
    color_bak = color
    color       = dr.antialias(color, rast_out, pos_transformed, pos_idx)
    # pdb.set_trace()
    return color

#----------------------------------------------------------------------------
# Cube pose fitter.
#----------------------------------------------------------------------------

def fit_pose(max_iter           = 10000,
             repeats            = 1,
             log_interval       = 10,
             display_interval   = None,
             display_res        = 512,
             lr_base            = 0.01,
             lr_falloff         = 1.0,
             nr_base            = 1.0,
             nr_falloff         = 1e-4,
             grad_phase_start   = 0.1,
             resolution         = 256,
             out_dir            = None,
             log_fn             = None,
             mp4save_interval   = None,
             mp4save_fn         = None,
             use_opengl         = False):

    log_file = None
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if log_fn:
            log_file = open(out_dir + '/' + log_fn, 'wt')
        if mp4save_interval != 0:
            writer = imageio.get_writer(f'{out_dir}/{mp4save_fn}', mode='I', fps=30, codec='libx264', bitrate='16M')
    else:
        mp4save_interval = None

    #
    datadir = f'{pathlib.Path(__file__).absolute().parents[1]}/data'
    with np.load(f'{datadir}/cube_p.npz') as f:
        pos_idx, pos, col_idx, col = f.values()
    print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))

    # Some input geometry contains vertex positions in (N, 4) (with v[:,3]==1).  Drop
    # the last column in that case.
    if pos.shape[1] == 4: pos = pos[:, 0:3]

    # Create position/triangle index tensors
    # pos_idx = torch.from_numpy(pos_idx.astype(np.int32)).cuda()
    # vtx_pos = torch.from_numpy(pos.astype(np.float32)).cuda()
    # col_idx = torch.from_numpy(col_idx.astype(np.int32)).cuda()
    # vtx_col = torch.from_numpy(col.astype(np.float32)).cuda()

    
    #create a unit rectangle
    
    pos_idx = tensor([[0,1,2],[0,2,3],[4,6,7],[4,5,6]], dtype = torch.int32)
    vtx_pos = tensor([[-0.5,-0.5,0,1],[0.5,-0.5,0,1],[0.5,0.5,0,1],[-0.5,0.5,0,1],
                    [0,-1,0.1,1],[1,-1,0.1,1],[1,0,0.1,1],[0,0,0.1,1]],dtype=torch.float32)
    vtx_col = tensor([[1.0000, 0.0000, 1.0000],
        [1.0000, 1.0000, 0.0000],
        [0.0000, 0.0000, 1.0000],
        [0.0000, 1.0000, 0.0000],
        [1.0000, 0.0000, 0.0000],
        [1.0000, 0.6470, 0.0000]],dtype=torch.float32)
    col_idx = tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3]],dtype=torch.int32)
    
    
    
    
    
    glctx = dr.RasterizeGLContext() if use_opengl else dr.RasterizeCudaContext()

    for rep in range(repeats):
        pose_target = torch.tensor(np.asarray([-0.25, 0.25, 0.5, 0.5,-0.3,0.3,0.5,0.5], np.float32), device='cuda')
        # recPose_rnd()
        pose_init   = torch.tensor(np.asarray([-0.2, 0.3, 0.5, 0.5,-0.4,0.4,0.5,0.5], np.float32), device='cuda')
        pose_opt = torch.tensor(pose_init, dtype=torch.float32, device='cuda', requires_grad=True)
        

        loss_best   = np.inf
        pose_best   = pose_opt.detach().clone()

        
        # Adam optimizer for texture with a learning rate ramp.
        # optimizer = torch.optim.Adam([pose_opt], betas=(0.9, 0.999), lr=lr_base)
        optimizer = torch.optim.SGD([pose_opt], lr=lr_base, momentum=0.9)

        # Render.
        for it in range(max_iter + 1):
            # Set learning rate.
            itf = 1.0 * it / max_iter
            nr = nr_base * nr_falloff**itf
            lr = lr_base * lr_falloff**itf
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

          
            # Render.
            color          = render_rec(glctx, pose_target, vtx_pos, pos_idx, vtx_col,col_idx, resolution)
            
            color_opt      = render_rec(glctx, pose_opt, vtx_pos, pos_idx, vtx_col,col_idx, resolution)

            # Image-space loss.
            diff = (color_opt - color)**2 # L2 norm.
            diff = torch.tanh(10*torch.sum(diff, dim=-1))
            loss =torch.mean(diff)
            

            # Measure image-space loss and update best found pose.
            loss_val = float(loss)
            if (loss_val < loss_best) and (loss_val > 0.0):
                pose_best = pose_opt.detach().clone()
                loss_best = loss_val
                if itf < grad_phase_start:
                    with torch.no_grad(): pose_opt[:] = pose_best

            # Print/save log.
            if log_interval and (it % log_interval == 0):
                err = q_angle_deg(pose_opt, pose_target)
                ebest = q_angle_deg(pose_best, pose_target)
                s = "rep=%d,iter=%d,err=%f,err_best=%f,loss=%f,loss_best=%f,lr=%f,nr=%f" % (rep, it, err, ebest, loss_val, loss_best, lr, nr)
                print(s)
                if log_file:
                    log_file.write(s + "\n")

            # Run gradient training step.
            if itf >= grad_phase_start:
                optimizer.zero_grad()
            
                loss.backward()
                
                
                optimizer.step()

            # with torch.no_grad():
            #     pose_opt /= torch.abs(torch.max(pose_opt))
            
            # Show/save image.
            display_image = display_interval and (it % display_interval == 0)
            save_mp4      = mp4save_interval and (it % mp4save_interval == 0)
            
            if display_image or save_mp4:
                img_ref  = color[0].detach().cpu().numpy()
                img_opt  = color_opt[0].detach().cpu().numpy()
                img_best = render_rec(glctx, pose_best,  vtx_pos, pos_idx, vtx_col,col_idx, resolution)[0].detach().cpu().numpy()
                result_image = np.concatenate([img_ref, img_best, img_opt], axis=1)[::-1]

                if display_image:
                    util.display_image(result_image, size=display_res, title='(%d) %d / %d' % (rep, it, max_iter))
                if save_mp4:
                    writer.append_data(np.clip(np.rint(result_image*255.0), 0, 255).astype(np.uint8))

    # Done.
    if writer is not None:
        writer.close()
    if log_file:
        log_file.close()

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Cube pose fitting example')
    parser.add_argument('--opengl', help='enable OpenGL rendering', action='store_true', default=False)
    parser.add_argument('--outdir', help='specify output directory', default='')
    parser.add_argument('--display-interval', type=int, default=0)
    parser.add_argument('--mp4save-interval', type=int, default=3)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()

    # Set up logging.
    if args.outdir:
        out_dir = f'{args.outdir}/pose'
        print (f'Saving results under {out_dir}')
    else:
        out_dir = None
        print ('No output directory specified, not saving log or images')

    # Run.
    fit_pose(
        max_iter=args.max_iter,
        repeats=args.repeats,
        log_interval=100,
        display_interval=args.display_interval,
        out_dir=out_dir,
        log_fn='log.txt',
        mp4save_interval=args.mp4save_interval,
        mp4save_fn='progress.mp4',
        use_opengl=args.opengl
    )

    # Done.
    print("Done.")

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
