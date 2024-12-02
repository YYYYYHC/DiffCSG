# take CSG program as input and render in a differnetiabl way, all gradient are kept

#demo case1: render one cube
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
import time
import nvdiffrast
import sys
import numpy as np
import torch
import json
import imageio
from tqdm import tqdm
import util
import numpy as np
import torch
from parser import build_tree_from_file_properly
import nvdiffrast.torch as dr
from GetIntersectionLine import GetIntersectionLine, GetIntersectionLineBatch, PaintLine, PaintLineWithData, GetIntersectionPoint_2D, PaintLinesWithData, PaintLines
from camera import Camera, look_at, perspective
from cylinder import cylinder
from sphere import sphere
from polyline import polyline

#create a gl context
glctx = dr.RasterizeGLContext()
glctx_line = dr.RasterizeGLContext(use_lines=True)
        
RESOLUTION = 512
ANTIALIAS_TH = 0.5
time1 =None
time2 = None
CYLINDER_RES = 15
POLYLINE_RES = 6
SPHERE_RES = 3

def generate_random_signs_like(input_tensor):
    # 获取输入tensor的形状
    shape = input_tensor.shape
    # 生成一个随机的0和1的序列，形状与输入tensor相同
    random_bits = torch.randint(0, 2, shape).to(input_tensor.device) * 2 - 1  # 0变为-1，1保持为1
    return random_bits


def pytorch_add_a_fake_line(pos_clip, pos_idx, rast_out, line_buffer, lines):
    rast_out_copy = rast_out.clone()
    
    for line in lines:
        
        midpoint = (line[0] + line[1]) / 2
        # midpoint = line[0]
        fake_triangle = torch.cat([line[0].unsqueeze(0), line[1].unsqueeze(0), midpoint.unsqueeze(0)], dim=0)
        #add a column of 1 to the end of the tensor
        fake_triangle = torch.cat([fake_triangle, torch.ones([3,1],dtype=torch.float32,device=pos_clip.device)],dim=1)
        pos_clip = torch.cat([pos_clip, fake_triangle.unsqueeze(0)], dim=1)
        pos_idx = torch.cat([pos_idx, torch.tensor([[pos_clip.shape[1]-1,pos_clip.shape[1]-2,pos_clip.shape[1]-3]],dtype=torch.int32,device=pos_clip.device)], dim=0)
        
        #modify the rast_out
        touched_pixels = torch.where(torch.sum(torch.sum((line_buffer[:RESOLUTION,:RESOLUTION] - line)**2,dim=-1),dim=-1) < 1e-7)
        rast_out_copy[0,touched_pixels[0],touched_pixels[1],:] = torch.tensor([0.5,0.5,0.9,pos_idx.shape[0]],dtype=torch.float32,device=pos_clip.device)
        # pdb.set_trace()
        
    return pos_clip, pos_idx, rast_out_copy

def replace_nan_gradients(grad):
    """
    replace nan gradients with zeros
    """
    if torch.isnan(grad).any():
        grad = torch.where(torch.isnan(grad), torch.zeros_like(grad), grad)
    return grad

def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to cartesian coordinates.
    
    Parameters:
    - r: Radius
    - theta: Polar angle (in radians)
    - phi: Azimuthal angle (in radians)
    
    Returns:
    - Cartesian coordinates (x, y, z)
    """
    z = r * torch.sin(theta) * torch.cos(phi)
    x = r * torch.sin(theta) * torch.sin(phi)
    y = r * torch.cos(theta)
    
    return x, y, z


        


class cube:
    #get the cube mesh, now it is a fixed cube
    def __init__(self, use_cuda = True, transform_mtx = None, points = None, colors = None, normals=None):
        
        if points is None:
            points = torch.tensor(
            [[0,0,0],
            [-0.5000, -0.5000, -0.5],
            [-0.5000, -0.5000,  0.5],
            [-0.5000,  0.5000, -0.5],
            [-0.5000,  0.5000,  0.5],
            [ 0.5000, -0.5000, -0.5],
            [ 0.5000, -0.5000,    0.5],
            [ 0.5000,  0.5000, -0.5],
            [ 0.5000,  0.5000,  0.5]],dtype=torch.float32)
            # points = torch.tensor(
            # [[0,0,0],
            # [-0.5000, -0.5000, 0.],
            # [-0.5000, -0.5000,  1.],
            # [-0.5000,  0.5000, 0.],
            # [-0.5000,  0.5000,  1],
            # [ 0.5000, -0.5000,0.],
            # [ 0.5000, -0.5000,  1],
            # [ 0.5000,  0.5000, 0.],
            # [ 0.5000,  0.5000,  1]],dtype=torch.float32)
        if colors is None:
            colors = torch.tensor([
            [0,0,0],
            [0.5, 0.5, 0.5],
            [0., 0., 1.],
            [0., 1., 0.],
            [0., 1., 1.],
            [1., 0., 0.],
            [1., 0., 1.],
            [1., 1., 0.],
            [1., 1., 1.]],dtype=torch.float32)
            
        
        self.vtx_pos = torch.vstack([points[1],points[2],points[3],points[4],
                                     points[1], points[2], points[5], points[6],
                                     points[5], points[6], points[7], points[8],
                                     points[7], points[8], points[3], points[4],
                                     points[2], points[4], points[6], points[8],
                                     points[1], points[3], points[5], points[7]])
        self.vtx_col = torch.vstack([colors[2],colors[2],colors[2],colors[2],
                                     colors[3], colors[3], colors[3], colors[3],
                                     colors[4], colors[4], colors[4], colors[4],
                                     colors[1], colors[1], colors[1], colors[1],
                                     colors[5], colors[5], colors[5], colors[5],
                                     colors[6], colors[6], colors[6], colors[6]])
        if normals is None:
            normals = torch.tensor([
            [0,0,0,0],
            [-1, -1, -1,0],
            [-1, -1, 1.,0],
            [-1, 1., -1,0],
            [-1, 1., 1.,0],
            [1., -1, -1,0],
            [1., -1, 1.,0],
            [1., 1., -1,0],
            [1., 1., 1.,0]],dtype=torch.float32)/np.sqrt(3)
            
            self.vtx_normal = torch.tensor([[-1,0,0,0],[-1,0,0,0],[-1,0,0,0],[-1,0,0,0],
                                            [0,-1,0,0],[0,-1,0,0],[0,-1,0,0],[0,-1,0,0],
                                            [1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
                                            [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
                                            [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],
                                            [0,0,-1,0],[0,0,-1,0],[0,0,-1,0],[0,0,-1,0]],dtype=torch.float32)
            
            # self.vtx_normal = torch.vstack([normals[1],normals[2],normals[3],normals[4],
            #                          normals[1], normals[2], normals[5], normals[6],
            #                          normals[5], normals[6], normals[7], normals[8],
            #                          normals[7], normals[8], normals[3], normals[4],
            #                          normals[2], normals[4], normals[6], normals[8],
            #                          normals[1], normals[3], normals[5], normals[7]])
            
            means = self.vtx_normal.view(-1,4,4).mean(dim=1)
            self.surface_normal = means.repeat(1,4).view(-1,4)
            self.surface_normal /= torch.max(self.surface_normal)
            
        if transform_mtx is not None:
            print('error: not transform normal')
            transform_mtx = transform_mtx.astype(np.float32)
            self.vtx_pos = transform_pos(transform_mtx, self.vtx_pos.cuda())
            #remove the last dimension
            self.vtx_pos = self.vtx_pos[0,:,:3]
        self.pos_idx = torch.tensor(
        [[0, 1, 2],
        [1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
        [8, 9, 10],
        [9, 10, 11],
        [12, 13, 14],
        [13, 14, 15],
        [16, 17, 18],
        [17, 18, 19],
        [20, 21, 22],
        [21, 22, 23]],dtype=torch.int32)
        
        
        
        col_idx2 = torch.tensor(
        [[0, 1, 2],
        [1, 2, 3],
        [4, 5, 6],
        [5, 6, 7],
        [8, 9, 10],
        [9, 10, 11],
        [12, 13, 14],
        [13, 14, 15],
        [16, 17, 18],
        [17, 18, 19],
        [20, 21, 22],
        [21, 22, 23]],dtype=torch.int32)
        self.col_idx = col_idx2
        if use_cuda:
            self.vtx_pos = self.vtx_pos.cuda()
            self.pos_idx = self.pos_idx.cuda()
            self.vtx_col = self.vtx_col.cuda()
            self.col_idx = self.col_idx.cuda()
            self.vtx_normal = self.vtx_normal.cuda()
            self.surface_normal = self.surface_normal.cuda()

    def merge_with(self, cubex):
        #merge the current cube with cubex, the result will be stored in the current cube
        
        #merge the pos_idx
        self.pos_idx = torch.cat([self.pos_idx,cubex.pos_idx+self.vtx_pos.shape[0]],dim=0)
        #merge the col_idx
        self.col_idx = torch.cat([self.col_idx,cubex.col_idx+self.vtx_col.shape[0]],dim=0)
        #merge the vertex position
        self.vtx_pos = torch.cat([self.vtx_pos,cubex.vtx_pos],dim=0)
        #merge the vertex color
        self.vtx_col = torch.cat([self.vtx_col,cubex.vtx_col],dim=0)
        #merge the vertex normal
        self.vtx_normal = torch.cat([self.vtx_normal,cubex.vtx_normal],dim=0)
        #merge the surface normal
        self.surface_normal = torch.cat([self.surface_normal,cubex.surface_normal],dim=0)
        
def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    # pdb.set_trace()
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

import torch.nn.functional as F

def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int, blur=False):
    if mtx is None:
        mtx = torch.eye(4).cuda()

    pos_clip = transform_pos(mtx, pos)
    rast_out, _ = dr.rasterize(glctx, pos_clip, pos_idx, resolution=[resolution, resolution])
    #get mask of trigangle idx
    mask = rast_out[..., 3]
    color, _ = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
    color = dr.antialias(color, rast_out, pos_clip, pos_idx)   

    return color,mask
def pytorch_get_objmask_from_buffer(buffer):
    cube1_mask = (buffer == 1)
    cube2_mask = (buffer == 2)

    cube1_count = cube1_mask.cumsum(dim=0)
    cube2_count = cube2_mask.cumsum(dim=0)

    return_mask = torch.zeros_like(buffer)
    return_mask[(buffer == 0) | (cube1_mask & (cube2_count % 2 == 1)) | (cube2_mask & (cube1_count % 2 == 0))] = 1

    return return_mask
def get_objmask_from_buffer(buffer):
    cube1_count = 0
    cube2_count = 0
    return_mask = torch.zeros_like(buffer)
    
    for id, objidx in enumerate(buffer):
        #check states
        if objidx == 0:
            return_mask[id] = 1
            return return_mask
        if objidx == 1:
            if cube2_count % 2 == 1:
                return_mask[id] = 1
                return return_mask
            else:
                cube1_count += 1
        if objidx == 2:
            if cube1_count % 2 == 0:
                return_mask[id] = 1
                return return_mask
            else:
                cube2_count += 1
    return 'error'
def pytorch_euler_to_mat(x=0, y=0, z=0):
    # Convert Euler angles to rotation matrix
    cos, sin = torch.cos, torch.sin
    
    x = x * np.pi / 180
    y = y * np.pi / 180
    z = z * np.pi / 180
    
    rotation_x = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    rotation_x[0, 0] = 1
    rotation_x[1, 1] = cos(x)
    rotation_x[1, 2] = -sin(x)
    rotation_x[2, 1] = sin(x)
    rotation_x[2, 2] = cos(x)
    

    # rotation_y = torch.tensor([[cy, 0, sy],
    #                            [0, 1, 0],
    #                            [-sy, 0, cy]])
    rotation_y = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    rotation_y[0, 0] = cos(y)
    rotation_y[0, 2] = sin(y)
    rotation_y[1, 1] = 1
    rotation_y[2, 0] = -sin(y)
    rotation_y[2, 2] = cos(y)
    
    
    # rotation_z = torch.tensor([[cz, -sz, 0],
    #                            [sz, cz, 0],
    #                            [0, 0, 1]])
    rotation_z = torch.zeros(3, 3, device=x.device, dtype=x.dtype)
    rotation_z[0, 0] = cos(z)
    rotation_z[0, 1] = -sin(z)
    rotation_z[1, 0] = sin(z)
    rotation_z[1, 1] = cos(z)
    rotation_z[2, 2] = 1
    
    rotation_matrix = torch.mm(rotation_x, torch.mm(rotation_y, rotation_z))
    # Extend to 4x4
    rotation_matrix = torch.cat((rotation_matrix, torch.zeros((1, 3)).cuda()), dim=0)
    rotation_matrix = torch.cat((rotation_matrix, torch.zeros((4, 1)).cuda()), dim=1)
    rotation_matrix[3, 3] = 1
    # Convert dtype to float32
    
    return rotation_matrix
def pytorch_translate(x, y, z):
    translate = torch.zeros(4,4,device=x.device, dtype=x.dtype)
    translate[0,0] = 1
    translate[1,1] = 1
    translate[2,2] = 1
    translate[3,3] = 1
    translate[0,3] = x
    translate[1,3] = y
    translate[2,3] = z
    return translate
    
class TransformCubes(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, pose):
        # 存储输入以供反向传播使用
        ctx.save_for_backward(params, pose)
        
        # 提取参数
        transform_matrix1 = params[:4, :]
        transform_matrix2 = params[4:, :]

        # 乘在pose的右边
        result1 = torch.matmul(pose, transform_matrix1.unsqueeze(0))
        result2 = torch.matmul(pose, transform_matrix2.unsqueeze(0))
        
        top_half_tensor1 = result1[:, :24, :]
        bottom_half_tensor2 = result2[:, 24:, :]

        # 合并张量
        result = torch.cat((top_half_tensor1, bottom_half_tensor2), dim=1)

        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        params, pose = ctx.saved_tensors

        grad_params = None
        grad_pose = None
        
        pose1 = pose[:24,:]
        pose2 = pose[24:,:]
        grad_output1 = grad_output[:,:24,:]
        grad_output2 = grad_output[:,24:,:]

        if ctx.needs_input_grad[0]:
            # Compute gradients for params
            grad_params = torch.zeros_like(params)
            grad_params[:3] = torch.sum(grad_output1[:, :3, :], dim=(0, 1))
            grad_params[3:6] = torch.sum(grad_output1[:, 3:6, :], dim=(0, 1))
            grad_params[6:9] = torch.sum(grad_output1[:, 6:9, :], dim=(0, 1))
            grad_params[9:12] = torch.sum(grad_output2[:, :3, :], dim=(0, 1))
            grad_params[12:15] = torch.sum(grad_output2[:, 3:6, :], dim=(0, 1))
            grad_params[15:18] = torch.sum(grad_output2[:, 6:9, :], dim=(0, 1))

        # Compute gradients for pose
        grad_pose = torch.zeros_like(pose)
        grad_pose[:24, :] = grad_output1[:, :, :]
        grad_pose[24:, :] = grad_output2[:, :, :]

        return grad_params, grad_pose
def transform_pos_and_normal(transform_mtx, pos, normal):
    rx,ry,rz,tx,ty,tz,sx,sy,sz= transform_mtx
    r_rot = pytorch_euler_to_mat(rx,ry,rz)
    trans = pytorch_translate(tx, ty, tz)
    scale = pytorch_scale_to_mat(sx,sy,sz)
    transform_matrix = torch.matmul(torch.matmul(trans,r_rot),scale)
    noraml = torch.matmul(normal, r_rot.t())
    pos = transform_pos(transform_matrix, pos)
    return pos, normal

def goldfeather_merge_buffer(rast_out_merged_1, rast_out_merged_2, color_1, color_2, op):
    #left: out1, right: out2
    if op == '+':
        print('union')
        pdb.set_trace()
    if op == '-':
        print('subtraction')
        pdb.set_trace()
    if op == '*':
        print('intersection')
        pdb.set_trace()
    
    return None, None



def goldfeather_render_tree(glctx, tree, resolution, variables, camera_view=None, render_intersection_line = True, requires_grad=False,global_transform=None, compute_intersection=True):    
    
    def get_pos_clip(node, poses_clip, names, pos_idxs, vtx_cols,vtx_normals_clip, col_idxs, operators, primitive_count,polyline_res, variables):
        if node.type == 'primitive':
            names.append(node.name)
            pos = node.render_data['primitive'].vtx_pos
            polyline_res_item=0
            if 'cube' in node.name:
                cube_interval = 24
                cube_idx_interval = 12
            elif 'cylinder' in node.name:
                cube_interval = 12 * CYLINDER_RES
                cube_idx_interval = 4 * CYLINDER_RES
                up_vtx_mask = node.render_data['primitive'].up_vtx_mask
                down_vtx_mask = node.render_data['primitive'].down_vtx_mask
            elif 'polyline' in node.name:
                # pdb.set_trace()
                polyline_res_item = (len(node.data) - 10)//2
                cube_interval = 12 * polyline_res_item
                point_mask = node.render_data['primitive'].point_mask
                cube_idx_interval = 4 * polyline_res_item
            elif 'sphere' in node.name:
                cube_interval = SPHERE_INTERVAL
                cube_idx_interval = SPHERE_IDX_INTERVAL
            polyline_res.append(polyline_res_item)
            
            pos_idxs.append(node.render_data['primitive'].pos_idx + primitive_count[0]) #this is only for cube
            vtx_cols.append(node.render_data['primitive'].vtx_col)
            col_idxs.append(node.render_data['primitive'].col_idx + primitive_count[0]) #this is only for cube
            primitive_count[0] = primitive_count[0] + cube_interval
            vtx_normal = node.render_data['primitive'].vtx_normal
            #parse node data
            node_data_lt = []
            
            for i in range(len(node.data)):
                item = node.data[i]
                if type(item) == str:
                    g = {'variables':variables}
                    item_data = eval(item,g)
                    
                    if not requires_grad:
                        item_data = item_data.detach()
                    node_data_lt.append(item_data)  
                else:
                    node_data_lt.append(item)  
                
            
            if not 'transform_mtx' in node.render_data.keys():
                
                transform_mtx = torch.stack(node_data_lt)
                #if use cyliner, should modify here, construct different transfrom mtx 
                
            else:
                transform_mtx = node.render_data['transform_mtx']
            #if use cyliner, should modify here, the transformation is different 
            if 'cube' in node.name or 'sphere' in node.name:
                centering=1
                if len(transform_mtx) == 9:
                    rx,ry,rz,tx,ty,tz,sx,sy,sz= transform_mtx
                elif len(transform_mtx) == 10:
                    rx,ry,rz,tx,ty,tz,sx,sy,sz, centering = transform_mtx
                    
                if centering==0:
                    pos = pos + torch.tensor([0,0,0.5],dtype=torch.float32,device=pos.device)
                if centering==2:
                    pos = pos + torch.tensor([0.5,0.5,0.5],dtype=torch.float32,device=pos.device)
                
                r_rot = pytorch_euler_to_mat(rx, ry, rz)
                trans = pytorch_translate(tx, ty, tz)
                scale = pytorch_scale_to_mat(sx, sy, sz)
                if centering==999:
                    transform_matrix = torch.matmul(torch.matmul(r_rot,trans),scale)
                else:
                    transform_matrix = torch.matmul(torch.matmul(trans,r_rot),scale)
                    
                if global_transform is not None:
                    transform_matrix = torch.matmul(global_transform, transform_matrix)
                node_vtx_normal = torch.matmul(vtx_normal, r_rot.t())
                if global_transform is not None:
                    node_vtx_normal = torch.matmul(node_vtx_normal, global_transform.t())
                node_pose = transform_pos(transform_matrix, pos)
                poses_clip.append(node_pose)
                vtx_normals_clip.append(node_vtx_normal)
            elif 'cylinder' in node.name:
                centering=1
                if len(transform_mtx) == 11:
                    rx,ry,rz,tx,ty,tz,sx,sy,sz, r_up, r_down= transform_mtx
                elif len(transform_mtx) == 12:
                    rx,ry,rz,tx,ty,tz,sx,sy,sz, r_up, r_down, centering= transform_mtx
                
                up_transform = torch.eye(3,device=rx.device)
                up_transform[0,0] = r_up
                up_transform[1,1] = r_up
                
                down_transform = torch.eye(3,device=rx.device)
                down_transform[0,0] = r_down
                down_transform[1,1] = r_down
                
                
                pos = torch.matmul(pos * up_vtx_mask, up_transform.t()) + torch.matmul(pos * down_vtx_mask, down_transform.t())
                if centering==0:
                    # pdb.set_trace()
                    pos = pos + torch.tensor([0,0,0.5],dtype=torch.float32,device=pos.device)
                
                
                
                # pdb.set_trace()
                r_rot = pytorch_euler_to_mat(rx, ry, rz)
                trans = pytorch_translate(tx, ty, tz)
                scale = pytorch_scale_to_mat(sx, sy, sz)
                transform_matrix = torch.matmul(torch.matmul(trans,r_rot),scale)
                if global_transform is not None:
                    transform_matrix = torch.matmul(global_transform, transform_matrix)
                node_vtx_normal = torch.matmul(vtx_normal, r_rot.t())
                if global_transform is not None:
                    node_vtx_normal = torch.matmul(node_vtx_normal, global_transform.t())
                node_pose = transform_pos(transform_matrix, pos)
                
                poses_clip.append(node_pose)
                vtx_normals_clip.append(node_vtx_normal)
            elif 'polyline' in node.name:
                rx,ry,rz,tx,ty,tz,sx,sy,sz = transform_mtx[-10: -1]
                
                
                
                h = transform_mtx[-1]
                n = (transform_mtx.shape[0] - 10)//2
                mx = 0
                my = 0
                for polyline_idx in range(n):
                    x = transform_mtx[2*polyline_idx]
                    
                    y = transform_mtx[2*polyline_idx+1]
                    mx += x
                    my += y
                for polyline_idx in range(n):
                    mask_idx = (point_mask == polyline_idx)
                    x = transform_mtx[2*polyline_idx]
                    y = transform_mtx[2*polyline_idx+1]
                    xn = (x-mx/n) / torch.sqrt((x-mx/n)**2 + (y-my/n)**2 + 1e-8)
                    yn = (y-my/n) / torch.sqrt((x-mx/n)**2 + (y-my/n)**2 + 1e-8)

                    pos = torch.where(mask_idx, pos + torch.stack([x,y,torch.zeros_like(x)],dim=-1), pos)
                    
                    mask_idx_normal = torch.cat((mask_idx, mask_idx[:,-1].unsqueeze(1)), dim=-1)
                    
                    vtx_normal = torch.where(mask_idx_normal, vtx_normal +torch.stack([xn,yn,torch.zeros_like(xn),torch.zeros_like(yn)],dim=-1) , vtx_normal)
                # pdb.set_trace()
                midx = (point_mask == -1)
                pos = torch.where(midx, pos + torch.stack([mx/n, my/n, torch.zeros_like(mx)],dim=-1), pos)
                
                ##debug the gradient problem
                # t= torch.sum(pos)
                
                
                # t.backward(retain_graph=True)
                
                # for i,e in enumerate(transform_mtx):
                #     print(len(transform_mtx)-i)
                #     print(torch.autograd.grad(t,e))
                
                # pdb.set_trace()
                r_rot = pytorch_euler_to_mat(rx, ry, rz)
                trans = pytorch_translate(tx, ty, tz)
                scale = pytorch_scale_to_mat(sx, sy, sz * h)
                transform_matrix = torch.matmul(torch.matmul(trans,r_rot),scale)
                node_vtx_normal = torch.matmul(vtx_normal, r_rot.t())
                node_pose = transform_pos(transform_matrix, pos)
                
                poses_clip.append(node_pose)
                vtx_normals_clip.append(node_vtx_normal)
                
                
        else:
            #add to the start of operators.
            operators.insert(0, node.name)
            get_pos_clip(node.left, poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res, variables)
            get_pos_clip(node.right, poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res, variables)
            
    #devide sub product based on '+'
    def devide_sub_product(node, poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt, vtx_normals_clip_lt, col_idxs_lt, operators_lt,primitive_counts_lt, polyline_res_lt, variables):
        if node.type == 'operator':
            
            if node.name == '+':
                devide_sub_product(node.left, poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt,vtx_normals_clip_lt, col_idxs_lt, operators_lt,primitive_counts_lt, polyline_res_lt, variables)
                devide_sub_product(node.right, poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt, vtx_normals_clip_lt, col_idxs_lt, operators_lt,primitive_counts_lt, polyline_res_lt, variables)
            else:
                poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res = [], [], [], [], [], [], [], [0], []
                get_pos_clip(node, poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res, variables)
                poses_clip_lt.append(poses_clip)
                names_lt.append(names)
                pos_idxs_lt.append(pos_idxs)
                vtx_cols_lt.append(vtx_cols)
                vtx_normals_clip_lt.append(vtx_normals_clip)
                col_idxs_lt.append(col_idxs)
                operators_lt.append(operators)
                primitive_counts_lt.append(primitive_count)
                polyline_res_lt.append(polyline_res)
        else:
            poses_clip, names, pos_idxs, vtx_cols,vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res = [], [], [], [], [], [], [], [0], []
            get_pos_clip(node, poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count,polyline_res, variables)
            poses_clip_lt.append(poses_clip)
            names_lt.append(names)
            pos_idxs_lt.append(pos_idxs)
            vtx_cols_lt.append(vtx_cols)
            vtx_normals_clip_lt.append(vtx_normals_clip)
            col_idxs_lt.append(col_idxs)
            operators_lt.append(operators)
            primitive_counts_lt.append(primitive_count)
            polyline_res_lt.append(polyline_res)
    
    time_pre_processing_start= time.time()
    poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt,vtx_normals_clip_lt, col_idxs_lt, operators_lt, primitive_counts_lt, polyline_res_lt= [], [], [], [], [], [],[], [], []
    devide_sub_product(tree, poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt, vtx_normals_clip_lt, col_idxs_lt, operators_lt, primitive_counts_lt, polyline_res_lt, variables)
    #get product shift
    product_interval_lt = []
    product_idx_interval_lt = []
    for names, polyline_res in zip(names_lt, polyline_res_lt):
        interval = 0
        idx_interval = 0
        
        for name, polyline_res_item in zip(names, polyline_res):
            
            if 'cube' in name:
                cube_interval = 24
                cube_idx_interval = 12
            elif 'cylinder' in name:
                cube_interval = 12 * CYLINDER_RES
                cube_idx_interval = 4 * CYLINDER_RES
            elif 'polyline' in name:
                cube_interval = 12 * polyline_res_item
                cube_idx_interval = 4 * polyline_res_item
            elif 'sphere' in name:
                cube_interval = SPHERE_INTERVAL
                cube_idx_interval = SPHERE_IDX_INTERVAL
            
            interval += cube_interval
            idx_interval += cube_idx_interval
        product_interval_lt.append(interval)
        product_idx_interval_lt.append(idx_interval)
        
                
    
    
    #get intervals from names_lt
    
    def merge_lt_into_tensor(lt):
        if len(lt) == 1:
            return lt[0]
        if lt[0].shape[0] == 1:
            return torch.cat(lt, dim=1)
        else:
            return torch.cat(lt, dim=0)
    
    #render all sub product
    final_rast_out = torch.zeros([1, resolution, resolution, 4], dtype=torch.float32, device=poses_clip_lt[0][0].device) + 10000
    final_pos_clip = []
    final_vtx_normals_clip = []
    final_pos_idxs = []
    final_vtx_cols = []
    final_col_idxs = []
    final_pos = []
    product_id = 0
    product_interval = 0
    product_idx_interval = 0
    time_pre_processing_end = time.time()
    for poses_clip, names, pos_idxs, vtx_cols, vtx_normals_clip, col_idxs, operators, primitive_count, polyline_res in zip(poses_clip_lt, names_lt, pos_idxs_lt, vtx_cols_lt, vtx_normals_clip_lt, col_idxs_lt, operators_lt, primitive_counts_lt, polyline_res_lt):
        #merge all data
        poses_clip = torch.cat(poses_clip, dim=1)
        vtx_normals_clip = torch.cat(vtx_normals_clip, dim=0).unsqueeze(0)
        
        pos_idxs = torch.cat(pos_idxs, dim=0)
        vtx_cols = torch.cat(vtx_cols, dim=0).unsqueeze(0)
        col_idxs = torch.cat(col_idxs, dim=0)

        #projection
        
        final_pos.append(poses_clip)
        
        poses_clip = torch.matmul(poses_clip, camera_view.unsqueeze(0))
        poses_clip = poses_clip / poses_clip[..., 3:]
        final_pos_clip.append(poses_clip)
        
        final_vtx_normals_clip.append(vtx_normals_clip)
        final_vtx_cols.append(vtx_cols)
        
        final_pos_idxs.append(pos_idxs + product_interval)
        final_col_idxs.append(col_idxs + product_interval)
            
        def get_pos_by_idx(primitive_idx):
            previous_interval = 0
            previous_idx_interval = 0
            current_cube_interval =0
            current_cube_idx_interval = 0
            for i in range(primitive_idx):
                name = names[i]
                polyline_res_item = polyline_res[i]
                if 'cube' in name:
                    cube_interval = 24
                    cube_idx_interval = 12
                elif 'cylinder' in name:
                    cube_interval = 12 * CYLINDER_RES
                    cube_idx_interval = 4 * CYLINDER_RES
                elif 'polyline' in name:
                    cube_interval = 12 * polyline_res_item
                    cube_idx_interval = 4 * polyline_res_item
                elif 'sphere' in name:
                    cube_interval = SPHERE_INTERVAL
                    cube_idx_interval = SPHERE_IDX_INTERVAL
                previous_interval += cube_interval
                previous_idx_interval += cube_idx_interval
            currentname = names[primitive_idx]
            current_polyline_res_item = polyline_res[primitive_idx]
            if 'cube' in currentname:
                current_cube_interval = 24
                current_cube_idx_interval = 12
            elif 'cylinder' in currentname:
                current_cube_interval = 12 * CYLINDER_RES
                current_cube_idx_interval = 4 * CYLINDER_RES
            elif 'polyline' in currentname:
                current_cube_interval = 12 * current_polyline_res_item
                current_cube_idx_interval = 4 * current_polyline_res_item
            elif 'sphere' in currentname:
                current_cube_interval = SPHERE_INTERVAL
                current_cube_idx_interval = SPHERE_IDX_INTERVAL
            
            return previous_interval, previous_idx_interval, poses_clip[:,previous_interval:previous_interval+current_cube_interval,:], pos_idxs[previous_idx_interval:previous_idx_interval+current_cube_idx_interval,:] - previous_interval
        
        #goldfeather render
        output_rast_out = torch.zeros([1, resolution, resolution, 4], dtype=torch.float32, device=poses_clip.device) + 10000
        for primitive_idx in range(len(names)):
            #if the primitive is substracted, draw back of A to rast_out
            previous_interval, previous_idx_interval, pos_clip, pos_idx = get_pos_by_idx(primitive_idx)
            # pos_clip = poses_clip[:,primitive_idx*cube_interval:(primitive_idx+1)*cube_interval,:]
            # pos_idx = pos_idxs[primitive_idx*cube_idx_interval:(primitive_idx+1)*cube_idx_interval,:] - primitive_idx * cube_interval
            if primitive_idx>0 and operators[primitive_idx-1] == '-':
                
                with nvdiffrast.torch.DepthPeeler(glctx, pos_clip, pos_idx, [resolution,resolution]) as peeler:
                    for i in range(2):
                        rast_out, _ = peeler.rasterize_next_layer()
                    
            else:
                #if the primitive is not substracted, draw front of A to rast_out
                with nvdiffrast.torch.DepthPeeler(glctx, pos_clip, pos_idx, [resolution,resolution]) as peeler:
                    for i in range(1):
                        rast_out, _ = peeler.rasterize_next_layer()
            shape_mask = torch.zeros_like(rast_out)
            shape_mask[rast_out[..., 3]>0] +=1
            
            rast_out_offset_mask = torch.zeros_like(rast_out)
            rast_out_offset_mask[..., 3] = (rast_out[...,3] >0)
            
            rast_out = torch.where(rast_out_offset_mask > 0, rast_out + previous_idx_interval, rast_out)
            
            #for each other primitive in the product, test visibility
            for primitive_idx_2 in range(len(names)):
                if primitive_idx_2 == primitive_idx:
                    continue
                #render two faces of the primitive2
                previous_interval, previous_idx_interval, pos_clip_2, pos_idx_2 = get_pos_by_idx(primitive_idx_2)
                # pos_clip_2 = poses_clip[:,primitive_idx_2*cube_interval:(primitive_idx_2+1)*cube_interval,:]
                # pos_idx_2 = pos_idxs[primitive_idx_2*cube_idx_interval:(primitive_idx_2+1)*cube_idx_interval,:]- primitive_idx_2 * cube_interval
                with nvdiffrast.torch.DepthPeeler(glctx, pos_clip_2, pos_idx_2, [resolution,resolution]) as peeler:
                    for i in range(2):
                        if i == 0:
                            rast_out_front, _ = peeler.rasterize_next_layer()
                        else:
                            rast_out_back, _ = peeler.rasterize_next_layer()
                
                z_primitive_1 = rast_out[..., 2]
                z_primitive_2_front = rast_out_front[..., 2]
                z_primitive_2_back = rast_out_back[..., 2]
                obj_mask = torch.zeros_like(rast_out) #parity check mask
                
                
                obj_mask[z_primitive_1 > z_primitive_2_front] += 1
                obj_mask[z_primitive_1 > z_primitive_2_back] += 1
                
                 
                #if substracted, accept even parity
                if primitive_idx_2 >0 and operators[primitive_idx_2-1] == '-':
                    rast_out = torch.where((shape_mask>0) & (obj_mask %2 ==1), rast_out*0, rast_out)
                    
                #if not substracted, accept odd parity
                else:
                    rast_out = torch.where((shape_mask>0) & (obj_mask %2 == 0) , rast_out*0, rast_out)
                
            #draw rast_out to output_rast_out with z-less
            rast_out = torch.where(rast_out == 0, torch.zeros_like(rast_out) + 10000, rast_out)
            
            output_rast_out = torch.where(output_rast_out == 0, torch.zeros_like(output_rast_out) + 10000, output_rast_out)
            zmask = (rast_out[..., 2] < output_rast_out[..., 2]).unsqueeze(-1).repeat(1,1,1,4)
            
            output_rast_out = torch.where(zmask, rast_out, output_rast_out)

        output_rast_out_offset_mask = torch.zeros_like(output_rast_out)
        output_rast_out_offset_mask[..., 3] = (output_rast_out[...,3] >0) & (output_rast_out[...,3] < 10000)
        output_rast_out = torch.where(output_rast_out_offset_mask > 0, output_rast_out +  product_idx_interval, output_rast_out)
        
        
        
        # output_rast_out = torch.where(output_rast_out == 0, torch.zeros_like(output_rast_out) + 10000, output_rast_out)
        final_zmask = (output_rast_out[..., 2] < final_rast_out[..., 2]).unsqueeze(-1).repeat(1,1,1,4)
        
        
        
        final_rast_out = torch.where(final_zmask, output_rast_out, final_rast_out)
        product_interval += product_interval_lt[product_id]
        product_idx_interval += product_idx_interval_lt[product_id]
        product_id +=1
        
    
    final_rast_out = torch.where(final_rast_out == 10000, 0*final_rast_out, final_rast_out)
    final_vtx_cols_ = merge_lt_into_tensor(final_vtx_cols)
    final_pos_idxs_ = merge_lt_into_tensor(final_pos_idxs)
    final_col_idxs_ = merge_lt_into_tensor(final_col_idxs)
    final_pos_clip = merge_lt_into_tensor(final_pos_clip)
    final_pos = merge_lt_into_tensor(final_pos)
    final_vtx_normals_clip_ = merge_lt_into_tensor(final_vtx_normals_clip)
    vtx_attr = torch.cat([final_vtx_cols_, final_vtx_normals_clip_], dim=2)
    
    time_gf_end = time.time()
    color_and_normal, _ = dr.interpolate(vtx_attr, final_rast_out, final_col_idxs_)
    
    
    time_interpolate_end = time.time()
    
    if compute_intersection:
        all_triangles = final_pos.squeeze(0)[final_pos_idxs_[:]][...,:-1]
        
        # all_triangles = all_triangles[:200]
        
        line_buffer = torch.zeros(resolution+1, resolution+1, 2, 3, device=poses_clip.device, dtype=poses_clip.dtype)
        N = all_triangles.shape[0]
        #construct two triangle matrix
        triangles_expanded_1 = all_triangles.unsqueeze(1).expand(-1, N, -1, -1)
        triangles_expanded_2 = all_triangles.unsqueeze(0).expand(N, -1, -1, -1)
        
        #only preserve the upper triangle
        triangles_expanded_1 = triangles_expanded_1[torch.where(torch.triu(torch.ones(N, N), diagonal=1)==1)]
        triangles_expanded_2 = triangles_expanded_2[torch.where(torch.triu(torch.ones(N, N), diagonal=1)==1)]
        comparison = (triangles_expanded_1.unsqueeze(2) == triangles_expanded_2.unsqueeze(1))
        has_common_vertex = comparison.any(dim=-1).any(dim=-1).any(dim=-1)
        triangles_expanded_1 = triangles_expanded_1[~has_common_vertex]
        triangles_expanded_2 = triangles_expanded_2[~has_common_vertex]

        ips = GetIntersectionLineBatch(triangles_expanded_1, triangles_expanded_2)
        # poses_clip = torch.matmul(poses_clip, camera_view.unsqueeze(0))
        #     poses_clip = poses_clip / poses_clip[..., 3:]
        # # 把每个点转换成齐次坐标，即在最后一个维度加上1
        ips = torch.cat([ips, torch.ones(ips.shape[0], 2, 1).to(ips.device)], dim=-1)  # 维度变为[1574, 2, 4]
        # 应用投影矩阵
        projected_points = torch.matmul(ips, camera_view)  # 注意转置矩阵以匹配形状

        # 执行透视除法，即用最后一个坐标除以前两个坐标
        ips = projected_points / projected_points[..., 3:4]


        #获取line rasterization的输入
        ips_reshape = projected_points.reshape(1,-1,4)
        
        #construct fake triangles
        midpoints = (ips[:,0,:] + ips[:,1,:])/2
        fake_triangles = torch.cat([ips,midpoints.unsqueeze(1)],dim=1)
        fake_triangles_count = fake_triangles.shape[0]
        ips = ips[...,:3]
        # line_buffer = PaintLinesWithData(resolution, ips)
        
        ips_idx = torch.tensor([range(ips_reshape.shape[1])], dtype=torch.int32).cuda().reshape(-1,2)
        
        
        
        ttt=time.time()
        
        if ips_reshape.shape[1] == 0:
            line_buffer_rast = torch.zeros([1, resolution, resolution, 4], dtype=torch.float32, device=poses_clip.device)
        else:
            line_buffer_rast, _ = dr.rasterize(glctx_line, ips_reshape,ips_idx, [resolution, resolution],grad_db=False)
    
    
        
        
        origin_pos_num = final_pos_clip.shape[1]
        origin_posid_num = final_pos_idxs_.shape[0]
        
        final_pos_clip = torch.cat([final_pos_clip, fake_triangles.reshape(-1,4).unsqueeze(0)],dim=1)
        
        update_final_pos_idxs_ = torch.tensor(range(fake_triangles_count*3),dtype=torch.int32,device=final_pos_idxs_[0].device).reshape(-1,3) + origin_pos_num
        # pdb.set_trace()
        final_pos_idxs_ = torch.cat([final_pos_idxs_, update_final_pos_idxs_],dim=0)
        
        #update rast out, the rast_out idx was offset by 1, so here should offset back
        
        mask_rast_out = (line_buffer_rast[...,3]>0).unsqueeze(-1).repeat(1,1,1,4)
        
        final_rast_out = torch.where(mask_rast_out, torch.stack([torch.zeros_like(line_buffer_rast[...,3]).cuda()+0.5, \
                                                                    torch.zeros_like(line_buffer_rast[...,3]).cuda()+0.5, \
                                                                    torch.zeros_like(line_buffer_rast[...,3]).cuda()+0.9, \
                                                                line_buffer_rast[...,3]+origin_posid_num],dim=-1), final_rast_out)
    
    time_intersection_compute_end = time.time()
    # print(time3-ts1, ts1-ts)
    # pdb.set_trace()
    # print('intersection time:', (t1-time2)/(time3-time2),(t2-t1)/(time3-time2), (t3-t2)/(time3-time2), (time3-t3)/(time3-time2))
    # print('intersection time:', time3-time2)
    # pdb.set_trace()
    color_and_normal= dr.antialias(color_and_normal, final_rast_out, final_pos_clip, final_pos_idxs_)
    time_antialias_end = time.time()
    # diff_color_and_normal = torch.where(torch.sum(torch.abs(color_and_normal_new - color_and_normal).squeeze(0),dim=-1)>0)
    # diff_img = torch.zeros([resolution, resolution, 3], dtype=torch.int32, device=poses_clip.device)
    # diff_img[diff_color_and_normal] = torch.tensor([128,128,128],dtype=torch.int32, device=poses_clip.device)
    #save diff_img
    #
    
    # imageio.imwrite('diff_img.png', diff_img.cpu().numpy().astype(np.uint8))
    
    
    color = color_and_normal[..., :3].contiguous()
    normal = color_and_normal[..., 3:6].contiguous()
    
    # with open(TIME_CSV, 'a') as f:
        
    #     f.write(str(time_pre_processing_end - time_pre_processing_start) + ',' + str(time_gf_end - time_pre_processing_end) + ',' + str(time_interpolate_end - time_gf_end) + ',' + str(time_intersection_compute_end - time_interpolate_end) + ',' + str(time_antialias_end - time_intersection_compute_end) + '\n')
    
    # color, _ = dr.interpolate(vtx_cols, output_rast_out, col_idxs)
    
    
    # color = dr.antialias(color, output_rast_out, poses_clip, pos_idxs)
    #render
    mask = None
    
    return color, mask, normal

def compute_edge_from_idx_mask(idx_mask):
    # Compute the binary mask based on idx_mask
    kernel = torch.tensor([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

    # Pad the idx_mask tensor
    padn = 1
    idx_mask_padded = torch.nn.functional.pad(idx_mask, (padn, padn, padn, padn), mode='constant', value=0)

    # Apply the convolution using the kernel
    binary_mask = torch.nn.functional.conv2d(idx_mask_padded.unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0).float(), padding=0)
    binary_mask = binary_mask.squeeze(0).squeeze(0)
    binary_mask = torch.abs(binary_mask/4 - idx_mask)
    
    # Set the values greater than 0 to 0 and the values less than 0 to 1
    binary_mask = torch.where(binary_mask > 0, torch.tensor(0), torch.tensor(1))

    return binary_mask
def euler_to_mat(x=0, y=0, z=0):
    # Convert Euler angles to rotation matrix
    cx = np.cos(x)
    sx = np.sin(x)
    cy = np.cos(y)
    sy = np.sin(y)
    cz = np.cos(z)
    sz = np.sin(z)

    rotation_x = np.array([[1, 0, 0],
                           [0, cx, -sx],
                           [0, sx, cx]])

    rotation_y = np.array([[cy, 0, sy],
                           [0, 1, 0],
                           [-sy, 0, cy]])

    rotation_z = np.array([[cz, -sz, 0],
                           [sz, cz, 0],
                           [0, 0, 1]])

    rotation_matrix = np.dot(rotation_x, np.dot(rotation_y, rotation_z))
    # Extend to 4x4
    rotation_matrix = np.concatenate((rotation_matrix, np.zeros((1, 3))), axis=0)
    rotation_matrix = np.concatenate((rotation_matrix, np.zeros((4, 1))), axis=1)
    rotation_matrix[3, 3] = 1
    # Convert dtype to float32
    rotation_matrix = rotation_matrix.astype(np.float32)
    return rotation_matrix

def scale_to_mat(scalex,sacley,scalez):
    #convert scale to scale matrix
    scale_matrix = np.array([[scalex, 0, 0,0],
                           [0, sacley, 0,0],
                           [0, 0, scalez,0],
                           [0, 0, 0,1]])
    return scale_matrix

def pytorch_scale_to_mat(scalex,sacley,scalez):
    #convert scale to scale matrix
    scale_matrix = torch.zeros(4,4,device=scalez.device, dtype=scalez.dtype)
    scale_matrix[0,0] = scalex
    scale_matrix[1,1] = sacley
    scale_matrix[2,2] = scalez
    scale_matrix[3,3] = 1
    
    return scale_matrix

def load_program(path, variables, require_grad, all_params=False, pertub_weight=0):
    if all_params:
        param_to_optimize = []
    # class TreeNode:
    # def __init__(self, type, name, data=None):
    #     self.type = type  # "operator" or "primitive"
    #     self.name = name
        # self.render_data = None
    #     self.data = data
    #     self.left = None
    #     self.right = None

    csg_tree = build_tree_from_file_properly(path, variables, require_grad, all_param=all_params)
    #generate primitives, colors, and transform matrix
    #travel the tree: if it is a leaf, generate a primitive, color, and transform matrix
    def gen_shape(node ):
        if node.type == 'primitive':
            #generate a primitive
            #generate a random color
            
            #load the transform matrix
            node.render_data = {}
            if all_params:
                
                transform_mtx = torch.stack(node.data)
                
                transform_mtx = transform_mtx.detach()
                #pertube transform mtx
                if pertub_weight > 0:
                    
                    mean=0
                    std_dev=pertub_weight
                    random_variable = torch.normal(mean, std_dev, size=(1,)).item()
                    # random_variable.item()  # 返回Python标准数据类型的随机数值
                    transform_mtx = transform_mtx* (1 + random_variable)
                node.render_data['transform_mtx'] = transform_mtx
                if require_grad:
                    node.render_data['transform_mtx'].requires_grad = True
                
                param_to_optimize.append(node.render_data['transform_mtx'])
            
            if 'cube' in node.name:
                colors = torch.rand([9,3],dtype=torch.float32).cuda()
                # #81a2be to rgb
                
                # colors = torch.tensor([129, 162, 190]).repeat(9,1).float()/255
                
                
                node.render_data['primitive'] = cube(transform_mtx=np.eye(4), colors=colors)
            elif 'cylinder' in node.name:
                resolution=CYLINDER_RES
                # #debug:2, release:4
                colors = torch.rand([4*resolution,3],dtype=torch.float32).cuda()
                colors = colors.unsqueeze(0)
                colors = colors.repeat(3,1,1)
                colors = colors.permute(1,0,2)
                colors = colors.reshape(-1,3)
                
                node.render_data['primitive'] = cylinder(resolution=resolution, colors=colors)
                # node.render_data['transform_mtx'] = transform_mtx
                # if variables is None and require_grad:
                #     transform_mtx.requires_grad = True
                #     param_to_optimize.append(transform_mtx)
            elif 'polyline' in node.name:
                polyline_res_item = (len(node.data) - 10)//2
                resolution=polyline_res_item
                
                colors = torch.rand([4*resolution,3],dtype=torch.float32).cuda()
                colors = colors.unsqueeze(0)
                colors = colors.repeat(3,1,1)
                colors = colors.permute(1,0,2)
                colors = colors.reshape(-1,3)
                node.render_data['primitive'] = polyline(resolution=resolution, colors=colors)
                # node.render_data['transform_mtx'] = transform_mtx
                # if variables is None and require_grad:
                #     transform_mtx.requires_grad = True
                #     param_to_optimize.append(transform_mtx)
            elif 'sphere' in node.name:
                res = SPHERE_RES
                # pure_color=torch.tensor([222, 147, 95])/255
                node.render_data['primitive'] = sphere(res=res)
                cube_interval, cube_idx_interval = node.render_data['primitive'].get_intervals()
                global SPHERE_INTERVAL
                global SPHERE_IDX_INTERVAL
                SPHERE_INTERVAL = cube_interval
                SPHERE_IDX_INTERVAL = cube_idx_interval
        else:
            if node.left is not None:
                gen_shape(node.left)
            if node.right is not None:
                gen_shape(node.right)
    gen_shape(csg_tree)
    
    if all_params:

        return csg_tree, param_to_optimize
    return csg_tree

def CSGDR(USE_ALL_PARAM=False,
          SPECIFICTEST=False,
          specific_test_source=None, 
          specific_test_code=None,
          compute_intersection=False,
          use_random=True,
          scale_level='0'):
    torch.autograd.set_detect_anomaly(True)
    SCALE={'0':0.05, '1':0.1, '2':0.2, '3':0.3} #[-0.1,+0.1], [-0.2,+0.2], [-0.4,+0.4]
    #test with a surface normal target
    print('compute intersection:', compute_intersection)    
    print('using random', use_random)
    print('scale level:', scale_level)
    print('use all param:', USE_ALL_PARAM)
    USE_RANDOM = True
    TUNE_SCALE =SCALE[scale_level]
    MAX_ITER = 1000
    PATIENCE = 50
    TEST_TIMES = 2
    # USE_ALL_PARAM = False
    CODEBASE_PATH = './CodeBase/benchmark/'
    # assert compute_intersection == False
    
    if USE_ALL_PARAM:
        TESTNAME = f'rev_benchmark_test_allparam_{TUNE_SCALE}_{MAX_ITER}'
    else:
        TESTNAME = f'rev_benchmark_test_hyperparam_{TUNE_SCALE}_{MAX_ITER}'
    TESTDIR = f'./exp/{TESTNAME}'
    res_store = f'./exp/{TESTNAME}/res.csv'
    os.system(f'rm -rf {res_store}')
    # SPECIFICTEST = True
    # specific_test_source = 'CADTalk'
    # specific_test_code = 'bike'
    # torch.manual_seed(1101)
    gtensor = torch.tensor([-90,0,0]).cuda()
    global_transform = pytorch_euler_to_mat(gtensor[0],gtensor[1],gtensor[2])
    
    from tqdm import tqdm as tqmd
    for codeSource in ['CADTalk', 'fusion', 'random']:
        if SPECIFICTEST:
            if codeSource != specific_test_source:
                continue
        codes_path = os.path.join(CODEBASE_PATH, codeSource)
        for filename in tqmd(os.listdir(codes_path)):
            if SPECIFICTEST:
                if filename != specific_test_code:
                    continue
            try:
                for times in range(TEST_TIMES):
                    print(f'processing: {codeSource} --- {filename}, times:{times}')
                    
                    datafrom = codeSource
                    dataid = filename
                    variable_path = f"./CodeBase/benchmark/{datafrom}/{dataid}/source.param"
                    Prog_path= f"./CodeBase/benchmark/{datafrom}/{dataid}/object.txt"
                    target_variable_path = f"./CodeBase/benchmark/{datafrom}/{dataid}/target.param"
                    config_path = f"./CodeBase/benchmark/{datafrom}/{dataid}/config.json"
                    if os.path.exists(config_path):
                        config = json.load(open(config_path))
                    else:
                        config = {'lr':1e-2, 'stop':5e-4, 'param_ignore':[], 'lr_all':1e-3}
                    lr = config['lr']
                    if USE_ALL_PARAM:
                        lr = config['lr_all']
                    stop = config['stop']
                    ignore = config['param_ignore']
                    current_test_dir = os.path.join(TESTDIR, codeSource,filename,f't_{times}')
                    os.makedirs(current_test_dir, exist_ok=True)
                    
                    variables = {}
                    with open(variable_path, 'r') as f:
                        for line in f:
                            name, value = line.strip().split('=')
                            
                            variables[name.strip()] = torch.tensor(float(value.strip()),dtype=torch.float32).cuda()
                            if name.strip() == 's' or name.strip() == 'scale':
                                continue
                            variables[name.strip()].requires_grad = False                    

                    globals().update(variables)
                    param_to_optimize = []
                    SELECTA = False
                    optimize_somethins = ['wheel_dia','spokes_length']
                    
                    
                    
                    
                    target_variables = {}
                    variables_count = len(variables)
                    size = (variables_count,)
                    if use_random:
                        random_numbers = torch.torch.normal(0, TUNE_SCALE, size=size)
                    else:
                        random_numbers = torch.tensor(np.load(f'{current_test_dir}/random_numbers.npy')).cuda()
                    
                    np.save(f'{current_test_dir}/random_numbers.npy', random_numbers.cpu().numpy())
                    with open(os.path.join(current_test_dir, 'rn.txt'), 'w') as f:
                        f.write(str(random_numbers.cpu().numpy()))
                    
                    for i,key in enumerate(variables):
                        if key == 's' or key == 'scale':
                            target_variables[key] = variables[key].clone().detach()
                            continue
                        
                        random_number = random_numbers[i]
                        if key in ignore:
                            random_number = 0
                        target_variables[key] = variables[key].clone()*(1 +random_number)
                        target_variables[key].requires_grad = True
                    
                    for key in variables:
                        
                        if SELECTA and not key in optimize_somethins:
                            continue
                        if key == 's' or key == 'scale':
                            continue
                        if key in ignore:
                            continue
                        param_to_optimize.append(target_variables[key])
                    
                    r = torch.tensor(93.0) # Radius
                    r = torch.tensor(103.0) # Radius for polyline
                    theta = torch.tensor(-torch.pi*0.8) # Polar angle, 45 degrees in radians
                    phi = torch.tensor(-torch.pi/1.5) # Azimuthal angle, 45 degrees in radians

                    # theta = torch.tensor(-torch.pi*0.8) # Polar angle, 45 degrees in radians
                    # phi = torch.tensor(-0) # Azimuthal angle, 45 degrees in radians

                    #man
                    if dataid == 'man':
                        theta = torch.tensor(torch.pi/3) # Polar angle, 45 degrees in radians
                        phi = torch.tensor(-0*torch.pi/1.8) # Azimuthal angle, 45 degrees in radians
                    else:
                        theta = torch.tensor(-torch.pi*0.8) # Polar angle, 45 degrees in radians
                        phi = torch.tensor(-torch.pi/1.5) # Azimuthal angle, 45 degrees in radians

                    camera_position = spherical_to_cartesian(r, theta, phi)
                    
                    target_point = [0, 0, 0]
                    up_vector = [0, 0, -1]
                    fov = 22.5  # Field of view
                    aspect_ratio =1  # Aspect ratio
                    near_plane = 0.1  # Near clipping plane
                    far_plane = 1000 # Far clipping plane
                    
                    view_matrix = look_at(camera_position, target_point, up_vector)
                    # pdb.set_trace()
                    projection_matrix = perspective(fov, aspect_ratio, near_plane, far_plane)
                    camera_view = torch.matmul(view_matrix.t(),projection_matrix.t()).cuda()
                
                    
                    os.makedirs(current_test_dir, exist_ok=True)

                    #load shapes from program
                    if USE_ALL_PARAM:
                        csg_tree, source_param = load_program(Prog_path, variables, require_grad=False,all_params=True)
                        
                    else:
                        
                        csg_tree = load_program(Prog_path, variables, require_grad=False)            
                    if USE_ALL_PARAM:
                        csg_tree_target, param_to_optimize = load_program(Prog_path, target_variables, require_grad=True,all_params=True, pertub_weight=0)
                        # np.save(f'{current_test_dir}/param_to_optimize.npy', [param_to_optimize[i].detach().cpu().numpy() for i in range(len(param_to_optimize))])
                        with open(os.path.join(current_test_dir, 'param_to_optimize.txt'), 'w') as f:
                            for param in param_to_optimize:
                                f.write(str(param))
                                f.write('\n')
                    else:
                        csg_tree_target = load_program(Prog_path, target_variables, require_grad=True)
                    # pdb.set_trace()
                    
                    
                    # color, mask, normal = goldfeather_render(glctx, None, cube_obj1.vtx_pos, cube_obj1.pos_idx, cube_obj1.vtx_col, cube_obj1.col_idx, 512, transform_cube= transform_cube, vtx_normal=cube_obj1.vtx_normal, camera_view=camera_view)
                    
                    time1 = time.time()
                    
                    color, mask, normal = goldfeather_render_tree(glctx, csg_tree, 512,variables, camera_view=camera_view, requires_grad=False, compute_intersection=compute_intersection)
                    time2 = time.time()
                    #save it to image
                    color_ = (color.detach().cpu().numpy()[0]*255).astype(np.uint8)
                    # normal_ = normal*0.5 + 0.5
                    normal_ = (normal.detach().cpu().numpy()[0]*255)
                    
                    normal_ = (normal_ - normal_.min())/2
                    normal_ = normal_.astype(np.uint8)[:,:,:3]
                    imageio.imwrite(f'{current_test_dir}/initial.png', color_)
                    imageio.imwrite(f'{current_test_dir}/initial_normal.png', normal_)
                    
                    
                    # target_color, mask, target_normal = goldfeather_render(glctx, None, target_cube_obj1.vtx_pos, target_cube_obj1.pos_idx, target_cube_obj1.vtx_col, target_cube_obj1.col_idx, 512, transform_cube= transform_cube_target, vtx_normal=target_cube_obj1.vtx_normal,camera_view=camera_view)
                    target_color, mask, target_normal = goldfeather_render_tree(glctx, csg_tree_target, 512,target_variables, camera_view=camera_view,requires_grad=True, compute_intersection=compute_intersection)
                    #save it to image
                    target_color_ = (target_color.detach().cpu().numpy()[0]*255).astype(np.uint8)
                    #render normal
                    target_normal_ = (target_normal.detach().cpu().numpy()[0]*255)
                    target_normal_ = (target_normal_ - target_normal_.min())/2
                    target_normal_ = target_normal_.astype(np.uint8)[:,:,:3]
                    
                    imageio.imwrite(f'{current_test_dir}/target.png', target_color_)
                    imageio.imwrite(f'{current_test_dir}/target_normal.png', target_normal_)
                    
                    optimizer = torch.optim.Adam(param_to_optimize, lr=lr)
                    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: max(1e-4, 10**(-x*0.0005)))
                    writer = imageio.get_writer(f'{current_test_dir}/resultdemo5.mp4', fps=20)
                    writer_compare  = imageio.get_writer(f'{current_test_dir}/resultdemo5_compare.mp4', fps=20)
                    writers_all = [imageio.get_writer('./debug/grads_all'+str(i)+'.mp4', fps=20) for i in range(18)]
                    writers_positive = [imageio.get_writer('./debug/grads_positive'+str(i)+'.mp4', fps=20) for i in range(18)]
                    writers_negative = [imageio.get_writer('./debug/grads_negative'+str(i)+'.mp4', fps=20) for i in range(18)]
                    # exit(0)
                    log_dir = f'{current_test_dir}/log.txt'
                    #clear the log dir
                    with open(log_dir, 'w') as f:
                        f.write('')
                    timeStart = time.time()
                    loss_10 = 0
                    last_loss=0
                    trigger_times=0
                    for it in range(MAX_ITER):
                        if it == MAX_ITER-1:
                            timeEnd = time.time()
                            with open(res_store, 'a') as f:
                                print(f'fail in {timeEnd} seconds\n')
                                f.write(f'{codeSource},{dataid},{timeEnd-timeStart}\n')
                            break
                        if USE_RANDOM:
                            if it % 5 == 0:
                                
                                #random theta_ and phi_
                                theta_ = torch.rand(1)*torch.pi
                                phi_ = torch.rand(1)*2*torch.pi
                                camera_position_random = spherical_to_cartesian(r, theta_, phi_)
                                camera_view_random = look_at(camera_position_random, target_point, up_vector)
                                camera_view_random = torch.matmul(camera_view_random.t(),projection_matrix.t()).cuda()
                                color, mask, normal = goldfeather_render_tree(glctx, csg_tree, 512, variables, camera_view=camera_view_random, requires_grad=False, compute_intersection=compute_intersection)
                    
                                target_color, mask, target_normal = goldfeather_render_tree(glctx, csg_tree_target, 512,target_variables,  camera_view=camera_view_random, requires_grad=True,compute_intersection=compute_intersection)

                                
                            else:
                                target_color, mask, target_normal = goldfeather_render_tree(glctx, csg_tree_target, 512,target_variables,  camera_view=camera_view_random, requires_grad=True, compute_intersection=compute_intersection)
                    
                    
                        else:
                            target_color, mask, target_normal = goldfeather_render_tree(glctx, csg_tree_target, 512,target_variables,  camera_view=camera_view, requires_grad=True,compute_intersection=compute_intersection)
                            # target_color, target_mask, target_normal = goldfeather_render_tree(glctx, csg_tree_target, 512, camera_view=camera_view)
                        
                        # print("sum color:"  ,torch.sum(color))
                        diffcolor = (target_color[:,:,:,:3] - color[:,:,:,:3])**2 # L2 norm.
                        diffnormal =  (target_normal[:,:,:,:3] - normal[:,:,:,:3])**2 # L2 norm.
                        diff = diffnormal
                        diff = torch.tanh(10*torch.sum(diff, dim=-1))
                        loss =torch.mean(diff)
                        
                        optimizer.zero_grad()
                        # pdb.set_trace()
                        loss.backward()
                        
                        optimizer.step()
                        
                        if  it % 10 == 0:
                            #render a fixed view for saving 
                            
                            color_, mask_, normal_2 = goldfeather_render_tree(glctx, csg_tree, 512,variables,  camera_view=camera_view, requires_grad=False, compute_intersection=compute_intersection)
                            target_color_2, mask_, target_normal_2 = goldfeather_render_tree(glctx, csg_tree_target, 512, target_variables, camera_view=camera_view,requires_grad=False,compute_intersection=compute_intersection)

                            
                            # target_color_, mask_, target_normal_ = goldfeather_render_tree(glctx, csg_tree_target, 512, camera_view=camera_view)
                            diffnormal =  (target_normal_2[:,:,:,:3] - normal_2[:,:,:,:3])**2 # L2 norm.
                            diffcolor = (target_color_2[:,:,:,:3] - color_[:,:,:,:3])**2 # L2 norm.
                            diff = diffnormal
                            loss_true =torch.mean(diff)
                            
                        
                            
                            with open(f'{current_test_dir}/log.txt', 'a') as f:
                                f.write(f'it:{it}, loss:{loss.item()}, loss_true:{loss_true.item()}\n')
                                #save current transform_cube to txt
                                f.write(f'transform_cube:{param_to_optimize}\n')
                            # if it % 10 == 0 and it > 0:
                            # loss_10 = loss_10/10
                            # loss_to_save = loss_10
                            
                        
                            if it % 100 == 0:
                                color_save = (color_.detach().cpu().numpy()[0]*255).astype(np.uint8)
                                writer.append_data(color_save)
                                color_save_compare = color_save.copy()
                                #save compare result
                                mask = np.where(np.sum(np.abs(color_save_compare-target_color_),axis=-1)>0)
                                # pdb.set_trace(  )
                                
                                color_save_compare[mask[0], mask[1]] =0.5 * target_color_[mask[0], mask[1]] + 0.5 * color_save_compare[mask[0], mask[1]]

                                normal_save = (target_normal_2.detach().cpu().numpy()[0]*255)
                                normal_save = (normal_save - normal_save.min())/2
                                normal_save = normal_save.astype(np.uint8)[:,:,:3]
                                writer_compare.append_data(np.concatenate((normal_save, normal_), axis=1))
                            if loss_true < stop:
                                timeEnd = time.time()
                                with open(res_store, 'a') as f:
                                    print(f'success in {timeEnd - timeStart} seconds\n')
                                    f.write(f'{codeSource},{dataid},{timeEnd-timeStart}\n')
                                    f.write(f'result: {param_to_optimize}\n')
                                    if USE_ALL_PARAM:
                                        f.write(f'target: {source_param}\n')
                                    else:
                                        f.write(f'target: {variables}\n')
                                break
                        
                
            except Exception as e:
                print(e)
                # 捕获其他所有类型的异常
                with open(f'./exp/{TESTNAME}/error.log', 'a+') as f:
                    f.write(f'error in {codeSource} --- {f}\n')
                continue  # 明确指出继续下一次迭代
import argparse
if __name__ == "__main__":
    #args
    
    #SPECIFICTEST=False,
        #   specific_test_source=None, 
        #   specific_test_code=None
    parser = argparse.ArgumentParser()
    parser.add_argument('--SPECIFICTEST', default=False,action='store_true')
    parser.add_argument('--specific_test_source', type=str, default=None)
    parser.add_argument('--specific_test_code', type=str, default=None)
    parser.add_argument('--compute_intersection', default=False, action='store_true')
    parser.add_argument('--use_random', default=False, action='store_true')
    parser.add_argument('--scale_level', type=str, default='0')
    parser.add_argument('--USE_ALL_PARAM', default=False, action='store_true')
    
    CSGDR(**vars(parser.parse_args()))