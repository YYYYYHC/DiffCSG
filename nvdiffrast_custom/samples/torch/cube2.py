
import torch
import open3d as o3d
import numpy as np
class cube:
    #get the cube mesh, now it is a fixed cube
    def __init__(self, use_cuda = True, transform_mtx = None, points = None, colors = None, normals=None):
        
        cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
        cube.translate([-cube.get_center()[0], -cube.get_center()[1], -cube.get_center()[2]])
        if not cube.has_vertex_normals():
            cube.compute_vertex_normals()
            
        cube.remove_duplicated_vertices()
        
        # 获取顶点数组（numpy格式）
        vertices = np.asarray(cube.vertices).astype(np.float16)
        
        # 获取面的顶点索引（numpy格式）
        triangles = np.asarray(cube.triangles).astype(np.int32)
        
        vertex_normals = np.asarray(cube.vertex_normals).astype(np.float16)
        vertex_normals = np.concatenate([vertex_normals, np.zeros((vertex_normals.shape[0],1),dtype=np.float16)],axis=1)
        
        
            
        self.vtx_pos = torch.tensor(vertices,dtype=torch.float32)
        self.vtx_normal = torch.tensor(vertex_normals,dtype=torch.float32)
        self.pos_idx = torch.tensor(triangles, dtype=torch.int32)
        #color should be 4 * resolution
        if colors is not None:
            self.vtx_col = colors[0].repeat(self.vtx_pos.shape[0],1)
        else:
            self.vtx_col = torch.tensor([1,0,0],dtype=torch.float32).repeat(4*3,1)
        self.col_idx = self.pos_idx.clone()
        if use_cuda:
            self.vtx_pos = self.vtx_pos.cuda()
            self.pos_idx = self.pos_idx.cuda()
            self.vtx_col = self.vtx_col.cuda()
            self.col_idx = self.col_idx.cuda()
            self.vtx_normal = self.vtx_normal.cuda()
            