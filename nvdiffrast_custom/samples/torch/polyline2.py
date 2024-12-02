import torch
import pdb
import numpy as np
import open3d as o3d
class polyline: 
    def __init__(self, use_cuda = True, resolution=4, colors=None):
        self.resolution = resolution
        
        #generate two circles
        r = 0
        h = 1.0
        polyline = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=1, resolution=resolution, split=1)

        # 将生成的圆柱体移到原点
        polyline.translate([-polyline.get_center()[0], -polyline.get_center()[1], -polyline.get_center()[2]])
        if not polyline.has_vertex_normals():
            polyline.compute_vertex_normals()

        polyline.remove_duplicated_vertices()
        # 获取顶点数组（numpy格式）
        vertices = np.asarray(polyline.vertices).astype(np.float16)
        # 获取面的顶点索引（numpy格式）
        triangles = np.asarray(polyline.triangles).astype(np.float16)

        vertex_normals = np.asarray(polyline.vertex_normals).astype(np.float16)

        
        self.vtx_pos = torch.tensor(vertices,dtype=torch.float32)
        self.vtx_normal = torch.tensor(vertex_normals,dtype=torch.float32)
        #extend to 4D
        self.vtx_normal = torch.cat([self.vtx_normal,torch.zeros((self.vtx_normal.shape[0],1),dtype=torch.float32)],dim=1)
        self.pos_idx = torch.tensor(triangles, dtype=torch.int32)
        #color should be 4 * resolution
        if colors is not None:
            self.vtx_col = colors[0].repeat(self.vtx_pos.shape[0],1)
        else:
            self.vtx_col = torch.tensor([1,0,0],dtype=torch.float32).repeat(4*resolution*3,1)
        
        self.col_idx = self.pos_idx.clone()
        
        self.up_vtx_mask = torch.zeros_like(self.vtx_pos,dtype=torch.int32)
        self.up_vtx_mask[self.vtx_pos[:,2] > 0] = 1
        
        self.down_vtx_mask = torch.zeros_like(self.vtx_pos,dtype=torch.int32)
        self.down_vtx_mask[self.vtx_pos[:,2] < 0] = 1
        
        self.point_mask = torch.zeros_like(self.vtx_pos,dtype=torch.int32)
        # self.point_mask[]
        self.point_mask[[0,1],:] = -1
        
        #self.point_mask[i+2,:] = i
        self.point_mask[2:2+resolution,:] = torch.tensor([i for i in range(resolution)],dtype=torch.int32).unsqueeze(1).expand(-1, 3).clone()
        self.point_mask[2+resolution:2+2*resolution,:] = torch.tensor([i for i in range(resolution)],dtype=torch.int32).unsqueeze(1).expand(-1, 3).clone()
        
        self.point_mask = self.point_mask[:, 0].unsqueeze(1).expand(-1, self.point_mask.size(1)).clone()
        # pdb.set_trace()
        self.vtx_pos[:,:2] = 0.
        
        self.vtx_pos = torch.where(torch.abs(self.vtx_pos)!= 0.5, torch.zeros_like(self.vtx_pos), self.vtx_pos)
        
        
        # #TBD: get correct normal
        # self.vtx_normal[torch.where(self.vtx_normal[:,2] == 0)] = torch.tensor([0,0,0,0],dtype=torch.float32)
        
        if use_cuda:
            self.vtx_pos = self.vtx_pos.cuda()
            self.vtx_normal = self.vtx_normal.cuda()
            self.pos_idx = self.pos_idx.cuda()
            self.vtx_col = self.vtx_col.cuda()
            self.col_idx = self.col_idx.cuda()
            self.up_vtx_mask = self.up_vtx_mask.cuda()
            self.down_vtx_mask = self.down_vtx_mask.cuda()
            self.point_mask = self.point_mask.cuda()
        
        
        
        
if __name__ == '__main__':
    polyline()
    