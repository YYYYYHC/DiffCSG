import torch
import pdb
import numpy as np
# import open3d as o3d
class polyline: 
    def __init__(self, use_cuda = True, resolution=4, colors=None):
        self.resolution = resolution
        
        #generate two circles
        r = 0
        h = 1.0
        # polyline = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=1, resolution=resolution, split=1)

        # # 将生成的圆柱体移到原点
        # polyline.translate([-polyline.get_center()[0], -polyline.get_center()[1], -polyline.get_center()[2]])

        #-1 is the hook
        center_down= torch.tensor([-1,0,-0.5],dtype=torch.float32)
        center_up = torch.tensor([-1,0,0.5],dtype=torch.float32)
        theta = torch.linspace(0, resolution, resolution+1)[:-1]
        x = theta #hook
        y = r * torch.sin(theta) + center_down[1]
        z = torch.zeros_like(x) + center_down[2]


        down_circle = torch.stack([x,y,z],dim=1)
        up_circle = torch.stack([x,y,z+h],dim=1)
        #[cos(theta),sin(theta),0,0]
        down_circle_normal = torch.stack([torch.cos(theta),torch.sin(theta),torch.zeros_like(theta),torch.zeros_like(theta)],dim=1)
        up_circle_normal = torch.stack([torch.cos(theta),torch.sin(theta),torch.zeros_like(theta),torch.zeros_like(theta)],dim=1)
        
        
        # merge two circles
        points = torch.stack([down_circle, up_circle])
        points = points.permute(1,0,2)
        points = points.reshape(-1,3)
        
        normals = torch.stack([down_circle_normal, up_circle_normal])
        normals = normals.permute(1,0,2)
        normals = normals.reshape(-1,4)
        
        
        vtx_index = [(i//3 +i%3) % (resolution*2) for i in range(3 * resolution*2) ]
        
        points_lateral = points[vtx_index]
        normals_lateral = normals[vtx_index]
        
        #add cneter
        up_circle = torch.cat([up_circle,center_up.unsqueeze(0) ],dim=0)
        down_circle = torch.cat([down_circle,center_down.unsqueeze(0)],dim=0)
        #normal
        up_circle_normal = torch.cat([up_circle_normal,torch.tensor([0,0,1,0],dtype=torch.float32).unsqueeze(0)],dim=0)
        down_circle_normal = torch.cat([down_circle_normal,torch.tensor([0,0,-1,0],dtype=torch.float32).unsqueeze(0)],dim=0)
        
        circle_index = [element for i in range(resolution) for element in [i % resolution, (i + 1) % resolution, resolution]]

        points_up_circle = up_circle[circle_index]
        points_down_circle = down_circle[circle_index]
        
        normals_up_circle = up_circle_normal[circle_index]
        normals_down_circle = down_circle_normal[circle_index]

        num_lateral = points_lateral.shape[0]
        num_circle = points_up_circle.shape[0]
        self.vtx_pos = torch.cat([points_lateral, points_up_circle,points_down_circle],dim=0)
        self.vtx_normal = torch.cat([normals_lateral, normals_up_circle,normals_down_circle],dim=0)
        # debug = True
        # if debug:
        #     self.vtx_pos = torch.cat([points_up_circle, points_down_circle, points_lateral],dim=0)
        #     self.vtx_normal = torch.cat([normals_up_circle, normals_down_circle, normals_lateral],dim=0)
        
        self.pos_idx = torch.tensor([i for i in range(self.vtx_pos.shape[0])],dtype=torch.int32)
        self.pos_idx = self.pos_idx.reshape(-1,3)
        
        #color should be 4 * resolution
        if colors is not None:
            self.vtx_col = colors
        else:
            self.vtx_col = torch.tensor([1,0,0],dtype=torch.float32).repeat(4*resolution*3,1)
        #[[0,0,0],[1,1,1],...]
        self.col_idx = torch.tensor([[3*i,3*i+1,3*i+2] for i in range(self.pos_idx.shape[0])],dtype=torch.int32)
        
        self.up_vtx_mask = torch.zeros_like(self.vtx_pos,dtype=torch.int32)
        self.up_vtx_mask[self.vtx_pos[:,2] > 0] = 1
        
        self.down_vtx_mask = torch.zeros_like(self.vtx_pos,dtype=torch.int32)
        self.down_vtx_mask[self.vtx_pos[:,2] < 0] = 1
        
        self.point_mask = self.vtx_pos
        self.point_mask = self.point_mask[:, 0].unsqueeze(1).expand(-1, self.point_mask.size(1)).clone()
        
        # mask_indices = (self.point_mask[:, 0] == 0) & (self.point_mask[:, 1] == 0)

        # # 在mask中将这些行对应的位置设置为-1
        # mask[mask_indices] = -1
        
        self.vtx_pos = torch.where(torch.abs(self.vtx_pos)!= 0.5, torch.zeros_like(self.vtx_pos), self.vtx_pos)
        
        #TBD: get correct normal
        self.vtx_normal[torch.where(self.vtx_normal[:,2] == 0)] = torch.tensor([0,0,0,0],dtype=torch.float32)
        
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
    