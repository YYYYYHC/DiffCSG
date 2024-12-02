
import torch
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
        