import torch
import pdb

def normalize(v, dim=None):
    """Normalize the vector."""
    if dim is not None:
        
        norm = torch.norm(v, dim=dim).unsqueeze(dim)
    else:
        norm = torch.norm(v)
    return v / norm

def midpoint(v1, v2):
    """Compute the midpoint of two vertices and normalize to the unit sphere."""
    return normalize((v1 + v2) / 2.0)

def tetrahedron_subdivision(vertices, triangles, n):
    """Perform n subdivisions on a tetrahedron, using lists to handle vertices."""
    for _ in range(n):
        new_triangles = []
        midpoints = {}  # Cache midpoints to avoid redundant computations.

        for tri in triangles:
            indices = []
            for edge in [(0, 1), (1, 2), (2, 0)]:
                edge_key = tuple(sorted([tri[edge[0]], tri[edge[1]]]))
                if edge_key not in midpoints:
                    mid_vertex = midpoint(torch.tensor(vertices[tri[edge[0]]]), torch.tensor(vertices[tri[edge[1]]]))
                    midpoints[edge_key] = len(vertices)
                    vertices.append(mid_vertex.tolist())
                indices.append(midpoints[edge_key])

            new_triangles += [[tri[0], indices[0], indices[2]],
                              [tri[1], indices[1], indices[0]],
                              [tri[2], indices[2], indices[1]],
                              [indices[0], indices[1], indices[2]]]

        triangles = new_triangles

    return vertices, triangles


class sphere:
    def __init__(self, use_cuda = True, transform_mtx = None, res = 2, colors = None, pure_color=None):
# Redefine initial vertices and faces, using lists.
        vertices_list = [
            normalize(torch.tensor([1., 1., 1.])).tolist(),
            normalize(torch.tensor([-1., -1., 1.])).tolist(),
            normalize(torch.tensor([-1., 1., -1.])).tolist(),
            normalize(torch.tensor([1., -1., -1.])).tolist()
        ]

        triangles_list = [
            [0, 1, 2],
            [0, 3, 1],
            [0, 2, 3],
            [1, 3, 2]
        ]

        # Perform the tetrahedron subdivision.
        n = res
        vertices_final, triangles_final = tetrahedron_subdivision(vertices_list, triangles_list, n)
        self.vtx_pos = torch.tensor(vertices_final)
        self.vtx_normal = normalize(self.vtx_pos, dim=-1)
        self.vtx_normal = torch.cat([self.vtx_normal, torch.zeros(self.vtx_normal.shape[0],1)], dim=1)
        self.pos_idx = torch.tensor(triangles_final, dtype=torch.int32)
        
        self.vtx_pos = self.vtx_pos[self.pos_idx].reshape(-1,3)
        self.vtx_normal = self.vtx_normal[self.pos_idx].reshape(-1,4)
        self.pos_idx = torch.tensor([i for i in range(self.vtx_pos.shape[0])],dtype=torch.int32)
        self.pos_idx = self.pos_idx.reshape(-1,3)
        
        # Generate triangle normals.
        # Convert back to PyTorch tensors for further processing.
        normals_final = torch.cross(self.vtx_pos[self.pos_idx[:, 1]] - self.vtx_pos[self.pos_idx[:, 0]],
                                    self.vtx_pos[self.pos_idx[:, 2]] - self.vtx_pos[self.pos_idx[:, 0]])
        normals_final = torch.stack([normalize(normal) for normal in normals_final])
        self.surface_normals = normals_final
        if colors is not None:
            self.vtx_col = colors
        else:
            
            colors = torch.rand(self.pos_idx.shape[0], 3)
            if pure_color is not None:
                
                colors[:]=pure_color
            colors = colors.unsqueeze(0)
            colors = colors.repeat(3,1,1)
            colors = colors.permute(1,0,2)
            colors = colors.reshape(-1,3)
            self.vtx_col = colors
        self.col_idx = torch.tensor([[3*i,3*i+1,3*i+2] for i in range(self.pos_idx.shape[0])],dtype=torch.int32)
        print("RES:",res, "Sphere vertices: ", self.vtx_pos.shape)
        if use_cuda:
            self.vtx_pos = self.vtx_pos.cuda()
            self.vtx_normal = self.vtx_normal.cuda()
            self.pos_idx = self.pos_idx.cuda()
            self.vtx_col = self.vtx_col.cuda()
            self.col_idx = self.col_idx.cuda()
            self.surface_normals = self.surface_normals.cuda()
            
    def get_intervals(self):
        vtx_interval = self.vtx_pos.shape[0]
        vtx_idx_interval = self.pos_idx.shape[0]
        return vtx_interval, vtx_idx_interval
        
if __name__ == "__main__":
    sphere = sphere()
    print(sphere.vtx_pos)
    print(sphere.vtx_normal)
    print(sphere.pos_idx)
    print(sphere.surface_normals)

    # cube_interval, cube_idx_interval = node.render_data['primitive'].get_intervals()