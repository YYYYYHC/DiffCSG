import torch
import pdb
import math
import torch.nn.functional as F


def look_at(camera_position, target_point, up_vector):
    camera_position = torch.tensor(camera_position, dtype=torch.float32)
    target_point = torch.tensor(target_point, dtype=torch.float32)
    up_vector = torch.tensor(up_vector, dtype=torch.float32)

    # Compute the forward vector from target to camera position
    forward = camera_position - target_point
    forward /= torch.linalg.norm(forward)
    
    # Compute the right vector
    right = torch.cross(up_vector, forward)
    right /= torch.linalg.norm(right)
    
    # Recompute the up vector to ensure orthogonality
    up = torch.cross(forward, right)
    
    # Create the view matrix
    view_matrix = torch.zeros((4, 4), dtype=torch.float32)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up
    view_matrix[2, :3] = forward
    view_matrix[:3, 3] = -torch.matmul(torch.stack([right, up, forward]), camera_position.unsqueeze(-1)).squeeze()
    view_matrix[3, 3] = 1.0
    
    return view_matrix

def perspective(fov, aspect, near, far):
    # scale = 1 / torch.tan(torch.radians(torch.tensor(fov, dtype=torch.float32)) / 2)
    scale = 1 / torch.tan(torch.tensor(fov, dtype=torch.float32) * 0.017453292519943295/ 2) 
    # Create the perspective projection matrix
    projection_matrix = torch.zeros((4, 4), dtype=torch.float32)
    projection_matrix[0, 0] = scale / aspect
    projection_matrix[1, 1] = scale
    projection_matrix[2, 2] = -(far + near) / (far - near)
    projection_matrix[2, 3] = -2 * far * near / (far - near)
    projection_matrix[3, 2] = -1
    
    return projection_matrix

class Camera:
    def __init__(self, position, target, up, fov=90, aspect_ratio=16/9, near=0.1, far=1000):
        self.position = position
        self.target = target
        self.up = up
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far

    
    def get_projection_matrix(self):
        aspect_ratio = self.aspect_ratio
        fov = self.fov
        near = self.near
        far = self.far
        return torch.tensor([
            [1 / (aspect_ratio * math.tan(fov / 2)), 0, 0, 0],
            [0, 1 / math.tan(fov / 2), 0, 0],
            [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
            [0, 0, -1, 0]
        ], dtype=torch.float32)
    
    def get_view_matrix(self):
        
        forward = F.normalize(self.target.unsqueeze(0)- self.position.unsqueeze(0))
        side = F.normalize(torch.cross(forward, self.up.unsqueeze(0)))
        up = torch.cross(side, forward)
        forward = forward.squeeze(0)
        side = side.squeeze(0)
        up = up.squeeze(0)
        return torch.tensor([
            [side[0], side[1], side[2], -torch.dot(side, self.position)],
            [up[0], up[1], up[2], -torch.dot(up, self.position)],
            [-forward[0], -forward[1], -forward[2], torch.dot(forward, self.position)],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
    
    def get_view_projection_matrix(self):
        return torch.matmul(self.get_view_matrix().t(), self.get_projection_matrix().t())