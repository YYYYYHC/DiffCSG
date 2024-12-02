import torch
import pdb
import time


def PaintLinesWithData(resolution, IntersectionPoints):
    assert IntersectionPoints.shape[1] == 2
    buffer = torch.zeros(resolution+1, resolution+1, 2, 3, device = IntersectionPoints.device)
    for i in range(IntersectionPoints.shape[0]):
        buffer = torch.where(buffer == 0, PaintLineWithData(resolution, IntersectionPoints[i]), buffer)
    return buffer


def PaintLineWithData(resolution, IntersectionPoint):
    assert IntersectionPoint.shape[0] == 2
    
    buffer = torch.zeros(resolution+1, resolution+1, 2, 3, device = IntersectionPoint.device)
    
    pointA, pointB = IntersectionPoint.detach()
    pointA = pointA[:2]
    pointB = pointB[:2]
    pointA = (pointA + 1) * resolution / 2
    pointB = (pointB + 1) * resolution / 2
    pointA = pointA.int()
    pointB = pointB.int()
    # Bresenham's line algorithm
    x0, y0 = pointA
    x1, y1 = pointB
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    # print("dx dy", dx, dy)
    if dx==0 and dy==0:
        return buffer
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while True:
        if y0 >= 0 and y0 <= resolution and x0 >= 0 and x0 <= resolution:
            buffer[y0, x0] = IntersectionPoint
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            err = err + dx
            y0 = y0 + sy
    return buffer
def PaintLines(resolution, IntersectionPoints):
    img = torch.zeros(resolution+1, resolution+1,3, device = IntersectionPoints.device)
    for i in range(IntersectionPoints.shape[0]):
        img = torch.where(img==0, PaintLine(resolution, IntersectionPoints[i]), img)
    return img

def PaintLine(resolution, IntersectionPoint):
    img = torch.zeros(resolution+1, resolution+1,3, device = IntersectionPoint.device)
    
    pointA, pointB = IntersectionPoint
    pointA = pointA[:2]
    pointB = pointB[:2]
    pointA = (pointA + 1) * resolution / 2
    pointB = (pointB + 1) * resolution / 2
    pointA = pointA.int()
    pointB = pointB.int()
    # print(pointA, pointB)
    
    # Bresenham's line algorithm
    x0, y0 = pointA
    x1, y1 = pointB
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if x0 < x1:
        sx = 1
    else:
        sx = -1
    if y0 < y1:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    while True:
        img[y0, x0] = torch.tensor([1.,1.,1.])
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err = err - dy
            x0 = x0 + sx
        if e2 < dx:
            err = err + dx
            y0 = y0 + sy
    return img


def GetIntersectionPoint_2D(edge1, edge2):
    # edge1 and edge2 are two 2x3 tensors, each row is a vertex of the edge, the third dimension is the z value, which is discarded
    # the output is a 2x2 tensor, each row is a vertex of the intersection point
    x1, y1, _ = edge1[0]
    x2, y2, _ = edge1[1]
    x3, y3, _ = edge2[0]
    x4, y4, _ = edge2[1]
    # get the intersection point
    x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / ((x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4))
    return [x, y]

#find the intersection line of two triangles, input is two triangles, output is the intersection line's two end points
def GetIntersectioinPoint(edge, triangle):
    # edge is a B*2*3 tensor, each row is a vertex of the edge
    # triangle is a B*3*3 tensor, each row is a vertex of the triangle
    # the output is a B*3 tensor, each row is a vertex of the intersection point
    
    
    
    # get the normal of the triangle
    normal = torch.cross(triangle[:, 0] - triangle[:, 1], triangle[:, 1] - triangle[:, 2])
    
    # get the distance from the origin to the plane
    distance = torch.einsum('bi,bi->b', normal, triangle[:, 0])
    # get the distance from the edge's two end points to the plane
    distance1 = torch.einsum('bi,bi->b', normal, edge[:, 0]) - distance
    distance2 = torch.einsum('bi,bi->b', normal, edge[:, 1]) - distance
    # get the intersection point
    alpha = distance1 / ((distance1 - distance2)+1e-10)

    
    intersectionPoint = edge[:, 0,:] - alpha.unsqueeze(1) * (edge[:, 0,:] - edge[:, 1,:])
    
    #check if the intersection point is inside the triangle
    #get the normal of the three edges of the triangle
    normal1 = torch.cross(triangle[:, 0] - triangle[:, 1], intersectionPoint - triangle[:, 1])
    normal2 = torch.cross(triangle[:, 1] - triangle[:, 2], intersectionPoint - triangle[:, 2])
    normal3 = torch.cross(triangle[:, 2] - triangle[:, 0], intersectionPoint - triangle[:, 0])
    #check if the intersection point is inside the triangle
    
    check = ((torch.einsum('bi,bi->b', normal1, normal2) > 0) & (torch.einsum('bi,bi->b', normal2, normal3) >0) & (torch.einsum('bi,bi->b', normal3, normal1) >0) & (alpha >= 0) & (alpha <= 1))
    
    # intersectionPoint[~((alpha >= 0) & (alpha <= 1))] = torch.nan
    #set the intersection point to nan if it is outside the edge
    intersectionPoint[~check] = torch.nan
    
    return intersectionPoint

#test the GetIntersectioinPoint function
example_edge = torch.tensor([[[0.1, 0.1, 3], [.15, .1, 2]],[[0.1, 0.1, -1], [.1, .1, 2]]], dtype=torch.float32)
example_triangle = torch.tensor([[[0, 0, 0.1], [1, 0, 0], [0, 1, 0]],[[0, 0, 0.1], [1, 0, 0], [0, 1, 0]]], dtype=torch.float32)

def GetIntersectionLineBatch(triangle1, triangle2):
    # triangle1 and triangle2 are two n*3*3 tensors, each row is a vertex of the triangle
    # the output is a n*2*3 tensor, each row is a vertex of the intersection line
    n = triangle1.shape[0]
        # get all the edges of the two triangles, each edge is a 2*3 tensor
    edges1 = torch.stack([triangle1[:, 0], triangle1[:, 1], triangle1[:, 1], triangle1[:, 2], triangle1[:, 2], triangle1[:, 0]], dim=1).view(n, 3, 2, 3)
    edges2 = torch.stack([triangle2[:, 0], triangle2[:, 1], triangle2[:, 1], triangle2[:, 2], triangle2[:, 2], triangle2[:, 0]], dim=1).view(n, 3, 2, 3)
    
    # get the edge matrix and the triangle matrix, it is a 6*2*3 tensor
    edgematrix = torch.cat([edges1, edges2], dim=1).view(n*6, 2, 3)
    # triangle matrix is a 6*3*3 tensor
    trianglematrix = torch.cat([torch.stack([triangle2, triangle2, triangle2], dim=1), torch.stack([triangle1,  triangle1, triangle1], dim=1)], dim=1).view(n*6, 3, 3)
    tt0=time.time()
    intersectPoints = GetIntersectioinPoint(edgematrix, trianglematrix)
    tt1=time.time()
    
    # get all the non-nan points
    intersectPoints = intersectPoints.view(n, 6, 3)
    
    ### only choice the rows with two non-nan points
    # print(torch.where(~torch.isnan(intersectPoints).any(dim=-1)))
    # pdb.set_trace()
    # intersectPoints = intersectPoints[~torch.isnan(intersectPoints).any(dim=-1)]
    # tt = torch.sum(intersectPoints)
    # tt.backward(retain_graph=True)
    # ss = torch.autograd.grad(tt, triangle1, retain_graph=True)
    # print(torch.sum(torch.isnan(ss[0])))
    # pdb.set_trace()
    
    intersectPoints = intersectPoints[torch.sum(torch.sum(~torch.isnan(intersectPoints), dim=2),dim=-1) == 6]
    
    
    intersectPoints = intersectPoints[~torch.isnan(intersectPoints).any(dim=2)]
    
    intersectPoints = intersectPoints.view(-1, 2, 3)
    
    #mask out the rows with two same points
    
    mask_same_points = ~(torch.sum((intersectPoints[:,0,:]-intersectPoints[:,1,:])**2, dim=1)<1e-5)
    
    intersectPoints = intersectPoints[mask_same_points]
    tt2 = time.time()
    return intersectPoints


def GetIntersectionLine(triangle1, triangle2):
    #triangle1 and triangle2 are two 3x3 tensors, each row is a vertex of the triangle
    #the output is a 2x3 tensor, each row is a vertex of the intersection line
    #get all the edges of the two triangles, each edge is a 2x3 tensor
    edges1 = torch.stack([triangle1[0], triangle1[1], triangle1[1], triangle1[2], triangle1[2], triangle1[0]]).view(3, 2, 3)
    edges2 = torch.stack([triangle2[0], triangle2[1], triangle2[1], triangle2[2], triangle2[2], triangle2[0]]).view(3, 2, 3)
    #get the edge matrix and the triangle matrix, it is a 6*2*3 tensor
    edgematrix = torch.concat([edges1, edges2], dim=0)
    #triangle matrix is a 6*3*3 tensor
    trianglematrix = torch.concat([torch.stack([triangle2, triangle2, triangle2]), torch.stack([triangle1,  triangle1, triangle1])], dim=0)
    intersectPoints = GetIntersectioinPoint(edgematrix, trianglematrix)
    #get all the non-nan points
    intersectPoints = intersectPoints[~torch.isnan(intersectPoints).any(dim=1)]
    
    # if intersectPoints.shape[0] != 2:
    #     print("The intersection line is not a line")
        # pdb.set_trace()
    # else:
    #     #check the grad of the intersectPoints with respect to the triangle1
    #     for i in range(2):
    #         for j in range(3):
    #             intersectPoints[i, j].backward(retain_graph=True)
    #             print(triangle1.grad)
    #             triangle1.grad.zero_()
        # print(intersectPoints)
    return intersectPoints
    


if __name__ == "__main__":
    triangle1 = torch.tensor([[0, 0., 0.], [0.5, 0, 0.], [0., 0.5, 0]], dtype=torch.float32) 
    
    triangle2 = torch.tensor([[0., 0., 0.3], [0.5, 0, -0.3], [0., 0.5, -0.3]], dtype=torch.float32)
    
    triangle1.requires_grad = True
    
    ip1 = GetIntersectionLine(triangle1, triangle2)
    ip2 = GetIntersectionLineBatch(triangle1.unsqueeze(0), triangle2.unsqueeze(0))
    
    print(ip1)
    print(ip2)