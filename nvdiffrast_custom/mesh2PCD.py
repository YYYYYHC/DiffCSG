import open3d as o3d
import numpy as np

def create_disk_mesh(center, normal, radius, resolution=20):
    """创建一个半径为radius、法向为normal的圆盘mesh。"""
    # 生成圆环的顶点
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    points = np.array([[np.cos(angle) * radius, np.sin(angle) * radius, 0] for angle in angles])

    # 添加中心点
    points = np.vstack(([[0, 0, 0]], points))

    # 计算法线向量与Z轴的旋转
    z_axis = np.array([0, 0, 1])
    if np.allclose(normal, z_axis):
        rotation_matrix = np.eye(3)
    else:
        axis = np.cross(z_axis, normal)
        axis_len = np.linalg.norm(axis)
        if axis_len > 0:
            axis /= axis_len
        angle = np.arccos(np.dot(z_axis, normal))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 旋转顶点
    points = points @ rotation_matrix.T

    # 创建三角面
    triangles = [[0, i + 1, (i + 1) % resolution + 1] for i in range(resolution)]

    # 创建mesh
    disk_mesh = o3d.geometry.TriangleMesh()
    disk_mesh.vertices = o3d.utility.Vector3dVector(points + center)
    disk_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    disk_mesh.compute_vertex_normals()

    return disk_mesh

# 加载网格并采样点云
mesh_path = "/home/yyyyyhc/nvdiffrast/exp_scad/app_test_hyperparam_0_5000/CADTalk/trafficcone_general_app/source.stl"
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(number_of_points=20000, use_triangle_normal=True)

# 创建每个点的圆盘mesh
disks = [create_disk_mesh(np.array(p), np.array(n), radius=0.1) for p, n in zip(pcd.points, pcd.normals)]

# 合并所有圆盘为一个mesh
combined_mesh = o3d.geometry.TriangleMesh()
for disk in disks:
    combined_mesh += disk

# 保存为.ply文件，包含法线
o3d.io.write_triangle_mesh("tfCone.ply", combined_mesh, write_ascii=True, write_vertex_normals=True)
