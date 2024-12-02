import open3d as o3d
import numpy as np

def genCylinder():
    file_path = "/home/yyyyyhc/nvdiffrast/cylinder.obj"

    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh.remove_duplicated_vertices()

    # 获取顶点数组（numpy格式）
    vertices = np.asarray(mesh.vertices)

    # 获取面的顶点索引（numpy格式）
    triangles = np.asarray(mesh.triangles)

    vertex_normals = np.asarray(mesh.vertex_normals)

    #save vertices and trianges
    np.save("vertices.npy", vertices)
    np.save("triangles.npy", triangles)
    np.save("vertex_normals.npy", vertex_normals)
    print("Vertices:\n", vertices)
    print("Triangles:\n", triangles)
    
def genSphere():
    file_path = "/home/yyyyyhc/nvdiffrast/sphere.ply"

    mesh = o3d.io.read_triangle_mesh(file_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh.remove_duplicated_vertices()

    # 获取顶点数组（numpy格式）
    vertices = np.asarray(mesh.vertices)

    # 获取面的顶点索引（numpy格式）
    triangles = np.asarray(mesh.triangles)

    vertex_normals = np.asarray(mesh.vertex_normals)

    #save vertices and trianges
    np.save("vertices_sphere.npy", vertices)
    np.save("triangles_sphere.npy", triangles)
    np.save("vertex_normals_sphere.npy", vertex_normals)
    print("Vertices:\n", vertices)
    print("Triangles:\n", triangles)
    print("Vertex Normals:\n", vertex_normals)
    
def get_polyline():
    # mesh = o3d.geometry.create_mesh_cylinder(radius=1.0, height=2.0, resolution=20, split=4)
    #save mesh
    # 创建圆柱体
    polyline = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=1, resolution=10, split=1)

    # 将生成的圆柱体移到原点
    polyline.translate([-polyline.get_center()[0], -polyline.get_center()[1], -polyline.get_center()[2]])

    o3d.io.write_triangle_mesh("polyline.obj", polyline)
if __name__ == "__main__":
    genCylinder()
    genSphere()
    get_polyline()