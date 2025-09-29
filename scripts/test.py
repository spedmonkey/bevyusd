from pxr import Usd, UsdGeom
import numpy as np

def triangulate_mesh(face_counts, face_indices):
    """
    Converts face counts and face indices into triangle indices.
    """
    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros((num_tri_vtx,), dtype=int)

    ctr = 0
    wedge_idx = 0
    for count in face_counts:
        for i in range(count - 2):
            tri_indices[ctr]     = face_indices[wedge_idx]
            tri_indices[ctr + 1] = face_indices[wedge_idx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedge_idx + i + 2]
            ctr += 3
        wedge_idx += count

    # Reshape into Nx3 triangles and fix winding
    return tri_indices.reshape(-1, 3)[:, [0, 2, 1]]

def get_triangle_vertex_positions(usd_file_path, mesh_path=None):
    """
    Loads a USD file and returns a list of triangle vertex positions.
    
    Returns:
        List of triangle vertex arrays (shape [3, 3] each)
    """
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    if mesh_path:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"Invalid mesh path: {mesh_path}")
        mesh = UsdGeom.Mesh(mesh_prim)
    else:
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                break
        else:
            raise RuntimeError("No UsdGeom.Mesh found in the stage.")

    face_counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
    face_indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
    points = np.array(mesh.GetPointsAttr().Get(), dtype=float)

    triangles = triangulate_mesh(face_counts, face_indices)

    # Extract per-triangle vertex positions
    triangle_vertex_positions = [points[tri] for tri in triangles]
    
    return triangle_vertex_positions
USD_FILE  = "C:/development/rust/bevy_usd/assets/mesh/box.usd"
a= ( get_triangle_vertex_positions(USD_FILE))
print (a)