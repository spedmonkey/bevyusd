
import sys
sys.path.append("C:/development/rust/bevy_usd/.venv/Lib/site-packages")
from pxr import Usd, UsdGeom
import numpy as np
def get_verts():
    # Open the stage
    stage = Usd.Stage.Open("C:/development/rust/bevy_usd/src/test.usd")

    # Get your mesh by path (replace with your mesh path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/rubbertoy/geo/shape"))

    # Get the points attribute
    points_attr = mesh.GetPointsAttr()

    # Read the points as a list of Gf.Vec3f
    points = points_attr.Get()

    return (points)  # list of Gf.Vec3f objects




def get_uvs():
    # Open the stage
    stage = Usd.Stage.Open("C:/development/rust/bevy_usd/src/test.usd")

    # Get your mesh by path (replace with your mesh path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/rubbertoy/geo/shape"))

    # Get the points attribute

    # Read the points as a list of Gf.Vec3f
    # Get the 'st' (UV) primvar
    primvar_api = UsdGeom.PrimvarsAPI(mesh)
    uvs_primvar =  primvar_api.GetPrimvar("primvars:st")

    if uvs_primvar:
        uvs = uvs_primvar.Get()
        vertex_count  = mesh.GetFaceVertexCountsAttr().Get()

        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        uvs = get_average_uvs(face_vertex_indices, uvs)

        return (uvs)
    else:
        return []





def get_normals():
    stage = Usd.Stage.Open("C:/development/rust/bevy_usd/src/test.usd")
        # Get your mesh by path (replace with your mesh path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/rubbertoy/geo/shape"))
    normals_attr = mesh.GetNormalsAttr()
    if normals_attr:
        normals = normals_attr.Get()
        return normals
    else:
        return []
    

def get_indicies():
        # Open the stage
    stage = Usd.Stage.Open("C:/development/rust/bevy_usd/src/test.usd")
    # Get your mesh by path (replace with your mesh path)
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/rubbertoy/geo/shape"))
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    vertex_count  = mesh.GetFaceVertexCountsAttr().Get()
    indicies = triangulate(vertex_count, face_vertex_indices)
    return indicies

def triangulate(counts, indices):
    # triangulate
    num_tris = np.sum(np.subtract(counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0
    for nb in counts:
        for i in range(nb - 2):
            tri_indices[ctr] = indices[wedgeIdx]
            tri_indices[ctr + 1] = indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb
    #the indicies need to be reversed otherwise it will be inside out
    tri_indices = tri_indices.reshape(-1, 3)
    tri_indices = tri_indices[:, [0, 2, 1]].flatten()
    return tri_indices

def triangulate_with_st(face_counts, face_indices, st_values):
    """
    Triangulate a polygon mesh with ST coordinates as tuples.

    face_counts: list of vertex counts per face
    face_indices: flat list of vertex indices per face
    st_values: list of (u, v) tuples, one per face-vertex
               must match the length of face_indices
    """
    tri_indices = []
    tri_st = []
    idx_offset = 0
    for count in face_counts:
        for i in range(1, count - 1):
            # Vertex indices for this triangle
            tri_indices.extend([
                face_indices[idx_offset + 0],
                face_indices[idx_offset + i],
                face_indices[idx_offset + i + 1],
            ])
            # UVs for this triangle
            tri_st.extend([
                st_values[idx_offset + 0],
                st_values[idx_offset + i],
                st_values[idx_offset + i + 1],
            ])
        idx_offset += count

    return tri_st

def get_average_uvs(faceVertexIndices,st_values):
    import numpy as np
    from collections import defaultdict

    # faceVertexIndices: flat list of vertex indices per face
    # st_values: list of (u,v) tuples, length same as faceVertexIndices

    vertex_uvs = defaultdict(list)

    for vert_idx, uv in zip(faceVertexIndices, st_values):
        vertex_uvs[vert_idx].append(uv)

    # Now average UVs per unique vertex
    unique_uvs = np.zeros((len(vertex_uvs), 2), dtype=float)

    for v_idx, uvs in vertex_uvs.items():
        unique_uvs[v_idx] = np.median(uvs, axis=0)
    unique_uvs = unique_uvs[::-1]
    unique_uvs =unique_uvs[::-1, ::-1]
    #unique_uvs = unique_uvs[:, ::-1]

    return unique_uvs


