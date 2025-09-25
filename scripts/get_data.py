
import sys
sys.path.append("C:/development/rust/bevy_usd/.venv/Lib/site-packages")

from pxr import Usd, UsdGeom, Gf
import numpy as np
from collections import defaultdict
USD_FILE  = "C:/development/rust/bevy_usd/assets/mesh/box.usd"


def get_uvs(mesh):
    # Open the stage
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

def get_average_uvs(faceVertexIndices,st_values):


    # faceVertexIndices: flat list of vertex indices per face
    # st_values: list of (u,v) tuples, length same as faceVertexIndices

    vertex_uvs = defaultdict(list)

    for vert_idx, uv in zip(faceVertexIndices, st_values):
        vertex_uvs[vert_idx].append(uv)

    # Now average UVs per unique vertex
    unique_uvs = np.zeros((len(vertex_uvs), 2), dtype=float)

    for v_idx, uvs in vertex_uvs.items():
        #unique_uvs[v_idx] = np.mean(uvs, axis=0)
        unique_uvs[v_idx] = uvs[-1]
    #unique_uvs = unique_uvs[::-1]
    #unique_uvs =unique_uvs[::-1, ::-1]
    unique_uvs = unique_uvs[:, ::-1]
    unique_uvs = np.column_stack((unique_uvs[:,1], 1 - unique_uvs[:,0]))

    return unique_uvs

def get_indicies(mesh):
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    vertex_count  =mesh.GetFaceVertexCountsAttr().Get()
    indicies = triangulate(vertex_count, face_vertex_indices)
    return indicies

def triangulate( counts, indices):
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

def get_normals(mesh):
    normals =mesh.GetNormalsAttr().Get()
    if normals is None:
        normals  = []
    return normals

def get_data(usd_file):
    #usd_file = "./assets/models/test.usd"
    #usd_file = "./assets/models/sphere_cube1.usd"
    usd_file = USD_FILE
    stage = Usd.Stage.Open(usd_file)
    prims=  ([str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Gprim)])
    data = []
    usd_scene = UsdScene()
    for prim in stage.Traverse():
        if  prim.IsA(UsdGeom.Gprim):
            mesh =UsdGeom.Mesh(stage.GetPrimAtPath(prim.GetPath()))
            points = mesh.GetPointsAttr().Get()
            if points is None:
                data.append([],[],[],[])
            else:
                xform = get_xform (prim)
                normals = get_normals(mesh)
                uvs =  get_uvs(mesh)
                indicies = get_indicies(mesh)
                prim =UsdPrim(points, normals, uvs, indicies, xform)
                usd_scene.add_prim(prim)
    return usd_scene

def get_xform(prim):
    # Wrap it as a UsdGeom.Xformable (if it's transformable)
    xformable = UsdGeom.Xformable(prim)
    # Get the local-to-world transform (flattened transform)
    # at time code Usd.TimeCode.Default() (default time)
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return (xform)

class UsdPrim():
    def __init__(self, points, normals, uvs, indicies, xform):
        self.points= points
        self.normals = normals
        self.uvs = uvs
        self.indicies = indicies
        self.xform  = xform 

class UsdScene():
    def __init__(self):
        self.usd_prims = []
    
    def add_prim(self, prim):
        self.usd_prims.append(prim)
    def get_prims(self):
        return self.usd_prims
    def get_points(self):
        return [prim.points for prim in self.usd_prims]
    def get_normals(self):
        return [prim.normals for prim in self.usd_prims]
    def get_uvs(self):
        return [prim.uvs for prim in self.usd_prims]
    def get_indicies(self):
        return [prim.indicies for prim in self.usd_prims]
    def get_matrix(self):
        return [prim.xform for prim in self.usd_prims]
    