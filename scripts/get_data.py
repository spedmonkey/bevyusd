import sys
import math
sys.path.append("C:/development/rust/bevy_usd/.venv/Lib/site-packages")
import trimesh
from pxr import Usd, UsdGeom, Gf
import numpy as np
from collections import defaultdict
import trimesh
USD_FILE = "C:/development/rust/bevy_usd/assets/mesh/test.usd"

def get_uvs(mesh):
    # Open the stage
    primvar_api = UsdGeom.PrimvarsAPI(mesh)
    uvs_primvar =  primvar_api.GetPrimvar("primvars:st")

    if uvs_primvar:
        uvs = uvs_primvar.Get()
        vertex_count  = mesh.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
        #uvs = get_average_uvs(face_vertex_indices, uvs)
        uvs = [(u, 1.0 - v) for (u, v) in uvs]
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
    n = 0
    tri_indices = []
    tri_list=[]
    for i in range(0, len(indices), 4):
        v0, v1, v2, v3 = indices[i:i+4]
        # Triangle 1
        tri_indices.append([v2, v0, v1][::-1])
        # Triangle 2
        tri_indices.append([v0, v2, v3][::-1])
    result =[]
    for i in tri_indices:
        for a in i:
            result.append(a)
    return result

def get_normals(mesh):
    normals =mesh.GetNormalsAttr().Get()
    if normals is None:
        normals  = []
    return normals

def get_data(usd_file):
    usd_file = "./assets/mesh/test.usd"
    #usd_file = "./assets/models/sphere_cube1.usd"
    stage = Usd.Stage.Open(usd_file)
    prims=  ([str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Gprim)])
    data = []
    usd_scene = UsdScene()
    for prim in stage.Traverse():
        if  prim.IsA(UsdGeom.Gprim):
            mesh =UsdGeom.Mesh(stage.GetPrimAtPath(prim.GetPath()))
            points = mesh.GetPointsAttr().Get()
            if points is None:
                print ("skipping")
            else:
                indices = get_indicies(mesh)
                indices  =mesh.GetFaceVertexIndicesAttr().Get()
                normals = mesh.GetNormalsAttr().Get()
                primvar_api = UsdGeom.PrimvarsAPI(mesh)
                uvs_primvar =  primvar_api.GetPrimvar("primvars:st")
                
                uvs = uvs_primvar.Get()
                uvs =[(u, 1.0 - v) for (u, v) in uvs]
                
                verts = get_verts(points, indices)
                normals = get_verts(normals, indices)
                #uvs = get_verts(uvs, indices)
                #normals =[(-x, -y, -z) for (x, y, z) in normals]
                #normals = get_verts(normals, indices)
                #uvs = get_verts(uvs, indices)
                tri_indices = [
                0,3,1 , 1,3,2,
                4,5,7 , 5,6,7,
                8,11,9 , 9,11,10,
                12,13,15 , 13,14,15,
                16,19,17 , 17,19,18,
                20,21,23 , 21,22,23,
                ]
                tri_indices = [
                0,3,1 , 1,3,2, #front   
                4,5,7 , 5,6,7, #back
                8,11,9 , 9,11,10, #top
                12,13,15 , 13,14,15,  # bottom
                16,19,17 , 17,19,18, #right
                20,21,23 , 21,22,23,  #left
                ]
                tri_indices=[0, 3, 1, 1, 3, 2, 4, 7, 5, 5, 7, 6, 8, 11, 9, 9, 11, 10, 12, 15, 13, 13, 15, 14, 16, 19, 17, 17, 19, 18, 20, 23, 21, 21, 23, 22]
                tri_indices = triangulate_and_reindex(indices)
                xform = get_xform(prim)
                prim =UsdPrim(verts, normals, uvs, tri_indices, xform)
                usd_scene.add_prim(prim)
    return usd_scene

def triangulate_and_reindex(quads):
    # quads: list of vertex indices, length multiple of 4
    triangles = []
    num_quads = len(quads) // 4

    for i in range(num_quads):
        base = i * 4
        # new quad vertex indices are [base, base+1, base+2, base+3]
        v0, v1, v2, v3 = base, base+1, base+2, base+3
        # add two triangles per quad
        triangles.extend([v0, v3, v1])
        triangles.extend([v1, v3, v2])
    return triangles
def get_verts(points, indices):
    #sindices = mesh.GetFaceVertexIndicesAttr().Get()
    vertices = []
    for index in indices:
        vertices.append(points[int(index)])
    return vertices
def triangulate_uvs(uvs, indices):
    vertices = []
    for index in indices[::-1]:
        vertices.append(uvs[int(index)])
    return vertices

def get_xform(prim):
    # Wrap it as a UsdGeom.Xformable (if it's transformable)
    xformable = UsdGeom.Xformable(prim)
    # Get the local-to-world transform (flattened transform)
    # at time code Usd.TimeCode.Default() (default time)
    xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return (xform)

class UsdPrim():
    def __init__(self, vertices, normals, uvs, indicies, xform):
        self.points= vertices
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
    def printer(self):
        print ("=======VERTS==========")
        print ([prim.points for prim in self.usd_prims])
        print ("======NORMALS=========")
        print ([prim.normals for prim in self.usd_prims])
        print ("======UVS=============")
        print ([prim.uvs for prim in self.usd_prims])
        print ("======INDICES=======")
        print ([prim.indicies for prim in self.usd_prims])

        
        print ("points",len([prim.points for prim in self.usd_prims][0]),
               "normals",len([prim.normals for prim in self.usd_prims][0]),
               "uvs",len([prim.uvs for prim in self.usd_prims][0]),
               "indices",len([prim.indicies for prim in self.usd_prims][0]))


a=get_data("asdf")