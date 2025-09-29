import trimesh
import numpy as np

# Example data
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0]
])

faceVertexCounts = [4]
faceVertexIndices = [0, 1, 2, 3]

# Convert to list of faces
faces = []
offset = 0
for count in faceVertexCounts:
    face = faceVertexIndices[offset:offset + count]
    faces.append(face)
    offset += count

# Create the mesh
mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)

# Triangulate the mesh
triangulated = mesh.triangles

# Get the new triangles
tri_faces = triangulated.faces
tri_vertices = triangulated.vertices

print("Triangulated faces:", tri_faces)
print("Vertices:", tri_vertices)