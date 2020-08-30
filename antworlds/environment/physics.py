import numpy as np


def query_tree(point, tree, triangles_array, k=1, return_distances=False, return_vertices=False):
    """Wrapper function for Scipy's tree.query method.
    Queries the closest k vertices to the given point.

    Returns an array of all the faces in which this vertex is included.
    Optionally returns the corresponding distances and the indexes of the k vertices."""

    distances, vert_indices = tree.query(point[:tree.m], k=k)
    faces_indices, _ = np.where(np.in1d(triangles_array, vert_indices).reshape(triangles_array.shape))

    if return_distances:
        if return_vertices:
            return faces_indices, distances, vert_indices
        else:
            return faces_indices, distances
    else:
        if return_vertices:
            return faces_indices, vert_indices
        else:
            return faces_indices


def facecoords(face_index, faces_array=None, vertices_array=None, mesh=None):
    """Computes the x, y, z coordinates of a face, from its index and either a raw PlyData mesh, or unpacked
    world data arrays."""

    if faces_array is not None and vertices_array is not None:
        vertexA = vertices_array[faces_array[face_index][0]]
        vertexB = vertices_array[faces_array[face_index][1]]
        vertexC = vertices_array[faces_array[face_index][2]]

    else:
        id_vertexA = mesh['face'].data['vertex_indices'][face_index][0]
        id_vertexB = mesh['face'].data['vertex_indices'][face_index][1]
        id_vertexC = mesh['face'].data['vertex_indices'][face_index][2]

        vertexA = mesh['vertex'][id_vertexA]
        vertexB = mesh['vertex'][id_vertexB]
        vertexC = mesh['vertex'][id_vertexC]

    Ax, Ay, Az = vertexA
    Bx, By, Bz = vertexB
    Cx, Cy, Cz = vertexC

    return np.array([[Ax, Ay, Az], [Bx, By, Bz], [Cx, Cy, Cz]])


def triangle_area(triangle_coords):
    """Extremely simple triangle area computation, from its coordinates."""

    a = triangle_coords[0, :]
    b = triangle_coords[1, :]
    c = triangle_coords[2, :]

    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def point_in_triangle(point_coords, triangle_coords=None, face_index=None, faces_array=None, vertices_array=None,
                      mesh=None):
    """For a given point and a list of faces: compute, for each face, if the point lays within that face or not IN TWO
    DIMENSIONS. This is used to compute over which face the agent is (true vertical projection)."""

    if face_index is not None:
        if faces_array is not None and vertices_array is not None:
            x_y_coords = facecoords(face_index, faces_array, vertices_array)[:, :2]
        else:
            x_y_coords = facecoords(face_index, mesh=mesh)[:, :2]
    else:
        x_y_coords = triangle_coords[:, :2]

    pt = point_coords[:2]
    v1 = x_y_coords[0, :]
    v2 = x_y_coords[1, :]
    v3 = x_y_coords[2, :]

    trianglearea = triangle_area(x_y_coords)
    area1 = triangle_area(np.vstack([pt, v2, v3]))
    area2 = triangle_area(np.vstack([pt, v1, v3]))
    area3 = triangle_area(np.vstack([pt, v1, v2]))

    if (area1 + area2 + area3) > trianglearea:
        return False

    return True


def get_closest_face(test_point, tree, faces_array, vertices_array, return_facesid=False):
    """For a given point, search from unpacked world data which is the face directly hovered by the point (true
    vertical projection)."""

    considered_faces = query_tree(test_point, tree, faces_array, k=1)

    faces_coords = [facecoords(face, faces_array=faces_array, vertices_array=vertices_array) for face in
                    considered_faces]

    current_triangle = False
    t = 0
    while current_triangle is False:
        current_triangle = [point_in_triangle(test_point, coords) for coords in faces_coords]
        t += 1

    hovered_face = faces_coords[t]

    if return_facesid is True:
        return hovered_face, considered_faces

    return hovered_face


def core_z_getter(test_point, closest_face_coords, return_normal=False):
    """Very simple function to calculate the missing z coordinate of a point, using the equation of the 3D plan.
     Takes the known x and y coords of the point, and the x,y,z coordinates of the face on which it projects
     vertically."""

    Px, Py = test_point

    Ax, Ay, Az = closest_face_coords[0]
    Bx, By, Bz = closest_face_coords[1]
    Cx, Cy, Cz = closest_face_coords[2]

    # Equation of the plan

    z_CA = Cz - Az
    z_BA = Bz - Az
    x_CA = Cx - Ax
    x_BA = Bx - Ax
    y_CA = Cy - Ay
    y_BA = By - Ay

    xy = (x_BA * y_CA) - (x_CA * y_BA)  # TODO : prevent xy to yeild 0 sometimes (to avoid dividing by 0)
    xz = (x_BA * z_CA) - (x_CA * z_BA)
    yz = (y_BA * z_CA) - (y_CA * z_BA)

    Pz = Az + (xz / xy) * (Py - Ay) - (yz / xy) * (Px - Ax)

    if return_normal:
        Ux = Bx - Ax
        Uy = By - Ay
        Uz = Bz - Az

        Vx = Cx - Ax
        Vy = Cy - Ay
        Vz = Cz - Az

        Nx = Uy * Vz - Uz * Vy
        Ny = Uz * Vx - Ux * Vz
        Nz = Ux * Vy - Uy * Vx

        normal = np.array([Nx, Ny, Nz])

    else:
        normal = 0.0

    return Pz, normal


def get_z(agent_x_y, ground, offset=0.0, return_normal=False):
    """Wrapper for z computation. Takes the x, y coordinates of the agent and the unpacked World dict,
    and returns the z. Be careful: x and y coordinates need to be in meters, and z is returned in meters
    An optional offset can be set to avoid blocking in the lower half area of the visual field."""

    tree = ground['tree']
    faces_array = ground['faces']
    vertices_array = ground['vertices']['pos']

    # Ensure offset is a float to avoid flooring to 0 (not needed anymore with Python 3.x)
    offset = float(offset)

    face_coords = get_closest_face(agent_x_y, tree, faces_array, vertices_array)

    z, normal = core_z_getter(agent_x_y, face_coords, return_normal=return_normal)

    return z + offset, normal