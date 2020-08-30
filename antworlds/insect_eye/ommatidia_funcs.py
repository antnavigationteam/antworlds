# Jan Stankiewicz, 24.7.2016
# Adapted from:
# Author: William G.K. Martin (wgm2111@cu where cu=columbia.edu)
# copyright (c) 2010
# liscence: BSD style
# Code base taken from: https://code.google.com/archive/p/mesh2d-mpl/source
# ================================================================================

# Notes
# phi = interommatidial angle
# rho = acceptance angle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from antworlds import utils
from antworlds import plotting

from antworlds import OMMATIDIA_CACHE_PATH


def get_barymat(n):
    """
    Define the matrix that will refine points on a triangle
    """

    # Define the values that will be needed
    ns = np.arange(n)
    vals = np.linspace(0, 1, n)

    # Initialize array
    numrows = int(n * (n + 1) / 2)
    bcmat = np.zeros((numrows, 3))

    # Loop over blocks to fill in the matrix
    shifts = np.arange(n, 0, -1)
    starts = np.zeros(n, dtype=int)
    starts[1:] += np.cumsum(shifts[:-1])    # starts are the cumulative shifts

    stops = starts + shifts
    for n_, start, stop, shift in zip(ns, starts, stops, shifts):
        bcmat[start:stop, 0] = vals[shift-1::-1]
        bcmat[start:stop, 1] = vals[:shift]
        bcmat[start:stop, 2] = vals[n_]

    return bcmat


def triangulate_bary(bary):
    """
    Triangulate a barycentric triangle using matplotlib
    :return: edges, triangles
    """
    x = np.cos(-np.pi / 4.0) * bary[:, 0] + np.sin(-np.pi / 4.0) * bary[:, 1]
    y = bary[:, 2]

    dely = Triangulation(x, y)

    return dely.edges, dely.triangles

# def get_icosahedron_points():
#     """
#     Define the 12 vertices on the z-axis aligned icosahedron
#     """
#
#     # Define the verticies with the golden ratio
#     a = (1 + np.sqrt(5)) / 2.0  # golden ratio
#
#     p = np.array([[a, -a, -a, a, 1, 1, -1, -1, 0, 0, 0, 0],
#                   [0, 0, 0, 0, a, -a, -a, a, 1, 1, -1, -1],
#                   [1, 1, -1, -1, 0, 0, 0, 0, a, -a, -a, a]]).T
#
#     p = p / np.sqrt(np.sum(p ** 2, axis=1))[0]
#
#     # Rotate top point to the z-axis
#     ang = np.arctan(p[0, 0] / p[0, 2])
#     ca, sa = np.cos(ang), np.sin(ang)
#     rotation = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
#     p = np.inner(rotation, p).T
#
#     # Reorder in a downward spiral
#     reorder_index = [0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]
#
#     return p[reorder_index]


# class Icosahedron(object):
#     """
#     Class that holds the vertices of an icosahedron, together with triangles, edges,
#     triangle midpoints, and edge midpoints.
#     """
#
#     # Define base vertices (points)
#     p = get_icosahedron_points()
#     px, py, pz = p.T
#
#     # Define faces (triangles)
#     tri = np.array([[1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
#                     [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
#                     [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]])
#
#     vert_coords = p[tri.T]
#
#     trimids = np.mean(vert_coords, axis=1)
#
#     # Make an array of sides
#     t_roll = np.roll(tri.T, -1, axis=1)   # Just roll the vertices indices by 1 left
#     sides = np.stack((tri.T, t_roll), axis=2)
#
#     # Put only positive clockwise sides into a 'bars' array
#     bars = sides[sides[:, :, 1] > sides[:, :, 0]]
#
#     barmids = np.mean(bars, axis=1)

def base_icosahedron():
    """
    Define the basic z-axis aligned icosahedron.
    A regular Icosahedron has 12 vertices and 20 faces.
    """

    # Define the vertices with the golden ratio G
    G = (1 + np.sqrt(5)) / 2.0

    p = np.array([[G, -G, -G, G, 1, 1, -1, -1, 0, 0, 0, 0],     # 12 vertices x 3 coords
                  [0, 0, 0, 0, G, -G, -G, G, 1, 1, -1, -1],
                  [1, 1, -1, -1, 0, 0, 0, 0, G, -G, -G, G]]).T

    p = p / np.sqrt(np.sum(p ** 2, axis=1))[0]

    # Rotate top point to the z-axis
    ang = np.arctan(p[0, 0] / p[0, 2])
    ca, sa = np.cos(ang), np.sin(ang)
    rotation = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])
    p = np.inner(rotation, p).T

    # Reorder in a downward spiral
    reorder_index = [0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]

    p = p[reorder_index]

    # Define the faces (20 triangles)
    tri = np.array([[1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
                    [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
                    [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]])

    verts = p[tri]

    return verts


# def get_triangulation(n, ico=Icosahedron()):
def subdivide_ico(n):
    """
    Compute the triangulation of the sphere by refining each face of the
    icosahedron to an nth order barycentric triangle. There are two key issues
    that this routine addresses :

    1) calculate the triangles (unique by construction)
    2) remove non-unique nodes and edges
    :param n: Level of division of the Icosahedron
    :param ico: Icosahedron object
    """

    # verts = ico.p[ico.tri]

    verts = base_icosahedron()
    bary = get_barymat(n)

    tensor = np.tensordot(verts, bary, axes=[(0,), (-1,)])
    newverts = tensor.swapaxes(1, 2).reshape(-1, 3)

    numtriangles = newverts.shape[0]
    if numtriangles > 1e6:
        print(f"/!\ New triangles nb is very high: {numtriangles}!!")

    flat_coordinates = np.arange(numtriangles).reshape(20, -1)

    barbary, tribary = triangulate_bary(bary)

    newtri = flat_coordinates[:, tribary].astype(int).reshape(-1, 3)
    newbar = flat_coordinates[:, barbary].astype(int).reshape(-1, 2)

    # Normalize verticies
    scalars = np.sqrt(np.sum(newverts ** 2, axis=1))
    newverts = (newverts.T / scalars).T

    # Remove repeated verticies
    cleanscaled_verts = np.matmul(newverts // 1e-8, 100 * np.arange(1, 4))
    _, iunique, irepeat = np.unique(cleanscaled_verts, return_index=True, return_inverse=True)

    univerts = newverts[iunique]
    unitri = irepeat[newtri]
    unibar = irepeat[newbar]

    mid = np.mean(univerts[unibar], axis=1)

    cleanscaled_mids = np.matmul(mid // 1e-8, 100 * np.arange(1, 4))
    _, iu = np.unique(cleanscaled_mids, return_index=True)

    return univerts.astype(np.float32), unitri.astype(np.float32), unibar[iu].astype(np.float32)


def subdiv_to_ommatidia(x):
    p = np.poly1d([10, -20, 12])
    y = p(x)
    return y


def ommatidia_to_subdiv(y):
    p = np.poly1d([10, -20, 12])
    x = (p - y).roots[0]
    return int(round(x))


def assert_ommatidia_nb(ommatidia_nb):

    subdivisions = ommatidia_to_subdiv(ommatidia_nb)
    new_om_nb = subdiv_to_ommatidia(subdivisions)

    if new_om_nb != ommatidia_nb:
        print(f"Invalid amount of ommatidia requested!\nClosest valid amount has been generated instead: {new_om_nb}")

    return new_om_nb


def ommatidia_builder(ommatidia=None, lod=None, print_specs=False, cache=False):

    if ommatidia is None and lod is None:
        raise AssertionError("You must pass either an ommatidia number, or a level of division!")

    elif ommatidia is not None:
        ommatidia = assert_ommatidia_nb(ommatidia)
        lod = ommatidia_to_subdiv(ommatidia)

    else:
        ommatidia = subdiv_to_ommatidia(lod)

    # om_dirs, _, _ = get_triangulation(lod)
    om_dirs, _, _ = subdivide_ico(lod)

    longs, lats = utils.eul2geo(om_dirs[:, 0], om_dirs[:, 1], om_dirs[:, 2])

    if cache:
        np.savez_compressed(OMMATIDIA_CACHE_PATH / f'om{ommatidia}.npz',
                            om_dirs=om_dirs,
                            longs=longs,
                            lats=lats)

    if print_specs:
        # Estimate interommatidial angle
        # NB. Analysis of the adopted method shows that there is some variance in the angle between neighbouring
        # points on the circle, hence the 'estimate' part in the name
        phi_estimate = utils.angle_between(om_dirs[0, :], om_dirs[1, :])

        print('Eye model specs (no pun intended...):')
        print(f'  Ommatidia: {ommatidia}')
        print(f'  Ideal acceptance angle: {np.rad2deg(phi_estimate):.3f} degs')

    return om_dirs.astype(np.float32), longs.astype(np.float32), lats.astype(np.float32)


def calculate_ommatidia_vs_acuity(start=3, end=71):

    lods = np.array(range(start, end))
    om_nbs = subdiv_to_ommatidia(lods)

    inter_om_angles = np.zeros_like(lods, dtype=np.float64)

    for i, val in enumerate(lods):
        inter_om_angles[i], _, _ = ommatidia_builder(lod=val, cache=False)

    fig, ax1 = plt.subplots()
    plt.semilogx(inter_om_angles, om_nbs, color='b', marker='x')
    ax1.set_xlabel('Interommatidial angle (' + u"\u00B0" + ')')
    ax1.set_xscale('log')
    ax1.set_ylim([np.amin(om_nbs), np.amax(om_nbs)])
    ax1.set_ylabel('Ommatidia', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    plt.semilogx(inter_om_angles, lods, color='r', marker='o')
    ax2.set_ylim([np.amin(lods), np.amax(lods)])
    ax2.set_ylabel('Level of division', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.show()


if __name__ == "__main__":

    # Options:
    level_of_division = 15       # Sets how many recursive triangle divisions our icosahedron undergoes
    ommatidia = 1960             # Or choose a number of ommatidia (it will produce the closest compatible number)
    plot = True
    save_result = True

    # Run this to view how level of acuity relates to interommatidial angle (estimated)
    # calculate_ommatidia_vs_acuity()

    # Build ommatidia for use with the eye model
    om_dirs, longs, lats = ommatidia_builder(ommatidia, cache=save_result, print_specs=True)

    if plot:
        plotting.plot_om_dir(om_dirs)
