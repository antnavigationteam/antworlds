import numpy as np
# from mayavi import mlab

import matplotlib.pyplot as plt


def scatter3d(xs, ys, zs, color='b', marker='o', msize=50):
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(xs, ys, zs, color=color, marker=marker, s=msize)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()


def plot_om_dir(rdirs, msize=10, greatcircle=False):
    """
    Plots the ommatidia directions
    :param rdirs: Receptors directions
    :param msize: Markers size
    :param greatcircle: Mark the great circle in another colour
    """

    scatter3d(rdirs[:, 0], rdirs[:, 1], rdirs[:, 2], msize=msize)

    if greatcircle:
        # This can be used to manually count the points on the great circle
        # - works best for odd values in level_of_division
        rdirs_z = rdirs[:, 2]
        idx_gc = np.where((rdirs_z < 0.01) & (rdirs_z > -0.01))     # find great circle
        rdirs_slice = rdirs[idx_gc, :]

        scatter3d(rdirs_slice[0, :, 0], rdirs_slice[0, :, 1], rdirs_slice[0, :, 2],
                  color='green',
                  marker='o',
                  msize=int(msize * 2)
                  )
    plt.show()


def plot_eye(eye, greatcircle=False):
    rdirs = eye.view_dirs
    plot_om_dir(rdirs, greatcircle=greatcircle)


# TODO - update the following plotting functions
def core_worldviewer_position(position, color=None, points_scale=0.01):

    if color is None:
        color = tuple(np.random.random(3))

    if position.ndim == 2:
        # We have a path

        # If third dimension contains 3 values, that means we know the Z coords
        if position.shape[1] == 3:
            x = position[:, 0]
            y = position[:, 1]
            z = position[:, 2]

        # If not, we fix Z to a constant height
        else:
            x = position[:, 0]
            y = position[:, 1]
            z = np.copy(x) * 0.0 - 0.99

    else:
        # We have a single point

        # If we have 3 values, that means we know the Z coord
        if len(position) == 3:
            x = position[0]
            y = position[1]
            z = position[2]

        # If not, we fix Z to a constant height
        else:
            x = position[0]
            y = position[1]
            z = np.copy(x) * 0.0 - 0.99

    mlab.points3d(x, y, z, color=color, scale_factor=points_scale)


def worldviewer(world=None, positions=None, faces=None, points_scale=0.05):
    """Takes a raw mesh or an unpacked world dictionary and generates a
        3D point representation with mayavi and wx.

    'faces' parameter displays also faces. Be aware that this option is very
    resource-hungry for real-life world scans!
    'positions' parameter can be a tuple containing the x, y or x, y, z
    coordinates arrays of a walked route, or a list of tuples
    (useful for plotting multiple routes, or multiple individual points)."""

    if world is None and positions is None:
        print("Nothing to plot...")

    else:
        if world is not None:

            if type(world) is dict:
                vertices_array = world['v']
                x, y, z = (vertices_array[:, d].astype('float32') for d in range(3))

                mlab.points3d(x, y, z, mode='point')

                if type(faces) is bool and faces is True:
                    triangles = world['i']
                    mlab.triangular_mesh(x, y, z, triangles, color=(1, 0, 0.4), opacity=0.5)

                elif type(faces) is not bool and faces is not None:
                    triangles = np.fromiter(faces, world['v'])
                    mlab.triangular_mesh(x, y, z, triangles, color=(1, 0, 0.4), opacity=0.5)

                else:
                    pass

            else:
                vertices = world['vertex']
                x, y, z = (vertices[d] for d in ('x', 'y', 'z'))

                mlab.points3d(x, y, z, mode='point')

                if ('face' in world) and type(faces) is bool and faces is True:
                    tri_idx = world['face']['vertex_indices']

                elif ('face' in world) and type(faces) is not bool and faces is not None:
                    tri_idx = faces

                    idx_dtype = tri_idx[0].dtype
                    triangles = np.fromiter(tri_idx, [('data', idx_dtype, (3,))], count=len(tri_idx))['data']

                    mlab.triangular_mesh(x, y, z, triangles, color=(1, 0, 0.4), opacity=0.5)

                else:
                    pass

        if positions is not None:

            if (type(positions) is tuple or type(positions) is list) or ('numpy' in str(type(positions)) and positions.ndim == 3):
                # We have multiple paths to plot (or multiple single points)

                for pos in positions:
                    pos = np.array(pos)

                    core_worldviewer_position(pos, color=None, points_scale=points_scale)

            else:
                # We have a single array of successive positions (= a path) or a single point

                core_worldviewer_position(positions, color=(1, 0, 0), points_scale=points_scale)

        print("Warning: Reload the insect_eye before re-running a simulation as this plotting method freezes the Pygame window")
        mlab.show()


def heatmap_vertices(vertices):
    X, Y, Z = np.hsplit(vertices, 3)
    plt.tricontour(X.squeeze(), Y.squeeze(), Z.squeeze(), cmap='Greys_r')
    plt.show()