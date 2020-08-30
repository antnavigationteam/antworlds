# coding=utf-8

import sys
import numpy as np


def unit_vector(vector, axis=0):
    """
    Normalizes the 'vector' so that its length is 1. 'vector' can have
    any number of components.
    """
    return np.divide(vector, np.linalg.norm(vector, axis=axis))


def angle_between(v1, v2, round_decimals=5):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1 = np.asanyarray(v1)
    v2 = np.asanyarray(v2)

    if v1.ndim == v2.ndim == 2:
        numerator = np.round(np.sum(v1 * v1, axis=1), decimals=round_decimals)
        denominator = np.round(np.linalg.norm(v1, axis=1) * np.linalg.norm(v1, axis=1), decimals=round_decimals)

        return np.arccos(numerator / denominator)

    elif v1.ndim == v2.ndim == 1:
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)

        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))    # TODO: remove the clipping here?

    else:
        raise AssertionError('The passed shapes mismatch!')


def rapid_angles_between(vec1, vec2, decimals=5):
    """
    Returns the angle in radians between an array of vectors 'v1' and 'v2'::
    v1 = np.array(((1, 0, 0), (1, 0, 0), (1, 0, 0)), dtype=np.float32)
    v2 = np.array(((0, 1, 0), (1, 0, 0), (-1, 0, 0)), dtype=np.float32)
    util.rapid_angle_between(v1,v2)
        [ 1.57079637  0.          3.14159274]
    Note: for a 1D vector use angle_between
    """
    # TODO - is setting decimals to 5 OK here?

    numerator = np.round(np.sum(vec1 * vec2, axis=1), decimals=decimals)
    denominator = np.round(np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1), decimals=5)
    return np.arccos(numerator / denominator)


def rotate_vector_list(vector_list, direction, angle_deg=None, angle_rad=None):
    """
    This function rotates a list of 1x3 vectors by a specified amount. Each row is a vector.
    :param vector_list - nx3 list of vectors, each row is a 1x3 vector
    :param direction, the axis about which the rotation is performed. For x axis use (1,0,0), for y-axis use (0,1,0)
    for z-axis use (0,0,1)
    :param angle_deg - the magnitude of the rotation in degrees
    :param angle_rad - the magnitude of the rotation in rad
    """
    vector_list = np.asanyarray(vector_list)

    if angle_deg is None and angle_rad is None:
        raise AssertionError('You must specify the angle (in rad or deg)')
    elif angle_rad is None:
        angle_rad = np.deg2rad(angle_deg)

    rot_mat = axis_rotation_matrix(direction, angle_rad)

    new_dirs = rot_mat @ vector_list[:, :3].T

    return new_dirs.T


def axis_rotation_matrix(direction, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general
    axis by a specified angle.

    R = dd^T + cos(a) (I - dd^T) + sin(a) skew(d)

    Parameters:
        angle : float a
        direction : array d
    """
    direction = np.asanyarray(direction)

    d = unit_vector(direction)

    diag = np.eye(3, dtype=np.float32)
    ddt = np.outer(d, d)
    skew = np.array([[0, d[2], -d[1]],
                     [-d[2], 0, d[0]],
                     [d[1], -d[0], 0]], dtype=np.float32)

    mtx = ddt + np.cos(angle) * (diag - ddt) + np.sin(angle) * skew
    return mtx.T


def rot2homo(rot_mat):
    """
    Converts a 3x3 rotation matrix into a 4x4 homogenous coordinates matrix
    :param rot_mat: the input 3x3 rot matrix
    :return: the output 4x4 homgeneous coordinates matrix
    """
    hom_mat = np.eye(4)
    hom_mat[:3, :3] = rot_mat
    return hom_mat


def gaussian(zeta, delta_rho_q):
    # Simple Gaussian
    # From Snyder (1979), as cited in Burton & Laughlin (2003)
    return np.exp(-4 * np.log(2) * np.abs(zeta)**2 / delta_rho_q**2)


def eul2geo(xx, yy, zz):
    """
    Converts euler rotation angles to a latitude longitude coordinate system
    """
    lon = np.arctan2(yy, xx)
    lat = np.arcsin(zz)
    return lon, lat


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    format_str = "{0:." + str(decimals) + "f}"
    percents = format_str.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def normalize(rawpoints, low=None, high=None):
    rawpoints = np.asanyarray(rawpoints)

    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)

    if high is None:
        high = np.max(maxs)

    if low is None:
        low = np.min(mins)

    input_range = maxs - mins
    wanted_range = high - low
    higher_fitted = maxs - rawpoints
    scaled_hghr_fitd = wanted_range * higher_fitted
    rescaled = scaled_hghr_fitd / input_range
    normalized = high - rescaled

    return normalized
