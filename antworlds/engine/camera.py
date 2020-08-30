'''
This module provides the functions required in order to configure an opengl camera for visualising 3D scenes
'''

import numpy as np
from antworlds.utils import unit_vector


class Camera(object):

    def __init__(self,
                 x=0,
                 y=0,
                 z=0,
                 rx=0,           # horizontal angle
                 ry=-np.pi/2,    # vertical angle - must initialise less than pi
                 rz=0,
                 mat=np.eye(4, dtype=np.float32)
                 ):

        self.x = x
        self.y = y
        self.z = z
        self.rx = rx     # horizontal angle
        self.ry = ry     # vertical angle - must initialise less than pi
        self.rz = rz
        self.mat = mat

    # def get_view_matrix(self, translate=True, mult_order='XYZ'):
    #     '''
    #     generates a view matrix based on the 6dof location of the camera
    #     :param translate:
    #     :return:
    #     '''
    #
    #     yaw = self.rz
    #     pitch = self.ry
    #     roll = self.rx
    #
    #     self.mat = np.eye(4, dtype=np.float32)
    #
    #     # TODO - these can be replaced with standard 6-DOF matrices
    #     yawMatrix = np.matrix([
    #         [np.cos(yaw), -np.sin(yaw), 0],
    #         [np.sin(yaw), np.cos(yaw), 0],
    #         [0, 0, 1]
    #     ])
    #     pitchMatrix = np.matrix([
    #         [np.cos(pitch), 0, np.sin(pitch)],
    #         [0, 1, 0],
    #         [-np.sin(pitch), 0, np.cos(pitch)]
    #     ])
    #     rollMatrix = np.matrix([
    #         [1, 0, 0],
    #         [0, np.cos(roll), -np.sin(roll)],
    #         [0, np.sin(roll), np.cos(roll)]
    #     ])
    #
    #     if mult_order == 'ZYX':
    #         R = yawMatrix * pitchMatrix * rollMatrix
    #     elif mult_order == 'XYZ':
    #         R = rollMatrix * pitchMatrix * yawMatrix
    #     else:
    #         raise ('Unknown multiplication order')
    #
    #     self.mat[0:3, 0:3] = R
    #
    #     if translate:
    #         self.mat = self.translate((-self.x, -self.y, -self.z))
    #     return self.mat

    def get_view_matrix(self, translate=True):

        self.mat = np.eye(4, dtype=np.float32)
        self.mat = self.rotate((np.cos(self.rx), 0, np.sin(self.rx)), self.ry)
        self.mat = self.rotate((0, -1, 0), -self.rx)
        if translate:
            self.mat = self.translate((-self.x, -self.y, -self.z))
        return self.mat

    def translate(self, value):
        x, y, z = value
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [x, y, z, 1]
        ], dtype=np.float32)
        return matrix @ self.mat

    def rotate(self, vector, angle):
        # TODO - is this column or row major?
        x, y, z = unit_vector(vector)
        s = np.sin(angle)
        c = np.cos(angle)
        m = 1 - c
        matrix = np.array([
            [m * x * x + c, m * x * y - z * s, m * z * x + y * s, 0],
            [m * x * y + z * s, m * y * y + c, m * y * z - x * s, 0],
            [m * z * x - y * s, m * y * z + x * s, m * z * z + c, 0],
            [0, 0, 0, 1]
        ])
        return matrix @ self.mat

    # def forward_vector(self):
    #     # derived from http://www.opengl-tutorial.org/beginners-tutorials/tutorial-6-keyboard-and-mouse/
    #     hor_angle = self.rx - np.pi/2
    #     return np.sin(hor_angle), 0, np.cos(hor_angle)
    #
    # def right_vector(self):
    #     m = np.cos(self.ry)
    #     vx = np.cos(self.rx - np.pi / 2) * m
    #     vy = np.sin(self.ry)
    #     vz = np.sin(self.rx - np.pi / 2) * m
    #     return vx, vy, vz

    def forward_vector(self):
        vx = np.sin(self.rx - np.pi/2) * (-np.sin(self.ry))
        vy = np.cos(self.rx - np.pi/2) * (-np.sin(self.ry))
        vz = np.cos(self.ry)
        return vx, vy, vz

    def right_vector(self):
        vx = -np.sin(self.rx)
        vy = -np.cos(self.rx)
        vz = np.cos(self.ry)
        return vx, vy, vz

    def up_vector(self):
        # derived from http://www.opengl-tutorial.org/beginners-tutorials/tutorial-6-keyboard-and-mouse/
        return np.cross(self.right_vector(), self.forward_vector())

    @staticmethod
    def lookat_mat(eye, lookat, up=np.array([0, 0, -1])):
        ez = unit_vector(np.subtract(eye, lookat))
        ex = unit_vector(np.cross(up, ez))
        ey = unit_vector(np.cross(ez, ex))

        rmat = np.eye(4)
        rmat[0, :3] = ex
        rmat[1, :3] = ey
        rmat[2, :3] = ez

        tmat = np.eye(4)
        tmat[:3, 3] = -eye

        return np.dot(rmat, tmat).T

    @staticmethod
    def ortho_mat(left, right, bottom, top, near, far):
        """Creates a perspective projection matrix using the specified near
        plane dimensions
        """
        dx = (right - left)
        dy = (top - bottom)
        dz = (far - near)

        a = -(right + left) / dx
        b = (top + bottom) / dy
        c = -2. / dz
        d = (-(far + near)) / dz
        e = 2. * near / dx
        f = 2. * near / dy

        return np.array([
            [e, 0., 0., 0.],
            [0., f, 0., 0.],
            [0., 0., c, 0.],
            [a, b, d, 1.]], dtype=np.float32)

    @staticmethod
    def perspective_mat(fovy=90, aspect=1.0, near=0.01, far=100):
        """Creates a perspective projection matrix using the specified near
        plane dimensions
        """
        y_max = near * np.tan(fovy * np.pi / 360.0)
        x_max = y_max * aspect

        left, right, bottom, top = -x_max, x_max, -y_max, y_max

        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = -(far + near) / (far - near)
        d = -2.0 * far * near / (far - near)
        e = 2.0 * near / (right - left)
        f = 2.0 * near / (top - bottom)

        return np.array([
            [e, 0., 0., 0.],
            [0., f, 0., 0.],
            [a, b, c, -1.],
            [0., 0., d, 0]], dtype=np.float32)

    @staticmethod
    def scale_mat(xyz):
        if np.asanyarray(xyz).ndim == 0:
            return np.eye(4, dtype=np.float32) * xyz

        x, y, z = xyz
        return np.array([[x, 0, 0, 0],
                         [0, y, 0, 0],
                         [0, 0, z, 0],
                         [0, 0, 0, 1]], dtype=np.float32)
