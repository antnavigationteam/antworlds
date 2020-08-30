#! /usr/bin/env python
import numpy as np
import imageio

import contextlib
with contextlib.redirect_stdout(None):
    import pygame

import OpenGL.GL as gl

from antworlds import utils

from antworlds.insect_eye.insect_eye import InsectEye
from antworlds.engine import camera as cam
from antworlds.engine.rendering import CubemapRenderer, SkyboxObject, RectangleRenderer, HabitatMesh, ConesRenderer


class RegularActor(object):

    def __init__(self,
                 mesh,
                 display_size=(1080, 720),
                 cube_res=64,
                 skybox=(0.0, 0.0, 1.0),
                 display_output=True):

        self.viewport_size = display_size
        self.display_output = display_output

        self.eye_output = np.empty((self.viewport_size[1], self.viewport_size[0], 3), dtype=np.uint8)

        self.count = 0
        self.clock = pygame.time.Clock()

        # Setup camera
        self.camera = cam.Camera()
        proj = self.camera.perspective_mat(fovy=90, aspect=1.0)
        # view = self.camera.lookat_mat(np.array([0, 0, -1]), np.array([1, 1, -1]))
        # scale = self.camera.scale_mat(1)

        # Initialise OpenGL context
        pygame.init()
        pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

        # Set general OpenGL params
        gl.glClearColor(0.2, 0.2, 0.2, 1.0)  # Gray
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_TEXTURE_CUBE_MAP_SEAMLESS)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glClearDepth(1.0)

        if type(skybox) is str:
            self.skybox_object = SkyboxObject(skybox, proj)
        else:
            skybox = np.asanyarray(skybox)
            array_norm = utils.normalize(skybox[:3], low=0, high=1)
            gl.glClearColor(*array_norm, 1.0)
            self.skybox_object = None

        self.backend_renderer = CubemapRenderer(cube_res, proj)
        self.mesh_object = HabitatMesh(mesh)

        self.render2d = RectangleRenderer(self.viewport_size)

        # self.update()

    def render_backend(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.backend_renderer.fbo_id)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        view = self.camera.get_view_matrix(translate=True)

        self.backend_renderer.draw_to_texture(view)
        self.mesh_object.draw(view, self.backend_renderer)

        if self.skybox_object is not None:
            view = self.camera.get_view_matrix(translate=False)
            self.skybox_object.draw(view, self.backend_renderer)

        self.finalize_render()

    def finalize_render(self):
        self.render2d.draw(self.backend_renderer)

    def get_eye_output(self):
        self.eye_output[...] = self.render2d.read()

    def update(self):
        self.render_backend()
        self.get_eye_output()

        if self.display_output:
            self.display()

    def snapshot(self, save=False):
        img_data = self.eye_output
        if save:
            imageio.imsave(f'snapshot{self.count}-x{self.camera.x},y{self.camera.y},th{self.camera.rx}.png', img_data)
        return img_data

    def pre_display(self):
        pass

    def display(self):
        self.pre_display()

        # self.count += 1
        self.clock.tick()

        if self.display_output:
            # Press Esc to quit function or P to screengrab
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        quit()
                    if event.key == pygame.K_p:
                        self.snapshot(save=True)

            pygame.display.set_caption(f"FPS: {self.clock.get_fps()}")

            pygame.display.flip()
            pygame.event.pump()


class InsectActor(RegularActor):

    def __init__(self,
                 *args,
                 eyemodel,
                 **kwargs):

        # Check if cube_res are the same
        if kwargs.get('cube_res', eyemodel.cube_res) != eyemodel.cube_res:
            print("Asked cube resolution different from the one in this eye object!\nUsing the eye one instead.")
            kwargs['cube_res'] = eyemodel.cube_res

        super(InsectActor, self).__init__(*args, **kwargs)

        self.eyemodel = eyemodel

        self.eye_output = np.empty((self.eyemodel.om_nb, 3), dtype=np.float32)

        self.ommatidia_coords = np.column_stack((utils.normalize(self.eyemodel.lon, low=-1.0, high=1.0),
                                                utils.normalize(self.eyemodel.lat, low=-1.0, high=1.0)))

        self.render2d = ConesRenderer(self.viewport_size)
        # self.render2d = RectangleRenderer(self.viewport_size)

    def finalize_render(self):
        pass

    def get_eye_output(self):
        colors_buffer = self.backend_renderer.read().reshape(6, -1, 4)
        colors = np.flip(colors_buffer, axis=1)
        # print(colors[:, :3])
        self.eye_output[...] = self.eyemodel.om_weights.dot(colors.reshape(-1, 4)[:, :3].astype(np.float32))/255.0

    def pre_display(self):
        self.render2d.draw(self.eye_output, self.ommatidia_coords)
        # self.render2d.draw(self.backend_renderer)

    def snapshot(self, save=False):
        img_data = self.render2d.read()
        if save:
            imageio.imsave(f'snapshot{self.count}-x{self.camera.x},y{self.camera.y},th{self.camera.rx}.png', img_data)
        return img_data


if __name__ == "__main__":

    # simulation = InsectActor(eyemodel=InsectEye(1962),
    #                     mesh="desert",
    #                     skybox='bright_day_ud')

    simulation = RegularActor(mesh="desert",
                              skybox='bright_day_ud')

    steps = 1000

    print("Press 'ESC' key to quit.")

    for theta in np.linspace(0, np.pi, steps):

        simulation.camera.x = 0
        simulation.camera.y = 0
        simulation.camera.z = 0.5
        simulation.camera.rx = theta
        simulation.camera.ry = -np.pi / 2

        simulation.update()
