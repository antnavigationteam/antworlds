'''
This module is included in order to manually explore a given mesh with given eye parameters. The module can be wrapped
for given parameters or the demo configuration can be used in order to check that the tool is working.
'''

# TODO - add joystick control functionality
# TODO - improve mouse functionality (currently where the initial click is made set the rotation

import pygame
from antworlds.insect_eye.insect_eye import InsectEye
from antworlds.engine import actors as s


class WASD(object):

    def __init__(self,
                 eye=None,
                 speed=0.25,
                 mesh='desert', skybox='bright_day_ud'):

        self.eye = eye

        if self.eye is not None:
            self.sim = s.InsectActor(eyemodel=self.eye, mesh=mesh, skybox=skybox)
        else:
            self.sim = s.RegularActor(mesh=mesh, skybox=skybox)

        self.translation_speed = speed
        self.mouse_sensitivity = speed * 0.1

        # Set controls
        self.forward = 'w'
        self.backward = 's'
        self.left = 'a'
        self.right = 'd'
        self.up = 't'
        self.down = 'g'

        # Set mouse properties
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        print("Press 'ESC' key to quit."
              "\n  z = move forward"
              "\n  s = move backward"
              "\n  q = move left"
              "\n  d = move right"
              "\n  t = climb up"
              "\n  g = descend")

    def controls_3d(self):

        keys = dict((chr(i), int(v)) for i, v in enumerate(pygame.key.get_pressed()) if i < 256)

        mouse_dx, mouse_dy = pygame.mouse.get_rel()

        # move forward-back or right-left
        fwd = self.translation_speed * (keys[self.forward] - keys[self.backward])
        strafe = self.translation_speed * (keys[self.left] - keys[self.right])
        up = self.translation_speed * (keys[self.up] - keys[self.down])

        return mouse_dx, mouse_dy, fwd, strafe, up

    def translate(self, vx, vy, vz, direction):
        vx = vx * direction
        vy = vy * direction
        vz = vz * direction
        self.sim.camera.x = self.sim.camera.x + vx  # TODO - cam module converts test vector to view
        self.sim.camera.y = self.sim.camera.y + vy
        self.sim.camera.z = self.sim.camera.z + vz

    def main(self):

        mouse_dx, mouse_dy, fwd, strafe, up = self.controls_3d()

        # calculate new viewing direction
        self.sim.camera.rx = self.sim.camera.rx + mouse_dx * self.mouse_sensitivity
        self.sim.camera.ry = self.sim.camera.ry + mouse_dy * self.mouse_sensitivity

        # translate to new position
        vx, vy, vz = self.sim.camera.forward_vector()
        self.translate(vx, vy, vz, -fwd)

        vx, vy, vz = self.sim.camera.right_vector()
        self.translate(vx, vy, vz, -strafe)

        vx, vy, vz = self.sim.camera.up_vector()
        self.translate(vx, vy, vz, -up)

        # update display
        self.sim.step()
        pygame.display.flip()
        pygame.event.pump()


class ZSQD(WASD):

    def __init__(self, *args, **kwargs):
        super(ZSQD, self).__init__(*args, **kwargs)

        self.forward = 'z'
        self.left = 'q'


if __name__ == "__main__":

    ie = InsectEye(1962)

    controls = ZSQD()

    while True:
        controls.main()
