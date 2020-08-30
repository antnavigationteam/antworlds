import contextlib
with contextlib.redirect_stdout(None):
    import pygame

import numpy as np
from skimage.transform import resize

from antworlds.environment import physics


def get_eye_image(sim, ground,
                  x=0.0, y=0.0, z=None,
                  extract_channel=None,
                  theta_radians=None,
                  theta_degrees=None,
                  save=False,
                  z_offset=0.03
                  ):

    if theta_radians is None and theta_degrees is None:
        theta = 0.0
    elif theta_radians is None:
        theta = np.deg2rad(theta_degrees)
    else:
        theta = theta_radians

    x, y = float(x), float(y)  # Needed to force floats and not 0d numpy arrays

    if z is None or np.isnan(float(z)):
        z, normal = physics.get_z((x, y), ground, offset=z_offset)
    else:
        z = float(z)

    # Set viewport position and aim
    sim.camera.x = x
    sim.camera.y = y
    sim.camera.z = z
    # sim.camera.rx = -theta - np.pi / 2  # for Pyeye's theta reference
    sim.camera.rx = -theta  # for Pyeye's theta reference
    sim.camera.ry = -np.pi / 2

    # Refresh the simulation
    sim.update()

    im_gen = sim.snapshot(save=save)

    if extract_channel is not None:
        return im_gen[:, :, extract_channel]

    return im_gen


def get_view_resized(x, y, z, th, sim, world_ground, resolution=360):
    """
    Outputs a downsampled and cropped snapshot of given coordinates
    """

    view = get_eye_image(sim, world_ground,
                         x=x, y=y, z=z,
                         theta_degrees=th, extract_channel=2,
                         save=False)

    scalefactor = resolution / sim.viewport_size[0]

    width = int(np.floor(sim.viewport_size[0] * scalefactor))
    height = int(np.floor(sim.viewport_size[1] * scalefactor))

    view_lowres = resize(view, (height, width))

    view_cut = view_lowres[:int(height / 2), :]

    return view_cut


