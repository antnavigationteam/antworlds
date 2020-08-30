# coding=utf-8

from pathlib import Path
import numpy as np
import scipy.sparse

from antworlds import utils
from antworlds.insect_eye import ommatidia_funcs

from antworlds import OMMATIDIA_CACHE_PATH, IE_MODELS_PATH


def cubemap_pixels_directions(res):
    """
    Generates a pixel-direction list from a cubemap of passed resolution
    :param res: The cube resolution
    :return: The n*n pixels directions (n = resolution) in x, y, z
    """
    vals = np.linspace(-1, 1, res + 1)[:-1]

    temp_face = np.empty((res, res, 3))
    X = 1
    for z in range(res):

        Z = vals[z]
        this_row_vecs = np.ones((res, 3))

        for y in range(res):

            Y = vals[y]

            vector3 = np.array([X, Y, Z])

            v3norm = utils.unit_vector(vector3)  # Get direction of each pixel
            this_row_vecs[y] = v3norm

        temp_face[z] = np.flip(this_row_vecs, axis=0)

    # Direction and angle
    rotation_matrix_params = [((0, 0, 1), np.pi / 2),
                              ((0, 0, 1), -np.pi / 2),
                              ((0, -1, 0), np.pi / 2),
                              ((0, 1, 0), np.pi / 2),
                              ((0, 1, 0), 0),
                              ((0, 0, 1), np.pi)
                              ]

    sq_res = res * res
    pixel_dirs = np.empty((sq_res*6, 3))

    # For each of the 6 faces: Calculate the rotation matrix, and rotate the pixel directions
    for i, (direction, angle) in enumerate(rotation_matrix_params):

        rot_mat = utils.axis_rotation_matrix(direction, angle)

        # Slicer to fill the pixelm_dirs array
        s = i * sq_res
        e = (i + 1) * sq_res

        rdirs_currentface = np.array([np.dot(rot_mat, elem) for elem in temp_face.reshape(-1, 3)])

        pixel_dirs[s:e, :] = rdirs_currentface.astype(np.float32)

    return pixel_dirs


def get_ommatidium_wm(pixel_dirs, om_dir, acc_ang):
    """
    Maps the cubemap pixel directions to an ommatidium (thanks to its direction and acceptance angle,
    and passed through a Gaussian).
    :param pixel_dirs: Default cubemap pixel directions
    :param om_dir: Current Ommatidium direction in x, y, z
    :param acc_ang: Current Ommatidium acceptance angle
    :return: The ommatidium's map
    """

    om_dir_array = np.tile(om_dir, (pixel_dirs.shape[0], 1))

    angles = utils.rapid_angles_between(pixel_dirs, om_dir_array)
    wm = utils.gaussian(angles, acc_ang)

    return wm


class InsectEye(object):
    """
    Class that holds the faceted eye model
    :param ommatidia: A number of ommatidia
    :param cube_res: Resolution for the cubemap to map to
    :return: An Insect Eye model
    """

    def __init__(self,
                 ommatidia,
                 cube_res=64,
                 name=''
                 ):

        if name != '':
            lookup_file = (IE_MODELS_PATH / f'{name}.npz')
        else:
            lookup_file = (IE_MODELS_PATH / f'eye_om{ommatidia}_phi-default_cube{cube_res}.npz')

        if lookup_file.exists():
            self._from_file(lookup_file)

        else:
            self._builder(ommatidia, cube_res=cube_res, name=name)

    def _builder(self,
                 ommatidia,
                 acc_angles=None,
                 cube_res=64,
                 wm_clip_thresh=1e-5,
                 name='',
                 save=False):

        self.name = name
        self.cube_res = cube_res

        if np.array(ommatidia).ndim == 0:
            self.om_nb = ommatidia_funcs.assert_ommatidia_nb(ommatidia)

            if (OMMATIDIA_CACHE_PATH / f'om{self.om_nb}').exists():
                with np.load((OMMATIDIA_CACHE_PATH / f'om{self.om_nb}')) as f:
                    self.om_dirs = f['om_dirs']
                    self.lat = f['lat']
                    self.lon = f['lon']
            else:
                self.om_dirs, self.lon, self.lat = ommatidia_funcs.ommatidia_builder(self.om_nb, cache=True)

        else:
            self.om_dirs = ommatidia
            self.om_nb = ommatidia.shape[0]

            self.lon, self.lat = utils.eul2geo(self.om_dirs[:, 0], self.om_dirs[:, 1], self.om_dirs[:, 2])

        acc_angles = np.atleast_1d(acc_angles)
        if len(acc_angles) != 1 and len(acc_angles) != self.om_nb:
            raise AttributeError("Incorrect length vector for acceptance angles!")

        if np.unique(acc_angles).size > 1:
            self.acc_angles = acc_angles
            self.is_uniform = False
            self.is_optimal = False

            file_lookup = None      # Can't reload from file because acceptance angles are multiple

        else:
            optimal_angl = utils.angle_between(self.om_dirs[0, :], self.om_dirs[1, :])

            if None in acc_angles:
                self.acc_angles = optimal_angl
                self.is_uniform = True
                self.is_optimal = True
                file_lookup = (IE_MODELS_PATH / f'eye_om{self.om_nb}_phi-default_cube{self.cube_res}.npz')

            else:
                self.acc_angles = np.unique(acc_angles)
                self.is_uniform = True
                if np.isclose(self.acc_angles, optimal_angl):
                    self.is_optimal = True
                    file_lookup = (IE_MODELS_PATH / f'eye_om{self.om_nb}_phi-default_cube{self.cube_res}.npz')
                else:
                    self.is_optimal = False
                    file_lookup = (IE_MODELS_PATH / f'eye_om{self.om_nb}_phi{np.rad2deg(self.acc_angles)}deg_cube{self.cube_res}.npz')

        if file_lookup is not None and file_lookup.exists():
            with np.load(file_lookup) as f:
                self.om_weights = scipy.sparse.csc_matrix((f['omw_spm_data'],
                                                           f['omw_spm_indices'],
                                                           f['omw_spm_indptr']),
                                                          shape=f['omw_spm_shape'])

            print('Previous corresponding eye found and recovered.')

        else:
            # Generate the weightmaps
            print(f'Generating Ommatidia weightmaps with compression\n')

            # Generate the default pixel directions for a cubemap (of resolution 'cube_res')
            px_dirs = cubemap_pixels_directions(cube_res)
            pixels_nb = px_dirs.shape[0]

            weight_maps = np.zeros((self.om_nb, pixels_nb))
            for o in range(self.om_nb):
                dq = self.om_dirs[o]
                drq = self.acc_angles if self.is_uniform else self.acc_angles[o]

                utils.print_progress(o, self.om_nb, prefix='Progress:', suffix='Complete', bar_length=20)

                # Get the ommatidium's weightmap that maps its visual field to a normal cubemap
                wm = get_ommatidium_wm(px_dirs, dq, drq)

                # Clip what is considered insignificant
                wm = np.choose(wm < wm_clip_thresh, (wm, 0))

                # Normalise so that the om. with more pixels do not have a stronger contribution to the output
                wm_norm = wm / wm.sum()

                weight_maps[o, :] = wm_norm

            # Convert to sparse matrix to spare a lot of memory
            self.om_weights = scipy.sparse.csc_matrix(weight_maps, dtype=np.float32)

            print(f'\n  Compressed to {self.om_weights.data.size * self.om_weights.data.itemsize * 1e-6:.2f} Mb'
                  f'(from {weight_maps.size * weight_maps.itemsize * 1e-6:.2f} Mb)')

        if save:
            self.export()

    def _from_file(self, file):

        with np.load(file) as f:
            self.name = f['name']
            self.om_dirs = f['om_dirs']
            self.om_nb = self.om_dirs.shape[0]
            self.acc_angles = f['acc_angles']
            self.cube_res = f['cube_res']
            self.lon = f['lon']
            self.lat = f['lat']
            self.om_weights = scipy.sparse.csc_matrix((f['omw_spm_data'], f['omw_spm_indices'], f['omw_spm_indptr']),
                                                      shape=f['omw_spm_shape'])
            self.is_uniform = f['is_uniform'].item()
            self.is_optimal = f['is_optimal'].item()

    @classmethod
    def load(cls, filename):

        to_load = Path(filename)

        file = IE_MODELS_PATH / (to_load.stem + '.npz')

        self = cls.__new__(cls)
        self._from_file(file)
        return self

    @classmethod
    def build(cls, ommatidia, acc_angles=None, cube_res=64, wm_clip_thresh=1e-5, name='', save=False):
        self = cls.__new__(cls)
        self._builder(ommatidia,
                      acc_angles=acc_angles,
                      cube_res=cube_res,
                      wm_clip_thresh=wm_clip_thresh,
                      name=name,
                      save=save)
        return self

    def export(self, savename=''):

        if self.name == '' and savename == '':
            if self.is_uniform and self.is_optimal:
                phi_str = '-default'
            elif self.is_uniform:
                phi_str = f'{np.rad2deg(self.acc_angles)}deg'
            else:
                phi_str = 'multi'
            savename = IE_MODELS_PATH / f'eye_om{self.om_nb}_phi{phi_str}_cube{self.cube_res}.npz'

        elif savename == '':
            savename = IE_MODELS_PATH / self.name

        else:
            savename = IE_MODELS_PATH / Path(savename).stem

        np.savez_compressed(savename,
                            om_dirs=self.om_dirs,
                            acc_angles=self.acc_angles,
                            cube_res=self.cube_res,
                            lon=self.lon,
                            lat=self.lat,
                            omw_spm_data=self.om_weights.data,
                            omw_spm_indices=self.om_weights.indices,
                            omw_spm_indptr=self.om_weights.indptr,
                            omw_spm_shape=self.om_weights.shape,
                            name=self.name,
                            is_uniform=self.is_uniform,
                            is_optimal=self.is_optimal)

    def __repr_(self):
        return f'InsectActor Eye {self.name}: ({self.om_nb} ommatidia)'


if __name__ == '__main__':

    # ## Make a weightmap from a file
    #
    # input_filename = 'buchner71.csv'
    # input_filepath = Path('data/ommatidia_data/' + input_filename)
    #
    # csv = np.genfromtxt(input_filepath, delimiter=",")
    #
    # assert (csv.ndim is 2), 'Data file has the wrong dimensionality, please check.'
    # assert (csv.shape[1] == 3), 'Data file must have 3 columns that specify ommatidia directions' \
    #                             'in (normalised) cartesian x, y, z coordinates.'
    #
    # droso_dirs = csv
    #
    # droso_eye = InsectEye.build(droso_dirs,
    #                             name='drosophila')

    ## Or test the module using a 6-axis eye that looks in each cardinal direction

    ortho_dirs = np.array([
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)
    ], dtype=np.float32)

    acceptance_angles = np.deg2rad(5)   # 0.1, # 745329
    filename_str = 'ortho_eye'

    ortho_eye = InsectEye.build(ortho_dirs,
                                acc_angles=acceptance_angles,
                                name=filename_str,
                                save=True)

