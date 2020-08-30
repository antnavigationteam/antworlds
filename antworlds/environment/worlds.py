import pathlib
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from antworlds import MESHES_PATH


def import_world(world_folder):
    """
    Loads compressed world data into two dicts
    """

    if pathlib.Path(world_folder).exists():
        path = pathlib.Path(world_folder)
    else:
        path = MESHES_PATH / str(world_folder).lower()

    loaded_fullworld = load_mesh(path / 'mesh_full.npz')
    loaded_ground = load_mesh(path / 'mesh_ground.npz')

    if 'tree' not in loaded_ground:
        loaded_ground['tree'] = generate_tree(loaded_ground['vertices']['pos'])

    return loaded_fullworld, loaded_ground


def generate_tree(vertices, tree_3d=False):
    """Generates a KD-Tree of all the vertices, to be used for quick z-coords computing.
        Default KD-Tree is 2D since we want vertical z projection, not 3D projection."""

    if tree_3d is True:
        tree = cKDTree(vertices, leafsize=10)
    else:
        tree = cKDTree(vertices[:, :2], leafsize=10)

    return tree


def extract_plydata(ply):

    vertex_count = ply['vertex'].count

    vertices = np.empty(vertex_count, dtype=[("pos", np.float32, 3),
                                             ("col", np.uint8, 4)])

    vertices['pos'][:, 0] = ply['vertex']['x']
    vertices['pos'][:, 1] = ply['vertex']['y']
    vertices['pos'][:, 2] = ply['vertex']['z']

    vertices['col'][:, 0] = ply['vertex']['red']
    vertices['col'][:, 1] = ply['vertex']['green']
    vertices['col'][:, 2] = ply['vertex']['blue']
    vertices['col'][:, 3] = 255

    faces = np.concatenate(ply['face']['vertex_indices']).reshape(-1, 3).astype(np.int32)

    return vertices, faces


def load_ply(ply_file):
    mesh = PlyData.read(ply_file)
    verts, faces = extract_plydata(mesh)
    loaded_dict = {'vertices': verts, 'faces': faces}
    return loaded_dict


def load_npz(npz_file):
    npz_file = np.load(npz_file, allow_pickle=False)
    return dict(npz_file)


def try_file(file):

    file = pathlib.Path(file)

    try_ply = (file.parent / (file.stem + '.ply'))
    try_npz = (file.parent / (file.stem + '.npz'))

    found = False
    if file.suffix == '.ply':
        if try_npz.is_file():
            print(f"Found already compressed version '{file.stem}.npz'")
            file = try_npz
            found = True

        elif file.is_file():
            file = file
            found = True

    elif file.suffix == '.npz':
        if not file.is_file() and try_ply.is_file():
            print(f"Found uncompressed version '{file.stem}.ply'")
            file = try_ply
            found = True

        elif file.is_file():
            file = file
            found = True

    elif file.suffix == '':
        if try_npz.is_file():
            file = try_npz
            found = True

        elif try_ply.is_file():
            print(f"Found uncompressed version '{file.stem}.ply'")
            file = try_ply
            found = True

    else:
        raise FileNotFoundError('This file type is not supported.')

    if found:
        return file
    return False


def load_mesh(file):

    # Try once directly
    found_file = try_file(file)

    # If failed, try in the default folder
    if not found_file:
        found_file = try_file(MESHES_PATH / file)

    # If failed, abandon
    if not found_file:
        raise FileNotFoundError('Cannot find specified file.')

    # If found, load it
    if found_file.suffix == '.ply':
        loaded_dict = load_ply(found_file)
        np.savez_compressed(found_file.parent / found_file.stem,
                            vertices=loaded_dict['vertices'],
                            faces=loaded_dict['faces'])
        print(f"Compressed file '{found_file.stem}.ply' into .npz")

    else:
        loaded_dict = load_npz(found_file)

    return loaded_dict


def subsample_vertices(world, level=1, xlims=None, ylims=None, zmin=None):

    vertices = world['vertices']['pos']
    shape = vertices.shape[0]

    r = shape % level

    subsampled = np.mean(vertices[:shape - r, :].reshape(-1, level, 3), axis=1)

    if zmin:
        subsampled = subsampled[subsampled[:, 2] > zmin]

    if xlims:
        subsampled = subsampled[(subsampled[:, 0] > xlims[0]) & (subsampled[:, 0] < xlims[1])]

    if ylims:
        subsampled = subsampled[(subsampled[:, 1] > ylims[0]) & (subsampled[:, 1] < ylims[1])]

    return subsampled


def compress_ply(where=MESHES_PATH):
    """
    Compress .ply file(s)
    """

    where = pathlib.Path(where)

    batch = where.rglob('*')

    for result in batch:
        if result.is_file() and result.suffix == '.ply':
            temp = load_mesh(result)
            np.savez_compressed(result.as_posix().replace('.ply', ''), vertices=temp['vertices'], faces=temp['faces'])
