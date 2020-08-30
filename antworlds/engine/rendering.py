import ctypes

from PIL.Image import open
import numpy as np

import OpenGL.GL as gl
import OpenGL.GL.shaders as shaders

from antworlds import utils
from antworlds.environment import worlds
from antworlds import CUBEMAPS_PATH, SHADERS_PATH


def make_program(vertex_code_file, fragment_code_file, geometry_code_file=None):

    vertex_code = (SHADERS_PATH / vertex_code_file).read_text()
    fragment_code = (SHADERS_PATH / fragment_code_file).read_text()

    if geometry_code_file is not None:
        geometry_code = (SHADERS_PATH / geometry_code_file).read_text()
        return shaders.compileProgram(shaders.compileShader(vertex_code, gl.GL_VERTEX_SHADER),
                                      shaders.compileShader(fragment_code, gl.GL_FRAGMENT_SHADER),
                                      shaders.compileShader(geometry_code, gl.GL_GEOMETRY_SHADER))

    return shaders.compileProgram(shaders.compileShader(vertex_code, gl.GL_VERTEX_SHADER),
                                  shaders.compileShader(fragment_code, gl.GL_FRAGMENT_SHADER))


def load_cubemap(pack_name):

    faces = [gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
             gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
             gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
             gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
             gl.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
             gl.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z]

    imfiles = ['posx.png',
               'negx.png',
               'posy.png',
               'negy.png',
               'negz.png',
               'posz.png']

    texture_id = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, texture_id)    # Bind to the previously activated texture unit (i.e. outside of this function)

    for i in range(6):
        image = open(CUBEMAPS_PATH / pack_name / imfiles[i])
        img_data = image.tobytes("raw", "RGBX", 0, -1)
        ix, iy = image.size

        gl.glTexImage2D(faces[i],
                        0,
                        gl.GL_RGBA8,
                        ix,
                        iy,
                        0,
                        gl.GL_RGBA,
                        gl.GL_UNSIGNED_BYTE,
                        img_data)

    gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

    return texture_id


class CubemapRenderer(object):

    rot = np.pi / 2
    matrices = [utils.rot2homo(utils.axis_rotation_matrix(np.array([0, 1, 0]), rot * 3)),   # Correct for 1 way
                utils.rot2homo(utils.axis_rotation_matrix(np.array([0, 1, 0]), -3 * rot)),
                utils.rot2homo(utils.axis_rotation_matrix(np.array([1, 0, 0]), -rot)),
                utils.rot2homo(utils.axis_rotation_matrix(np.array([1, 0, 0]), rot)),
                utils.rot2homo(np.eye(3, dtype=np.float32)),                                # Correct side
                utils.rot2homo(utils.axis_rotation_matrix(np.array([0, 1, 0]), 2 * rot))]   # Correct for 1 way

    eyes_up = matrices[2]

    # TODO - remove these

    def __init__(self, cube_res, cam_proj):

        dummy_data = np.zeros((cube_res * cube_res, 4), dtype=np.uint8)     # 4 because RGBA

        self.cube_res = cube_res

        self.program = make_program('cubemap.vert', 'cubemap.frag', 'cubemap.glsl')
        gl.glUseProgram(self.program)

        # Get uniforms: handles for projection and scale do not change in one simulation. The view does.
        proj_handle = gl.glGetUniformLocation(self.program, "projection")
        gl.glUniformMatrix4fv(proj_handle, 1, False, cam_proj)

        self.view_handle = gl.glGetUniformLocation(self.program, "view")

        # Create Framebuffer
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)

        # make sure we're on the default unit 0
        gl.glActiveTexture(gl.GL_TEXTURE0)

        # Create a texture
        self.texobj_colour_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texobj_colour_id)

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        for i in range(6):
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0,                      # level-of-detail (n th mipmap, default = 0)
                            gl.GL_RGBA8,            # Best to also specifiy precision (8) explicitlty
                            self.cube_res,          # Width
                            self.cube_res,          # Height
                            0,                      # Must be 0. Documentation says so.
                            gl.GL_RGBA,
                            gl.GL_UNSIGNED_BYTE,
                            dummy_data)

        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER,
                                gl.GL_COLOR_ATTACHMENT0,    # Attachment point
                                self.texobj_colour_id,      # Texture ID
                                0)                          # Mipmap level of texture to attach

        # Release the texture
        # gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        # Bind a second texture for the depth
        self.texobj_depth_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texobj_depth_id)

        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)

        for i in range(6):
            gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                            0,                          # level-of-detail (n th mipmap, default = 0)
                            gl.GL_DEPTH_COMPONENT24,    # Best to also specifiy precision (8) explicitlty
                            self.cube_res,              # Width
                            self.cube_res,              # Height
                            0,                          # Must be 0. Documentation says so.
                            gl.GL_DEPTH_COMPONENT,
                            gl.GL_UNSIGNED_INT,
                            dummy_data)

        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER,
                                gl.GL_DEPTH_ATTACHMENT,     # Attachment point
                                self.texobj_depth_id,       # Texture ID
                                0)                          # Mipmap level of texture to attach

        # Release the texture
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        # Release framebuffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Release bound program
        gl.glUseProgram(0)

        self.img_buffer = np.zeros((6, self.cube_res * self.cube_res * 4), dtype=np.uint8)

    def draw_to_texture(self, view):

        gl.glUseProgram(self.program)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        gl.glViewport(0, 0, self.cube_res, self.cube_res)

        gl.glUniformMatrix4fv(self.view_handle,
                              1,
                              gl.GL_FALSE,
                              view)

        # gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, self.texobj_colour_id, 0)
        # gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, self.texobj_depth_id, 0)

        for i in range(6):
            rot_mat = self.eyes_up @ view @ self.matrices[i]
            gl.glUniformMatrix4fv(self.view_handle,
                                  1,
                                  gl.GL_FALSE,
                                  rot_mat)

            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,
                                      gl.GL_COLOR_ATTACHMENT0,
                                      gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                      self.texobj_colour_id,
                                      0)

            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,
                                      gl.GL_DEPTH_ATTACHMENT,
                                      gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                      self.texobj_depth_id,
                                      0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

    def read(self):

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, self.texobj_colour_id)

        for i in range(6):
            gl.glGetTexImage(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                             0,
                             gl.GL_RGBA,
                             gl.GL_UNSIGNED_BYTE,
                             self.img_buffer[i, :])

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        return self.img_buffer


class HabitatMesh(object):
    def __init__(self, mesh):

        if type(mesh) is dict:
            mesh_data = mesh
        else:
            mesh_data = worlds.load_mesh(mesh)

        vertex_data = mesh_data['vertices']
        faces_data = mesh_data['faces']

        self.nb_faces = faces_data.shape[0]

        # Make a VAO
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)

        # Make a VBO to hold vertex coords and vertex indices
        vbo_ids = gl.glGenBuffers(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_ids[0])

        # Data format
        stride = vertex_data.strides[0]
        offset_pos = ctypes.c_void_p(0)
        offset_col = ctypes.c_void_p(vertex_data.dtype['pos'].itemsize)

        coords_loc = 0              # Must match the layout in the cubemap shader

        gl.glEnableVertexAttribArray(coords_loc)
        gl.glVertexAttribPointer(coords_loc,
                                 3,                     # XYZ so size 3
                                 gl.GL_FLOAT,
                                 True,
                                 stride,
                                 offset_pos)

        colour_loc = 1              # Must match the layout in the cubemap shader

        gl.glEnableVertexAttribArray(colour_loc)
        gl.glVertexAttribPointer(colour_loc,
                                 4,                     # RGBA so size 4
                                 gl.GL_UNSIGNED_BYTE,
                                 True,                  # Unsigned byte values [0, 255] need to be [0, 1], so normalize
                                 stride,
                                 offset_col)

        # Send the data over to the VBO
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        vertex_data.nbytes,
                        vertex_data,
                        gl.GL_STATIC_DRAW)

        # Send indices to the IBO
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER,
                        vbo_ids[1])

        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                        faces_data.nbytes,
                        faces_data,
                        gl.GL_STATIC_DRAW)

        gl.glBindVertexArray(0)

    def draw(self, view, renderer):

        gl.glUseProgram(renderer.program)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, renderer.fbo_id)
        gl.glViewport(0, 0, renderer.cube_res, renderer.cube_res)

        gl.glUniformMatrix4fv(renderer.view_handle,
                              1,
                              gl.GL_FALSE,
                              view)

        # Select mesh VAO
        gl.glBindVertexArray(self.vao_id)

        # Bind to the active texture
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, renderer.texobj_colour_id)

        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, renderer.texobj_colour_id, 0)
        gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, renderer.texobj_depth_id, 0)

        gl.glDrawElements(gl.GL_TRIANGLES,          # Mode
                          self.nb_faces * 3,        # Count
                          gl.GL_UNSIGNED_INT,       # Type
                          ctypes.c_void_p(0))       # Element array buffer offset (None)

        # Release VAO, FBO and program
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)


class SkyboxObject(object):

    # We are forced to use 36 vertices and not 8 because the normals are different on each face
    cube_vert_coords = np.array(
        [-1., 1., -1., -1., -1., -1., 1., -1., -1., 1., -1., -1., 1., 1., -1., -1., 1., -1., -1., -1., 1., -1., -1.,
         -1., -1., 1., -1., -1., 1., -1., -1., 1., 1., -1., -1., 1., 1., -1., -1., 1., -1., 1., 1., 1., 1., 1., 1., 1.,
         1., 1., -1., 1., -1., -1., -1., -1., 1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., -1., -1., 1., -1.,
         1., -1., 1., 1., -1., 1., 1., 1., 1., 1., 1., -1., 1., 1., -1., 1., -1., -1., -1., -1., -1., -1., 1., 1., -1.,
         -1., 1., -1., -1., -1., -1., 1., 1., -1., 1.], dtype=np.float32)

    # TODO - Look at clipping distance

    eyes_up = utils.rot2homo(utils.axis_rotation_matrix(np.array([1, 0, 0]), -np.pi/2))

    # TODO - remove this matrix

    def __init__(self, skybox_name, cam_proj):

        # self.program = make_program('skybox_simple.vert', 'skybox_simple.frag')
        self.program = make_program('skybox.vert', 'skybox.frag', 'skybox.glsl')
        gl.glUseProgram(self.program)

        # Get uniforms: handles for projection and resolution do not change in one simulation. The view does.
        proj_handle = gl.glGetUniformLocation(self.program, "projection")
        self.skybox_view_handle = gl.glGetUniformLocation(self.program, "view")

        gl.glUniformMatrix4fv(proj_handle, 1, False, cam_proj)

        ### Activate texture unit 1
        tex_unit = 1

        gl.glActiveTexture(gl.GL_TEXTURE0 + tex_unit)
        self.texobj_skybox_id = load_cubemap(skybox_name)

        skybox_texture_loc = gl.glGetUniformLocation(self.program, "skybox_cubemap")
        gl.glUniform1i(skybox_texture_loc, tex_unit)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        ###

        # Make a VAO
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        self.vao_len = 36

        # Make a VBO
        self.vbo_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)

        # Bind VBO to VAO with the index of the correct attribute (loc)
        loc = gl.glGetAttribLocation(self.program, 'position')
        gl.glEnableVertexAttribArray(loc)

        # Describe the position data layout in the buffer
        gl.glVertexAttribPointer(loc,
                                 3,                                 # 3 vertices per triangle
                                 gl.GL_FLOAT,                       # Type
                                 False,                             # Values already normalised, no need to
                                 0,                                 # Stride
                                 ctypes.c_void_p(0))                # Array buffer offset (None)

        # Send the data over to the buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        self.cube_vert_coords.nbytes,
                        self.cube_vert_coords,
                        gl.GL_STATIC_DRAW)

        # Unbind from the VAO first (Important ! Will fail otherwise)
        gl.glBindVertexArray(0)

        # Unbind the buffer
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # Release bound program
        gl.glUseProgram(0)

    def draw(self, view, renderer):

        # gl.glDepthMask(False)

        gl.glUseProgram(self.program)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, renderer.fbo_id)
        gl.glViewport(0, 0, renderer.cube_res, renderer.cube_res)

        # Select skybox VAO
        gl.glBindVertexArray(self.vao_id)

        rot_mat = self.eyes_up @ view
        gl.glUniformMatrix4fv(self.skybox_view_handle, 1, gl.GL_FALSE, rot_mat)

        # # Bind to the active texture
        # gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, renderer.texobj_colour_id)
        #
        # gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, renderer.texobj_colour_id, 0)
        # gl.glFramebufferTexture(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, renderer.texobj_depth_id, 0)

        gl.glDrawArrays(gl.GL_TRIANGLES,
                        0,
                        self.vao_len)

        # Release VAO, FBO and program
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glUseProgram(0)

        # gl.glDepthMask(True)


class RectangleRenderer(object):

    def __init__(self, screen_size):

        rect_coords = np.array([[-1, -1, 0],
                                [ 1, -1, 0],
                                [-1,  1, 0],
                                [ 1,  1, 0]], dtype=np.float32)

        rect_idx = np.array([0, 1, 2,
                             1, 2, 3], dtype=np.uint32)

        self.screen_size = screen_size

        self.program = make_program('rectangle.vert', 'rectangle.frag')
        gl.glUseProgram(self.program)

        # Get uniforms: Quad resolution and environment map texture
        resolution = gl.glGetUniformLocation(self.program, "resolution")
        env_map_texture = gl.glGetUniformLocation(self.program, "EnvMap")

        gl.glUniform2f(resolution, self.screen_size[0], self.screen_size[1])
        gl.glUniform1i(env_map_texture, 0)

        # Create VAO
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        self.vao_len = 6                    # 3 vertices * 2 triangles = 6 vertices

        # Create VBO
        self.vbo_id = gl.glGenBuffers(2)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id[0])

        # Bind VBO to VAO with the index of the correct attribute (loc)
        loc = gl.glGetAttribLocation(self.program, 'position')
        gl.glEnableVertexAttribArray(loc)

        # Describe the position data layout in the buffer
        gl.glVertexAttribPointer(loc,
                                 rect_coords.shape[1],  # Size
                                 gl.GL_FLOAT,           # Type
                                 False,                 # Values already normalised, no need to
                                 0,                     # Stride
                                 ctypes.c_void_p(0))    # Array buffer offset (None)

        # Send the data over to the buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        rect_coords.nbytes,
                        rect_coords,
                        gl.GL_STATIC_DRAW)

        # Element Buffer
        self.elementbuffer = self.vbo_id[1]
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER,
                        self.vbo_id[1])
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER,
                        rect_idx.nbytes,
                        rect_idx,
                        gl.GL_STATIC_DRAW)

        # Unbind from the VAO first (Important ! Will fail otherwise)
        gl.glBindVertexArray(0)

        # Unbind the others
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # Release bound program
        gl.glUseProgram(0)

        self.img_buffer = np.empty((self.screen_size[0], self.screen_size[1], 3), dtype=np.uint8)

    def draw(self, renderer):

        # Bind program and VAO
        gl.glUseProgram(self.program)

        gl.glViewport(0, 0, self.screen_size[0], self.screen_size[1])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glBindVertexArray(self.vao_id)

        # We have to bind to the framebuffer in order to read the colours from the environment cubemap
        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, renderer.texobj_colour_id)

        gl.glDrawElements(gl.GL_TRIANGLES,      # Mode
                          self.vao_len,         # 2 triangles * 3 vertices = 6
                          gl.GL_UNSIGNED_INT,   # Type
                          ctypes.c_void_p(0))   # Element array buffer offset (None)

        # Release program and VAO
        gl.glBindVertexArray(0)

        gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)

        gl.glUseProgram(0)

    def read(self):
        self.img_buffer[...] = gl.glReadPixels(0,
                                               0,
                                               self.screen_size[0],
                                               self.screen_size[1],
                                               gl.GL_RGB,
                                               gl.GL_FLOAT) * 255

        return np.flip(self.img_buffer.reshape(self.screen_size[1], self.screen_size[0], -1), axis=0)


class ConesRenderer(object):

    def __init__(self, screen_size):

        self.screen_size = screen_size

        self.program = make_program('cone.vert', 'cone.frag')
        gl.glUseProgram(self.program)

        vert_coords = ConesRenderer.make_base_cone()

        # Get where colour and center coords are in the shader
        self.unif_colour = gl.glGetUniformLocation(self.program, "fragment_color")
        self.unif_centre_coord = gl.glGetUniformLocation(self.program, "cone_centre")

        # Make a VAO
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        self.vao_len = vert_coords.shape[0]  # Number of vertices in our cone

        # Make a VBO
        self.vbo_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)

        # Bind VBO to VAO with the index of the correct attribute (loc)
        loc = gl.glGetAttribLocation(self.program, 'verts_in')
        gl.glEnableVertexAttribArray(loc)

        # Describe the position data layout in the buffer
        gl.glVertexAttribPointer(loc,
                                 vert_coords.shape[1],      # Size
                                 gl.GL_FLOAT,               # Type
                                 False,                     # Values already normalised, no need to
                                 0,                         # Stride
                                 ctypes.c_void_p(0))        # Array buffer offset (None)

        # Send the data over to the buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER,
                        vert_coords.nbytes,
                        vert_coords,
                        gl.GL_STATIC_DRAW)

        # Unbind from the VAO first (Important ! Will fail otherwise)
        gl.glBindVertexArray(0)

        # Unbind the others
        gl.glDisableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        # Release bound program
        gl.glUseProgram(0)

        self.img_buffer = np.empty((self.screen_size[0], self.screen_size[1], 3), dtype=np.uint8)

    def draw(self, eye_output, cone_centres):

        # Bind program and VAO
        gl.glUseProgram(self.program)

        gl.glViewport(0, 0, self.screen_size[0], self.screen_size[1])
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glBindVertexArray(self.vao_id)

        # Here no need to bind to the unit 0 texture because we already extracted each ommatidium's view

        for om_nb, cone_XYZ in enumerate(cone_centres):     # For each cone :
            eye_percieved_colour = eye_output[om_nb]

            # For all vertices of current cone, color and center coords (om position) are the same, so we use uniforms
            gl.glUniform3f(self.unif_colour, *eye_percieved_colour)
            gl.glUniform2f(self.unif_centre_coord, *cone_XYZ)

            # Draw all the vertices of the current cone
            gl.glDrawArrays(gl.GL_TRIANGLE_FAN,
                            0,                      # Start at vertex 0
                            self.vao_len)           # Number of vertices per cone

        # Release program and VAO
        gl.glBindVertexArray(0)

        gl.glUseProgram(0)

    def read(self):
        self.img_buffer[...] = gl.glReadPixels(0,
                                               0,
                                               self.screen_size[0],
                                               self.screen_size[1],
                                               gl.GL_RGB,
                                               gl.GL_FLOAT) * 255

        return np.flip(self.img_buffer.reshape(self.screen_size[1], self.screen_size[0], -1), axis=0)

    @staticmethod
    def make_base_cone(nb_points=66, radius=0.3):
        """
        Generate verts for a cone using the fan strip vertex method
        """
        circle = np.linspace(0, 2 * np.pi, nb_points - 1, endpoint=True)

        # Overlap the first and the last triangles
        thetas = np.insert(circle, nb_points - 1, circle[-1] + circle[1])

        vertices_coords = np.empty((nb_points + 1, 3), dtype=np.float32)

        # Fill with coords of the cone base (start at 1 because 0 is the tip)
        vertices_coords[1:, 0] = radius * np.sin(thetas)  # X
        vertices_coords[1:, 1] = radius * np.cos(thetas)  # Y
        vertices_coords[1:, 2] = 1                        # Z

        # Cone tip coords
        vertices_coords[0, :] = 0

        return vertices_coords

