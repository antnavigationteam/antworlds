#version 450

in vec3 gs_TexCoords;
out vec4 color;

layout(binding=1) uniform samplerCube skybox_cubemap;

void main()
{
    color = texture(skybox_cubemap, gs_TexCoords);
}