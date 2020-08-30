#version 450
in vec3 TexCoords;
out vec4 color;

uniform samplerCube skybox_cubemap;

void main()
{
    color = texture(skybox_cubemap, TexCoords);
}