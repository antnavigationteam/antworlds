#version 450

in vec3 position;


out vec4 gl_Position;

void main()
{
    gl_Position =  vec4(position, 1.0);
}

