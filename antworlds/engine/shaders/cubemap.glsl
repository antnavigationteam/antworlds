#version 450 core
#define M_PI 3.1415926535897932384626433832795

 // Triangles (strips) out, 3 vertices each
layout (triangles) in;
layout (triangle_strip, max_vertices = 18) out;

in vec4 ColorV[];
out vec4 ColorG;

uniform mat4 projection;
uniform mat4 view;

int gl_ViewportIndex;
int gl_Layer;


const mat4 mods[6] = mat4[6]    // TODO perform this in python - opengl bible p553
(
    mat4(cos(M_PI/2),0,-sin(M_PI/2),0,     0,1,0,0,    sin(M_PI/2),0,cos(M_PI/2),0,   0,0,0,1),    //-x
    mat4(cos(-M_PI/2),0,-sin(-M_PI/2),0,     0,1,0,0,    sin(-M_PI/2),0,cos(-M_PI/2),0,   0,0,0,1),      //-z

    // y is top and bottom
    mat4(1,0,0,0,   0,cos(-M_PI/2),-sin(-M_PI/2),0,     0, sin(-M_PI/2),cos(-M_PI/2),0,    0,0,0,1),    //+y
    mat4(1,0,0,0,   0,cos(M_PI/2),-sin(M_PI/2),0,     0, sin(M_PI/2),cos(M_PI/2),0,    0,0,0,1),       //-y

    mat4(1),         //+x
    mat4(cos(M_PI),0,-sin(M_PI),0,     0,1,0,0,    sin(M_PI),0,cos(M_PI),0,   0,0,0,1)              // +z

);

void main()
{
    for (int face = 0; face < 6; face++) {
        gl_Layer = face;  // gl_Layer: built-in variable that specifies which layer (= face of the cubemap) we render to

        for (int i = 0; i < gl_in.length(); i++) {  // for each triangle's vertices
            gl_Position = projection * mods[face] * view * gl_in[i].gl_Position;


            ColorG = ColorV[i];
            EmitVertex();
        }
        EndPrimitive();
    }
}
