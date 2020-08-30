#version 330 core
layout(location = 8) in vec3 verts_in;
uniform vec2 cone_centre;

void main(){

    gl_Position = vec4(verts_in.x + cone_centre.x, verts_in.y + cone_centre.y, verts_in.z, 1.0);

}