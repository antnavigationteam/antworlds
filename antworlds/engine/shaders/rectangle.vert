#version 450 core

attribute vec3 position;

//uniform mat4 model;
//uniform mat4 view;
//uniform mat4 projection;

void main(){

//    gl_Position = vec4(projection * view * model * vec4(position, 1.0));
    gl_Position = vec4(position, 1.0);

}
