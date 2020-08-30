#version 450 core
uniform vec3 fragment_color = vec3(0.0, 1.0, 0.0);
out vec4 color;

void main(){
  color = vec4(fragment_color, 1);   // TODO - use a vec3 here instead?
}