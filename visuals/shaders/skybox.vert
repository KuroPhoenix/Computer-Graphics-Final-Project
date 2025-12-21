#version 330 core
layout(location = 0) in vec3 aPos;
out vec3 TexCoords;
uniform mat4 Projection;
uniform mat4 ViewMatrix;
// TODO#3-1: vertex shader
// Note:
//           1. Pass the vertex position through as TexCoords for cubemap sampling
//           2. Make the skybox unaffected by camera translation by zeroing the view translation
//              before computing gl_Position
void main() {
  TexCoords = aPos;
  mat4 rotationView = mat4(mat3(ViewMatrix));
  vec4 pos = Projection * rotationView * vec4(aPos, 1.0);
  gl_Position = vec4(pos.xy, pos.w, pos.w);
}
