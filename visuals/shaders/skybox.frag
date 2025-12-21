#version 330 core
in vec3 TexCoords;
out vec4 FragColor;
uniform samplerCube skybox;
// TODO#3-2: fragment shader
// Note:
//           1. Sample the cubemap
void main() {
  FragColor = texture(skybox, TexCoords);
}
