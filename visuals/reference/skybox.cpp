#include <vector>
#include <iostream>

#include "context.h"
#include "program.h"
#include "gl_helper.h"
#include "stb_image.h"
#include <string>
#include <glm/gtc/type_ptr.hpp>

class SkyBox {
 public:
  static std::vector<GLfloat> vertices;
  static std::vector<GLuint> indices;
  GLuint VAO = 0;
  GLuint VBO = 0;
  GLuint EBO = 0;

  void setup() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GLfloat), vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint), indices.data(), GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void*)0);

    glBindVertexArray(0);
  }
};

std::vector<GLfloat> SkyBox::vertices = {-1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
                                         -1.0f, -1.0f, 1.0f,  1.0f, -1.0f, 1.0f,  1.0f, 1.0f, 1.0f,  -1.0f, 1.0f, 1.0f};
std::vector<GLuint> SkyBox::indices = {0, 1, 3, 3, 1, 2, 1, 5, 2, 2, 5, 6, 5, 4, 6, 6, 4, 7,
                                       4, 0, 7, 7, 0, 3, 3, 2, 7, 7, 2, 6, 4, 5, 0, 0, 5, 1};

// Load a cubemap from 6 textures ordered: right, left, top, bottom, front, back
static GLuint loadCubemap(const std::vector<std::string>& faces) {
  GLuint textureID;
  glGenTextures(1, &textureID);
  glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

  // Do not flip cubemap images
  stbi_set_flip_vertically_on_load(false);

  for (unsigned int i = 0; i < faces.size(); i++) {
    int width, height, nrChannels;
    unsigned char* data = stbi_load(faces[i].c_str(), &width, &height, &nrChannels, 0);
    if (data) {
      GLenum format = GL_RGB;
      if (nrChannels == 4) format = GL_RGBA;
      glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
      stbi_image_free(data);
    } else {
      std::cout << "Cubemap texture failed to load at path: " << faces[i] << std::endl;
      stbi_image_free(data);
    }
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  return textureID;
}


bool SkyboxProgram::load() {
  programId = quickCreateProgram(vertProgramFile, fragProgramFIle);
  if (programId == 0) return false;

  SkyBox* sb = new SkyBox();
  sb->setup();

  VAO = new GLuint[1];
  VAO[0] = sb->VAO;

  std::vector<std::string> faces = {
      std::string("../assets/skybox/skybox0.png"), std::string("../assets/skybox/skybox1.png"),
      std::string("../assets/skybox/skybox2.png"), std::string("../assets/skybox/skybox3.png"),
      std::string("../assets/skybox/skybox4.png"), std::string("../assets/skybox/skybox5.png"),
  };

  ctx->cubemapTexture = loadCubemap(faces);

  glUseProgram(programId);
  GLint loc = glGetUniformLocation(programId, "skybox");
  if (loc >= 0) glUniform1i(loc, 0);
  glUseProgram(0);

  return true;
}

void SkyboxProgram::doMainLoop() {
  glDepthFunc(GL_LEQUAL); // ensure skybox rendered behind
  glUseProgram(programId);

  const float* p = ctx->camera->getProjectionMatrix();
  GLint pmatLoc = glGetUniformLocation(programId, "Projection");
  glUniformMatrix4fv(pmatLoc, 1, GL_FALSE, p);

  const float* v = ctx->camera->getViewMatrix();
  GLint vmatLoc = glGetUniformLocation(programId, "ViewMatrix");

  glm::mat4 view = glm::make_mat4(v);
  view[3][0] = 0.0f;
  view[3][1] = 0.0f;
  view[3][2] = 0.0f;
  const float* viewPtr = glm::value_ptr(view);
  glUniformMatrix4fv(vmatLoc, 1, GL_FALSE, viewPtr);

  glBindVertexArray(VAO[0]);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, ctx->cubemapTexture);
  glDepthMask(GL_FALSE);
  glDrawElements(GL_TRIANGLES, (GLsizei)SkyBox::indices.size(), GL_UNSIGNED_INT, (void*)0);
  glDepthMask(GL_TRUE);
  glBindVertexArray(0);

  glUseProgram(0);
  glDepthFunc(GL_LESS);
}
