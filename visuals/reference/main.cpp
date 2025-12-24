#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <unordered_map>
#include <GLFW/glfw3.h>
#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#undef GLAD_GL_IMPLEMENTATION
#include <glm/glm.hpp>

#include <glm/ext/matrix_transform.hpp>

#include "camera.h"
#include "context.h"
#include "gl_helper.h"
#include "model.h"
#include "opengl_context.h"
#include "program.h"
#include "utils.h"

#include "glm/gtc/matrix_access.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

void initOpenGL();
void resizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int, int action, int);
void triggerDiscoMode();
void toggleDiscoMode();
void updateDiscoMode();

Context ctx;
static std::mt19937 gRng(static_cast<unsigned int>(
    std::chrono::high_resolution_clock::now().time_since_epoch().count()));

namespace {
float randomFloat(float minValue, float maxValue) {
  std::uniform_real_distribution<float> dist(minValue, maxValue);
  return dist(gRng);
}

glm::vec3 randomColor() {
  return glm::vec3(randomFloat(0.2f, 1.0f), randomFloat(0.2f, 1.0f), randomFloat(0.2f, 1.0f));
}

glm::vec3 randomPointLightPosition() {
  return glm::vec3(randomFloat(0.0f, 8.192f), randomFloat(0.5f, 3.5f), randomFloat(0.0f, 5.12f));
}

glm::vec3 randomSpotPosition() {
  return glm::vec3(randomFloat(-1.0f, 9.0f), randomFloat(2.0f, 6.5f), randomFloat(-1.0f, 6.5f));
}

glm::vec3 randomDirectionTowardsPlane(const glm::vec3& origin) {
  glm::vec3 target(randomFloat(0.0f, 8.192f), 0.0f, randomFloat(0.0f, 5.12f));
  glm::vec3 dir = target - origin;
  if (glm::length(dir) < 1e-4f) dir = glm::vec3(0.0f, -1.0f, 0.0f);
  return glm::normalize(dir);
}

glm::vec3 randomDirectionalDirection() {
  glm::vec3 dir(randomFloat(-1.0f, 1.0f), randomFloat(-1.0f, -0.2f), randomFloat(-1.0f, 1.0f));
  if (glm::length(dir) < 1e-4f) dir = glm::vec3(-0.3f, -0.5f, -0.2f);
  return glm::normalize(dir);
}
}  // namespace

struct LightState {
  int directionLightEnable = 0;
  glm::vec3 directionLightDirection = glm::vec3(0.0f);
  glm::vec3 directionLightColor = glm::vec3(0.0f);
  int pointLightEnable = 0;
  glm::vec3 pointLightPosition = glm::vec3(0.0f);
  glm::vec3 pointLightColor = glm::vec3(0.0f);
  int spotLightEnable = 0;
  glm::vec3 spotLightPosition = glm::vec3(0.0f);
  glm::vec3 spotLightDirection = glm::vec3(0.0f);
  glm::vec3 spotLightColor = glm::vec3(0.0f);
};

static LightState gSavedLightState;
static bool gDiscoModeActive = false;
static bool gLightStateSaved = false;
static double gNextDiscoShuffleTime = 0.0;

void triggerDiscoMode() {
  ctx.directionLightEnable = 1;
  ctx.pointLightEnable = 1;
  ctx.spotLightEnable = 1;
  ctx.directionLightColor = randomColor();
  ctx.directionLightDirection = randomDirectionalDirection();
  ctx.pointLightColor = randomColor();
  ctx.pointLightPosition = randomPointLightPosition();
  ctx.spotLightColor = randomColor();
  ctx.spotLightPosition = randomSpotPosition();
  ctx.spotLightDirection = randomDirectionTowardsPlane(ctx.spotLightPosition);
}
void toggleDiscoMode() {
  if (gDiscoModeActive) {
    gDiscoModeActive = false;
    if (gLightStateSaved) {
      ctx.directionLightEnable = gSavedLightState.directionLightEnable;
      ctx.directionLightDirection = gSavedLightState.directionLightDirection;
      ctx.directionLightColor = gSavedLightState.directionLightColor;
      ctx.pointLightEnable = gSavedLightState.pointLightEnable;
      ctx.pointLightPosition = gSavedLightState.pointLightPosition;
      ctx.pointLightColor = gSavedLightState.pointLightColor;
      ctx.spotLightEnable = gSavedLightState.spotLightEnable;
      ctx.spotLightPosition = gSavedLightState.spotLightPosition;
      ctx.spotLightDirection = gSavedLightState.spotLightDirection;
      ctx.spotLightColor = gSavedLightState.spotLightColor;
    }
    return;
  }
  gSavedLightState.directionLightEnable = ctx.directionLightEnable;
  gSavedLightState.directionLightDirection = ctx.directionLightDirection;
  gSavedLightState.directionLightColor = ctx.directionLightColor;
  gSavedLightState.pointLightEnable = ctx.pointLightEnable;
  gSavedLightState.pointLightPosition = ctx.pointLightPosition;
  gSavedLightState.pointLightColor = ctx.pointLightColor;
  gSavedLightState.spotLightEnable = ctx.spotLightEnable;
  gSavedLightState.spotLightPosition = ctx.spotLightPosition;
  gSavedLightState.spotLightDirection = ctx.spotLightDirection;
  gSavedLightState.spotLightColor = ctx.spotLightColor;
  gLightStateSaved = true;
  gDiscoModeActive = true;
  triggerDiscoMode();
  gNextDiscoShuffleTime = glfwGetTime() + randomFloat(0.15f, 0.45f);
}

void updateDiscoMode() {
  if (!gDiscoModeActive) return;
  double now = glfwGetTime();
  if (now >= gNextDiscoShuffleTime) {
    triggerDiscoMode();
    gNextDiscoShuffleTime = now + randomFloat(0.15f, 0.45f);
  }
}

Material mFlatwhite;
Material mShinyred;
Material mClearblue;
Material mMirror;

void loadMaterial() {
  mFlatwhite.ambient = glm::vec3(0.0f, 0.0f, 0.0f);
  mFlatwhite.diffuse = glm::vec3(1.0f, 1.0f, 1.0f);
  mFlatwhite.specular = glm::vec3(0.0f, 0.0f, 0.0f);
  mFlatwhite.shininess = 10;

  mShinyred.ambient = glm::vec3(0.1985f, 0.0000f, 0.0000f);
  mShinyred.diffuse = glm::vec3(0.5921f, 0.0167f, 0.0000f);
  mShinyred.specular = glm::vec3(0.5973f, 0.2083f, 0.2083f);
  mShinyred.shininess = 100.0f;

  mClearblue.ambient = glm::vec3(0.0394f, 0.0394f, 0.3300f);
  mClearblue.diffuse = glm::vec3(0.1420f, 0.1420f, 0.9500f);
  mClearblue.specular = glm::vec3(0.1420f, 0.1420f, 0.9500f);
  mClearblue.shininess = 10;

  mMirror.ambient = glm::vec3(0.8f, 0.8f, 0.8f);
  mMirror.diffuse = glm::vec3(0.2f, 0.2f, 0.2f);
  mMirror.specular = glm::vec3(0.9f, 0.9f, 0.9f);
  mMirror.shininess = 200.0f;
  mMirror.reflectivity = 0.9f;
}

void loadPrograms() {
  ctx.programs.push_back(new SkyboxProgram(&ctx));
  ctx.programs.push_back(new LightProgram(&ctx));

  for (auto iter = ctx.programs.begin(); iter != ctx.programs.end(); iter++) {
    if (!(*iter)->load()) {
      std::cout << "Load program fail, force terminate" << std::endl;
      exit(1);
    }
  }
  glUseProgram(0);
}


Model* createBottle() {
  /* TODO#1-1: Add the bottle model
   *           1. Create a model by reading the model file "../assets/models/bottle/bottle.obj" with the object loader(Model::fromObjectFile()) you write.
   *           2. Add the texture "../assets/models/bottle/bottle.jpg" to the model.
   *           3. Do transform(rotation & scale) to the model.
   *           4. Set the drawMode for this model
   * Note:
   *           You should finish implement the object loader(Model::fromObjectFile()) first.
   *           You can refer to the Model class structure in model.h.
   * Hint:
   *           Model* m = Model::fromObjectFile();
   *           m->textures.push_back();
   *           m->modelMatrix = glm::scale(m->modelMatrix, glm::vec3(0.05f, 0.05f, 0.05f));
   *           m->modelMatrix = glm::rotate(m->modelMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
   *           m->modelMatrix = glm::rotate(m->modelMatrix, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
   *           m->drawMode = 
   */
  std::unordered_map<std::string, int> dummy;
  Model* m = Model::fromObjectFile("../assets/models/bottle/bottle.obj", dummy);
  m->textures.push_back(createTexture("../assets/models/bottle/bottle.jpg"));
  m->modelMatrix = glm::scale(m->modelMatrix, glm::vec3(0.05f, 0.05f, 0.05f));
  m->modelMatrix = glm::rotate(m->modelMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
  m->modelMatrix = glm::rotate(m->modelMatrix, glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
  m->drawMode = GL_QUADS;
  return m;
}

Model* createPaimon() {
  const std::vector<std::pair<std::string, const char*>> materialTextures = {
      {"$Material_0", "../assets/models/Paimon/脸.jpg"},
      {"$Material_1", "../assets/models/Paimon/头发.jpg"},
      {"$Material_2", "../assets/models/Paimon/脸.jpg"},
      {"$Material_3", "../assets/models/Paimon/衣服.jpg"},
      {"$Material_4", "../assets/models/Paimon/衣服.jpg"},
      {"$Material_5", "../assets/models/Paimon/头发.jpg"},
      {"$Material_6", "../assets/models/Paimon/披风2.jpg"},
      {"en", "../assets/models/Paimon/表情.png"}};

  std::unordered_map<std::string, int> matToTex;
  matToTex.reserve(materialTextures.size());

  // Assign stable texture indices so submeshes can reference them.
  std::unordered_map<std::string, int> textureIndexByPath;
  for (const auto& entry : materialTextures) {
    const std::string pathKey(entry.second);
    auto iter = textureIndexByPath.find(pathKey);
    if (iter == textureIndexByPath.end()) {
      int idx = static_cast<int>(textureIndexByPath.size());
      textureIndexByPath[pathKey] = idx;
      matToTex[entry.first] = idx;
    } else {
      matToTex[entry.first] = iter->second;
    }
  }

  Model* m = Model::fromObjectFile("../assets/models/Paimon/paimon.obj", matToTex);
  if (!m) return nullptr;

  // textures must be inserted according to the deduplicated path order
  std::vector<std::pair<int, std::string>> indexedTextures;
  indexedTextures.reserve(textureIndexByPath.size());
  for (const auto& kv : textureIndexByPath) {
    indexedTextures.push_back({kv.second, kv.first});
  }
  std::sort(indexedTextures.begin(), indexedTextures.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& tex : indexedTextures) {
    m->textures.push_back(createTexture(tex.second.c_str()));
  }
  m->modelMatrix = glm::scale(m->modelMatrix, glm::vec3(0.1f));
  return m;
}

Model* createPlane() {
  /* TODO#1-2: Add a plane model
   *           1. Create a model and manually set plane positions, normals, texcoords
   *           2. Add texure "../assets/models/Wood_maps/AT_Wood.jpg"
   *           3. Set m->numVertex, m->drawMode
   * Note:
   *           GL_TEXTURE_WRAP is set to GL_REPEAT in createTexture, you may need to know
   *           what this means to set m->textures correctly
   */
  /*
    Size 8.192 * 5.12
    Center position (4.096, 0, 2.56) (green point)
    Normal (0, 1, 0)
    Note: texture is map to size 4.096 * 2.56
    Extend texture by correctly set texcoords
  */
  Model* m = new Model();
  m->textures.push_back(createTexture("../assets/models/Wood_maps/AT_Wood.jpg"));
  m->positions.insert(m->positions.end(), {0, 0, 0});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {0, 0});
  m->positions.insert(m->positions.end(),{0,0,5.12});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {0, 2});
  m->positions.insert(m->positions.end(),{8.192, 0, 5.12});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {2, 2});

  m->positions.insert(m->positions.end(), {0, 0, 0});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {0, 0});
  m->positions.insert(m->positions.end(),{8.192, 0, 5.12});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {2, 2});
  m->positions.insert(m->positions.end(),{8.192, 0, 0});
  m->normals.insert(m->normals.end(), {0, 1, 0});
  m->texcoords.insert(m->texcoords.end(), {2, 0});
  m->numVertex = 6;
  m->drawMode = GL_TRIANGLES;
  return m;
}

float bezier(float t, float p0, float p1, float p2, float p3) {
  return pow(1 - t, 3) * p0 + 3 * pow(1 - t, 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t * t * t * p3;
}

void emit(Model* m, int idx, std::vector<glm::vec3> &gridPos, std::vector<glm::vec3> &gridNormal, std::vector<glm::vec2> &gridTex) {
  const auto& p = gridPos[idx];
  const auto& n = gridNormal[idx];
  const auto& uv = gridTex[idx];
  m->positions.insert(m->positions.end(), {p.x, p.y, p.z});
  m->normals.insert(m->normals.end(), {n.x, n.y, n.z});
  m->texcoords.insert(m->texcoords.end(), {uv.x, uv.y});
  ++m->numVertex;
}

Model* createBezierVaseModel() {
  const int segments = 36;         // Circular segments
  const int height_segments = 50;  // Height segments
  float height = 1.0f;             // Vase Height

  // Control points for the Bezier curve (you can try adjusting these to shape the vase)
  float p0 = 0.2f;  // Radius at base
  float p1 = 1.0f;  // Control point 1
  float p2 = 0.2f;  // Control point 2
  float p3 = 0.1f;  // Radius at neck
  //Use cubic bezier curve

  /* TODO#1-3: Add a vase outer surface model
   *           1. Create a model and manually set vase positions, normals, texcoords
   *           2. Add texure "../assets/models/Vase/Vase.jpg"
   *           3. Set m->numVertex, m->drawMode
   * Note:
   *           You should refer to the cubic bezier curve function bezier().
   */
  Model* vase = new Model();
  std::vector<glm::vec3> grid_pos(segments * (height_segments + 1));
  std::vector<glm::vec3> grid_normal(segments * (height_segments + 1));
  std::vector<glm::vec2> grid_tex(segments * (height_segments + 1));
  vase->textures.push_back(createTexture("../assets/models/Vase/Vase.jpg"));
  int rowSize = segments;
  for (int i = 0; i <= height_segments; i++) {
    float t = static_cast<float>(i) / height_segments;
    float y = height * t;
    float r = bezier(t, p0, p1, p2, p3);
    for (int j = 0; j < segments; j++) {
      float theta = 2 * M_PI * j / segments;
      float t_next = static_cast<float>(std::min(i + 1, height_segments)) / height_segments;
      float t_prev = static_cast<float>(std::max(i - 1, 0)) / height_segments;
      float r_next = bezier(t_next, p0, p1, p2, p3);
      float r_prev = bezier(t_prev, p0, p1, p2, p3);
      float dr = (i == height_segments) ? (r - r_prev) : (r_next - r);
      float u = static_cast<float>(j) / segments;
      float v = t;
      glm::vec3 tangentTheta(-glm::sin(theta), 0, glm::cos(theta));
      glm::vec3 tangentHeight(dr * glm::cos(theta), height / height_segments, dr * glm::sin(theta));
      glm::vec3 normal = -glm::normalize(glm::cross(tangentTheta, tangentHeight));
      grid_pos[i * segments + j] = glm::vec3(r * glm::cos(theta), y, r * glm::sin(theta));
      grid_normal[i * segments + j] = glm::vec3(normal.x, normal.y, normal.z);
      grid_tex[i * segments + j] = glm::vec2(u * 2, v * 3);
    }
  }
  for (int i = 0 ; i < height_segments ; i++) {
    for (int j = 0; j < segments; j++) {
      int j_next = (j + 1) % segments;
      int idx0 = i * rowSize + j;
      int idx1 = (i + 1) * rowSize + j;
      int idx2 = (i + 1) * rowSize + j_next;
      int idx3 = i * rowSize + j_next;
      emit(vase, idx0, grid_pos, grid_normal, grid_tex);
      emit(vase, idx1, grid_pos, grid_normal, grid_tex);
      emit(vase, idx2, grid_pos, grid_normal, grid_tex);
      emit(vase, idx0, grid_pos, grid_normal, grid_tex);
      emit(vase, idx2, grid_pos, grid_normal, grid_tex);
      emit(vase, idx3, grid_pos, grid_normal, grid_tex);
    }
  }
  vase->drawMode = GL_TRIANGLES;
  return vase;
}

Model* createBezierVaseInnerModel() {
  const int segments = 36;         // Circular segments
  const int height_segments = 50;  // Height segments
  float height = 1.0f;             // Vase Height

  // Control points for the Bezier curve (adjust these to shape the vase)
  float p0 = 0.2f;  // Radius at base
  float p1 = 1.0f;  // Control point 1
  float p2 = 0.2f;  // Control point 2
  float p3 = 0.1f;  // Radius at neck

  /* TODO#1-4: Add a vase inner surface model
   *           1. Create a model and manually set vase positions, normals, texcoords
   *           2. Add texure "../assets/models/Vase/Vase2.jpg"
   *           3. Set m->numVertex, m->drawMode
   * Note:
   *           You should refer to the cubic bezier curve function bezier().
   */
  Model* vase = new Model();
  std::vector<glm::vec3> grid_pos(segments * (height_segments + 1));
  std::vector<glm::vec3> grid_normal(segments * (height_segments + 1));
  std::vector<glm::vec2> grid_tex(segments * (height_segments + 1));
  vase->textures.push_back(createTexture("../assets/models/Vase/Vase2.jpg"));
  int rowSize = segments;
  for (int i = 0; i <= height_segments; i++) {
    float epsilon = 0.01f;
    float t = static_cast<float>(i) / height_segments;
    float y = height * t;
    float r = bezier(t, p0, p1, p2, p3) - epsilon;
    for (int j = 0; j < segments; j++) {
      float theta = 2 * M_PI * j / segments;
      float t_next = static_cast<float>(std::min(i + 1, height_segments)) / height_segments;
      float t_prev = static_cast<float>(std::max(i - 1, 0)) / height_segments;
      float r_next = bezier(t_next, p0, p1, p2, p3);
      float r_prev = bezier(t_prev, p0, p1, p2, p3);
      float dr = (i == height_segments) ? (r - r_prev) : (r_next - r);
      float u = static_cast<float>(j) / segments;
      float v = t;
      glm::vec3 tangentTheta(-glm::sin(theta), 0, glm::cos(theta));
      glm::vec3 tangentHeight(dr * glm::cos(theta), height / height_segments, dr * glm::sin(theta));
      // Flip the normal inward so the interior shades correctly
      glm::vec3 normal = glm::normalize(glm::cross(tangentTheta, tangentHeight));
      grid_pos[i * segments + j] = glm::vec3(r * glm::cos(theta), y, r * glm::sin(theta));
      grid_normal[i * segments + j] = glm::vec3(normal.x, normal.y, normal.z);
      grid_tex[i * segments + j] = glm::vec2(u * 2, v * 2 * (440.f / 480.f));
    }
  }
  for (int i = 0 ; i < height_segments ; i++) {
    for (int j = 0; j < segments; j++) {
      int j_next = (j + 1) % segments;
      int idx0 = i * rowSize + j;
      int idx1 = (i + 1) * rowSize + j;
      int idx2 = (i + 1) * rowSize + j_next;
      int idx3 = i * rowSize + j_next;
      emit(vase, idx2, grid_pos, grid_normal, grid_tex);
      emit(vase, idx1, grid_pos, grid_normal, grid_tex);
      emit(vase, idx0, grid_pos, grid_normal, grid_tex);
      emit(vase, idx3, grid_pos, grid_normal, grid_tex);
      emit(vase, idx2, grid_pos, grid_normal, grid_tex);
      emit(vase, idx0, grid_pos, grid_normal, grid_tex);
    }
  }
  vase->drawMode = GL_TRIANGLES;
  return vase;
}

Model* createBezierVaseBottomModel() {
  /* TODO#1-5: Add a vase bottom surface model
   *           1. Create a model and manually set vase positions, normals, texcoords
   *           2. Add texure "../assets/models/Vase/Vase2.jpg"
   *           3. Set m->numVertex, m->drawMode
   * Note:
   *           You should refer to the cubic bezier curve function bezier().
   */
    float p0 = 0.2f;
    float p1 = 1.0f;
    float p2 = 0.2f;
    float p3 = 0.1f;
    float height = 1.0f;

    Model* m = new Model();
    m->textures.push_back(createTexture("../assets/models/Vase/Vase2.jpg"));

    const int segments = 36;
    const float radius = std::max(bezier(0.f, p0, p1, p2, p3), 0.01f);

    const glm::vec3 center(0.f, 0.f, 0.f);
    const glm::vec2 centerUV(0.5f, 0.5f);
    const float repeatU = 2.f;
    const float repeatV = repeatU * (440.f / 480.f);

    std::vector<glm::vec3> ring(segments);
    std::vector<glm::vec2> ringUV(segments);

    for (int j = 0; j < segments; ++j) {
        float theta = 2.f * M_PI * j / segments;
        float x = radius * cosf(theta);
        float z = radius * sinf(theta);
        ring[j] = glm::vec3(x, 0.f, z);

        float u = (cosf(theta) * 0.5f + 0.5f) * repeatU;
        float v = (sinf(theta) * 0.5f + 0.5f) * repeatV;
        ringUV[j] = glm::vec2(u, v);
    }

    auto emitCapLayer = [&](const glm::vec3& n, bool normalUp) {
        const float layerOffset = normalUp ? 0.0f : -0.0005f;  // separate front/back to avoid z-fight
        for (int j = 0; j < segments; ++j) {
            int next = (j + 1) % segments;

            // center
            m->positions.insert(m->positions.end(), {center.x, center.y + layerOffset, center.z});
            m->normals.insert(m->normals.end(),  {n.x, n.y, n.z});
            m->texcoords.insert(m->texcoords.end(), {centerUV.x, centerUV.y});
            ++m->numVertex;

            if (normalUp) {
                m->positions.insert(m->positions.end(), {ring[j].x, ring[j].y + layerOffset, ring[j].z});
                m->normals.insert(m->normals.end(),  {n.x, n.y, n.z});
                m->texcoords.insert(m->texcoords.end(), {ringUV[j].x, ringUV[j].y});
                ++m->numVertex;

                m->positions.insert(m->positions.end(), {ring[next].x,  ring[next].y + layerOffset,  ring[next].z});
                m->normals.insert(m->normals.end(),  {n.x, n.y, n.z});
                m->texcoords.insert(m->texcoords.end(), {ringUV[next].x, ringUV[next].y});
                ++m->numVertex;
            } else {
                m->positions.insert(m->positions.end(), {ring[next].x, ring[next].y + layerOffset, ring[next].z});
                m->normals.insert(m->normals.end(),  {n.x, n.y, n.z});
                m->texcoords.insert(m->texcoords.end(), {ringUV[next].x, ringUV[next].y});
                ++m->numVertex;

                m->positions.insert(m->positions.end(), {ring[j].x,  ring[j].y + layerOffset,  ring[j].z});
                m->normals.insert(m->normals.end(),  {n.x, n.y, n.z});
                m->texcoords.insert(m->texcoords.end(), {ringUV[j].x, ringUV[j].y});
                ++m->numVertex;
            }
        }
    };

    emitCapLayer(glm::vec3(0.f, 1.f, 0.f), false);
    emitCapLayer(glm::vec3(0.f, -1.f, 0.f), true);

    m->drawMode = GL_TRIANGLES;
    return m;
}

void loadModels() {
  /* TODO#2-1: Push the model to ctx.models
   * Note:
   *    You can refer to the context class in context.h and model class in model.h
   * Hint:
        ctx.models.push_back();
   */

  ctx.models.push_back(createPlane());        // 0
  ctx.models.push_back(createBottle());       // 1
  ctx.models.push_back(createBezierVaseModel());      // 2 outer
  ctx.models.push_back(createBezierVaseInnerModel()); // 3 inner
  ctx.models.push_back(createBezierVaseBottomModel()); // 4 bottom
  ctx.models.push_back(createPaimon()); // 5
}

void setupObjects() {
  /* TODO#2-2: Set up the object by the model vector
   * Note:
   *    You can refer to the context class in context.h and objects structure in model.h
   * Hint:
   *    ctx.objects.push_back(new Object(0, glm::translate(glm::identity<glm::mat4>(), glm::vec3(1.5, 0.4, 3))));
   *    (*ctx.objects.rbegin())->material = mMirror;
   */

  // Plane
  ctx.objects.push_back(new Object(0, glm::mat4(1.0f)));
  ctx.objects.back()->material = mFlatwhite;
  ctx.objects.back()->material.reflectivity = 0.3f;

  // Bottle
  ctx.objects.push_back(new Object(1, glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 1.5f))));
  ctx.objects.back()->material = mFlatwhite;

  // Vase outer
  glm::vec3 vasePosition(6.0f, 0.01f, 3.0f);
  glm::mat4 vaseTransform = glm::translate(glm::mat4(1.0f), vasePosition);
  glm::vec3 paimonPosition(2.0f, 0.0f, 1.0f);
  glm::mat4 paimonTransform = glm::translate(glm::mat4(1.0f), paimonPosition);
  ctx.objects.push_back(new Object(2, vaseTransform));
  ctx.objects.back()->material = mFlatwhite;

  ctx.objects.push_back(new Object(3, vaseTransform));
  ctx.objects.back()->material = mFlatwhite;

  ctx.objects.push_back(new Object(4, vaseTransform));
  ctx.objects.back()->material = mFlatwhite;

  //Paimon
  ctx.objects.push_back(new Object(5, paimonTransform));
  ctx.objects.back()->material = mFlatwhite;
  ctx.paimonIndex = static_cast<int>(ctx.objects.size()) - 1;
}

int main() {
  initOpenGL();
  GLFWwindow* window = OpenGLContext::getWindow();
  /* TODO#0: Change window title to "HW2 - `your student id`"
   *         Ex. HW2 - 311550000
   */
  glfwSetWindowTitle(window, "HW2 - 112550019");

  // Init Camera helper
  Camera camera(glm::vec3(0, 2, 5));
  camera.initialize(OpenGLContext::getAspectRatio());
  // Store camera as glfw global variable for callbacks use
  glfwSetWindowUserPointer(window, &camera);
  ctx.camera = &camera;
  ctx.window = window;

  loadMaterial();
  loadModels();
  loadPrograms();
  setupObjects();


  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330 core");
  // Main rendering loop
  while (!glfwWindowShouldClose(window)) {
    // Polling events.
    glfwPollEvents();
    updateDiscoMode();
    // Update camera position and view
    camera.move(window);
    // GL_XXX_BIT can simply "OR" together to use.
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    /// TO DO Enable DepthTest
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glClearDepth(1.0f);

    for (size_t i = 0; i < ctx.programs.size(); i++) {
      ctx.programs[i]->doMainLoop();
    }

    if (ctx.paimonMovable) {
      Object* paimon = ctx.objects[ctx.paimonIndex];
      const float step = 0.01f; // adjust speed
      const float flyStep = 0.01f;
      const float rotStep = glm::radians(1.0f);
      glm::vec3 delta(0.0f);
      if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) delta.z -= step;
      if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) delta.z += step;
      if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) delta.x -= step;
      if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) delta.x += step;
      if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) delta.y += flyStep;
      if (glfwGetKey(window, GLFW_KEY_J) == GLFW_PRESS) delta.y -= flyStep;
      if (glfwGetKey(window, GLFW_KEY_K) == GLFW_PRESS) paimon->transformMatrix = glm::rotate(paimon->transformMatrix, rotStep, glm::vec3(0.f, 1.f, 0.f));
      if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS) paimon->transformMatrix = glm::rotate(paimon->transformMatrix, -rotStep, glm::vec3(0.f, 1.f, 0.f));
      if (delta != glm::vec3(0.0f)) {
        Object* paimon = ctx.objects[ctx.paimonIndex]; // store/run-time index when you push it
        paimon->transformMatrix = glm::translate(paimon->transformMatrix, delta);
      }
  }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // Lights control panel
    {
      ImGui::Begin("Lights Control");

      // --- Directional Light ---
      ImGui::Text("Directional Light");
      {
        ImGui::SameLine();
        bool enable = (ctx.directionLightEnable != 0);
        if (ImGui::Checkbox("Enable##dir", &enable)) ctx.directionLightEnable = enable ? 1 : 0;
        ImGui::SliderFloat3("Dir X/Y/Z##dir", &ctx.directionLightDirection.x, -50.0f, 50.0f);
        ImGui::ColorEdit3("Color##dir", &ctx.directionLightColor[0]);
      }
      ImGui::Separator();

      // --- Point Light ---
      ImGui::Text("Point Light");
      {
        ImGui::SameLine();
        bool enable = (ctx.pointLightEnable != 0);
        if (ImGui::Checkbox("Enable##point", &enable)) ctx.pointLightEnable = enable ? 1 : 0;
        ImGui::SliderFloat3("Pos X/Y/Z##point", &ctx.pointLightPosition.x, -10.0f, 10.0f);
        ImGui::ColorEdit3("Color##point", &ctx.pointLightColor[0]);
      }
      ImGui::Separator();

      // --- Spot Light ---
      ImGui::Text("Spot Light");
      {
        ImGui::SameLine();
        bool enable = (ctx.spotLightEnable != 0);
        if (ImGui::Checkbox("Enable##spot", &enable)) ctx.spotLightEnable = enable ? 1 : 0;
        ImGui::SliderFloat3("Pos X/Y/Z##spot", &ctx.spotLightPosition.x, -10.0f, 10.0f);
        ImGui::ColorEdit3("Color##spot", &ctx.spotLightColor[0]);
      }
      ImGui::Separator();

      {
        const char* hint = "Use F1 to toggle cursor";
        ImGui::Separator();
        ImVec2 winSize = ImGui::GetWindowSize();
        ImVec2 txtSize = ImGui::CalcTextSize(hint);
        float y = winSize.y - txtSize.y - ImGui::GetStyle().FramePadding.y - ImGui::GetStyle().ItemSpacing.y;
        if (y > ImGui::GetCursorPosY()) ImGui::SetCursorPosY(y);
        ImGui::TextColored(ImVec4(0.9f, 0.9f, 0.6f, 1.0f), "%s", hint);
      }
      ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

#ifdef __APPLE__
    // Some platform need explicit glFlush
    glFlush();
#endif
    glfwSwapBuffers(window);
  }
  // Cleanup ImGui
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  return 0;
}

void keyCallback(GLFWwindow* window, int key, int, int action, int) {
  // Press ESC to close the window.
  if (key == GLFW_KEY_ESCAPE) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    return;
  }
  if (action == GLFW_PRESS) {
    switch (key) {
      case GLFW_KEY_F1: {
        // Toggle cursor
        ctx.paimonMovable = true;
        Camera* cam = static_cast<Camera*>(glfwGetWindowUserPointer(window));
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_DISABLED) {
          glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
          int ww, hh;
          glfwGetWindowSize(window, &ww, &hh);
          glfwSetCursorPos(window, static_cast<double>(ww) / 2.0, static_cast<double>(hh) / 2.0);
          if (cam) cam->setLastMousePos(window);
        } else {
          glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
          if (cam) cam->setLastMousePos(window);
        }
        break;
      }
      case GLFW_KEY_G: {
        if (glfwGetInputMode(window, GLFW_CURSOR) == GLFW_CURSOR_NORMAL) {
          toggleDiscoMode();
        }
        break;
      }
      default:
        break;
    }
  }
}

void resizeCallback(GLFWwindow* window, int width, int height) {
  OpenGLContext::framebufferResizeCallback(window, width, height);
  auto ptr = static_cast<Camera*>(glfwGetWindowUserPointer(window));
  if (ptr) {
    ptr->updateProjectionMatrix(OpenGLContext::getAspectRatio());
  }
}

void initOpenGL() {
  // Initialize OpenGL context, details are wrapped in class.
#ifdef __APPLE__
  // MacOS need explicit request legacy support
  OpenGLContext::createContext(21, GLFW_OPENGL_ANY_PROFILE);
#else
  OpenGLContext::createContext(21, GLFW_OPENGL_ANY_PROFILE);
//  OpenGLContext::createContext(43, GLFW_OPENGL_COMPAT_PROFILE);
#endif
  GLFWwindow* window = OpenGLContext::getWindow();
  glfwSetKeyCallback(window, keyCallback);
  glfwSetFramebufferSizeCallback(window, resizeCallback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
#ifndef NDEBUG
  OpenGLContext::printSystemInfo();
  // This is useful if you want to debug your OpenGL API calls.
  OpenGLContext::enableDebugCallback();
#endif
}
