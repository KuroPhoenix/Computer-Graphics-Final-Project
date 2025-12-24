#include "model.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <glm/vec3.hpp>

Model* Model::fromObjectFile(const char* obj_file, std::unordered_map<std::string, int>&matToTex) {
  Model* m = new Model();

  std::ifstream ObjFile(obj_file);

  if (!ObjFile.is_open()) {
    std::cout << "Can't open File !" << std::endl;
    return NULL;
  }

  /* TODO#1: Load model data from OBJ file
   *         You only need to handle v, vt, vn, f
   *         Other fields you can directly ignore
   *         Fill data into m->positions, m->texcoords m->normals and m->numVertex
   *         Data format:
   *           For positions and normals
   *         | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | ...
   *         | face 1                                                       | face 2               ...
   *         | v1x  | v1y  | v1z  | v2x  | v2y  | v2z  | v3x  | v3y  | v3z  | v1x  | v1y  | v1z  | ...
   *         | vn1x | vn1y | vn1z | vn1x | vn1y | vn1z | vn1x | vn1y | vn1z | vn1x | vn1y | vn1z | ...
   *           For texcoords
   *         | 0    | 1    | 2    | 3    | 4    | 5    | 6    | 7    | ...
   *         | face 1                                  | face 2        ...
   *         | v1x  | v1y  | v2x  | v2y  | v3x  | v3y  | v1x  | v1y  | ...
   * Note:
   *        OBJ File Format (https://en.wikipedia.org/wiki/Wavefront_.obj_file)
   *        Vertex per face = 3 or 4
   */
  std::string line = "";
  std::string prefix = "";
  std::string mtlFileName = "";
  std::stringstream ss;
  Submesh submesh = {0, -1, -1};
  std::vector<glm::vec3> vList(1), tList(1), nList(1);
  float vec[3];
  while (getline(ObjFile, line)) {
    ss.clear();
    ss.str(line);
    ss >> prefix;
    if (prefix == "usemtl") {
      std::string mtl = "";
      ss >> mtl;
      if (submesh.textureIndex >= 0 && m->numVertex > submesh.start) {
        m->submeshes.push_back({submesh.start, m->numVertex - submesh.start, submesh.textureIndex});
      }
      submesh.start = m->numVertex;
      submesh.textureIndex = matToTex[mtl];
      submesh.count++;
    }
    else if (prefix == "v") {
      for (int i = 0; i < 3; i++) {
        ss >> prefix;
        vec[i] = stof(prefix);
      }
      vList.push_back(glm::vec3(vec[0], vec[1], vec[2]));
    } else if (prefix == "vt") {
      for (int i = 0; i < 2; i++) {
        ss >> prefix;
        vec[i] = stof(prefix);
      }
      tList.push_back(glm::vec3(vec[0], vec[1], vec[2]));
    } else if (prefix == "vn") {
      for (int i = 0; i < 3; i++) {
        ss >> prefix;
        vec[i] = stof(prefix);
      }
      nList.push_back(glm::vec3(vec[0], vec[1], vec[2]));
    } else if (prefix == "f") {
      std::string tmp = "";
      size_t num_face = 0;
      while (ss >> prefix) {
        size_t i = 0;
        num_face++;
        bool Istexture = false;
        while (i < prefix.size()) {
          if (prefix[i] == '/') {
            if (Istexture) {
              m->texcoords.push_back(tList[atoi(tmp.c_str())][0]);
              m->texcoords.push_back(tList[atoi(tmp.c_str())][1]);
            } else {
              for (int i = 0; i < vList[atoi(tmp.c_str())].length(); i++)
                m->positions.push_back(vList[atoi(tmp.c_str())][i]);
            }
            tmp = "";
            Istexture = !Istexture;
          } else
            tmp += prefix[i];
          i++;
        }
        m->normals.push_back(nList[atoi(tmp.c_str())][0]);
        m->normals.push_back(nList[atoi(tmp.c_str())][1]);
        m->normals.push_back(nList[atoi(tmp.c_str())][2]);
        tmp = "";
      }
      if (num_face == 4) {
        m->numVertex += 4;
      } else {
        m->numVertex += 3;
      }
    }
  }
  if (submesh.textureIndex >= 0 && m->numVertex > submesh.start) {
    m->submeshes.push_back({submesh.start, m->numVertex - submesh.start, submesh.textureIndex});
  }
  if (ObjFile.is_open()) ObjFile.close();
  return m;
}
