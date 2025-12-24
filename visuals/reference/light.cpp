#include <iostream>
#include "context.h"
#include "program.h"

bool LightProgram::load() {
  programId = quickCreateProgram(vertProgramFile, fragProgramFIle);

  int num_model = (int)ctx->models.size();
  VAO = new GLuint[num_model];

  glGenVertexArrays(num_model, VAO);
  for (int i = 0; i < num_model; i++) {
    glBindVertexArray(VAO[i]);
    Model* model = ctx->models[i];

    GLuint VBO[3];
    glGenBuffers(3, VBO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * model->positions.size(), model->positions.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * model->normals.size(), model->normals.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * model->texcoords.size(), model->texcoords.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
  }

  return programId != 0;
}

void LightProgram::doMainLoop() {
  glUseProgram(programId);
  int obj_num = (int)ctx->objects.size();
  for (int i = 0; i < obj_num; i++) {
    int modelIndex = ctx->objects[i]->modelIndex;
    glBindVertexArray(VAO[modelIndex]);

    Model* model = ctx->models[modelIndex];
    const float* p = ctx->camera->getProjectionMatrix();
    GLint pmatLoc = glGetUniformLocation(programId, "Projection");
    glUniformMatrix4fv(pmatLoc, 1, GL_FALSE, p);

    const float* v = ctx->camera->getViewMatrix();
    GLint vmatLoc = glGetUniformLocation(programId, "ViewMatrix");
    glUniformMatrix4fv(vmatLoc, 1, GL_FALSE, v);

    glm::mat4 fullModel = ctx->objects[i]->transformMatrix * model->modelMatrix;
    const float* m = glm::value_ptr(fullModel);

    GLint mmatLoc = glGetUniformLocation(programId, "ModelMatrix");
    glUniformMatrix4fv(mmatLoc, 1, GL_FALSE, m);

    glm::mat4 TIMatrix = glm::transpose(glm::inverse(fullModel));
    const float* ti = glm::value_ptr(TIMatrix);
    mmatLoc = glGetUniformLocation(programId, "TIModelMatrix");
    glUniformMatrix4fv(mmatLoc, 1, GL_FALSE, ti);

    const float* vp = ctx->camera->getPosition();
    mmatLoc = glGetUniformLocation(programId, "viewPos");
    glUniform3f(mmatLoc, vp[0], vp[1], vp[2]);

    glUniform1i(glGetUniformLocation(programId, "dl.enable"), ctx->directionLightEnable);
    glUniform3fv(glGetUniformLocation(programId, "dl.direction"), 1, glm::value_ptr(ctx->directionLightDirection));
    glUniform3fv(glGetUniformLocation(programId, "dl.lightColor"), 1, glm::value_ptr(ctx->directionLightColor));

    glUniform1i(glGetUniformLocation(programId, "pl.enable"), ctx->pointLightEnable);
    glUniform3fv(glGetUniformLocation(programId, "pl.position"), 1, glm::value_ptr(ctx->pointLightPosition));
    glUniform3fv(glGetUniformLocation(programId, "pl.lightColor"), 1, glm::value_ptr(ctx->pointLightColor));
    glUniform1f(glGetUniformLocation(programId, "pl.constant"), ctx->pointLightConstant);
    glUniform1f(glGetUniformLocation(programId, "pl.linear"), ctx->pointLightLinear);
    glUniform1f(glGetUniformLocation(programId, "pl.quadratic"), ctx->pointLightQuardratic);

    glUniform1i(glGetUniformLocation(programId, "sl.enable"), ctx->spotLightEnable);
    glUniform3fv(glGetUniformLocation(programId, "sl.position"), 1, glm::value_ptr(ctx->spotLightPosition));
    glUniform3fv(glGetUniformLocation(programId, "sl.direction"), 1, glm::value_ptr(ctx->spotLightDirection));
    glUniform3fv(glGetUniformLocation(programId, "sl.lightColor"), 1, glm::value_ptr(ctx->spotLightColor));
    glUniform1f(glGetUniformLocation(programId, "sl.cutOff"), ctx->spotLightCutOff);
    glUniform1f(glGetUniformLocation(programId, "sl.constant"), ctx->spotLightConstant);
    glUniform1f(glGetUniformLocation(programId, "sl.linear"), ctx->spotLightLinear);
    glUniform1f(glGetUniformLocation(programId, "sl.quadratic"), ctx->spotLightQuardratic);

    glUniform3fv(glGetUniformLocation(programId, "material.ambient"), 1,
                 glm::value_ptr(ctx->objects[i]->material.ambient));
    glUniform3fv(glGetUniformLocation(programId, "material.diffuse"), 1,
                 glm::value_ptr(ctx->objects[i]->material.diffuse));
    glUniform3fv(glGetUniformLocation(programId, "material.specular"), 1,
                 glm::value_ptr(ctx->objects[i]->material.specular));
    glUniform1f(glGetUniformLocation(programId, "material.shininess"), ctx->objects[i]->material.shininess);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, model->textures[ctx->objects[i]->textureIndex]);
    glUniform1i(glGetUniformLocation(programId, "ourTexture"), 0);

    if (ctx->cubemapTexture != 0) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_CUBE_MAP, ctx->cubemapTexture);
      glUniform1i(glGetUniformLocation(programId, "skybox"), 1);
    }
    glUniform1f(glGetUniformLocation(programId, "material.reflectivity"), ctx->objects[i]->material.reflectivity);

    glDrawArrays(model->drawMode, 0, model->numVertex);
  }
  glUseProgram(0);
}
