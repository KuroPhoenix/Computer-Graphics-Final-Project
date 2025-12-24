#include "camera.h"

Camera::Camera(glm::vec3 _position)
    : position(_position),
      up(0, 1, 0),
      front(0, 0, -1),
      right(1, 0, 0),
      rotation(glm::identity<glm::quat>()),
      projectionMatrix(1),
      viewMatrix(1) {}

void Camera::initialize(float aspectRatio) {
  updateProjectionMatrix(aspectRatio);
  updateViewMatrix();
}

void Camera::move(GLFWwindow* window) {
  bool ismoved = false;
  if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    lastMouseX = xpos;
    lastMouseY = ypos;
    return;
  }
  // Mouse part
  if (lastMouseX == 0 && lastMouseY == 0) {
    glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
  } else {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    float dx = mouseMoveSpeed * static_cast<float>(xpos - lastMouseX);
    float dy = mouseMoveSpeed * static_cast<float>(lastMouseY - ypos);
    lastMouseX = xpos;
    lastMouseY = ypos;
    if (dx != 0 || dy != 0) {
      ismoved = true;
      glm::quat rx(glm::angleAxis(dx, glm::vec3(0, -1, 0)));
      glm::quat ry(glm::angleAxis(dy, glm::vec3(1, 0, 0)));
      rotation = rx * rotation * ry;
    }
  }
  // Keyboard part
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
    position += front * keyboardMoveSpeed;
    ismoved = true;
  } else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
    position -= front * keyboardMoveSpeed;
    ismoved = true;
  } else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
    position -= right * keyboardMoveSpeed;
    ismoved = true;
  } else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
    position += right * keyboardMoveSpeed;
    ismoved = true;
  }
  // Update view matrix if moved
  if (ismoved) {
    updateViewMatrix();
  }
}

void Camera::setLastMousePos(GLFWwindow* window) {
  if (!window) return;
  glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
}

void Camera::updateViewMatrix() {
  constexpr glm::vec3 original_front(0, 0, -1);
  constexpr glm::vec3 original_up(0, 1, 0);

  front = rotation * original_front;
  up = rotation * original_up;
  right = glm::cross(front, up);
  viewMatrix = glm::lookAt(position, position + front, up);
}

void Camera::updateProjectionMatrix(float aspectRatio) {
  constexpr float FOV = glm::radians(45.0f);
  constexpr float zNear = 0.1f;
  constexpr float zFar = 100.0f;

  projectionMatrix = glm::perspective(FOV, aspectRatio, zNear, zFar);
}