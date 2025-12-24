project(HW2 C CXX)

set(HW2_SOURCE
  ${HW2_SOURCE_DIR}/camera.cpp
  ${HW2_SOURCE_DIR}/gl_helper.cpp
  ${HW2_SOURCE_DIR}/main.cpp
  ${HW2_SOURCE_DIR}/model.cpp
  ${HW2_SOURCE_DIR}/opengl_context.cpp
  ${HW2_SOURCE_DIR}/Programs/example.cpp
#  ${HW2_SOURCE_DIR}/Programs/basic.cpp
  ${HW2_SOURCE_DIR}/Programs/light.cpp
  ${HW2_SOURCE_DIR}/Programs/skybox.cpp
)

set(HW2_HEADER
  ${HW2_SOURCE_DIR}/../include/camera.h
  ${HW2_SOURCE_DIR}/../include/context.h
  ${HW2_SOURCE_DIR}/../include/gl_helper.h
  ${HW2_SOURCE_DIR}/../include/model.h
  ${HW2_SOURCE_DIR}/../include/opengl_context.h
  ${HW2_SOURCE_DIR}/../include/program.h
  ${HW2_SOURCE_DIR}/../include/utils.h
)
add_executable(HW2 ${HW2_SOURCE} ${HW2_HEADER})
target_include_directories(HW2
  PRIVATE ${HW2_SOURCE_DIR}/../include
  PRIVATE ${HW2_SOURCE_DIR}/../extern/imgui
)

set(IMGUI_DIR ${HW2_SOURCE_DIR}/../extern/imgui)

add_library(imgui STATIC
  ${IMGUI_DIR}/imgui.cpp
  ${IMGUI_DIR}/imgui_draw.cpp
  ${IMGUI_DIR}/imgui_tables.cpp
  ${IMGUI_DIR}/imgui_widgets.cpp
  ${IMGUI_DIR}/imgui_impl_glfw.cpp
  ${IMGUI_DIR}/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
  ${IMGUI_DIR}
  ${HW2_SOURCE_DIR}/../extern/glfw/include
  ${HW2_SOURCE_DIR}/../extern/glad/include
)

# Dear ImGui backend needs the same loader/GLFW settings and headers
target_compile_definitions(imgui
  PUBLIC
    GLFW_INCLUDE_NONE
    IMGUI_IMPL_OPENGL_LOADER_GLAD
)

target_link_libraries(imgui
  PUBLIC glfw
  PUBLIC glad
)

add_dependencies(HW2 glad glfw glm stb imgui)
# Can include glfw and glad in arbitrary order
target_compile_definitions(HW2 PRIVATE GLFW_INCLUDE_NONE IMGUI_IMPL_OPENGL_LOADER_GLAD)
# More warnings
if (NOT MSVC)
  target_compile_options(HW2
    PRIVATE "-Wall"
    PRIVATE "-Wextra"
    PRIVATE "-Wpedantic"
  )
endif()
# Prefer std c++20, at least need c++17 to compile
set_target_properties(HW2 PROPERTIES
  CXX_STANDARD 20
  CXX_EXTENSIONS OFF
)

target_link_libraries(HW2
  PRIVATE glad
  PRIVATE glfw
  PRIVATE stb
  PRIVATE imgui
)

if (TARGET glm::glm_shared)
  target_link_libraries(HW2 PRIVATE glm::glm_shared)
elseif(TARGET glm::glm_static)
  target_link_libraries(HW2 PRIVATE glm::glm_static)
else()
  target_link_libraries(HW2 PRIVATE glm::glm)
endif()
