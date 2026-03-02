cmake_minimum_required(VERSION 3.24)
project(AdvancedRenderingEngineGL LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Avoid accidentally picking MSYS/MinGW packages from environment paths.
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)

find_package(OpenGL REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h" REQUIRED)

add_executable(advanced_render_engine
    src/main.cpp
    src/tinygltf_impl.cpp
)

target_compile_definitions(advanced_render_engine PRIVATE
    GLM_ENABLE_EXPERIMENTAL
)

target_link_libraries(advanced_render_engine PRIVATE
    OpenGL::GL
    glfw
    glad::glad
    glm::glm
)

target_include_directories(advanced_render_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TINYGLTF_INCLUDE_DIRS}
)
