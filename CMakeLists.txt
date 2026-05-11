cmake_minimum_required(VERSION 3.18)
project(ScaffoldChunGS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "ARM architecture detected (Jetson)")
    set(CUDA_ARCH "87")

    # JetPack 5.x / 6.x Torch paths (pip-installed PyTorch)
    find_package(Torch REQUIRED PATHS
        /usr/local/lib/python3.8/dist-packages/torch/share/cmake/Torch
        /usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch
        /usr/local/lib/python3.11/dist-packages/torch/share/cmake/Torch
        ${TORCH_CMAKE_PREFIX_PATH})
else()
    message(STATUS "x86_64 architecture detected")
    set(CUDA_ARCH "86")
    find_package(Torch REQUIRED PATHS
        /workspace/third_party/libtorch
        ${TORCH_CMAKE_PREFIX_PATH})
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(CUDA REQUIRED)

find_package(Eigen3 REQUIRED)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Jetson: use system OpenCV (pre-installed via JetPack SDK)
    find_package(OpenCV REQUIRED)
else()
    find_package(OpenCV REQUIRED
        PATHS /workspace/third_party/install/opencv/
        NO_DEFAULT_PATH)
endif()

find_package(OpenMP REQUIRED)

# Optional: INRIA diff_gaussian_rasterization CUDA kernel
# When available, link against this for real-time rendering.
# Clone from: https://github.com/graphdeco-inria/diff-gaussian-rasterization
set(DIFF_GAUSSIAN_RASTERIZATION_DIR
    ${CMAKE_SOURCE_DIR}/third_party/diff_gaussian_rasterization
    CACHE PATH "Path to diff_gaussian_rasterization")
if(EXISTS ${DIFF_GAUSSIAN_RASTERIZATION_DIR})
  message(STATUS "Found CUDA rasterizer at ${DIFF_GAUSSIAN_RASTERIZATION_DIR}")
  add_definitions(-DHAVE_CUDA_RASTERIZER)
else()
  message(STATUS "CUDA rasterizer not found — using pure-LibTorch fallback")
endif()

# Optional: GLFW + OpenGL for viewer
find_package(glfw3 QUIET)
find_package(OpenGL QUIET)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUDA_ARCH}")

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${TORCH_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})

link_directories(
    ${TORCH_LIBRARY_DIRS})

# ---------------------------------------------------------------------------
# Scaffold-ChunGS Library
# ---------------------------------------------------------------------------
set(SCAFFOLD_CHUNGS_SOURCES
    # Model
    src/model/model_core.cpp
    src/model/model_mlp.cpp
    src/model/model_storage.cpp
    src/model/model_memory.cpp
    src/model/model_optimization.cpp
    # Rendering
    src/rendering/gaussian_renderer.cpp
    src/rendering/frustum_culler.cpp
    # Scene
    src/scene/gaussian_keyframe.cpp
    src/scene/gaussian_scene.cpp
    src/scene/keyframe_selection.cpp
    # Training
    src/training/trainer.cpp
    src/training/losses.cpp
    # Depth Estimation
    src/depth/depth_estimator.cpp
    # Mapper Pipeline
    src/mapper/mapper_core.cpp)

add_library(scaffold_chungs SHARED ${SCAFFOLD_CHUNGS_SOURCES})

target_link_libraries(scaffold_chungs
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    OpenMP::OpenMP_CXX)

target_compile_features(scaffold_chungs PUBLIC cxx_std_17)

# ---------------------------------------------------------------------------
# Example executables
# ---------------------------------------------------------------------------
add_executable(scaffold_chunks_demo examples/scaffold_chunks_demo.cpp)
target_link_libraries(scaffold_chunks_demo scaffold_chungs)

# ---------------------------------------------------------------------------
# Configuration output
# ---------------------------------------------------------------------------
set_target_properties(scaffold_chungs PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set_target_properties(scaffold_chunks_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS "Scaffold-ChunGS configured:")
message(STATUS "  Torch include: ${TORCH_INCLUDE_DIRS}")
message(STATUS "  CUDA arch: sm_${CUDA_ARCH}")
message(STATUS "  OpenCV: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
