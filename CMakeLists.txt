cmake_minimum_required(VERSION 3.18)
project(softserve_cuda_task LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Ampere, Ada Lovelace, and Hopper
set(CMAKE_CUDA_ARCHITECTURES "80;89;90")
string(APPEND CMAKE_CUDA_FLAGS " -lineinfo --use_fast_math -Xptxas=-v")

# Force explicit location tracking for the CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# 1. Unified Single Executable Target
# We pass ALL source files (both CPU .cpp and GPU .cu) directly to one target
add_executable(softserve_task
    src/main.cpp
    src/cpu_reference.cpp
    src/cuda_kernels.cu
    tests/correctness.cpp
    tests/benchmarks.cpp
)

# 2. Bind includes and system paths directly to the application
target_include_directories(softserve_task PRIVATE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tests>
)

# 3. Link required libraries directly
target_link_libraries(softserve_task PRIVATE 
    CUDA::cudart 
    CUDA::cuda_driver
)
