cmake_minimum_required(VERSION 3.18)
project(cuda_tile_stencil_task LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target Ampere (sm_80), Ada Lovelace (sm_89), and Hopper (sm_90)
set(CMAKE_CUDA_ARCHITECTURES "80;89;90")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --use_fast_math -Xptxas=-v")

include_directories(include)

# Core Library
add_library(stencil_core
    src/cpu_reference.cpp
    src/cuda_kernels.cu
)
target_include_directories(stencil_core PUBLIC include)

# Correctness Tests Executable
add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE stencil_core)

# Benchmark Executable
add_executable(benchmark_run src/main_benchmark.cpp)
target_link_libraries(benchmark_run PRIVATE stencil_core)
