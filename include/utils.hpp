#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA Failure");                    \
        }                                                                \
    } while (0)

