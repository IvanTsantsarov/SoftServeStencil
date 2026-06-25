#include <iostream>
#include <vector>
#include <iomanip>

#include "defines.h"
#include "cuda_kernels.cuh"
#include "../include/utils.hpp"


void benchmark(int size) {
    int all = size * size;
    std::vector<float> h_input(all, 1.25f);

    float h_coeffs[COEF_S][COEF_S];
    
    for(int i=0; i < COEF_S; ++i) {
        for(int j=0; j < COEF_S; ++j) {
            // Balanced mock constants
            h_coeffs[i][j] = 0.0039f; 
        } 
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, all * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), all * sizeof(float), cudaMemcpyHostToDevice));

    float baseline_time = 0.0f;
    float optimized_time = 0.0f;

    
    #if  !USING_NCU
        // Warm-up
        launch_baseline_kernel(d_input, d_output, size, size, h_coeffs, baseline_time);
        launch_optimized_kernel(d_input, d_output, size, size, h_coeffs, optimized_time);
        // Real Execution Timing Runs
        launch_baseline_kernel(d_input, d_output, size, size, h_coeffs, baseline_time);
    #endif

    launch_optimized_kernel(d_input, d_output, size, size, h_coeffs, optimized_time);

    // Compute metrics
    // Read Input + Write Output + Stencil overlapping footprint
    double bytes_transferred = 2.0 * all * sizeof(float);
    double effective_bandwidth_gb_s = (bytes_transferred / (optimized_time / 1000.0)) / 1e9;

    // Arithmetic estimation: 16x16=256 steps. Each step has ~5 floating point operations
    double estimated_flop = double(all) * 256.0 * 5.0;
    double gflops = (estimated_flop / (optimized_time / 1000.0)) / 1e9;

    std::cout << std::setw(10) << size
            << std::setw(19) << (std::to_string(baseline_time) + " ms")
            << std::setw(19) << (std::to_string(optimized_time) + " ms")
            << std::setw(14) << (std::to_string(baseline_time / optimized_time) + "x")
            << std::setw(16) << effective_bandwidth_gb_s
            << std::setw(16) << gflops << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}


bool all_benchmarks() {
        std::cout << std::left
              << std::setw(12) << "Resolution"
              << std::setw(18) << "Baseline Time"
              << std::setw(18) << "Optimized Time"
              << std::setw(13) << "Speedup"
              << std::setw(18) << "Eff. BW (GB/s)"
              << std::setw(18) << "Est. GFLOPS" << std::endl;
    std::cout << std::string(95, '-') << std::endl;

    #if !USING_NCU
        benchmark(1024);
        benchmark(4096);
    #endif
    benchmark(8192);

    return true;
}