#include "cuda_kernels.cuh"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

void run_benchmark(int size) {
    int N = size * size;
    std::vector<float> h_input(N, 1.25f);

    float h_coeffs[16][16];
    for(int i=0; i<16; ++i) {
        for(int j=0; j<16; ++j) h_coeffs[i][j] = 0.0039f; // Balanced mock constants
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float baseline_time = 0.0f;
    float optimized_time = 0.0f;

    // Warm-up
    launch_baseline_kernel(d_input, d_output, size, size, h_coeffs, baseline_time);
    launch_optimized_kernel(d_input, d_output, size, size, h_coeffs, optimized_time);

    // Real Execution Timing Runs
    launch_baseline_kernel(d_input, d_output, size, size, h_coeffs, baseline_time);
    launch_optimized_kernel(d_input, d_output, size, size, h_coeffs, optimized_time);

    // Compute metrics
    // Read Input + Write Output + Stencil overlapping footprint
    double bytes_transferred = 2.0 * N * sizeof(float);
    double effective_bandwidth_gb_s = (bytes_transferred / (optimized_time / 1000.0)) / 1e9;

    // Arithmetic estimation: 16x16=256 steps. Each step has ~5 floating point operations
    double estimated_flop = double(N) * 256.0 * 5.0;
    double gflops = (estimated_flop / (optimized_time / 1000.0)) / 1e9;

    std::cout << std::setw(10) << size << "x" << size
              << std::setw(15) << baseline_time << " ms"
              << std::setw(15) << optimized_time << " ms"
              << std::setw(12) << (baseline_time / optimized_time) << "x"
              << std::setw(15) << effective_bandwidth_gb_s
              << std::setw(15) << gflops << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    std::cout << std::left
              << std::setw(12) << "Resolution"
              << std::setw(18) << "Baseline Time"
              << std::setw(18) << "Optimized Time"
              << std::setw(13) << "Speedup"
              << std::setw(18) << "Eff. BW (GB/s)"
              << std::setw(18) << "Est. GFLOPS" << std::endl;
    std::cout << std::string(95, '-') << std::endl;

    run_benchmark(1024);
    run_benchmark(4096);
    run_benchmark(8192);

    return 0;
}

