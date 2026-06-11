#include "cpu_reference.hpp"
#include "cuda_kernels.cuh"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>

void execute_test(int size) {
    int N = size * size;
    std::vector<float> h_input(N);
    std::vector<float> h_cpu_output(N, 0.0f);
    std::vector<float> h_gpu_output(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i % 100) * 0.1f + 0.5f;
    }

    float h_coeffs[16][16];
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) h_coeffs[i][j] = 1.0f / 256.0f;
    }

    // Execute reference validation pipeline
    cpu_stencil_transform(h_input.data(), h_cpu_output.data(), size, size, h_coeffs);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    float dummy_t = 0.0f;
    launch_optimized_kernel(d_input, d_output, size, size, h_coeffs, dummy_t);

    CUDA_CHECK(cudaMemcpy(h_gpu_output.data(), d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify mathematical bounds criteria
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < N; ++i) {
        float abs_err = std::fabs(h_cpu_output[i] - h_gpu_output[i]);
        float rel_err = abs_err / (std::fabs(h_cpu_output[i]) + 1e-6f);

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std::cout << "Test Dimensions: " << size << "x" << size << " -> ";
    if (max_abs_err < 1e-3f && max_rel_err < 1e-3f) {
        std::cout << "PASSED (Max Abs Error: " << max_abs_err << ", Max Rel Error: " << max_rel_err << ")\n";
    } else {
        std::cout << "FAILED (Max Abs Error: " << max_abs_err << ", Max Rel Error: " << max_rel_err << ")\n";
        exit(1);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

int main() {
    execute_test(256);
    execute_test(1024);
    execute_test(4096);
    std::cout << "\nAll Correctness Validations Passed Successfully.\n";
    return 0;
}

