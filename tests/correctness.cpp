#include "cpu_reference.hpp"
#include "cuda_kernels.cuh"
#include "defines.h"
#include "utils.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include "tests.h"

void correctness(int side) {
    int all = side * side;

    std::cout << "Correctness test on " << side << "by" << side << "image...\n";
    
    std::vector<float> cpu_output(all, 0.0f);
    std::vector<float> gpu_output(all, 0.0f);

    // Generate image file
    std::vector<float> input = generate_image(side);
    normalize(side, input);
    char file_path[256] = {0};
    sprintf(file_path, "../res/image%d.ppm", side);
    write_ppm(file_path, side, input);
    
    std::cout << "Generating coefficients " << COEF_S << "by" << COEF_S << "matrix...\n";

    // generate gradient coeficients
    float coeffs[COEF_S][COEF_S];
    for (int i = 0; i < COEF_S; ++i) {
        for (int j = 0; j < COEF_S; ++j) coeffs[i][j] = (float)i*j * (1.0f / COEF_ALL);
    }

    std::cout << "Performing CPU stencil transform..."<< "\n";

    // Execute reference validation pipeline
    cpu_stencil_transform(input.data(), cpu_output.data(), side, side, coeffs);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, all * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, all * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), all * sizeof(float), cudaMemcpyHostToDevice));

    float dummy_t = 0.0f;

    std::cout << "Performing GPU kernel exec..."<< "\n";
    // launcoptimized_kernel(d_input, d_output, size, size, coeffs, dummy_t);
    launch_baseline_kernel(d_input, d_output, side, side, coeffs, dummy_t);

    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, all * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Calculating diviation errors..."<< "\n";
    // Verify mathematical bounds criteria
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < all; ++i) {
        float abs_err = std::fabs(cpu_output[i] - gpu_output[i]);
        float rel_err = abs_err / (std::fabs(cpu_output[i]) + 1e-6f);

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;
    }

    std::cout << "Test Dimensions: " << side << "x" << side << " -> ";
    if (max_abs_err < MAX_ERR && max_rel_err < MAX_ERR) {
        std::cout << "PASSED (Max Abs Error: " << max_abs_err << ", Max Rel Error: " << max_rel_err << ")\n";
    } else {
        std::cout << "FAILED (Max Abs Error: " << max_abs_err << ", Max Rel Error: " << max_rel_err << ")\n";
        exit(1);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "Correctnes test finished!"<< "\n";
}

bool all_correctness() {
    correctness(256);
    correctness(1024);
    correctness(4096);
    std::cout << "\nAll Correctness Validations Passed Successfully.\n";
    return true;
}

