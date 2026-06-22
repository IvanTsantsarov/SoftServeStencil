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

    std::cout << "Correctness test on " << side << "x" << side << " image...\n";
    
    std::vector<float> cpu_output(all, 0.0f);
    std::vector<float> gpu_output(all, 0.0f);

    // Generate image file
    std::vector<float> input = generate_image(side);
    
    normalize(side, input); // trash comment

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

    float *d_input = nullptr;
    float *d_output = nullptr;
    int all_floats = all * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, all_floats));
    CUDA_CHECK(cudaMalloc(&d_output, all_floats));
    CUDA_CHECK(cudaMemcpy(d_input, input.data(), all_floats, cudaMemcpyHostToDevice));

    // Currently not measured
    float elapsed_ms = 0.0f;

    std::cout << "Performing GPU kernel exec..."<< "\n";
    launch_optimized_kernel(d_input, d_output, side, side, coeffs, elapsed_ms);
    // launch_baseline_kernel(d_input, d_output, side, side, coeffs, elapsed_ms);

    CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, all * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Calculating diviation errors..."<< "\n";
    // Verify mathematical bounds criteria
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    int wrong_pixels = 0;
    const int max_wrong_pixels = 100;
    for (int i = 0; i < all; ++i) {
        float abs_err = std::fabs(cpu_output[i] - gpu_output[i]);
        float rel_err = abs_err / (std::fabs(cpu_output[i]) + BOTTOM_ERR);

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        // printout wrong values
        if (abs_err >= MAX_ERR && rel_err >= MAX_ERR) { 
            wrong_pixels ++;
            if( wrong_pixels < max_wrong_pixels) {
                std::cout << i << "=>" << cpu_output[i] << ',' << gpu_output[i] << '|';// << std::endl;
            }
        }
    }

    std::cout << "Test Dimensions: " << side << "x" << side << " -> ";
    if (wrong_pixels == 0) {
        std::cout << "PASSED (Max Abs Error: " << max_abs_err 
        << ", Max Rel Error: " << max_rel_err << ")" << std::endl;
    } else {
        std::cout << "FAILED (Max Abs Error: " << max_abs_err << ",Wrong pixels:" << wrong_pixels
        << ", Max Rel Error: " << max_rel_err <<  ")" << std::endl;
        exit(1);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    char ppm_path[256] = {0};

    sprintf( ppm_path, "../res/cpu%d.ppm", side);
    write_ppm( ppm_path, side, cpu_output );

    sprintf( ppm_path, "../res/gpu%d.ppm", side);
    write_ppm( ppm_path, side, gpu_output );

    std::cout << "Correctnes test finished!"<< "\n";
}

bool all_correctness() {
    correctness(256);
    correctness(1024);
    correctness(4096);
    std::cout << "\nAll Correctness Validations Passed Successfully.\n";
    return true;
}

