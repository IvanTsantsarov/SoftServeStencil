#pragma once

void launch_baseline_kernel(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float h_coeffs[16][16],
    float& elapsed_ms
);

void launch_optimized_kernel(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float h_coeffs[16][16],
    float& elapsed_ms
);

