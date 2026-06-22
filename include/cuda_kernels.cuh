#pragma once
#include "defines.h"

void launch_baseline_kernel(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float h_coeffs[COEF_S][COEF_S],
    float& elapsed_ms
);

void launch_optimized_kernel(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    const float h_coeffs[COEF_S][COEF_S],
    float& elapsed_ms
);

