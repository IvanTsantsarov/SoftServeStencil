#pragma once

void cpu_stencil_transform(
    const float* input,
    float* output,
    int width,
    int height,
    const float coeffs[16][16]
);

