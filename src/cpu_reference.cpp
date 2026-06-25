#include "../include/defines.h"
#include "../include/cpu_reference.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>


void cpu_stencil_transform(
    const float* input,
    float* output,
    int width,
    int height,
    const float coeffs[COEF_S][COEF_S])
{
    int height_1 = height - 1;
    int width_1 = width - 1;

    std::cout << "Precalculate squares of the input..." << std::endl;

    // First, compute the square root of the absolute value of the input for all pixels
    int all = height * width;
    float* inputSquareRoots = new float[all];
    for (int i = 0; i < all; ++i) {
        inputSquareRoots[i] = std::sqrt(std::fabs(input[i]));
    }

    std::cout << "Tiling on CPU (can be very slow)..." << std::endl;

    for (int ty = 0; ty < height; ty += TILE_H) {
        int endTileY = ty + TILE_H;
        for (int tx = 0; tx < width; tx += TILE_W) {
            int endTileX = tx + TILE_W;

            // 1. Compute tile minimum
            float tile_min = MAX_FLT;
            for (int y = ty; y < std::min(endTileY, height); ++y) {
                for (int x = tx; x < std::min(endTileX, width); ++x) {
                    tile_min = std::min(tile_min, input[y * width + x]);
                }
            }

                // but I've changed it like just to create smooth result
            #if BEAUTY_RESULT
                float inv_norm_factor = 0.5f;
            #else
                float inv_norm_factor = 1.0f / std::max(tile_min, MIN_FLT);
            #endif

            // 2. Compute stencil transformation & normalization
            for (int y = ty; y < std::min(endTileY, height); ++y) {
                for (int x = tx; x < std::min(endTileX, width); ++x) {

                    float acc = 0.0f;
                    for (int dy = -7; dy <= 8; ++dy) {
                        
                        int ny = y + dy;
                        // out-of-bounds clamp y
                        ny = std::max(0, std::min(height_1, ny));
                        int offsetY = ny * width;
                        
                        for (int dx = -7; dx <= 8; ++dx) {
                            int nx = x + dx;

                            // out-of-bounds clamp x
                            nx = std::max(0, std::min(width_1, nx));

                            int offset = offsetY + nx;
                            float v = input[offset];
                            float sqrt_v = inputSquareRoots[offset];
                            float transformed = v * v + 0.25f * v + sqrt_v;
                            acc += coeffs[dy + 7][dx + 7] * transformed;
                        }
                    }
                    output[y * width + x] = acc * inv_norm_factor;
                }
            }

        }
    }

    delete [] inputSquareRoots;

    std::cout << "CPU stencil finished." << std::endl;
}

