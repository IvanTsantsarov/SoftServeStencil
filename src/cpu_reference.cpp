#include "../include/cpu_reference.hpp"
#include <cmath>
#include <algorithm>

void cpu_stencil_transform(
    const float* input,
    float* output,
    int width,
    int height,
    const float coeffs[16][16])
{
    const int TILE_W = 128;
    const int TILE_H = 32;
    int height_1 = height - 1;
    int width_1 = width - 1;

    // First, compute the square root of the absolute value of the input for all pixels
    float inputSquareRoots[height * width];
    for (int i = 0; i < height * width; ++i) {
        inputSquareRoots[i] = std::sqrt(std::fabs(input[i]));
    }

    for (int ty = 0; ty < height; ty += TILE_H) {
        int endTileY = ty + TILE_H;
        for (int tx = 0; tx < width; tx += TILE_W) {
            int endTileX = tx + TILE_W;

            // 1. Compute tile minimum
            float tile_min = 1e37f;
            for (int y = ty; y < std::min(endTileY, height); ++y) {
                for (int x = tx; x < std::min(endTileX, width); ++x) {
                    tile_min = std::min(tile_min, input[y * width + x]);
                }
            }
            float norm_factor_1 = 1.0f / std::max(tile_min, 1e-6f);

            // 2. Compute stencil transformation & normalization
            for (int y = ty; y < std::min(endTileY, height); ++y) {
                for (int x = tx; x < std::min(endTileX, width); ++x) {
                    float acc = 0.0f;

                    for (int dy = -7; dy <= 8; ++dy) {
                        int ny = y + dy;
                        int offsetY = ny * width;
                        for (int dx = -7; dx <= 8; ++dx) {
                            int nx = x + dx;

                            // Out-of-bounds clamp to edge strategy
                            ny = std::max(0, std::min(height_1, ny));
                            nx = std::max(0, std::min(width_1, nx));

                            int offset = offsetY + nx;
                            float v = input[offset];
                            float sqrt_v = inputSquareRoots[offset];
                            float transformed = v * v + 0.25f * v + sqrt_v;
                            acc += coeffs[dy + 7][dx + 7] * transformed;
                        }
                    }
                    output[y * width + x] = acc * norm_factor_1;
                }
            }

        }
    }
}

