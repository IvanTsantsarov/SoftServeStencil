#include "cuda_kernels.cuh"
#include "utils.hpp"
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constant Memory allocation for Coefficients
__constant__ float c_coeffs[16][16];

// Helper to safely fetch clamp-to-edge global values
__device__ inline float fetch_pixel(const float* input, int x, int y, int width, int height) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return input[y * width + x];
}

// ==========================================
// BASELINE KERNEL IMPLEMENTATION
// ==========================================
__global__ void baseline_kernel(const float* input, float* output, int width, int height) {
    int tx = blockIdx.x * 128;
    int ty = blockIdx.y * 32;

    if (tx >= width || ty >= height) return;

    // Phase 1: Compute tile minimum via unoptimized loop scan
    float tile_min = 1e37f;
    for (int y = ty; y < ty + 32 && y < height; ++y) {
        for (int x = tx; x < tx + 128 && x < width; ++x) {
            float v = input[y * width + x];
            if (v < tile_min) tile_min = v;
        }
    }
    float norm_factor = max(tile_min, 1e-6f);

    // Phase 2: Compute Stencil
    int x = blockIdx.x * 128 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    if (x < width && y < height) {
        float acc = 0.0f;
        for (int dy = -7; dy <= 8; ++dy) {
            for (int dx = -7; dx <= 8; ++dx) {
                float v = fetch_pixel(input, x + dx, y + dy, width, height);
                float transformed = v * v + 0.25f * v + sqrtf(fabsf(v));
                acc += c_coeffs[dy + 7][dx + 7] * transformed;
            }
        }
        output[y * width + x] = acc / norm_factor;
    }
}

// ==========================================
// OPTIMIZED KERNEL IMPLEMENTATION (AMPERE+)
// ==========================================
#define BLK_W 128
#define BLK_H 32
#define HALO_L 7
#define HALO_R 8
#define SHARED_W (BLK_W + HALO_L + HALO_R) // 128 + 7 + 8 = 143
#define SHARED_H (BLK_H + HALO_L + HALO_R) // 32 + 7 + 8 = 47

__global__ void __launch_bounds__(128, 4)
optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int width, int height) {

    // Allocate shared memory for async loading
    __shared__ float smem_input[SHARED_H][SHARED_W];
    __shared__ float smem_min_pool[4]; // One min value per warp (4 warps total)

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 127
    int block_origin_x = blockIdx.x * BLK_W;
    int block_origin_y = blockIdx.y * BLK_H;

    // 1. Asynchronous Global to Shared Memory Copy (Ampere Feature)
    // Total elements to load: 143 * 47 = 6721. Shared out across 128 threads.
    int total_elements = SHARED_W * SHARED_H;
    for (int i = tid; i < total_elements; i += 128) {
        int smem_y = i / SHARED_W;
        int smem_x = i % SHARED_W;

        int glob_x = block_origin_x - HALO_L + smem_x;
        int glob_y = block_origin_y - HALO_L + smem_y;

        // Secure clamp-to-edge
        glob_x = max(0, min(width - 1, glob_x));
        glob_y = max(0, min(height - 1, glob_y));

        // Direct pipeline asynchronous memory transfer
        cuda::memcpy_async(&smem_input[smem_y][smem_x], &input[glob_y * width + glob_x], sizeof(float));
    }
    cuda::memcpy_async_wait();
    __syncthreads();

    // 2. Intra-Warp Shuffle Reductions for Tile Minimum
    // Step A: Each thread finds its local min across sequential rows assigned to it
    float thread_min = 1e37f;
    int local_x = threadIdx.x; // Threads mapped 0 to 31 directly matches tile columns
    int local_y_start = threadIdx.y * 8; // Thread block is 32x4. Each warp works on 8 rows.

    for (int row = 0; row < 8; ++row) {
        int ly = local_y_start + row;
        if (block_origin_x + local_x < width && block_origin_y + ly < height) {
            float v = smem_input[ly + HALO_L][local_x + HALO_L];
            thread_min = min(thread_min, v);
        }
    }

    // Step B: Native Warp shuffle down reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_min = min(thread_min, __shfl_down_sync(0xFFFFFFFF, thread_min, offset));
    }

    // Step C: Master thread of each warp writes to shared cache pool
    if (threadIdx.x == 0) {
        smem_min_pool[threadIdx.y] = thread_min;
    }
    __syncthreads();

    // Step D: Warp 0 resolves final tile minimum
    float global_tile_min = smem_min_pool[0];
    if (threadIdx.y == 0 && threadIdx.x < 4) {
        global_tile_min = min(global_tile_min, smem_min_pool[threadIdx.x]);
        for (int offset = 2; offset > 0; offset /= 2) {
            global_tile_min = min(global_tile_min, __shfl_down_sync(0xF, global_tile_min, offset));
        }
        if (threadIdx.x == 0) {
            smem_min_pool[0] = global_tile_min;
        }
    }
    __syncthreads();

    float norm_factor = max(smem_min_pool[0], 1e-6f);

    // 3. Stencil Computation using Cached Shared Memory Matrix
    // Process 32x128 Tile using a 32x4 thread layout (Looping 8 times over Y dimension)
    int target_x = block_origin_x + threadIdx.x;
    for (int step = 0; step < 8; ++step) {
        int target_ly = threadIdx.y * 8 + step;
        int target_y = block_origin_y + target_ly;

        if (target_x < width && target_y < height) {
            float acc = 0.0f;
            int smem_center_y = target_ly + HALO_L;
            int smem_center_x = threadIdx.x + HALO_L;

            #pragma unroll 16
            for (int dy = -7; dy <= 8; ++dy) {
                #pragma unroll 16
                for (int dx = -7; dx <= 8; ++dx) {
                    float v = smem_input[smem_center_y + dy][smem_center_x + dx];
                    float transformed = v * v + 0.25f * v + sqrtf(fabsf(v));
                    acc += c_coeffs[dy + 7][dx + 7] * transformed;
                }
            }
            output[target_y * width + target_x] = acc / norm_factor;
        }
    }
}

// ==========================================
// DRIVER ROUTINES
// ==========================================
void launch_baseline_kernel(const float* d_input, float* d_output, int width, int height, const float h_coeffs[16][16], float& elapsed_ms) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, h_coeffs, 256 * sizeof(float)));

    dim3 threadsPerBlock(32, 4); // Fixed layout targeting 128 elements inside loop structures
    dim3 numBlocks((width + 127) / 128, (height + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    baseline_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

void launch_optimized_kernel(const float* d_input, float* d_output, int width, int height, const float h_coeffs[16][16], float& elapsed_ms) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, h_coeffs, 256 * sizeof(float)));

    dim3 threadsPerBlock(32, 4); // 128 total threads matching optimized architecture bounds
    dim3 numBlocks((width + 127) / 128, (height + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    optimized_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

