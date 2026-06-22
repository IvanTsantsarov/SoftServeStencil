// Supress warning about the barrier
#pragma nv_diag_suppress 20054


#include "cuda_kernels.cuh"
#include "defines.h"
#include "utils.hpp"
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Constant Memory allocation for Coefficients
__constant__ float c_coeffs[COEF_S][COEF_S];

// Helper to safely fetch clamp-to-edge global values
__device__ inline float fetch_pixel(const float* input, int x, int y, int width, int height) {
    x = max(0, min(width - 1, x));
    y = max(0, min(height - 1, y));
    return input[y * width + x];
}

////////////////////////////////////////////////////////////////////////////////
// NON OPTIMIZED (BASELINE) KERNEL IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////
__global__ void baseline_kernel(const float* input, float* output, int width, int height) {
    int tx = blockIdx.x * TILE_W;
    int ty = blockIdx.y * TILE_H;

    if (tx >= width || ty >= height) return;

    // Phase 1: Fixed sequential scan bounds to match CPU exactly (only 1 TILE_W)
    float tile_min = MAX_FLT;
    int y_end = ty + TILE_H;
    int x_end = tx + TILE_W; 
    
    for (int y = ty; y < y_end && y < height; ++y) {
        for (int x = tx; x < x_end && x < width; ++x) {
            float v = input[y * width + x];
            if (v < tile_min) tile_min = v;
        }
    }
    float norm_factor = max(tile_min, MIN_FLT);

    // Phase 2: Grid-stride loop to make 32x4 threads process all 128x32 pixels
    for (int local_y = threadIdx.y; local_y < TILE_H; local_y += blockDim.y) {
        for (int local_x = threadIdx.x; local_x < TILE_W; local_x += blockDim.x) {
            
            int x = tx + local_x;
            int y = ty + local_y;

            if (x < width && y < height) {
                float acc = 0.0f;
                for (int dy = -HALO_L; dy <= HALO_R; ++dy) {
                    for (int dx = -HALO_L; dx <= HALO_R; ++dx) {
                        // Ensure fetch_pixel uses identical clamp-to-edge logic as CPU!
                        float v = fetch_pixel(input, x + dx, y + dy, width, height);
                        float transformed = v * v + 0.25f * v + sqrtf(fabsf(v));
                        acc += c_coeffs[dy + 7][dx + 7] * transformed;
                    }
                }
                output[y * width + x] = acc / norm_factor;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED KERNEL IMPLEMENTATION (AMPERE+)
////////////////////////////////////////////////////////////////////////////////
__global__ void __launch_bounds__(128, 4)
optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int width, int height) {

    // Allocate without explicit initialization to avoid warnings
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // Allocate shared memory for async loading
    __shared__ float smem_input[SHARED_H][SHARED_W];
    __shared__ float smem_min_pool[WARPS_C]; // One min value per warp

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // Global thread ID in block (0 to 127)
    
    // Initialize the barrier on thread 0 BEFORE anyone uses it
    if (tid == 0) {
        init(&bar, blockDim.x * blockDim.y); 
    }
    // Sync the block to ensure the barrier object is fully initialized
    __syncthreads();     
    
    int block_origin_x = blockIdx.x * TILE_W;
    int block_origin_y = blockIdx.y * TILE_H;

    // Asynchronous global to shared memory (Ampere+)
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
        cuda::memcpy_async(
            &smem_input[smem_y][smem_x], 
            &input[glob_y * width + glob_x], 
            sizeof(float),
            bar);
    }
    
    bar.arrive_and_wait();

    //////////////////////////////////////////////////////////
    // Warp Shuffle Reductions to calculate tile minimum value

    // FIX 1: Map all 128 threads sequentially to cover all 128 tile columns
    float thread_min = MAX_FLT;
    int local_x = tid; // 0 to 127 covers TILE_W completely

    for (int row = 0; row < TILE_H; ++row) {
        if (block_origin_x + local_x < width && block_origin_y + row < height) {
            float v = smem_input[row + HALO_L][local_x + HALO_L];
            thread_min = min(thread_min, v);
        }
    }

    // FIX 2: Use an explicit full mask (0xFFFFFFFF) to prevent shuffle stalls
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_min = min(thread_min, __shfl_down_sync(0xFFFFFFFF, thread_min, offset));
    }

    // Step 3: Master thread of each warp writes to shared cache pool
    if (threadIdx.x == 0) {
        smem_min_pool[threadIdx.y] = thread_min;
    }
    __syncthreads();

    // FIX 3: Safe, sequential resolution on thread 0 to eliminate inter-warp shuffle bugs
    if (tid == 0) {
        float final_min = smem_min_pool[0];
        for (int i = 1; i < WARPS_C; ++i) {
            final_min = min(final_min, smem_min_pool[i]);
        }
        smem_min_pool[0] = final_min;
    }
    __syncthreads();

    float norm_factor = max(smem_min_pool[0], MIN_FLT);

    // Allocate a register array for the current row
    float local_row[HALO_ALL]; 

    // 3. Stencil Computation using Cached Shared Memory Matrix
    // Use a grid-stride loop over X so 32x4 threads span the 128-column tile width
    for (int col_step = 0; col_step < 4; ++col_step) {
        int target_lx = col_step * 32 + threadIdx.x; // Marches across 0 to 127
        int target_x = block_origin_x + target_lx;
        int smem_center_x = target_lx + HALO_L;

        for (int step = 0; step < 8; ++step) {
            int target_ly = threadIdx.y * 8 + step;
            int target_y = block_origin_y + target_ly;

            if (target_x < width && target_y < height) {
                float acc = 0.0f;
                int smem_center_y = target_ly + HALO_L;

                for (int dy = -HALO_L; dy <= HALO_R; ++dy) {
                    int current_smem_y = smem_center_y + dy;

                    // Copy current row into registers to reuse across dx iterations
                    #pragma unroll
                    for (int dx = -7; dx <= 8; ++dx) {
                        local_row[dx + 7] = smem_input[current_smem_y][smem_center_x + dx];
                    }

                    #pragma unroll
                    for (int dx = -HALO_L; dx <= HALO_R; ++dx) {
                        int dx_offset = dx + 7;
                        float v = local_row[dx_offset]; 
                        
                        // SFU evaluates fast sqrt calculations
                        float transformed = v * v + 0.25f * v + sqrtf(fabsf(v)); 
                        
                        acc += c_coeffs[dy + 7][dx_offset] * transformed;
                    }
                }
                output[target_y * width + target_x] = acc / norm_factor;
            }
        }
    }
}

//////////////////////////////////////////////
// NON OPTIMAZED (BASELINE) KERNEL LAUNCH
//////////////////////////////////////////////
// not using std::function or CUfunction just to intuitive
void launch_baseline_kernel(const float* d_input, float* d_output, int width, int height, 
    const float h_coeffs[COEF_S][COEF_S], float& elapsed_ms) {

    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, h_coeffs, COEF_ALL * sizeof(float)));

    dim3 threadsPerBlock(32, 4); // Fixed layout targeting 128 elements inside loop structures
    dim3 numBlocks((width + 127) / 128, (height + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    baseline_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

//////////////////////////////////////////////
// OPTIMAZED KERNEL (AMPERE+) LAUNCH
//////////////////////////////////////////////
void launch_optimized_kernel(const float* d_input, float* d_output, int width, int height, const float h_coeffs[16][16], float& elapsed_ms) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, h_coeffs, COEF_ALL * sizeof(float)));

    dim3 threadsPerBlock(32, 4); // 128 total threads matching optimized architecture bounds
    dim3 numBlocks((width + 127) / 128, (height + 31) / 32);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    optimized_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

