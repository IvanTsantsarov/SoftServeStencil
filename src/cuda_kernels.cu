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

    // Allocate without an explicit initialization expression (ugly)
    // to avoid the warning or just suppress it like I did
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;

    // Allocate shared memory for async loading
    __shared__ float smem_input[SHARED_H][SHARED_W];
    __shared__ float smem_min_pool[WARPS_C]; // One min value per warp

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 127
    
    // Initialize the barrier on thread 0 BEFORE anyone uses it
    if (tid == 0) {
        init(&bar, blockDim.x * blockDim.y); // Pass total block threads
    }
    // We must sync the block here so all threads know the barrier is ready
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
    // __syncthreads(); // not needed when there is a barrier

    //////////////////////////////////////////////////////////
    // Warp Shuffle Reductions to calculate tile minimum value

    // Step 1: Each thread finds its local min
    float thread_min = MAX_FLT;
    int local_x = threadIdx.x; // Threads mapped 0 to 31 directly matches tile columns
    int local_y_start = threadIdx.y * 8; // Thread block is 32x4. Each warp works on 8 rows.

    for (int row = 0; row < HALO_S; ++row) {
        int ly = local_y_start + row;
        if (block_origin_x + local_x < width && block_origin_y + ly < height) {
            float v = smem_input[ly + HALO_L][local_x + HALO_L];
            thread_min = min(thread_min, v);
        }
    }

    unsigned int active_mask = __activemask();
    // #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_min = min(thread_min, __shfl_down_sync(active_mask, thread_min, offset));
    }

    // Step 3: Master thread of each warp writes to shared cache pool
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

    float norm_factor = max(smem_min_pool[0], MIN_FLT);

    // Allocate a register array for x row
    float local_row[HALO_ALL]; 

    // 3. Stencil Computation using Cached Shared Memory Matrix
    // Process 32x128 Tile using a 32x4 thread layout (Looping 8 times over Y dimension)
    int target_x = block_origin_x + threadIdx.x;
    int smem_center_x = threadIdx.x + HALO_L;
    for (int step = 0; step < 8; ++step) {
        int target_ly = threadIdx.y * 8 + step;
        int target_y = block_origin_y + target_ly;

        if (target_x < width && target_y < height) {
            float acc = 0.0f;
            int smem_center_y = target_ly + HALO_L;

            // #pragma unroll 16 // Too many unrols alongside with dx loop
            // it can put pressure on registers file
            for (int dy = -HALO_L; dy <= HALO_R; ++dy) {
                int current_smem_y = smem_center_y + dy;

                // Copy current row into registers for each warp to use it
                // in the upcomming dx itterations
                #pragma unroll
                for (int dx = -7; dx <= 8; ++dx) {
                    local_row[dx + 7] = smem_input[current_smem_y][smem_center_x + dx];
                }

                #pragma unroll
                for (int dx = -HALO_L; dx <= HALO_R; ++dx) {
                    int dx_offset = dx + 7;
                    // Read directly out of our zero-latency register array
                    float v = local_row[dx_offset]; 
                    
                    // SFU will elliminate the need of sqrt lookup table like in CPU
                    float transformed = v * v + 0.25f * v + sqrtf(fabsf(v)); 
                    
                    // accumilate the result
                    acc += c_coeffs[dy + 7][dx_offset] * transformed;
                }
            }
            output[target_y * width + target_x] = acc / norm_factor;
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

