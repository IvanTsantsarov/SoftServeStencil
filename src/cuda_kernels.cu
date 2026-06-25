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

    // but I've changed it like just to create smooth result
    #if BEAUTY_RESULT
        float inv_norm_factor = 0.5f;
    #else
        float inv_norm_factor = 1.0f / max(tile_min, MIN_FLT);
    #endif


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
                output[y * width + x] = acc * inv_norm_factor;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED KERNEL IMPLEMENTATION (AMPERE+)
////////////////////////////////////////////////////////////////////////////////
__global__ void __launch_bounds__(128, 3)
optimized_kernel(const float* __restrict__ input, float* __restrict__ output, int width, int height)
{

    #if HALF_FLOAT
        __shared__ half smem_input[SHARED_H][SHARED_W];
    #else
        __shared__ float smem_input[SHARED_H][SHARED_W];
    #endif
    
    __shared__ float smem_min_pool[WARPS_C];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 to 127
    
    #if !HALF_FLOAT
        __shared__ cuda::barrier<cuda::thread_scope_block> bar;
        if (tid == 0) {
            init(&bar, blockDim.x * blockDim.y); 
        }
        __syncthreads();     
    #endif
    
    
    int block_origin_x = blockIdx.x * TILE_W;
    int block_origin_y = blockIdx.y * TILE_H;

    // 1. Asynchronous Copy (Unchanged)
    int total_elements = SHARED_W * SHARED_H;
    for (int i = tid; i < total_elements; i += 128) {
        int smem_y = i / SHARED_W;
        int smem_x = i % SHARED_W;
        int glob_x = max(0, min(width - 1, block_origin_x - HALO_L + smem_x));
        int glob_y = max(0, min(height - 1, block_origin_y - HALO_L + smem_y));

        #if HALF_FLOAT
            smem_input[smem_y][smem_x] = input[glob_y * width + glob_x];
        #else
            cuda::memcpy_async(&smem_input[smem_y][smem_x], &input[glob_y * width + glob_x], sizeof(float), bar);
        #endif
    }
    
    #if !HALF_FLOAT
        bar.arrive_and_wait();
    #endif

    // 2. Tile minimum calc
    float thread_min = MAX_FLT;
    for (int row = 0; row < TILE_H; ++row) {
        if (block_origin_x + tid < width && block_origin_y + row < height) {
            thread_min = min(thread_min, smem_input[row + HALO_L][tid + HALO_L]);
        }
    }

    // Reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_min = min(thread_min, __shfl_down_sync(0xFFFFFFFF, thread_min, offset));
    }

    if (threadIdx.x == 0) {
        smem_min_pool[threadIdx.y] = thread_min;
    }
    __syncthreads();

    // Final min for 1st thread
    if (tid == 0) {
        float final_min = smem_min_pool[0];
        for (int i = 1; i < WARPS_C; ++i) {
            final_min = min(final_min, smem_min_pool[i]);
        }
        smem_min_pool[0] = final_min;
    }
    __syncthreads(); 

    
    // but I've changed it like just to create smooth result
    #if BEAUTY_RESULT
        float inv_norm_factor = 0.5f;
    #else
        float inv_norm_factor = 1.0f / max(smem_min_pool[0], MIN_FLT);
    #endif

    // 3. Using vector computation instead
    // Thread block layout (32x4) maps to 32 independent processing units.
    // Instead of grid-striding by 32 single pixels, we stride by 32 chunks of *4 pixels* (float4).
    // Loop steps 0 -> (128 total width / 4 elements per vector / 32 threads) = 1 iteration required per row!
    
    int target_lx = threadIdx.x * 4; // Thread 0 checks 0,1,2,3; Thread 1 checks 4,5,6,7... up to 127
    int target_x = block_origin_x + target_lx;
    int smem_center_x = target_lx + HALO_L;

    // Loop 8 rows
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        int target_ly = threadIdx.y * 8 + step;
        int target_y = block_origin_y + target_ly;

        // Check bounds
        if (target_y < height && target_x < width) {
            
            // Using a 4D vector
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            int smem_center_y = target_ly + HALO_L;

            // Unrolling 
            // *** Warning, this connected to HALO size!
            #pragma unroll 16
            for (int dy = -7; dy <= 8; ++dy) {
                half* smem_input_y = smem_input[smem_center_y + dy];
                float4 v;

                #pragma unroll 16
                for (int dx = -7; dx <= 8; ++dx) {
                    float coeff = c_coeffs[dy + 7][dx + 7];
                    // Manual unroll

                    // Sequental share mem reading
    
                    #if HALF_FLOAT
                        v.x = smem_input_y[smem_center_x + dx];
                        v.y = smem_input_y[smem_center_x + dx + 1];
                        v.z = smem_input_y[smem_center_x + dx + 2];
                        v.w = smem_input_y[smem_center_x + dx + 3];
                    #else                    
                        memcpy(&v, &smem_input_y[smem_center_x + dx], sizeof(float2));
                    #endif

                    // TODO: SQRT is a serios bottleneck
                    // try to use Taylor series 
                    // with first 3 members of the row: sqrt(x) = 1 + 1/2(x-1) - 1/8(x-1)^2;
                    // for values close to 1 it gives 1e-4 error (bellow MAX_ERR)

                    #define ACC_ADD(__dim__) acc.__dim__ += \
                        coeff * ((v.__dim__ + 0.25f) * v.__dim__ + \
                        sqrtf(fabsf(v.__dim__))); 
                        // fabsf(v.__dim__) * rsqrtf(fabsf(v.__dim__)));
                    
                    ACC_ADD(x);
                    ACC_ADD(y);
                    ACC_ADD(z);
                    ACC_ADD(w);
                }
            }

            // Normalize all vector values
            acc.x *= inv_norm_factor;
            acc.y *= inv_norm_factor;
            acc.z *= inv_norm_factor;
            acc.w *= inv_norm_factor;

            // Perform vectorized global writeback safely with boundary awareness
            int out_idx = target_y * width + target_x;
            
            if (target_x + 3 < width) {
                // Perfect alignment match: Cast and execute direct 128-bit store instruction
                reinterpret_cast<float4*>(output)[out_idx / 4] = acc;
            } else {
                // Edge tail correction: Clean individual fallbacks for fractional boundaries
                output[out_idx] = acc.x;
                if (target_x + 1 < width) output[out_idx + 1] = acc.y;
                if (target_x + 2 < width) output[out_idx + 2] = acc.z;
            }
        } // Bounds checking
    } // for (int step = 0; step < 8; ++step)
}

//////////////////////////////////////////////
// NON OPTIMAZED (BASELINE) KERNEL LAUNCH
//////////////////////////////////////////////
// not using std::function or CUfunction just to intuitive
void launch_baseline_kernel(const float* d_input, float* d_output, int width, int height, 
    const float h_coeffs[COEF_S][COEF_S], float& elapsed_ms) {

    CUDA_CHECK(cudaMemcpyToSymbol(c_coeffs, h_coeffs, COEF_ALL * sizeof(float)));

    dim3 numBlocks((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H ); 
    dim3 threadsPerBlock(32, 4); // 128 total threads
    
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

    dim3 numBlocks((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H );
    dim3 threadsPerBlock(32, 4); // 128 total threads

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

