import numpy as np
from numba import cuda, float32
import math

# Define the coefficients for the stencil.
# These would typically be loaded from a file or generated.
# For this example, let's use a 16x16 array of ones as specified by the stencil definition (dy, dx from -7 to 8).
COEFFS_DIM = 16 # (8 - (-7) + 1)
coeffs_cpu_default = np.ones((COEFFS_DIM, COEFFS_DIM), dtype=np.float32)

@cuda.jit(device=True)
def transform_value(v):
    """Device function to compute the transformation part of the stencil."""
    # Matches: v * v + 0.25f * v + sqrtf(fabsf(v))
    return v * v + 0.25 * v + math.sqrt(math.fabs(v))

@cuda.jit(device=True)
def stencil16x16_device(input_gpu, x, y, width, height, coeffs_gpu):
    """Device function to compute the 16x16 stencil for a given pixel (x, y)."""
    acc = 0.0
    # Iterate through dy from -7 to 8, dx from -7 to 8
    for dy_offset in range(COEFFS_DIM):
        dy = dy_offset - 7
        for dx_offset in range(COEFFS_DIM):
            dx = dx_offset - 7

            current_x = x + dx
            current_y = y + dy

            # Handle image boundaries: if a tap falls outside, treat value as 0.0
            v = 0.0
            if 0 <= current_x < width and 0 <= current_y < height:
                v = input_gpu[current_y * width + current_x] # Assuming flattened 1D array

            transformed = transform_value(v)
            # coeffs[dy + 7][dx + 7] maps to coeffs_gpu[dy_offset, dx_offset]
            acc += coeffs_gpu[dy_offset, dx_offset] * transformed
    return acc

@cuda.jit
def process_image_kernel(input_gpu, output_gpu, width, height, coeffs_gpu):
    """
    Main CUDA kernel to process the image using 128x32 tiles.
    Each block processes one tile.
    """
    TILE_WIDTH = 128
    TILE_HEIGHT = 32
    TILE_SIZE = TILE_WIDTH * TILE_HEIGHT

    # Calculate the tile index for the current block
    tile_idx_x = cuda.blockIdx.x
    tile_idx_y = cuda.blockIdx.y

    # Calculate the starting pixel coordinates of the current tile
    tile_start_x = tile_idx_x * TILE_WIDTH
    tile_start_y = tile_idx_y * TILE_HEIGHT

    # Guard against processing tiles that are entirely out of image bounds
    if tile_start_x >= width or tile_start_y >= height:
        return

    # Shared memory for minimum reduction within the block.
    # The size is dynamically allocated based on `sharedmem` argument in kernel launch.
    shared_min_temp = cuda.shared.array(shape=(0,), dtype=float32)

    # Each thread computes a local minimum over the elements it is responsible for within the tile.
    thread_min = float32(np.inf)

    # Calculate global thread ID within the block
    thread_id_in_block = cuda.threadIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    num_threads_in_block = cuda.blockDim.x * cuda.blockDim.y

    # First pass: Compute tile minimum
    # Each thread iterates over a subset of the tile's pixels to find its local minimum
    for k in range(thread_id_in_block, TILE_SIZE, num_threads_in_block):
        # Calculate relative coordinates (tx_rel, ty_rel) within the current tile
        ty_rel = k // TILE_WIDTH
        tx_rel = k % TILE_WIDTH

        # Calculate global image coordinates
        global_y = tile_start_y + ty_rel
        global_x = tile_start_x + tx_rel

        # Ensure global coordinates are within image bounds before accessing `input_gpu`
        if global_x < width and global_y < height:
            current_val = input_gpu[global_y * width + global_x]
            if current_val < thread_min:
                thread_min = current_val

    # Store the thread's local minimum in shared memory
    shared_min_temp[thread_id_in_block] = thread_min
    cuda.syncthreads() # Synchronize to ensure all local minima are written to shared memory

    # Perform parallel reduction in shared memory to find the overall tile minimum
    s = num_threads_in_block // 2
    while s > 0:
        if thread_id_in_block < s:
            shared_min_temp[thread_id_in_block] = min(shared_min_temp[thread_id_in_block], shared_min_temp[thread_id_in_block + s])
        s //= 2
        cuda.syncthreads() # Synchronize after each reduction step

    # The final tile minimum is now stored in shared_min_temp[0]
    tile_min = shared_min_temp[0]

    # Normalize tile_min to prevent division by zero or extremely small numbers
    normalized_tile_min = max(tile_min, 1e-6)

    # Second pass: Compute stencil and normalize for each pixel in the tile
    # Each thread iterates over its assigned subset of tile pixels again
    for k in range(thread_id_in_block, TILE_SIZE, num_threads_in_block):
        ty_rel = k // TILE_WIDTH
        tx_rel = k % TILE_WIDTH

        global_y = tile_start_y + ty_rel
        global_x = tile_start_x + tx_rel

        # Ensure global coordinates are within image bounds for output
        if global_x < width and global_y < height:
            # Compute the stencil value for the current pixel
            acc = stencil16x16_device(input_gpu, global_x, global_y, width, height, coeffs_gpu)
            # Store the normalized result in the output image
            output_gpu[global_y * width + global_x] = acc / normalized_tile_min

def process_image(input_image: np.ndarray, coeffs: np.ndarray = coeffs_cpu_default) -> np.ndarray:
    """
    Host function to launch the CUDA kernel for image processing.

    Args:
        input_image (np.ndarray): The input floating-point image (2D NumPy array).
        coeffs (np.ndarray): The 16x16 stencil coefficients (2D NumPy array, float32).

    Returns:
        np.ndarray: The processed output image (2D NumPy array).
    """
    if input_image.ndim != 2:
        raise ValueError("Input image must be a 2D NumPy array.")

    height, width = input_image.shape

    # Ensure input image and coefficients are float32 for CUDA compatibility
    if input_image.dtype != np.float32:
        input_image = input_image.astype(np.float32)
    if coeffs.dtype != np.float32:
        coeffs = coeffs.astype(np.float32)

    # Allocate and transfer data to GPU memory
    # Flatten images for 1D access in CUDA kernel
    input_gpu = cuda.to_device(input_image.ravel())
    output_gpu = cuda.device_array_like(input_gpu) # Output array of same shape and type
    coeffs_gpu = cuda.to_device(coeffs)

    TILE_WIDTH = 128
    TILE_HEIGHT = 32

    # Calculate grid dimensions (number of blocks in X and Y)
    blocks_x = (width + TILE_WIDTH - 1) // TILE_WIDTH
    blocks_y = (height + TILE_HEIGHT - 1) // TILE_HEIGHT
    grid_dim = (blocks_x, blocks_y)

    # Define block dimensions (threads per block).
    # A common choice is 32x32 = 1024 threads, which is usually the maximum.
    # This allows each thread to process a subset of the TILE_SIZE elements (4096 elements).
    threadsperblock = (32, 32) # Total 1024 threads

    # Calculate shared memory size needed for the reduction array.
    # It needs to hold `num_threads_in_block` float32 values.
    num_threads_in_block = threadsperblock[0] * threadsperblock[1]
    shared_mem_size = num_threads_in_block * np.float32().itemsize

    # Launch the CUDA kernel
    process_image_kernel[grid_dim, threadsperblock, 0, shared_mem_size](
        input_gpu, output_gpu, width, height, coeffs_gpu
    )
    cuda.synchronize() # Wait for the kernel to complete execution

    # Copy the result back from GPU to host memory and reshape to original 2D image shape
    return output_gpu.copy_to_host().reshape(height, width)
