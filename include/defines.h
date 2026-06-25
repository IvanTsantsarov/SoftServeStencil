// When profiling with NCU is used onlu optimized kernel runs 
// only on highest resolution
// with no correctness tests 
#define USING_NCU 0

// Beauty result using inv_norm_factor = 0.5f, not max(tile_min, MIN_FLT)
#define BEAUTY_RESULT 1

// Half float tendst to use float16 instead of float32
// for inputs shared memory
#define HALF_FLOAT 1


#define THREADS_C 128
#define WARPS_C (THREADS_C/32)

#define TILE_W 128
#define TILE_H 32
#define COEF_S 16
#define COEF_ALL (COEF_S*COEF_S)
#define HALO_S 8
#define HALO_L 7
#define HALO_R 8
#define HALO_ALL 16
#define SHARED_W (TILE_W + HALO_L + HALO_R) // 128 + 7 + 8 = 143
#define SHARED_H (TILE_H + HALO_L + HALO_R) // 32 + 7 + 8 = 47
#define MAX_FLT 1e37f
#define MIN_FLT 1e-6f


// When float16 is used precision must be shifted
// or correctness test will fail
#if HALF_FLOAT
    #define MAX_ERR 1e-1
#else
    #define MAX_ERR 1e-3
#endif

#define BOTTOM_ERR 1e-6f

#define RND_CENTERS_C 10
#define RND_AMP 2.0f
#define RND_PHASE 0.1f

