/**
 * NVIDIA CUDA helper functions.
 */

#ifndef SRC__CUDA_CUH
#define SRC__CUDA_CUH

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "util.h"

#define CUDA_ERRCHK( err ) {\
    if (err != cudaSuccess) {\
        std::cerr << "[" << __FILE__ << ", " << __LINE__ << "] " \
            << cudaGetErrorName(err) << ": " \
            << cudaGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

// Mask for all warps.
#define ALLWARP (1 << warpSize - 1) 

/**
 * Only thread of warp id = 0 receive a warp-level minimum.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to minimize.
 * Returns:
 *   minimum value across all warps (only for thread of warp id = 0).
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__inline__ __device__
T warp_min(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val = min(val, __shfl_down_sync(ALLWARP, val, offset));
    return val;
}

/*****************************************************************************
 ***** Warp-level Operations *************************************************
 *****************************************************************************/

/**
 * Each thread in a warp will receive a warp-level minimum.
 * __shfl_xor_sync creates a butterfly pattern.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to minimize.
 * Returns:
 *   minimum value across all warps.
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__device__ __inline__
T warp_all_min(T val) {
    for (int mask = warpSize >> 1; mask > 0; mask >>= 1) 
        val = min(val, __shfl_xor_sync(ALLWARP, val, mask));
    return val;
}

/**
 * Only thread of warp id = 0 receive a warp-level maximum.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to maximum.
 * Returns:
 *   maximum value across all warps (only for thread of warp id = 0).
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__device__ __inline__
T warp_max(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val = max(val, __shfl_down_sync(ALLWARP, val, offset));
    return val;
}

/**
 * Each thread in a warp will receive a warp-level maximum.
 * __shfl_xor_sync creates a butterfly pattern.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to maximum.
 * Returns:
 *   maximum value across all warps.
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__device__ __inline__
T warp_all_max(T val) {
    for (int mask = warpSize >> 1; mask > 0; mask >>= 1) 
        val = max(val, __shfl_xor_sync(ALLWARP, val, mask));
    return val;
}

/**
 * Only thread of warp id = 0 receive a warp-level or.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to or.
 * Returns:
 *   or value across all warps (only for thread of warp id = 0).
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int,
             long, unsigned long, long long, unsigned long long, float,
             double>::value>>
__device__ __inline__
T warp_or(T val) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) 
        val |= __shfl_down_sync(ALLWARP, val, offset);
    return val;
}

/**
 * Each thread in a warp will receive a warp-level or.
 * __shfl_xor_sync creates a butterfly pattern.
 * Supported val types: 
 *   int, uint, long, ulong, long long, ulong long, float, double.
 * Parameters:
 *   - val <- value to or.
 * Returns:
 *   or value across all warps.
 */
template <typename T,
          typename = typename std::enable_if<is_contained<T, int, unsigned int, 
             long, unsigned long, long long, unsigned long long, float, 
             double>::value>>
__device__ __inline__
T warp_all_or(T val) {
    for (int mask = warpSize >> 1; mask > 0; mask >>= 1) 
        val |= __shfl_xor_sync(ALLWARP, val, mask);
    return val;
}

#endif // SRC__CUDA_CUH
