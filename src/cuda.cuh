/**
 * NVIDIA CUDA helper functions.
 */

#ifndef SRC__CUDA_CUH
#define SRC__CUDA_CUH

#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "graph.h"

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

// https://stackoverflow.com/questions/16252902/sfinae-set-of-types-contains-the-type
template <typename T, typename ...> struct is_contained : std::false_type {};
template <typename T, typename Head, typename ...Tail>
struct is_contained<T, Head, Tail...> : std::integral_constant<bool,
    std::is_same<T, Head>::value || is_contained<T, Tail...>::value> {};

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

/*****************************************************************************
 ***** Host Helper Functions *************************************************
 *****************************************************************************/

/**
 * Copy specified subgraph to cu_index and cu_neighbors.
 * Parameters:
 *   - g               <- graph.
 *   - cu_index_to     <- pointer to target cuda index array.
 *   - cu_neighbors_to <- pointer to traget cuda neighbors array.
 *   - start_id        <- starting node id.
 *   - end_id          <- ending node id (exclusive).
 */
__inline__
void copy_subgraph_to_device(const CSRWGraph &g, 
        offset_t **cu_index_to, wnode_t **cu_neighbors_to, 
        const nid_t start_id, const nid_t end_id
) {
    // Local copies of cu_index and cu_neighbors.
    offset_t *cu_index     = nullptr;
    wnode_t  *cu_neighbors = nullptr;

    // Get number of nodes and edges.
    size_t index_size     = (end_id - start_id + 1) * sizeof(offset_t);
    size_t neighbors_size = (g.index[end_id] - g.index[start_id])
                            * sizeof(wnode_t);

    // Make a down-shifted copy of the index. Index values in the graph will no 
    // longer point to the same location in memory.
    offset_t *subgraph_index = new offset_t[end_id - start_id + 1];
    #pragma omp parallel for
    for (int i = start_id; i <= end_id; i++)
        subgraph_index[i - start_id] = g.index[i] - g.index[start_id];

    // Allocate memory and copy subgraph over.
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, subgraph_index, 
                index_size, cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, &g.neighbors[g.index[start_id]],
                neighbors_size, cudaMemcpyHostToDevice));

    // Delete main memory copy of subgraph index.
    delete[] subgraph_index;

    // Assign output.
    *cu_index_to     = cu_index;
    *cu_neighbors_to = cu_neighbors;
}

#endif // SRC__CUDA_CUH
