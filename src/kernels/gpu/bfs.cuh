/**
 * GPU implementations of BFS kernels.
 */

#ifndef SRC_KERNELS_GPU__BFS_CUH
#define SRC_KERNELS_GPU__BFS_CUH

#include <algorithm>

#include "../kernel_types.cuh"
#include "../../bitmap.cuh"
#include "../../cuda.cuh"
#include "../../graph.cuh"
#include "../../util.h"

/*****************************************************************************
 ***** BFS Epoch Kernels *****************************************************
 *****************************************************************************/

double bfs_gpu(
        const CSRUWGraph &g, const nid_t source_id, 
        nid_t ** const ret_parents, bfs_gpu_epoch_func epoch_kernel,
        int block_count = 64, int thread_count = 1024
) {
    // Copy graph.
    offset_t *cu_index      = nullptr;
    nid_t    *cu_neighbors  = nullptr;
    size_t   index_size     = (g.num_nodes + 1) * sizeof(offset_t);
    size_t   neighbors_size = g.num_edges * sizeof(nid_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, g.index, index_size,
                cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g.neighbors, neighbors_size,
                cudaMemcpyHostToDevice));

    // Allocate parents array.
    nid_t *parents = new nid_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        parents[i] = INVALID_NODE;
    parents[source_id] = source_id;

    nid_t *cu_parents = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_parents, g.num_nodes * sizeof(nid_t)));
    CUDA_ERRCHK(cudaMemcpy(cu_parents, parents, g.num_nodes * sizeof(nid_t),
                cudaMemcpyHostToDevice));

    // Allocate frontiers' bitmaps.
    Bitmap::Bitmap *frontier         = Bitmap::cu_cpu_constructor(g.num_nodes);
    Bitmap::Bitmap *next_frontier    = Bitmap::cu_cpu_constructor(g.num_nodes);
    Bitmap::Bitmap *cu_frontier      = Bitmap::cu_constructor(frontier);
    Bitmap::Bitmap *cu_next_frontier = Bitmap::cu_constructor(next_frontier);

    Bitmap::cu_cpu_set_bit(frontier, source_id);

    // Allocate num nodes counter.
    nid_t num_nodes, *cu_num_nodes = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_num_nodes, sizeof(nid_t)));

    // Run kernel.
    Timer t; t.Start();
    do {
        CUDA_ERRCHK(cudaMemset(cu_num_nodes, 0, sizeof(nid_t)));

        (*epoch_kernel)<<<block_count, thread_count>>>(
                cu_index, cu_neighbors, cu_parents, 0, g.num_nodes,
                cu_frontier, cu_next_frontier, cu_num_nodes);

        CUDA_ERRCHK(cudaMemcpy(&num_nodes, cu_num_nodes, sizeof(nid_t),
                    cudaMemcpyDeviceToHost));
        std::swap(frontier, next_frontier);
        std::swap(cu_frontier, cu_next_frontier);
        Bitmap::cu_cpu_reset(next_frontier);
    } while (num_nodes != 0);
    t.Stop();

    // Copy results.
    CUDA_ERRCHK(cudaMemcpy(parents, cu_parents, g.num_nodes * sizeof(nid_t),
                cudaMemcpyDeviceToHost));
    *ret_parents = parents;

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_parents));
    CUDA_ERRCHK(cudaFree(cu_num_nodes));
    Bitmap::cu_cpu_destructor(&frontier);
    Bitmap::cu_cpu_destructor(&next_frontier);
    Bitmap::cu_destructor(&cu_frontier);
    Bitmap::cu_destructor(&cu_next_frontier);

    return t.Millisecs();
}

/*****************************************************************************
 ***** BFS Epoch Kernels *****************************************************
 *****************************************************************************/

/**
 * Runs BFS pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each thread is assigned to a single node.
 *
 * Parameters:
 *   - index         <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                      ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors     <- graph neighbors corresponding to the indexed nodes.
 *   - parents       <- node parent list.
 *   - start_id      <- starting node id.
 *   - end_id        <- ending node id (exclusive).
 *   - frontier      <- current frontier.
 *   - next_frontier <- nodes in the next frontier.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
__global__ 
void epoch_bfs_pull_gpu_one_to_one(
        const offset_t * const index, const nid_t * const neighbors, 
        nid_t * const parents,
        const nid_t start_id, const nid_t end_id, 
        const Bitmap::Bitmap * const frontier,
        Bitmap::Bitmap * const next_frontier, 
        nid_t * const num_nodes
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_num_nodes = 0;

    for (nid_t u = start_id + tid; u < end_id; u += num_threads) {
        // If current node hasn't been explored yet.
        if (parents[u] == INVALID_NODE) {
            nid_t index_id = u - start_id;
            for (offset_t i = index[index_id]; i < index[index_id + 1]; i++) {
                nid_t nei = neighbors[i];
                // If parent has been explored.
                if (Bitmap::get_bit(frontier, nei)) {
                    parents[u] = nei;
                    Bitmap::cu_set_bit_atomic(next_frontier, u);
                    local_num_nodes++;
                    break; // Early exit.
                }
            }
        }
    }

    // Push update count.
    atomicAdd(num_nodes, local_num_nodes);
}

/**
 * Runs BFS pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each thread is assigned to a warp.
 *
 * Parameters:
 *   - index         <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                      ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors     <- graph neighbors corresponding to the indexed nodes.
 *   - parents       <- node parent list.
 *   - start_id      <- starting node id.
 *   - end_id        <- ending node id (exclusive).
 *   - frontier      <- current frontier.
 *   - next_frontier <- nodes in the next frontier.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
__global__ 
void epoch_bfs_pull_gpu_warp(
        const offset_t * const index, const nid_t * const neighbors, 
        nid_t * const parents,
        const nid_t start_id, const nid_t end_id, 
        const Bitmap::Bitmap * const frontier,
        Bitmap::Bitmap * const next_frontier, 
        nid_t * const num_nodes
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid & (warpSize - 1); // ID within a warp.
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_num_nodes = 0;

    for (nid_t u = start_id + tid / warpSize; u < end_id; 
            u += (num_threads / warpSize)
    ) {
        // If current node hasn't been explored yet.
        if (parents[u] == INVALID_NODE) {
            nid_t index_id = u - start_id;
            bool in_next_frontier = false;

            offset_t iters = (index[index_id + 1] - index[index_id] + warpSize - 1)
                                / warpSize; // Number of warpSize iters.
            
            for (offset_t iter_i = 0; iter_i < iters; iter_i++) {
                offset_t i = index[index_id] + iter_i * warpSize + warpid;
                if (i < index[index_id + 1] and 
                        Bitmap::get_bit(frontier, neighbors[i])
                ) {
                    parents[u] = neighbors[i];
                    Bitmap::cu_set_bit_atomic(next_frontier, u);
                    in_next_frontier = true;
                }

                in_next_frontier = warp_all_or(in_next_frontier);

                if (in_next_frontier) {
                    local_num_nodes++;
                    break;
                }
            }
        }
    }

    // Push update count.
    if (warpid == 0)
        atomicAdd(num_nodes, local_num_nodes);
}

/**
 * Runs BFS pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Each thread is assigned to a warp.
 *
 * Parameters:
 *   = sync_iters    <- number of iters through neighbors list until warp
 *                      synchronization.
 *   - index         <- graph index where index 0 = start_id, 1 = start_id + 1,
 *                      ... , (end_id - start_id - 1) = end_id - 1.
 *   - neighbors     <- graph neighbors corresponding to the indexed nodes.
 *   - parents       <- node parent list.
 *   - start_id      <- starting node id.
 *   - end_id        <- ending node id (exclusive).
 *   - frontier      <- current frontier.
 *   - next_frontier <- nodes in the next frontier.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
template <offset_t sync_iters = 1>
__global__
void epoch_bfs_sync_pull_gpu_warp(
        const offset_t * const index, const nid_t * const neighbors, 
        nid_t * const parents,
        const nid_t start_id, const nid_t end_id, 
        const Bitmap::Bitmap * const frontier,
        Bitmap::Bitmap * const next_frontier, 
        nid_t * const num_nodes
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid & (warpSize - 1); // ID within a warp.
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_num_nodes = 0;

    for (nid_t u = start_id + tid / warpSize; u < end_id; 
            u += (num_threads / warpSize)
    ) {
        // If current node hasn't been explored yet.
        if (parents[u] == INVALID_NODE) {
            nid_t index_id = u - start_id;
            bool in_next_frontier = false;

            offset_t iters = (index[index_id + 1] - index[index_id] + warpSize - 1)
                                / warpSize; // Number of warpSize iters.
            
            for (offset_t gen_i = 0; gen_i < iters; gen_i += sync_iters) {
                for (offset_t sync_i = 0; sync_i < sync_iters; sync_i++) {
                    offset_t i = index[index_id] + (gen_i + sync_i) * warpSize 
                            + warpid;

                    if (i < index[index_id + 1]
                            and Bitmap::get_bit(frontier, neighbors[i])
                    ) {
                        parents[u] = neighbors[i];
                        Bitmap::cu_set_bit_atomic(next_frontier, u);
                        in_next_frontier = true;                        
                    }
                }

                in_next_frontier = warp_all_or(in_next_frontier);

                if (in_next_frontier) {
                    local_num_nodes++;
                    break;
                }
            }
        }
    }

    // Push update count.
    if (warpid == 0)
        atomicAdd(num_nodes, local_num_nodes);
}

#endif // SRC_KERNELS_GPU__BFS_CUH
