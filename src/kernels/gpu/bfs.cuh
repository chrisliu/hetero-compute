/**
 * GPU implementations of BFS kernels.
 */

#ifndef SRC_KERNELS_GPU__BFS_CUH
#define SRC_KERNELS_GPU__BFS_CUH

#include "../../bitmap.cuh"
#include "../../graph.h"
#include "../../util.h"

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
 *   - next_froniter <- nodes in the next frontier.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
__global__ 
void epoch_bfs_pull_gpu_one_to_one(
        const offset_t * const index, const wnode_t * const neighbors, 
        nid_t * const parents,
        const nid_t start_id, const nid_t end_id, 
        Bitmap::Bitmap * const next_frontier, nid_t * const num_nodes
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_num_nodes = 0;

    for (nid_t nid = start_id + tid; nid < end_id; nid += num_threads) {
        // If current node hasn't been explored yet.
        if (parents[nid] == INVALID_NODE) {
            for (offset_t i = index[nid]; i < index[nid + 1]; i++) {
                nid_t nei = neighbors[i];
                // If parent has been explored.
                if (parents[nei] != INVALID_NODE) {
                    parents[nid] = nei;
                    Bitmap::set_bit(next_frontier, nid);
                    local_num_nodes++;
                    break; // Early exit.
                }
            }
        }
    }

    // Push update count.
    atomicAdd(num_nodes, local_num_nodes);
}


#endif // SRC_KERNELS_GPU__BFS_CUH
