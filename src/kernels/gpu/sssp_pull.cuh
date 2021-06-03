/**
 * GPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH
#define SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH

#include <omp.h> 

#include "../kernel_types.h"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"

/**
 * Runs SSSP kernel on GPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 */
void sssp_pull_gpu(const CSRWGraph &g, sssp_gpu_epoch_func epoch_kernel, 
        weight_t **ret_dist
) {
    /// Setup.
    // Copy graph.
    offset_t *cu_index      = nullptr;
    wnode_t  *cu_neighbors  = nullptr;
    size_t   index_size     = g.num_nodes * sizeof(offset_t);
    size_t   neighbors_size = g.num_edges * sizeof(wnode_t);
    cudaMalloc((void **) &cu_index, index_size);
    cudaMalloc((void **) &cu_neighbors, neighbors_size);
    cudaMemcpy(cu_index, g.index, index_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_neighbors, g.neighbors, neighbors_size, 
            cudaMemcpyHostToDevice);

    // Update counter.
    nid_t *cu_updated = nullptr;
    cudaMalloc((void **) &cu_updated, sizeof(nid_t));
    
    // Distance.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = MAX_WEIGHT;
    dist[0] = 0; // Arbitrarily set start.

    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes * sizeof(weight_t);
    cudaMalloc((void **) &cu_dist, dist_size);

    // Actual kernel run.
    std::cout << "Starting kernel ..." << std::endl;

    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        // Reset updated counter and distances.
        nid_t updated = 1;
        cudaMemcpy(cu_dist, dist, dist_size, cudaMemcpyHostToDevice);

        Timer timer; timer.Start();
        while (updated != 0) {
            cudaMemset(cu_updated, 0, sizeof(nid_t));

            // Note: Must run with thread count >= 32 since warp level
            //       synchronization is performed.
            (*epoch_kernel)<<<64, 1024>>>(cu_index, cu_neighbors, 0,
                    g.num_nodes, cu_dist, cu_updated);

            cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                    cudaMemcpyDeviceToHost);
        }
        timer.Stop();
        
        total_time += timer.Millisecs();
    }

    std::cout << "Kernel completed in: " << (total_time / BENCHMARK_TIME_ITERS)
        << " ms." << std::endl;

    // Copy distances.
    cudaMemcpy(dist, cu_dist, dist_size, cudaMemcpyDeviceToHost);
    *ret_dist = dist;

    // Free memory.
    cudaFree(cu_index);
    cudaFree(cu_neighbors);
    cudaFree(cu_updated);
    cudaFree(cu_dist);
}

/**
 * Runs SSSP pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__ 
void sssp_pull_gpu_warp_min(const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid % warpSize; // ID within a warp.
    int num_threads = gridDim.x * blockDim.x;

    nid_t local_updated = 0;

    for (int nid = start_id + tid / warpSize; nid < end_id; 
            nid += (num_threads / warpSize)
    ) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (int i = index[nid] + warpid; i < index[nid + 1]; i += warpSize) {
            weight_t prop_dist = dist[neighbors[i].v] + neighbors[i].w;
            new_dist = min(prop_dist, new_dist);
        }

        new_dist = warp_min(new_dist);

        // Update distance if applicable.
        if (warpid == 0 and new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    atomicAdd(updated, local_updated);
}

/**
 * Runs SSSP pull on GPU for one epoch on a range of nodes [start_id, end_id).
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - start_id  <- starting node id.
 *   - end_id    <- ending node id (exclusive).
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__ 
void sssp_pull_gpu_naive(const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    int local_updated = 0;

    for (int nid = start_id + tid; nid < end_id; nid += num_threads) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (int i = index[nid]; i < index[nid + 1]; i++) {
            weight_t prop_dist = dist[neighbors[i].v] + neighbors[i].w;
            new_dist = min(prop_dist, new_dist);
        }

        // Update distance if applicable.
        if (new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    atomicAdd(updated, local_updated);
}

#endif // SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH
