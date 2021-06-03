/**
 * GPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH
#define SRC_KERNELS_GPU__KERNEL_SSSP_PULL_GPU_CUH

#include <iostream>
#include <omp.h> 

#include "../kernel_types.h"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"
#include "../../benchmarks/benchmark.h"

/** Forward decl. */
__global__ 
void epoch_sssp_pull_gpu_naive(const offset_t *index, const wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
);

/******************************************************************************
 ***** SSSP Kernel ************************************************************
 ******************************************************************************/

/**
 * Runs SSSP kernel on GPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 * Returns:
 *   Execution results for the entire graph.
 */
segment_res_t sssp_pull_gpu(const CSRWGraph &g, sssp_gpu_epoch_func epoch_kernel, 
        weight_t **ret_dist, int block_count = 64, int thread_count = 1024
) {
    CONDCHK(epoch_kernel != epoch_sssp_pull_gpu_naive and thread_count < 32, 
            "thread count must be greater or equal warp size")
    /// Setup.
    // Copy graph.
    offset_t *cu_index      = nullptr;
    wnode_t  *cu_neighbors  = nullptr;
    size_t   index_size     = g.num_nodes * sizeof(offset_t);
    size_t   neighbors_size = g.num_edges * sizeof(wnode_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, g.index, index_size, 
            cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g.neighbors, neighbors_size, 
            cudaMemcpyHostToDevice));

    // Update counter.
    nid_t *cu_updated = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));
    
    // Distance.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = MAX_WEIGHT;
    dist[0] = 0; // Arbitrarily set start.

    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes * sizeof(weight_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, dist_size));

    // Return data structure.
    segment_res_t res;
    res.start_id   = 0;
    res.end_id     = g.num_nodes;
    res.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;

    // Actual kernel run.
    std::cout << "Starting kernel ..." << std::endl;
    
    double total_epochs = 0.0;
    double total_time   = 0.0;

    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        // Reset updated counter and distances.
        nid_t updated = 1;
        CUDA_ERRCHK(cudaMemcpy(cu_dist, dist, dist_size, 
                cudaMemcpyHostToDevice));

        Timer timer; timer.Start();
        while (updated != 0) {
            CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

            // Note: Must run with thread count >= 32 since warp level
            //       synchronization is performed.
            (*epoch_kernel)<<<64, 1024>>>(cu_index, cu_neighbors, 0,
                    g.num_nodes, cu_dist, cu_updated);

            CUDA_ERRCHK(cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                    cudaMemcpyDeviceToHost));

            total_epochs++;
        }
        timer.Stop();
        
        total_time += timer.Millisecs();
    }

    // Save results.
    std::cout << total_epochs << std::endl;
    res.num_edges = (total_epochs / BENCHMARK_TIME_ITERS) * g.num_edges;
    res.millisecs = total_time / BENCHMARK_TIME_ITERS;
    res.gteps     = res.num_edges / (res.millisecs / 1000) / 1e9;

    std::cout << "Kernel completed in (avg): " << res.millisecs << " ms." 
        << std::endl;

    // Copy distances.
    CUDA_ERRCHK(cudaMemcpy(dist, cu_dist, dist_size, cudaMemcpyDeviceToHost));
    *ret_dist = dist;

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_updated));
    CUDA_ERRCHK(cudaFree(cu_dist));

    return res;
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

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
void epoch_sssp_pull_gpu_warp_min(const offset_t *index, const wnode_t *neighbors, 
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
void epoch_sssp_pull_gpu_naive(const offset_t *index, const wnode_t *neighbors, 
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
