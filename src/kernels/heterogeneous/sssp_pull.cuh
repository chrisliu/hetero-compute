/**
 * Heterogeneous implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
#define SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH

#include <omp.h>
#include <vector>

#include "../kernel_types.h"
#include "../gpu/sssp_pull.cuh"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../schedule.h"
#include "../../util.h"

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. The CPU kernel will tackle nodes [cpu_start_id, cpu_end_id)
 * and the GPU kernel will tackle nodes [gpu_start_id, gpu_end_id).
 * TODO: currently only supports single CPU and single GPU.
 * TODO: only supports one contiguous block for each the CPU and GPU kernels.
 *
 * Parameters:
 *   - g            <- graph.
 *   - schedule     <- heterogeneous SSSP schedule.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 *   - block_count  <- (optional) number of blocks.
 *   - thread_count <- (optional) number of threads.
 * Returns:
 *   Execution time in milliseconds.
 */
double sssp_pull_heterogeneous(
        const CSRWGraph &g, SSSPHeteroSchedule &schedule,
        const weight_t *init_dist, weight_t ** const ret_dist,
) {
    CONDCHK(gpu_epoch_kernel != epoch_sssp_pull_gpu_one_to_one
                and thread_count % 32 != 0, 
            "thread count must be divisible by 32");

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
    
    // Distance.
    size_t dist_size = g.num_nodes * sizeof(weight_t);
    weight_t *dist = nullptr; 
    CUDA_ERRCHK(cudaMallocHost((void **) &dist, dist_size));
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    weight_t *cu_dist = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, dist_size));
    CUDA_ERRCHK(cudaMemcpy(cu_dist, init_dist, dist_size, 
            cudaMemcpyHostToDevice));

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updated = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));

    // Get CPU and GPU block sizes.
    size_t cpu_dist_size = (cpu_range.end_id - cpu_range.start_id) 
        * sizeof(weight_t);
    std::vector<size_t> gpu_dist_sizes(gpu_ranges.size());
    for (int i = 0; i < gpu_ranges.size(); i++)
        gpu_dist_sizes[i] = (gpu_ranges[i].end_id - gpu_ranges[i].start_id)
            * sizeof(weight_t);

    // Start kernel!
    Timer timer; timer.Start();
    while (updated != 0) {
        updated = cpu_updated = 0;
        CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

        // Launch GPU epoch kernel.
        for (int i = 0; i < gpu_ranges.size(); i++)
            (*gpu_epoch_kernel)<<<block_count, thread_count>>>(
                    cu_index, cu_neighbors, 
                    gpu_ranges[i].start_id, gpu_ranges[i].end_id,
                    cu_dist, cu_updated);

        // Launch CPU epoch kernel.
        #pragma omp parallel
        {
            (*cpu_epoch_kernel)(g, dist, cpu_range.start_id, cpu_range.end_id,
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }

        // Get updated (implicit cudaDeviceSynchronize).
        CUDA_ERRCHK(cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                cudaMemcpyDeviceToHost));
        updated += cpu_updated;
        
        // Synchronize distances.
        for (int i = 0; i < gpu_ranges.size(); i++)
            CUDA_ERRCHK(cudaMemcpyAsync(
                        dist + gpu_ranges[i].start_id, 
                        cu_dist + gpu_ranges[i].start_id,
                        gpu_dist_sizes[i], cudaMemcpyDeviceToHost));
        // Only perform HtoD memcpy if next epoch is to run (very expensive).
        if (updated != 0)
            CUDA_ERRCHK(cudaMemcpyAsync(
                        cu_dist + cpu_range.start_id, dist + cpu_range.start_id,
                        cpu_dist_size, cudaMemcpyHostToDevice));
    }
    timer.Stop();

    // Copy output.
    *ret_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_dist)[i] = dist[i];

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_updated));
    CUDA_ERRCHK(cudaFree(cu_dist));
    CUDA_ERRCHK(cudaFreeHost(dist));

    return timer.Millisecs();
}

/******************************************************************************
 ***** Helper Functions *******************************************************
 ******************************************************************************/


#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
