/**
 * Heterogeneous implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
#define SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH

#include <omp.h>

#include "../kernel_types.h"
#include "../gpu/sssp_pull.cuh"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. The CPU kernel will tackle nodes [cpu_start_id, cpu_end_id)
 * and the GPU kernel will tackle nodes [gpu_start_id, gpu_end_id).
 * TODO: currently only supports single CPU and single GPU.
 * TODO: only supports one contiguous block for each the CPU and GPU kernels.
 *
 * Parameters:
 *   - g                <- graph.
 *   - cpu_epoch_kernel <- cpu epoch kernel.
 *   - gpu_epoch_kernel <- gpu epoch kernel.
 *   - cpu_start_id     <- cpu kernel starting node id.
 *   - cpu_end_id       <- cpu kernel ending node id (exclusive).
 *   - gpu_start_id     <- gpu kernel starting node id.
 *   - gpu_end_id       <- gpu kernel ending node id (exclusive).
 *   - init_dist        <- initial distance array.
 *   - ret_dist         <- pointer to the address of the return distance array.
 *   - block_count      <- (optional) number of blocks.
 *   - thread_count     <- (optional) number of threads.
 * Returns:
 *   Execution time in milliseconds.
 */
double sssp_pull_heterogeneous(
        const CSRWGraph &g, 
        sssp_cpu_epoch_func cpu_epoch_kernel, 
        sssp_gpu_epoch_func gpu_epoch_kernel,
        const nid_t cpu_start_id, const nid_t cpu_end_id,
        const nid_t gpu_start_id, const nid_t gpu_end_id,
        const weight_t *init_dist, weight_t ** const ret_dist,
        int block_count = 64, int thread_count = 1024
) {
    CONDCHK(gpu_epoch_kernel != epoch_sssp_pull_gpu_naive 
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
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes * sizeof(weight_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, dist_size));
    CUDA_ERRCHK(cudaMemcpy(cu_dist, dist, dist_size, 
            cudaMemcpyHostToDevice));

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updated = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));

    // Get CPU and GPU block sizes.
    size_t cpu_dist_size = (cpu_end_id - cpu_start_id) * sizeof(weight_t);
    size_t gpu_dist_size = (gpu_end_id - gpu_start_id) * sizeof(weight_t);

    // Start kernel!
    Timer timer; timer.Start();
    while (updated != 0) {
        updated = cpu_updated = 0;
        CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

        // Launch GPU epoch kernel.
        (*gpu_epoch_kernel)<<<block_count, thread_count>>>(cu_index, 
                cu_neighbors, gpu_start_id, gpu_end_id, cu_dist, cu_updated);

        // Launch CPU epoch kernel.
        #pragma omp parallel
        {
            (*cpu_epoch_kernel)(g, dist, cpu_start_id, cpu_end_id, 
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }

        // Get updated (implicit cudaDeviceSynchronize).
        CUDA_ERRCHK(cudaMemcpy(&updated, cu_updated, sizeof(nid_t),
                cudaMemcpyDeviceToHost));
        updated += cpu_updated;
        
        // Synchronize distances.
        CUDA_ERRCHK(cudaMemcpy(cu_dist + cpu_start_id, dist + cpu_start_id,
                    cpu_dist_size, cudaMemcpyHostToDevice));
        // Only updated GPU distances if another epoch will be performed.
        /*if (updated != 0) */
        CUDA_ERRCHK(cudaMemcpy(dist + gpu_start_id, cu_dist + gpu_start_id,
                    gpu_dist_size, cudaMemcpyDeviceToHost));
    }
    timer.Stop();

    // Assign output.
    *ret_dist = dist;

    // Free memory.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));
    CUDA_ERRCHK(cudaFree(cu_updated));
    CUDA_ERRCHK(cudaFree(cu_dist));

    return timer.Millisecs();
}

#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
