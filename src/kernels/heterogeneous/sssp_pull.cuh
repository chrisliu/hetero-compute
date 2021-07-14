/**
 * Heterogeneous implementations of SSSP pull kernel.
 * This is generated by util/scheduler/scheduler/kernelgen/sssp_hetero.py.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH
#define SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH

#include <omp.h>
#include <vector>

#include "../kernel_types.h"
#include "../cpu/sssp_pull.h"
#include "../gpu/sssp_pull.cuh"
#include "../../cuda.cuh"
#include "../../graph.h"
#include "../../util.h"

// Toggle timing on/off.
#define TIMING_ON

/**
 * Runs SSSP kernel heterogeneously across the CPU and GPU. Synchronization 
 * occurs in serial. 
 * Configuration:
 *   - 1x Intel i7-9700K
 *   - 1x NVIDIA Quadro RTX 4000
 *
 * Parameters:
 *   - g         <- graph.
 *   - init_dist <- initial distance array.
 *   - ret_dist  <- pointer to the address of the return distance array.
 * Returns:
 *   Execution time in milliseconds.
 */
double sssp_pull_heterogeneous(const CSRWGraph &g, 
        const weight_t *init_dist, weight_t ** const ret_dist
) {
    // Configuration.
    constexpr int num_gpus     = 1;
    constexpr int num_blocks   = 5;
    constexpr int num_segments = 16;
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    constexpr int gpu_blocks[] = {0, 5};
    nid_t block_ranges[num_blocks * 2];

    block_ranges[0] = seg_ranges[0]; // Block 0 Start 0
    block_ranges[1] = seg_ranges[1]; // Block 0 End 1 (excl.)
    block_ranges[2] = seg_ranges[1]; // Block 1 Start 1
    block_ranges[3] = seg_ranges[2]; // Block 1 End 2 (excl.)
    block_ranges[4] = seg_ranges[2]; // Block 2 Start 2
    block_ranges[5] = seg_ranges[11]; // Block 2 End 11 (excl.)
    block_ranges[6] = seg_ranges[13]; // Block 3 Start 13
    block_ranges[7] = seg_ranges[15]; // Block 3 End 15 (excl.)
    block_ranges[8] = seg_ranges[15]; // Block 4 Start 15
    block_ranges[9] = seg_ranges[16]; // Block 4 End 16 (excl.)

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        for (int block = gpu_blocks[cur_gpu]; block < gpu_blocks[cur_gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }

    // Distance.
    size_t   dist_size = g.num_nodes * sizeof(weight_t);
    weight_t *dist     = nullptr; 

    /// CPU Distance.
    CUDA_ERRCHK(cudaMallocHost((void **) &dist, dist_size));
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    /// GPU Distances.
    weight_t *cu_dists[num_gpus];
    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_dists[cur_gpu], dist_size));
        CUDA_ERRCHK(cudaMemcpy(cu_dists[cur_gpu], init_dist, dist_size,
            cudaMemcpyHostToDevice));
    }

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus];
    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_updateds[cur_gpu], 
                sizeof(nid_t)));
    }
        
    // Intitialize timing related variables.
#ifdef TIMING_ON
    Timer cpu_timer;
    cudaEvent_t gpu_start[num_gpus];
    cudaEvent_t gpu_stop[num_gpus];
    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        CUDA_ERRCHK(cudaEventCreate(&gpu_start[cur_gpu]));
        CUDA_ERRCHK(cudaEventCreate(&gpu_stop[cur_gpu]));
    }
    cudaEvent_t mem_start, mem_stop;
    CUDA_ERRCHK(cudaSetDevice(0));
    CUDA_ERRCHK(cudaEventCreate(&mem_start));
    CUDA_ERRCHK(cudaEventCreate(&mem_stop));
    
    int epoch = 0;
#endif // TIMING_ON
    
    // Start kernel!
    Timer timer; timer.Start();
    while (updated != 0) {
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
            CUDA_ERRCHK(cudaSetDevice(cur_gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[cur_gpu], 0, 
                    sizeof(nid_t)));
        }

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        CUDA_ERRCHK(cudaSetDevice(0));
#ifdef TIMING_ON
        CUDA_ERRCHK(cudaEventRecord(gpu_start[0]));
#endif // TIMING_ON
        epoch_sssp_pull_gpu_block_min<<<64, 1024>>>(
                cu_indices[0], cu_neighbors[0],
                block_ranges[0], block_ranges[1],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_block_min<<<512, 128>>>(
                cu_indices[1], cu_neighbors[1],
                block_ranges[2], block_ranges[3],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_warp_min<<<64, 1024>>>(
                cu_indices[2], cu_neighbors[2],
                block_ranges[4], block_ranges[5],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_warp_min<<<64, 1024>>>(
                cu_indices[3], cu_neighbors[3],
                block_ranges[6], block_ranges[7],
                cu_dists[0], cu_updateds[0]);
        epoch_sssp_pull_gpu_one_to_one<<<64, 1024>>>(
                cu_indices[4], cu_neighbors[4],
                block_ranges[8], block_ranges[9],
                cu_dists[0], cu_updateds[0]);
#ifdef TIMING_ON
        CUDA_ERRCHK(cudaEventRecord(gpu_stop[0]));
#endif // TIMING_ON

        // Launch CPU epoch kernels.
#ifdef TIMING_ON
        cpu_timer.Start();
#endif // TIMING_ON
        #pragma omp parallel
        {
            epoch_sssp_pull_cpu_one_to_one(g, dist, 
                    seg_ranges[11], seg_ranges[13],
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }
#ifdef TIMING_ON
        cpu_timer.Stop();
#endif // TIMING_ON

        // Copy GPU distances back to main memory.
        for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
            CUDA_ERRCHK(cudaSetDevice(cur_gpu));
            for (int block = gpu_blocks[cur_gpu];
                    block < gpu_blocks[cur_gpu + 1]; block++
            ) {
                int start = block_ranges[2 * block];
                int end   = block_ranges[2 * block + 1];
                CUDA_ERRCHK(cudaMemcpyAsync(
                            dist + start, cu_dists[cur_gpu] + start,
                            (end - start) * sizeof(weight_t),
                            cudaMemcpyDeviceToHost));
            }
        }
                
        // Start memory transfer timing.
#ifdef TIMING_ON
        CUDA_ERRCHK(cudaSetDevice(0));
        CUDA_ERRCHK(cudaEventRecord(mem_start));
#endif // TIMING_ON
        
        // Synchronize updates.
        nid_t tmp_updated;
        for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
            CUDA_ERRCHK(cudaSetDevice(cur_gpu));
            CUDA_ERRCHK(cudaMemcpy(&tmp_updated, cu_updateds[cur_gpu], 
                    sizeof(nid_t), cudaMemcpyDeviceToHost));
            updated += tmp_updated;
        }
        updated += cpu_updated;

        // Only update GPU distances if another epoch will be run.
        if (updated != 0) {
            // Copy CPU distances to all GPUs.
            for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
                CUDA_ERRCHK(cudaMemcpyAsync(
                    cu_dists[cur_gpu] + seg_ranges[11],
                    dist + seg_ranges[11],
                    (seg_ranges[13] - seg_ranges[11]) * sizeof(weight_t),
                    cudaMemcpyHostToDevice));
            }

            // Copy GPU distances peer-to-peer.
            for (int src_gpu = 0; src_gpu < num_gpus; src_gpu++) {
                CUDA_ERRCHK(cudaSetDevice(src_gpu));
                for (int dst_gpu = 0; dst_gpu < num_gpus; dst_gpu++) {
                    if (src_gpu == dst_gpu) continue;
                    
                    for (int block = gpu_blocks[src_gpu];
                            block < gpu_blocks[src_gpu + 1]; block++
                    ) {
                        int start = block_ranges[2 * block];
                        int end   = block_ranges[2 * block + 1];
                        CUDA_ERRCHK(cudaMemcpyAsync(
                                    cu_dists[dst_gpu] + start,
                                    cu_dists[src_gpu] + start,
                                    (end - start) * sizeof(weight_t),
                                    cudaMemcpyDeviceToDevice));
                    }
                }
            }
        }
        
        // Stop memory transfer timing.
        CUDA_ERRCHK(cudaSetDevice(0));
        CUDA_ERRCHK(cudaEventRecord(mem_stop));
        
        // Display timing results.
#ifdef TIMING_ON
        // Get times.
        float gpu_times[num_gpus];
        for (int i = 0; i < num_gpus; i++) {
            CUDA_ERRCHK(cudaSetDevice(i));
            CUDA_ERRCHK(cudaEventSynchronize(gpu_stop[i]));
            CUDA_ERRCHK(cudaEventElapsedTime(&gpu_times[i], gpu_start[i], 
                    gpu_stop[i]))
        }
        CUDA_ERRCHK(cudaSetDevice(0));
        CUDA_ERRCHK(cudaEventSynchronize(mem_stop));
        float transfer_time;
        CUDA_ERRCHK(cudaEventElapsedTime(&transfer_time, mem_start, mem_stop));
        
        // Print results.
        std::cout << "Epoch " << epoch << " results" << std::endl
                  << " > CPU time: " << cpu_timer.Millisecs() << " ms" 
                      << std::endl;
        for (int i = 0; i < num_gpus; i++) {
            std::cout
                  << " > GPU " << i << " time: " << gpu_times[i] << " ms" 
                  << std::endl;
        }
        std::cout << " > Memory transfer time: " << transfer_time << " ms" 
            << std::endl;
        epoch++;
#endif // TIMING_ON
        
    }
    timer.Stop();

    // Copy output.
    *ret_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_dist)[i] = dist[i];

    // Free memory.
    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[cur_gpu]));
        CUDA_ERRCHK(cudaFree(cu_dists[cur_gpu]));
        
        for (int block = gpu_blocks[cur_gpu]; block < gpu_blocks[cur_gpu + 1];
                block++
        ) {
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }
    }
    CUDA_ERRCHK(cudaFreeHost(dist));
    delete[] seg_ranges;
        
#ifdef TIMING_ON
    CUDA_ERRCHK(cudaSetDevice(0));
    CUDA_ERRCHK(cudaEventDestroy(mem_start));
    CUDA_ERRCHK(cudaEventDestroy(mem_stop));
    for (int cur_gpu = 0; cur_gpu < num_gpus; cur_gpu++) {
        CUDA_ERRCHK(cudaSetDevice(cur_gpu));
        CUDA_ERRCHK(cudaEventDestroy(gpu_start[cur_gpu]));
        CUDA_ERRCHK(cudaEventDestroy(gpu_stop[cur_gpu]));
    }
#endif // TIMING_ON
    
    return timer.Millisecs();
}

#endif // SRC_KERNELS_HETEROGENEOUS__SSSP_PULL_CUH