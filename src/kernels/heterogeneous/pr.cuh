/**
 * Heterogeneous implementation of the PR pull kernel.
 * This is generated by util/scheduler/scheduler/kernelgen/pr_hetero.py.
 */

#ifndef SRC_KERNELS_HETEROGENEOUS__PR_CUH
#define SRC_KERNELS_HETEROGENEOUS__PR_CUH

#include <omp.h>
#include <vector>

#include "../kernel_types.cuh"
#include "../cpu/pr.cuh"
#include "../gpu/pr.cuh"
#include "../../cuda.cuh"
#include "../../graph.cuh"
#include "../../util.h"

constexpr int num_gpus_pr = 1;

/** Forward decl. */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_dists, 
        cudaStream_t *memcpy_streams);

/**
 * Runs PR kernel heterogeneously across the CPU and GPU. Synchronization 
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
double pr_pull_heterogeneous(const CSRWGraph &g, 
        const weight_t *init_dist, weight_t ** const ret_dist
) {
    // Configuration.
    constexpr int num_blocks   = 11;
    constexpr int num_segments = 24;
    
    // Copy graph.
    nid_t *seg_ranges = compute_equal_edge_ranges(g, num_segments);
    
    /// Block ranges to reduce irregular memory acceses.
    constexpr int gpu_blocks[] = {0, 11};
    nid_t block_ranges[num_blocks * 2];

    block_ranges[0] = seg_ranges[0]; // Block 0 Start 0
    block_ranges[1] = seg_ranges[1]; // Block 0 End 1 (excl.)
    block_ranges[2] = seg_ranges[1]; // Block 1 Start 1
    block_ranges[3] = seg_ranges[3]; // Block 1 End 3 (excl.)
    block_ranges[4] = seg_ranges[3]; // Block 2 Start 3
    block_ranges[5] = seg_ranges[5]; // Block 2 End 5 (excl.)
    block_ranges[6] = seg_ranges[5]; // Block 3 Start 5
    block_ranges[7] = seg_ranges[6]; // Block 3 End 6 (excl.)
    block_ranges[8] = seg_ranges[6]; // Block 4 Start 6
    block_ranges[9] = seg_ranges[8]; // Block 4 End 8 (excl.)
    block_ranges[10] = seg_ranges[8]; // Block 5 Start 8
    block_ranges[11] = seg_ranges[10]; // Block 5 End 10 (excl.)
    block_ranges[12] = seg_ranges[10]; // Block 6 Start 10
    block_ranges[13] = seg_ranges[15]; // Block 6 End 15 (excl.)
    block_ranges[14] = seg_ranges[15]; // Block 7 Start 15
    block_ranges[15] = seg_ranges[16]; // Block 7 End 16 (excl.)
    block_ranges[16] = seg_ranges[17]; // Block 8 Start 17
    block_ranges[17] = seg_ranges[18]; // Block 8 End 18 (excl.)
    block_ranges[18] = seg_ranges[20]; // Block 9 Start 20
    block_ranges[19] = seg_ranges[23]; // Block 9 End 23 (excl.)
    block_ranges[20] = seg_ranges[23]; // Block 10 Start 23
    block_ranges[21] = seg_ranges[24]; // Block 10 End 24 (excl.)

    //degrees
    offset_t *cu_degrees      = nullptr;
    offset_t *degrees = new offset_t[g.num_nodes];
    for(int i=0; i<g.num_nodes; i++){
        degrees[i]=g.get_degree(i);
    }
    size_t deg_size = g.num_nodes * sizeof(offset_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_degrees, deg_size));
    CUDA_ERRCHK(cudaMemcpy(cu_degrees, degrees, deg_size,
            cudaMemcpyHostToDevice));

    /// Actual graphs on GPU memory.
    offset_t *cu_indices[num_blocks];
    wnode_t  *cu_neighbors[num_blocks];

    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++) 
            copy_subgraph_to_device(g,
                    &cu_indices[block], &cu_neighbors[block],
                    block_ranges[2 * block], block_ranges[2 * block + 1]);
    }

    // Initialize memcopy streams.
    // idx = from_gpu * num_gpus_pr + to_gpu;
    cudaStream_t memcpy_streams[num_gpus_pr * num_gpus_pr];
    for (int from = 0; from < num_gpus_pr; from++) {
        CUDA_ERRCHK(cudaSetDevice(from));
        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamCreate(&memcpy_streams[from * num_gpus_pr + to]));
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
    weight_t *cu_dists[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {        
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_dists[gpu], dist_size));
        CUDA_ERRCHK(cudaMemcpyAsync(cu_dists[gpu], dist, dist_size,
            cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr]));
    }
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr]));
    }

    // Update counter.
    nid_t updated     = 1;
    nid_t cpu_updated = 0;
    nid_t *cu_updateds[num_gpus_pr];
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaMalloc((void **) &cu_updateds[gpu], 
                sizeof(nid_t)));
    }

    // Create compute streams and markers.
    cudaStream_t compute_streams[num_blocks]; // Streams for compute.
    cudaEvent_t  compute_markers[num_blocks]; // Compute complete indicators.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {
            CUDA_ERRCHK(cudaStreamCreate(&compute_streams[b]));
            CUDA_ERRCHK(cudaEventCreate(&compute_markers[b]));
        }
    }

    // Get init vertex.
    // TODO: add this as a parameter.
    nid_t start;
    for (nid_t i = 0; i < g.num_nodes; i++)
        if (init_dist[i] != 1.0f/g.num_nodes) start = i;

    // Start kernel!
    Timer timer; timer.Start();
    int epochs = 0;

    /*
    // Push for the first iteration.
    // TODO: implement push for more than one epoch. Requires parallel queue.
    for (wnode_t nei : g.get_neighbors(start)) {
        if (nei.v == start) continue;

        dist[nei.v] = nei.w;       
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                cu_dists[gpu] + nei.v, dist + nei.v,
                sizeof(weight_t), cudaMemcpyHostToDevice));
        }
    }
    epochs++;
    */

    int iters=0;
    while (updated != 0) {
        if(iters>200){
            break;
        }
        iters++;
        // Reset update counters.
        updated = cpu_updated = 0;          
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemsetAsync(cu_updateds[gpu], 0, 
                    sizeof(nid_t)));
        }

        // Launch GPU epoch kernels.
        // Implicit CUDA device synchronize at the start of kernels.
        CUDA_ERRCHK(cudaSetDevice(0));
        epoch_pr_pull_gpu_block_min<<<256, 1024, 0, compute_streams[0]>>>(
                cu_indices[0], cu_neighbors[0],
                block_ranges[0], block_ranges[1],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[0], compute_streams[0]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[0], cu_dists[0] + block_ranges[0],
                (block_ranges[1] - block_ranges[0]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[0]));
        epoch_pr_pull_gpu_block_min<<<1024, 256, 0, compute_streams[1]>>>(
                cu_indices[1], cu_neighbors[1],
                block_ranges[2], block_ranges[3],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[1], compute_streams[1]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[2], cu_dists[0] + block_ranges[2],
                (block_ranges[3] - block_ranges[2]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[1]));
        epoch_pr_pull_gpu_block_min<<<4096, 64, 0, compute_streams[2]>>>(
                cu_indices[2], cu_neighbors[2],
                block_ranges[4], block_ranges[5],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[2], compute_streams[2]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[4], cu_dists[0] + block_ranges[4],
                (block_ranges[5] - block_ranges[4]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[2]));
        epoch_pr_pull_gpu_block_min<<<1024, 256, 0, compute_streams[3]>>>(
                cu_indices[3], cu_neighbors[3],
                block_ranges[6], block_ranges[7],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[3], compute_streams[3]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[6], cu_dists[0] + block_ranges[6],
                (block_ranges[7] - block_ranges[6]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[3]));
        epoch_pr_pull_gpu_block_min<<<4096, 64, 0, compute_streams[4]>>>(
                cu_indices[4], cu_neighbors[4],
                block_ranges[8], block_ranges[9],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[4], compute_streams[4]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[8], cu_dists[0] + block_ranges[8],
                (block_ranges[9] - block_ranges[8]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[4]));
        epoch_pr_pull_gpu_warp_min<<<256, 1024, 0, compute_streams[5]>>>(
                cu_indices[5], cu_neighbors[5],
                block_ranges[10], block_ranges[11],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[5], compute_streams[5]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[10], cu_dists[0] + block_ranges[10],
                (block_ranges[11] - block_ranges[10]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[5]));
        epoch_pr_pull_gpu_block_min<<<2048, 128, 0, compute_streams[6]>>>(
                cu_indices[6], cu_neighbors[6],
                block_ranges[12], block_ranges[13],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[6], compute_streams[6]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[12], cu_dists[0] + block_ranges[12],
                (block_ranges[13] - block_ranges[12]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[6]));
        epoch_pr_pull_gpu_block_min<<<4096, 64, 0, compute_streams[7]>>>(
                cu_indices[7], cu_neighbors[7],
                block_ranges[14], block_ranges[15],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[7], compute_streams[7]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[14], cu_dists[0] + block_ranges[14],
                (block_ranges[15] - block_ranges[14]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[7]));
        epoch_pr_pull_gpu_block_min<<<4096, 64, 0, compute_streams[8]>>>(
                cu_indices[8], cu_neighbors[8],
                block_ranges[16], block_ranges[17],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[8], compute_streams[8]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[16], cu_dists[0] + block_ranges[16],
                (block_ranges[17] - block_ranges[16]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[8]));
        epoch_pr_pull_gpu_warp_min<<<256, 1024, 0, compute_streams[9]>>>(
                cu_indices[9], cu_neighbors[9],
                block_ranges[18], block_ranges[19],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[9], compute_streams[9]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[18], cu_dists[0] + block_ranges[18],
                (block_ranges[19] - block_ranges[18]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[9]));
        epoch_pr_pull_gpu_one_to_one<<<256, 1024, 0, compute_streams[10]>>>(
                cu_indices[10], cu_neighbors[10],
                block_ranges[20], block_ranges[21],
                cu_dists[0], cu_updateds[0], g.num_nodes, cu_degrees);
        CUDA_ERRCHK(cudaEventRecord(compute_markers[10], compute_streams[10]));
        CUDA_ERRCHK(cudaMemcpyAsync(
                dist + block_ranges[20], cu_dists[0] + block_ranges[20],
                (block_ranges[21] - block_ranges[20]) * sizeof(weight_t),
                cudaMemcpyDeviceToHost, compute_streams[10]));

        // Launch CPU epoch kernels.
        #pragma omp parallel
        {
            epoch_pr_pull_cpu_one_to_one(g, dist, 
                    seg_ranges[16], seg_ranges[17],
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }
#pragma omp parallel
        {
            epoch_pr_pull_cpu_one_to_one(g, dist, 
                    seg_ranges[18], seg_ranges[20],
                    omp_get_thread_num(), omp_get_num_threads(), cpu_updated);
        }

        // Sync compute streams.
        for (int b = 0; b < num_blocks; b++)
            CUDA_ERRCHK(cudaEventSynchronize(compute_markers[b]));

        // Synchronize updates.
        nid_t gpu_updateds[num_gpus_pr];
        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaMemcpyAsync(
                    &gpu_updateds[gpu], cu_updateds[gpu],  sizeof(nid_t), 
                    cudaMemcpyDeviceToHost, memcpy_streams[gpu * num_gpus_pr + gpu]));
        }
        updated += cpu_updated;

        for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
            CUDA_ERRCHK(cudaSetDevice(gpu));
            CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
            updated += gpu_updateds[gpu];
        }

        // Only update GPU distances if another epoch will be run.
        if (updated != 0) {
            // Copy CPU distances to all GPUs.
            for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
                CUDA_ERRCHK(cudaMemcpyAsync(
                    cu_dists[gpu] + seg_ranges[16],
                    dist + seg_ranges[16],
                    (seg_ranges[17] - seg_ranges[16]) * sizeof(weight_t),
                    cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr + gpu]));
            }
            for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
                CUDA_ERRCHK(cudaMemcpyAsync(
                    cu_dists[gpu] + seg_ranges[18],
                    dist + seg_ranges[18],
                    (seg_ranges[20] - seg_ranges[18]) * sizeof(weight_t),
                    cudaMemcpyHostToDevice, memcpy_streams[gpu * num_gpus_pr + gpu]));
            }

            // Copy GPU distances peer-to-peer.
            // Not implmented if INTERLEAVE=true.
            gpu_butterfly_P2P_pr(seg_ranges, cu_dists, memcpy_streams); 

            // Synchronize HtoD async calls.
            for (int gpu = 0; gpu < num_gpus_pr; gpu++)
                CUDA_ERRCHK(cudaStreamSynchronize(memcpy_streams[gpu * num_gpus_pr + gpu]));
        }

        // Sync DtoH copies.
        for (int b = 0; b < num_blocks; b++)
            CUDA_ERRCHK(cudaStreamSynchronize(compute_streams[b]));
        

        
        epochs++;
    }
    
    timer.Stop();

    // Copy output.
    *ret_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        (*ret_dist)[i] = dist[i];

    // Free streams.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        for (int b = gpu_blocks[gpu]; b < gpu_blocks[gpu + 1]; b++) {
            CUDA_ERRCHK(cudaStreamDestroy(compute_streams[b]));
            CUDA_ERRCHK(cudaEventDestroy(compute_markers[b]));
        }

        for (int to = 0; to < num_gpus_pr; to++)
            CUDA_ERRCHK(cudaStreamDestroy(memcpy_streams[gpu * num_gpus_pr + to]));
    }

    // Free memory.
    for (int gpu = 0; gpu < num_gpus_pr; gpu++) {
        CUDA_ERRCHK(cudaSetDevice(gpu));
        CUDA_ERRCHK(cudaFree(cu_updateds[gpu]));
        CUDA_ERRCHK(cudaFree(cu_dists[gpu]));
        
        for (int block = gpu_blocks[gpu]; block < gpu_blocks[gpu + 1];
                block++
        ) {
            CUDA_ERRCHK(cudaFree(cu_indices[block]));
            CUDA_ERRCHK(cudaFree(cu_neighbors[block]));
        }
    }
    CUDA_ERRCHK(cudaFreeHost(dist));
    delete[] seg_ranges;

    return timer.Millisecs();
}

/**
 * Enable peer access between all compatible GPUs.
 */
void enable_all_peer_access_pr() {
    int can_access_peer;
    for (int from = 0; from < num_gpus_pr; from++) {
        CUDA_ERRCHK(cudaSetDevice(from));

        for (int to = 0; to < num_gpus_pr; to++) {
            if (from == to) continue;

            CUDA_ERRCHK(cudaDeviceCanAccessPeer(&can_access_peer, from, to));
            if(can_access_peer) {
                CUDA_ERRCHK(cudaDeviceEnablePeerAccess(to, 0));
                std::cout << from << " " << to << " yes" << std::endl;
            } else {
                std::cout << from << " " << to << " no" << std::endl;
            }
        }
    }
}

/**
 * Butterfly GPU P2P transfer.
 */
void gpu_butterfly_P2P_pr(nid_t *seg_ranges, weight_t **cu_dists, 
    cudaStream_t *memcpy_streams
) {
    
}

#endif // SRC_KERNELS_HETEROGENEOUS__PR_CUH