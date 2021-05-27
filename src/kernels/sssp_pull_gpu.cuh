#ifndef SSSP_PULL_GPU_H
#define SSSP_PULL_GPU_H

/*#include <nvfunctional>*/
#include <functional>
#include <omp.h> 

#include "../cuda.h"
#include "../gapbs.h"
#include "../util.h"

// Epoch kernel type.
typedef void (*sssp_gpu_epoch_func)(const nid_t *, 
        const cu_wnode_t *, const nid_t, const nid_t, weight_t *, nid_t *);

/** Forward decl. */
__global__ void sssp_pull_gpu_impl(const nid_t *index,
    const cu_wnode_t *neighbors, const nid_t start_id, const nid_t end_id,
    weight_t *dist, nid_t *updated);

/**
 * Runs SSSP kernel on GPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 */
void sssp_pull_gpu(const wgraph_t &g, weight_t **ret_dist) {
    /// Setup.
    // Copy graph.
    nid_t      *index     = nullptr;
    cu_wnode_t *neighbors = nullptr;
    wgraph_to_cugraph(g, &index, &neighbors);

    size_t     index_size     = g.num_nodes() * sizeof(nid_t);
    size_t     neighbors_size = 2 * g.num_edges() * sizeof(cu_wnode_t);
    nid_t      *cu_index      = nullptr;
    cu_wnode_t *cu_neighbors  = nullptr;
    cudaMalloc((void **) &cu_index, index_size);
    cudaMalloc((void **) &cu_neighbors, neighbors_size);
    cudaMemcpy(cu_index, index, index_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_neighbors, neighbors, neighbors_size, cudaMemcpyHostToDevice);

    delete[] index; delete[] neighbors;

    // Update counter.
    nid_t *cu_updated = nullptr;
    cudaMalloc((void **) &cu_updated, sizeof(nid_t));
    
    // Distance.
    weight_t *dist = new weight_t[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
        dist[i] = MAX_WEIGHT;
    dist[0] = 0; // Arbitrarily set start.

    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes() * sizeof(weight_t);
    cudaMalloc((void **) &cu_dist, dist_size);
    cudaMemcpy(cu_dist, dist, dist_size, cudaMemcpyHostToDevice);

    // Actual kernel run.
    nid_t updated = 1;

    std::cout << "Starting kernel ..." << std::endl;
    Timer timer; timer.Start();

    while (updated != 0) {
        cudaMemset(cu_updated, 0, sizeof(nid_t));

        // Note: Must run with thread count >= 32 since warp level 
        //       synchronization is performed.
        sssp_pull_gpu_impl<<<64, 1024>>>(cu_index, cu_neighbors, 0, 
                g.num_nodes(), cu_dist, cu_updated);

        cudaMemcpy(&updated, cu_updated, sizeof(nid_t), cudaMemcpyDeviceToHost);
    }

    timer.Stop();
    std::cout << "Kernel completed in: " << timer.Millisecs() << " ms."
        << std::endl;

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
void sssp_pull_gpu_impl(const nid_t *index, const cu_wnode_t *neighbors, 
        const nid_t start_id, const nid_t end_id, weight_t *dist, nid_t *updated
) {
    int tid         = blockIdx.x * blockDim.x + threadIdx.x;
    int warpid      = tid % warpSize;
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

#endif // SSSP_PULL_GPU_H
