/**
 * Implementations of CPU and GPU SSSP pull with data race.
 */

#include <omp.h> 


#include "gapbs.h"
#include "util.h"

// Forward decl.
void kernel_sssp_pull_cpu(const wgraph_t &g, weight_t *dist, const int tid, 
        const int num_threads, int &updated);
void sssp_pull_cpu(const wgraph_t &g, weight_t **ret_dist);
void sssp_pull_gpu(const wgraph_t &g, weight_t **ret_dist);
__global__ void sssp_pull_gpu_impl(const nid_t *index, 
    const cu_wnode_t *neighbors, const int num_nodes, weight_t *dist, 
    int *updated);

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);
    //wgraph_t ordered_g = b.MakeGraph();

    // Run SSSP.
    weight_t *distances = nullptr;
    sssp_pull_cpu(ordered_g, &distances);
    /*sssp_pull_gpu(ordered_g, &distances);*/

    if (cli.scale() <= 4) {
        std::cout << "node neighbors" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++) {
            std::cout << " > node " << i << std::endl;
            for (auto &out_nei : ordered_g.out_neigh(i)) {
                std::cout << "    > node " << out_nei.v << ": " << out_nei.w
                    << std::endl;
            }
        }

        std::cout << "node: distance" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++)
            std::cout << " > " << i << ": " << distances[i] << std::endl;
    }

    //WeightedWriter w(ordered_g);
    //w.WriteGraph("graph.wel");

    //WeightedReader r("graph.wel");

    return EXIT_SUCCESS;
}

/******************************************************************************
 ***** Kernels ****************************************************************
 ******************************************************************************/

/**
 * Runs SSSP kernel on CPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 */
void sssp_pull_cpu(const wgraph_t &g, weight_t **ret_dist) {
    weight_t *dist = new weight_t[g.num_nodes()];

    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
        dist[i] = MAX_WEIGHT;

    // Arbitrary: Set lowest degree node as source.
    dist[0] = 0;

    // Start kernel.
    std::cout << "Starting kernel ..." << std::endl;
    Timer timer; timer.Start();

    int updated = 1;

    while (updated != 0) {
        updated = 0;

        #pragma omp parallel
        {
            kernel_sssp_pull_cpu(g, dist, omp_get_thread_num(), 
                    omp_get_num_threads(), updated);
        }

        // Implicit OMP BARRIER here (see "implicit barrier at end of parallel 
        // region").
    }

    timer.Stop();
    std::cout << "Kernel completed in: " << timer.Millisecs() << " ms."
        << std::endl;

    // Assign output.
    *ret_dist = dist;
}

void sssp_pull_gpu(const wgraph_t &g, weight_t **ret_dist) {
    /// Setup.
    std::cout << "Setting up ..." << std::endl;
    // Copy graph.
    std::cout << " > Copying graph ..." << std::endl;
    nid_t      *index     = nullptr;
    cu_wnode_t *neighbors = nullptr;
    wgraph_to_cugraph(g, &index, &neighbors);
    size_t index_size     = g.num_nodes() * sizeof(nid_t);
    size_t neighbors_size = 2 * g.num_edges() * sizeof(cu_wnode_t);

    nid_t      *cu_index     = nullptr;
    cu_wnode_t *cu_neighbors = nullptr;
    cudaMalloc((void **) &cu_index, index_size);
    cudaMalloc((void **) &cu_neighbors, neighbors_size);
    cudaMemcpy(cu_index, index, index_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_neighbors, neighbors, neighbors_size, cudaMemcpyHostToDevice);

    delete[] index; delete[] neighbors;

    // Distance and update counter.
    std::cout << " > Initializing distance and update counter ..." << std::endl;
    int *cu_updated = nullptr;
    cudaMalloc((void **) &cu_updated, sizeof(int));
    
    weight_t *dist = new weight_t[g.num_nodes()];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
        dist[i] = MAX_WEIGHT;
    dist[0] = 0;
    weight_t *cu_dist = nullptr;
    size_t dist_size = g.num_nodes() * sizeof(weight_t);
    cudaMalloc((void **) &cu_dist, dist_size);
    cudaMemcpy(cu_dist, dist, dist_size, cudaMemcpyHostToDevice);

    // Actual kernel run.
    int updated = 1;

    std::cout << "Starting kernel ..." << std::endl;
    Timer timer; timer.Start();

    while (updated != 0) {
        cudaMemset(cu_updated, 0, sizeof(int));

        sssp_pull_gpu_impl<<<1, 8>>>(cu_index, cu_neighbors, g.num_nodes(),
                cu_dist, cu_updated);

        cudaMemcpy(&updated, cu_updated, sizeof(int), cudaMemcpyDeviceToHost);
    }

    timer.Stop();
    std::cout << "Kernel completed in: " << timer.Millisecs() << " ms."
        << std::endl;

    // Copy distances.
    std::cout << "Copying output ..." << std::endl;
    cudaMemcpy(dist, cu_dist, dist_size, cudaMemcpyDeviceToHost);
    *ret_dist = dist;

    // Free memory.
    std::cout << "Freeing memory ..." << std::endl;
    cudaFree(cu_index);
    cudaFree(cu_neighbors);
    cudaFree(cu_updated);
    cudaFree(cu_dist);
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

/**
 * Runs SSSP pull on CPU for one epoch.
 * Parameters:
 *   - g           <- graph.
 *   - dist        <- input distances and output distances computed this 
 *                    epoch.
 *   - tid         <- processor id.
 *   - num_threads <- number of processors.
 *   - updated     <- global counter of number of nodes updated.
 */
void kernel_sssp_pull_cpu(const wgraph_t &g, weight_t *dist, const int tid,
        const int num_threads, int &updated
) {
    int local_updated = 0;

    // Propagate, reduce, and apply.
    for (int nid = tid; nid < g.num_nodes(); nid += num_threads) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (wnode_t nei : g.in_neigh(nid)) {
            weight_t prop_dist = dist[nei.v] + nei.w;
            new_dist = std::min(prop_dist, new_dist);
        }

        // Update distance if applicable.
        if (new_dist != dist[nid]) {
            dist[nid] = new_dist;
            local_updated++;
        }
    }

    // Push update count.
    #pragma omp atomic
    updated += local_updated;
}

/**
 * Runs SSSP pull on GPU for one epoch.
 * Parameters:
 *   - index     <- graph index returned by deconstruct_wgraph().
 *   - neighbors <- graph neighbors returned by deconstruct_wgraph().
 *   - num_nodes <- number of nodes in this graph.
 *   - dist      <- input distance and output distances computed this epoch.
 *   - updated   <- global counter on number of nodes updated.
 */
__global__ 
void sssp_pull_gpu_impl(const nid_t *index, const cu_wnode_t *neighbors, 
        const int num_nodes, weight_t *dist, int *updated
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;

    int local_updated = 0;

    for (int nid = tid; nid < num_nodes; nid += num_threads) {
        weight_t new_dist = dist[nid];

        // Find shartest candiadte distance.
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
