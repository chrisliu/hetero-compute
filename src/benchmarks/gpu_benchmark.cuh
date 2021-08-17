/**
 * Benchmarks for GPU implementations.
 */

#ifndef SRC_BENCHMARKS__GPU_BENCHMARK_CUH
#define SRC_BENCHMARKS__GPU_BENCHMARK_CUH

#include <random>

#include "benchmark.h"
#include "../cuda.cuh"
#include "../graph.h"
#include "../kernels/kernel_types.cuh"
#include "../kernels/gpu/bfs.cuh"
#include "../kernels/gpu/sssp_pull.cuh"

/**
 * Tree based microbenchmark for GPU implementations.
 */
class SSSPGPUTreeBenchmark : public SSSPTreeBenchmark {
public:
    SSSPGPUTreeBenchmark(
            const CSRWGraph *g_, sssp_gpu_epoch_func epoch_kernel_,
            const int block_count_ = 64, const int thread_count_ = 1024); 
    ~SSSPGPUTreeBenchmark();

    void set_epoch_kernel(sssp_gpu_epoch_func epoch_kernel_);

    // Exposed block count and thread count to allow dynamic configuration.
    int block_count;   // Number of blocks to launch kernel.
    int thread_count;  // Number of threads to launch kernel.

protected:
    sssp_gpu_epoch_func epoch_kernel; // GPU epoch kernel.

    offset_t   *cu_index;     // (GPU) subgraph indices.
    wnode_t    *cu_neighbors; // (GPU) subgraph neighbors and weights.
    weight_t   *cu_dist;      // (GPU) distances.
    nid_t      *cu_updated;   // (GPU) update counter.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};

/**
 * Benchmarks a full SSSP GPU run.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch_kernel.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 *   - block_count  <- (optional) number of blocks.
 *   - thread_count <- (optional) number of threads.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_gpu(
        const CSRWGraph &g, 
        sssp_gpu_epoch_func epoch_kernel,
        const weight_t *init_dist, weight_t **ret_dist,
        int block_size = 64, int thread_count = 1024);

/*****************************************************************************
 ***** Tree Benchmark Implementations ****************************************
 *****************************************************************************/

SSSPGPUTreeBenchmark::SSSPGPUTreeBenchmark(
        const CSRWGraph *g_, sssp_gpu_epoch_func epoch_kernel_,
        const int block_count_, const int thread_count_)
    : SSSPTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
    , block_count(block_count_)
    , thread_count(thread_count_)
    , cu_index(nullptr)
    , cu_neighbors(nullptr)
{
    // Initialize update counter.
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));

    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, 
            g->num_nodes * sizeof(weight_t)));
}

SSSPGPUTreeBenchmark::~SSSPGPUTreeBenchmark() {
    CUDA_ERRCHK(cudaFree(cu_dist));
    CUDA_ERRCHK(cudaFree(cu_updated));
}

/**
 * Sets GPU epoch kernel benchmark will run.
 * Parameters:
 *   - epoch_kernel_ <- new SSSP epoch kernel.
 */
void SSSPGPUTreeBenchmark::set_epoch_kernel(sssp_gpu_epoch_func epoch_kernel_) {
    epoch_kernel = epoch_kernel_;
}

/**
 * Performs benchmark on a single slice of nodes of range [start, end).
 * Parameters:
 *   - start_id <- starting node ID.
 *   - end_id   <- ending node ID (exclusive).
 * Returns:
 *   result segment data structure.
 */
segment_res_t SSSPGPUTreeBenchmark::benchmark_segment(
        const nid_t start_id, const nid_t end_id
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start_id;
    result.end_id     = end_id;
    result.num_edges  = g->index[end_id] - g->index[start_id];
    result.avg_degree = static_cast<float>(result.num_edges) 
        / (end_id- start_id);

    // Copy subgraph.
    copy_subgraph_to_device(*g, &cu_index, &cu_neighbors, start_id, end_id);
    /*copy_subgraph_to_device(start_id, end_id);*/

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    double total_time = 0.0;
    float  millis     = 0.0f;

    // CUDA timer.
    cudaEvent_t start_t, stop_t;
    CUDA_ERRCHK(cudaEventCreate(&start_t));
    CUDA_ERRCHK(cudaEventCreate(&stop_t));

    // Run benchmark for this segment!
    for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {
        // Setup kernel.
        CUDA_ERRCHK(cudaMemcpy(cu_dist, init_dist, 
                g->num_nodes * sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

        // Run epoch kernel.
        CUDA_ERRCHK(cudaEventRecord(start_t));
        (*epoch_kernel)<<<block_count, thread_count>>>(cu_index, cu_neighbors,
                start_id, end_id, cu_dist, cu_updated);
        CUDA_ERRCHK(cudaEventRecord(stop_t));

        // Save time.
        CUDA_ERRCHK(cudaEventSynchronize(stop_t));
        CUDA_ERRCHK(cudaEventElapsedTime(&millis, start_t, stop_t));

        total_time += millis;
    }

    CUDA_ERRCHK(cudaEventDestroy(start_t));
    CUDA_ERRCHK(cudaEventDestroy(stop_t));

    // Save results.
    result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    // Free subgraph.
    CUDA_ERRCHK(cudaFree(cu_index));
    CUDA_ERRCHK(cudaFree(cu_neighbors));

    return result;
}

/*****************************************************************************
 ***** Kernel Benchmark Implementations **************************************
 *****************************************************************************/

segment_res_t benchmark_sssp_gpu(
        const CSRWGraph &g, 
        sssp_gpu_epoch_func epoch_kernel,
        SourcePicker<CSRWGraph> &sp,
        int block_size, int thread_count
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    // Define initial and return distances.
    weight_t *init_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_dist[i] = INF_WEIGHT;
    weight_t *ret_dist = nullptr;

    // Run kernel!
    nid_t previous_source = 0;
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
        nid_t cur_source = sp.next_vertex();
        init_dist[previous_source] = INF_WEIGHT;
        init_dist[cur_source]      = 0;
        previous_source = cur_source;

        total_time += sssp_pull_gpu(g, epoch_kernel, init_dist, &ret_dist,
                block_size, thread_count);
        delete[] ret_dist;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

segment_res_t benchmark_bfs_gpu(
        const CSRUWGraph &g,
        bfs_gpu_epoch_func epoch_kernel,
        SourcePicker<CSRUWGraph> &sp,
        int block_size, int thread_count
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    nid_t *parents;

    // Run kernel!
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
        nid_t cur_source = sp.next_vertex();

        total_time += bfs_gpu(g, cur_source, &parents, epoch_kernel,
                block_size, thread_count);

        delete[] parents;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

#endif // SRC_BENCHMARKS__GPU_BENCHMARK_CUH
