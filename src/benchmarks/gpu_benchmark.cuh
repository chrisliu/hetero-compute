/**
 * Benchmarks for GPU implementations.
 */

#ifndef SRC_BENCHMARKS__GPU_BENCHMARK_CUH
#define SRC_BENCHMARKS__GPU_BENCHMARK_CUH

#include <omp.h>
#include <random>

#include "benchmark.h"
#include "../cuda.cuh"
#include "../graph.h"
#include "../kernels/kernel_types.h"

/**
 * Tree based microbenchmark for GPU implementations.
 */
class SSSPGPUTreeBenchmark : public SSSPTreeBenchmark {
public:
    SSSPGPUTreeBenchmark(const CSRWGraph *g_, sssp_gpu_epoch_func epoch_kernel_,
            const int block_count_ = 8, const int thread_count_ = 1024); 
    ~SSSPGPUTreeBenchmark();

    void set_epoch_kernel(sssp_gpu_epoch_func epoch_kernel_);

    // Exposed block count and thread count to allow dynamic configuration.
    int block_count;   // Number of blocks to launch kernel.
    int thread_count;  // Number of threads to launch kernel.

protected:
    sssp_gpu_epoch_func epoch_kernel; // GPU epoch kernel.

    offset_t   *cu_index;     // (GPU) graph indices.
    wnode_t    *cu_neighbors; // (GPU) graph neighbors and weights.
    weight_t   *cu_dist;      // (GPU) distances.
    nid_t      *cu_updated;   // (GPU) update counter.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};

/******************************************************************************
 ***** Microbenchmark Implementations *****************************************
 ******************************************************************************/

SSSPGPUTreeBenchmark::SSSPGPUTreeBenchmark(const CSRWGraph *g_,
        sssp_gpu_epoch_func epoch_kernel_,
        const int block_count_, const int thread_count_)
    : SSSPTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
    , block_count(block_count_)
    , thread_count(thread_count_)
{
    // Initialize GPU copy of graph.
    size_t   index_size     = g->num_nodes * sizeof(offset_t);
    size_t   neighbors_size = g->num_edges * sizeof(wnode_t);
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, g->index, index_size, 
            cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, g->neighbors, neighbors_size, 
            cudaMemcpyHostToDevice));

    // Initialize update counter.
    CUDA_ERRCHK(cudaMalloc((void **) &cu_updated, sizeof(nid_t)));

    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, 
            g->num_nodes * sizeof(weight_t)));
}

SSSPGPUTreeBenchmark::~SSSPGPUTreeBenchmark() {
    cudaFree(cu_index);
    cudaFree(cu_neighbors);
    cudaFree(cu_dist);
    cudaFree(cu_updated);
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
segment_res_t SSSPGPUTreeBenchmark::benchmark_segment(const nid_t start_id,
        const nid_t end_id
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start_id;
    result.end_id     = end_id;
    result.num_edges  = g->index[end_id] - g->index[start_id];
    result.avg_degree = static_cast<float>(result.num_edges) 
        / (end_id- start_id);

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    double total_time = 0.0;
    float  millis     = 0.0f;

    // CUDA timer.
    cudaEvent_t start_t, stop_t;
    CUDA_ERRCHK(cudaEventCreate(&start_t));
    CUDA_ERRCHK(cudaEventCreate(&stop_t));

    // Run benchmark for this segment!
    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        // Setup kernel.
        CUDA_ERRCHK(cudaMemcpy(cu_dist, init_dist, g->num_nodes * sizeof(weight_t),
                cudaMemcpyHostToDevice));
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
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.teps      = result.num_edges / (result.millisecs * 1000);

    return result;
}

#endif // SRC_BENCHMARKS__GPU_BENCHMARK_CUH
