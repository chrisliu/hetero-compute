/**
 * Benchmarks for GPU implementations.
 */

#ifndef SRC_BENCHMARKS__GPU_BENCHMARK_CUH
#define SRC_BENCHMARKS__GPU_BENCHMARK_CUH

#include <omp.h>
#include <random>

#include "benchmark.h"
#include "../cuda.cuh"
#include "../util.h"
#include "../kernels/kernel_types.h"

/**
 * Tree based microbenchmark for GPU implementations.
 */
class SSSPGPUBenchmark : public TreeBenchmark {
public:
    SSSPGPUBenchmark(const CSRWGraph *g_, sssp_gpu_epoch_func epoch_kernel_,
            const int block_count_ = 8, const int thread_count_ = 1024); 
    ~SSSPGPUBenchmark();

    void set_epoch_kernel(sssp_gpu_epoch_func epoch_kernel_);

    // Exposed block count and thread count to allow dynamic configuration.
    int block_count;   // Number of blocks to launch kernel.
    int thread_count;  // Number of threads to launch kernel.

protected:
    sssp_gpu_epoch_func epoch_kernel; // GPU epoch kernel.

    weight_t   *init_dist;    // (CPU) initial distances. 
    offset_t   *cu_index;     // (GPU) graph indices.
    wnode_t    *cu_neighbors; // (GPU) graph neighbors and weights.
    weight_t   *cu_dist;      // (GPU) distances.
    nid_t      *cu_updated;   // (GPU) update counter.

    segment_res_t benchmark_segment(const nid_t start, const nid_t end);
};

/******************************************************************************
 ***** Microbenchmark Implementations *****************************************
 ******************************************************************************/

SSSPGPUBenchmark::SSSPGPUBenchmark(const CSRWGraph *g_,
        sssp_gpu_epoch_func epoch_kernel_,
        const int block_count_, const int thread_count_)
    : TreeBenchmark(g_)
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

    // Initialize distances.
    init_dist = new weight_t[g->num_nodes];
    unsigned init_seed = 1024; // TODO: make this random?
    #pragma omp parallel
    {
        std::mt19937_64 gen(init_seed + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = omp_get_thread_num(); i < g->num_nodes;
                i += omp_get_num_threads()
        )

        init_dist[i] = dist(gen);          
    }

    CUDA_ERRCHK(cudaMalloc((void **) &cu_dist, 
            g->num_nodes * sizeof(weight_t)));
}

SSSPGPUBenchmark::~SSSPGPUBenchmark() {
    delete[] init_dist;
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
void SSSPGPUBenchmark::set_epoch_kernel(sssp_gpu_epoch_func epoch_kernel_) {
    epoch_kernel = epoch_kernel_;
}

/**
 * Performs benchmark on a single slice of nodes of range [start, end).
 * Parameters:
 *   - start <- starting node ID.
 *   - end   <- ending node ID (exclusive).
 * Returns:
 *   result segment data structure.
 */
segment_res_t SSSPGPUBenchmark::benchmark_segment(const nid_t start,
        const nid_t end
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start;
    result.end_id     = end;
    result.num_edges  = g->index[end] - g->index[start];
    result.avg_degree = static_cast<float>(result.num_edges) / (end - start);

    // Setup kernel.
    CUDA_ERRCHK(cudaMemcpy(cu_dist, init_dist, g->num_nodes * sizeof(weight_t),
            cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemset(cu_updated, 0, sizeof(nid_t)));

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    double total_time = 0.0;
    float  millis     = 0.0f;

    // CUDA timer.
    cudaEvent_t start_t, stop_t;
    CUDA_ERRCHK(cudaEventCreate(&start_t));
    CUDA_ERRCHK(cudaEventCreate(&stop_t));
    Timer timer;

    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        CUDA_ERRCHK(cudaEventRecord(start_t));
        timer.Start();
        (*epoch_kernel)<<<block_count, thread_count>>>(cu_index, cu_neighbors,
                start, end, cu_dist, cu_updated);
        timer.Stop();
        CUDA_ERRCHK(cudaEventRecord(stop_t));

        CUDA_ERRCHK(cudaEventSynchronize(stop_t));
        CUDA_ERRCHK(cudaEventElapsedTime(&millis, start_t, stop_t));

        total_time += millis;
        total_time += timer.Millisecs();
    }

    CUDA_ERRCHK(cudaEventDestroy(start_t));
    CUDA_ERRCHK(cudaEventDestroy(stop_t));

    // Save results.
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.teps      = result.num_edges / (result.millisecs * 1000);

    return result;
}

#endif // SRC_BENCHMARKS__GPU_BENCHMARK_CUH
