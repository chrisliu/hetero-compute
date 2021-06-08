/**
 * Benchmarks for heterogeneous implementations.
 */

#ifndef SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH
#define SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH

#include "../graph.h"
#include "../kernels/heterogeneous/sssp_pull.cuh"

/**
 * Benchmarks a full SSSP heterogeneous run.
 * Parameters:
 *   - g                <- graph.
 *   - cpu_epoch_kernel <- cpu epoch kernel.
 *   - gpu_epoch_kernel <- gpu epoch kernel.
 *   - cpu_range        <- range of CPU kernel.
 *   - gpu_ranges       <- ranges for GPU kernel.
 *   - init_dist        <- initial distance array.
 *   - ret_dist         <- pointer to the address of the return distance array.
 *   - block_count      <- (optional) number of blocks.
 *   - thread_count     <- (optional) number of threads.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_heterogeneous(
        const CSRWGraph &g,
        sssp_cpu_epoch_func cpu_epoch_kernel, 
        sssp_gpu_epoch_func gpu_epoch_kernel,
        const graph_range_t cpu_range,
        const std::vector<graph_range_t> gpu_ranges,
        const weight_t *init_dist, weight_t ** const ret_dist,
        int block_count = 64, int thread_count = 1024);

/******************************************************************************
 ***** Microbenchmark Implementations *****************************************
 ******************************************************************************/

segment_res_t benchmark_sssp_heterogeneous(
        const CSRWGraph &g,
        sssp_cpu_epoch_func cpu_epoch_kernel, 
        sssp_gpu_epoch_func gpu_epoch_kernel,
        const graph_range_t cpu_range,
        const std::vector<graph_range_t> gpu_ranges,
        const weight_t *init_dist, weight_t ** const ret_dist,
        int block_count, int thread_count
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    // Run kernel!
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        total_time += sssp_pull_heterogeneous(g, 
                cpu_epoch_kernel, gpu_epoch_kernel, cpu_range, gpu_ranges, 
                init_dist, ret_dist, block_count, thread_count);

        if (iter != BENCHMARK_TIME_ITERS - 1)
            delete[] (*ret_dist);
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9;

    return result;
}

#endif // SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH
