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
 *   - init_dist        <- initial distance array.
 *   - ret_dist         <- pointer to the address of the return distance array.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_heterogeneous(const CSRWGraph &g,
        const weight_t *init_dist, weight_t ** const ret_dist
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
        total_time += sssp_pull_heterogeneous(g, init_dist, ret_dist);

        if (iter != BENCHMARK_TIME_ITERS - 1)
            delete[] (*ret_dist);
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9;

    return result;
}

#endif // SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH
