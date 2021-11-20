/**
 * Benchmarks for heterogeneous implementations.
 */

#ifndef SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH
#define SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH

#include "../graph.cuh"
#include "../kernels/heterogeneous/bfs.cuh"
#include "../kernels/heterogeneous/sssp.cuh"

/**
 * Benchmarks a full SSSP heterogeneous run.
 * Parameters:
 *   - g   <- graph.
 *   - sp  <- source node picker.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_heterogeneous(const CSRWGraph &g,
        SourcePicker<CSRWGraph> sp
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    /*// Compute min and max degree.*/
    /*float min_degree, max_degree;*/
    /*min_degree = max_degree = static_cast<float>(g.index[1] - g.index[0]);*/
    /*#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)*/
    /*for (int nid = 1; nid < g.num_nodes; nid++) {*/
        /*float ndeg = static_cast<float>(g.index[nid + 1] - g.index[nid]);*/
        /*min_degree = min(min_degree, ndeg);*/
        /*max_degree = max(max_degree, ndeg);*/
    /*}*/
    result.min_degree = 0;
    result.max_degree = 0;

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

        total_time += sssp_pull_heterogeneous(g, init_dist, &ret_dist);

        delete[] ret_dist;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

/**
 * Benchmarks a full BFS heterogeneous run.
 * Parameters:
 *   - g <- graph.
 *   - sp <- source node picker.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_bfs_heterogeneous(const CSRUWGraph &g,
        SourcePicker<CSRUWGraph> sp
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    /*// Compute min and max degree.*/
    /*float min_degree, max_degree;*/
    /*min_degree = max_degree = static_cast<float>(g.index[1] - g.index[0]);*/
    /*#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)*/
    /*for (int nid = 1; nid < g.num_nodes; nid++) {*/
        /*float ndeg = static_cast<float>(g.index[nid + 1] - g.index[nid]);*/
        /*min_degree = min(min_degree, ndeg);*/
        /*max_degree = max(max_degree, ndeg);*/
    /*}*/
    result.min_degree = 0;
    result.max_degree = 0;

    nid_t *parents;

    // Run kernel!
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
        nid_t cur_source = sp.next_vertex();

        total_time += bfs_do_heterogeneous(g, cur_source, &parents);

        delete[] parents;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

#endif // SRC_BENCHMARKS__HETEROGENEOUS_BENCHMARK_CUH
