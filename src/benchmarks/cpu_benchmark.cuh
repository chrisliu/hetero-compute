/**
 * Benchamrks for CPU implementations.
 */

#ifndef SRC_BENCHMARKS__CPU_BENCHMARK_CUH
#define SRC_BENCHMARKS__CPU_BENCHMARK_CUH

#include <omp.h>

#include "benchmark.cuh"
#include "../graph.cuh"
#include "../util.h"
#include "../kernels/kernel_types.cuh"
#include "../kernels/cpu/sssp.cuh"

/*****************************************************************************
 ***** Benchmarks ************************************************************
 *****************************************************************************/

class SSSPCPUTreeBenchmark : public SSSPTreeBenchmark {
public:
    SSSPCPUTreeBenchmark(
            const CSRWGraph *g_, sssp_cpu_epoch_func epoch_kernel_);

protected:
    sssp_cpu_epoch_func epoch_kernel; // CPU epoch kernel.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};

/**
 * Benchmarks a full SSSP CPU run.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch_kernel.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_cpu(
        const CSRWGraph &g, 
        sssp_cpu_epoch_func epoch_kernel,
        const weight_t *init_dist, weight_t **ret_dist);

/*****************************************************************************
 ***** Tree Benchmark Implementations ****************************************
 *****************************************************************************/

SSSPCPUTreeBenchmark::SSSPCPUTreeBenchmark(
        const CSRWGraph *g_, sssp_cpu_epoch_func epoch_kernel_)
    : SSSPTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
{}

segment_res_t SSSPCPUTreeBenchmark::benchmark_segment(
        const nid_t start_id, const nid_t end_id
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start_id;
    result.end_id     = end_id;
    result.num_edges  = g->index[end_id] - g->index[start_id];
    result.avg_degree = static_cast<float>(result.num_edges) 
        / (end_id - start_id);

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    Timer timer;
    double total_time = 0.0;

    for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {
        // Setup kernel.
        weight_t *dist = new weight_t[g->num_nodes];
        #pragma omp parallel for
        for (int i = 0; i < g->num_nodes; i++)
            dist[i] = init_dist[i];
        nid_t updated = 0;

        // Run kernel.
        timer.Start();
        #pragma omp parallel
        {
            (*epoch_kernel)(*g, dist, start_id, end_id, omp_get_thread_num(),
                    omp_get_num_threads(), updated);
        }
        timer.Stop();

        // Save time.
        total_time += timer.Millisecs();
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

/*****************************************************************************
 ***** Kernel Benchmark Implementations **************************************
 *****************************************************************************/

segment_res_t benchmark_sssp_cpu(
        const CSRWGraph &g, sssp_cpu_epoch_func epoch_kernel,
        SourcePicker<CSRWGraph> &sp
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

        total_time += sssp_pull_cpu(g, epoch_kernel, init_dist, &ret_dist);
        delete[] ret_dist;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}

segment_res_t benchmark_bfs_cpu(
        const CSRUWGraph &g, bfs_cpu_kernel kernel, SourcePicker<CSRUWGraph> &sp
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    nid_t *parents = nullptr;

    // Run kernel!
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
        nid_t cur_source = sp.next_vertex();

        total_time += kernel(g, cur_source, &parents);
        
        delete[] parents;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;    
}

#endif // SRC_BENCHMARKS__CPU_BENCHMARK_CUH
