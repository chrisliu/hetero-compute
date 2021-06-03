/**
 * Benchamrks for CPU implementations.
 */

#ifndef SRC_BENCHMARKS__CPU_BENCHMARK_H
#define SRC_BENCHMARKS__CPU_BENCHMARK_H

#include <omp.h>

#include "benchmark.h"
#include "../graph.h"
#include "../util.h"
#include "../kernels/kernel_types.h"

class SSSPCPUTreeBenchmark : public SSSPTreeBenchmark {
public:
    SSSPCPUTreeBenchmark(const CSRWGraph *g_, 
            sssp_cpu_epoch_func epoch_kernel_);

protected:
    sssp_cpu_epoch_func epoch_kernel; // CPU epoch kernel.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};

/******************************************************************************
 ***** Microbenchmark Implementations *****************************************
 ******************************************************************************/

SSSPCPUTreeBenchmark::SSSPCPUTreeBenchmark(const CSRWGraph *g_,
        sssp_cpu_epoch_func epoch_kernel_)
    : SSSPTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
{}

segment_res_t SSSPCPUTreeBenchmark::benchmark_segment(const nid_t start_id,
        const nid_t end_id
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

    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
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
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9;

    return result;
}

#endif // SRC_BENCHMARKS__CPU_BENCHMARK_H
