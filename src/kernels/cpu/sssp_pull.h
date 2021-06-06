/**
 * CPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_CPU__KERNEL_SSSP_PULL_H
#define SRC_KERNELS_CPU__KERNEL_SSSP_PULL_H

#include <iostream>
#include <omp.h>

#include "../kernel_types.h"
#include "../../graph.h"
#include "../../util.h"
#include "../../benchmarks/benchmark.h"

/**
 * Runs SSSP kernel on CPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 */
segment_res_t sssp_pull_cpu(const CSRWGraph &g, 
        sssp_cpu_epoch_func epoch_kernel, weight_t **ret_dist
) {
    // Setup.
    weight_t *dist = new weight_t[g.num_nodes];


    // Return data structure.
    segment_res_t res;
    res.start_id   = 0;
    res.end_id     = g.num_nodes;
    res.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    res.num_edges = g.num_edges;

    // Start kernel.
    std::cout << "Starting kernel ..." << std::endl;

    double total_epochs = 0.0;
    double total_time   = 0.0;

    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        #pragma omp parallel for
        for (int i = 0; i < g.num_nodes; i++)
            dist[i] = MAX_WEIGHT;

        // Arbitrary: Set highest degree node as source.
        dist[0] = 0;

        nid_t updated = 1;

        Timer timer; timer.Start();
        while (updated != 0) {
            updated = 0;

            #pragma omp parallel
            {
                (*epoch_kernel)(g, dist, 0, g.num_nodes, omp_get_thread_num(), 
                        omp_get_num_threads(), updated);
            }

            // Implicit OMP BARRIER here (see "implicit barrier at end of 
            // parallel region").

            total_epochs++;
        }
        timer.Stop();

        total_time += timer.Millisecs();
    }
    // Save results.
    res.millisecs = total_time / BENCHMARK_TIME_ITERS;
    res.gteps     = res.num_edges / (res.millisecs / 1000) / 1e9;

    std::cout << "Kernel completed in (avg): " << res.millisecs << " ms." 
        << std::endl;

    // Assign output.
    *ret_dist = dist;

    return res;
}

/**
 * Runs SSSP pull on CPU for one epoch.
 * Parameters:
 *   - g           <- graph.
 *   - dist        <- input distances and output distances computed this 
 *                    epoch.
 *   - start_id    <- starting node id.
 *   - end_id      <- ending node id (exclusive).
 *   - tid         <- processor id.
 *   - num_threads <- number of processors.
 *   - updated     <- global counter of number of nodes updated.
 */
void epoch_sssp_pull_cpu(const CSRWGraph &g, weight_t *dist, 
        const nid_t start_id, const nid_t end_id, const int tid,
        const int num_threads, nid_t &updated
) {
    nid_t local_updated = 0;

    // Propagate, reduce, and apply.
    for (int nid = start_id + tid; nid < end_id; nid += num_threads) {
        weight_t new_dist = dist[nid];

        // Find shortest candidate distance.
        for (wnode_t nei : g.get_neighbors(nid)) {
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

#endif // SRC_KERNELS_CPU__KERNEL_SSSP_PULL_H
