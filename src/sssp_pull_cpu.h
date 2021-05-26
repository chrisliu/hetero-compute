#ifndef SSSP_PULL_CPU_H
#define SSSP_PULL_CPU_H

#include "gapbs.h"
#include "util.h"

/** Forward decl. */
void kernel_sssp_pull_cpu(const wgraph_t &g, weight_t *dist, const int tid, 
        const int num_threads, int &updated);
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

#endif // SSSP_PULL_CPU_H
