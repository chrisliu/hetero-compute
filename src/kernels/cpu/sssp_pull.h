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

/******************************************************************************
 ***** SSSP Kernels ***********************************************************
 ******************************************************************************/

/**
 * Runs SSSP kernel on CPU in parallel. Synchronization occurs in serial.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch kernel.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 * Returns:
 *   Execution time in milliseconds.
 */
double sssp_pull_cpu(const CSRWGraph &g, sssp_cpu_epoch_func epoch_kernel, 
        const weight_t *init_dist, weight_t **ret_dist
) {
    // Setup computed distances.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = init_dist[i];

    nid_t updated = 1;

    // Start kernel!
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
    }
    timer.Stop();

    // Assign output.
    *ret_dist = dist;

    return timer.Millisecs();
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

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
