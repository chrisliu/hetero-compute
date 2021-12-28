/**
 * CPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_CPU__KERNEL_SSSP_CUH
#define SRC_KERNELS_CPU__KERNEL_SSSP_CUH

#include <iostream>
#include <omp.h>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "../kernel_types.cuh"
#include "../../graph.cuh"
#include "../../util.h"

/*****************************************************************************
 ***** SSSP Kernels **********************************************************
 *****************************************************************************/

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
double sssp_pull_cpu(
        const CSRWGraph &g, sssp_cpu_epoch_func epoch_kernel, 
        const weight_t *init_dist, weight_t ** const ret_dist
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

/**
 * Run SSSP kernel on CPU in serial.
 * Parameters:
 *   - g         <- graph.
 *   - source_id <- initial point. 
 *   - ret_dist  <- pointer to the address of the return distance array.
 */
void sssp_dijkstras_cpu_serial(
        const CSRWGraph &g, nid_t source_id, weight_t **ret_dist
) {
    // Setup.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = INF_WEIGHT;
    dist[source_id] = 0.0f;

    using wnode_pair_t = std::pair<weight_t, nid_t>;
    std::priority_queue<wnode_pair_t, std::vector<wnode_pair_t>, 
        std::greater<wnode_pair_t>> frontier;
    frontier.push(std::make_pair(0.0f, source_id));

    while (not frontier.empty()) {
        weight_t weight = frontier.top().first;
        nid_t    nid    = frontier.top().second;
        frontier.pop();

        // If this is the latest distance.
        if (weight == dist[nid]) {
            for (wnode_t nei : g.get_neighbors(nid)) {
                weight_t new_weight = weight + nei.w;

                // If possible, update weight.
                if (new_weight < dist[nei.v]) {
                    dist[nei.v] = new_weight;
                    frontier.push(std::make_pair(new_weight, nei.v));
                }
            }
        }
    }

    *ret_dist = dist;
}

void sssp_pull_cpu_serial(
        const CSRWGraph &g, nid_t source_id, weight_t **ret_dist
) {
    // Setup.
    weight_t *dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        dist[i] = INF_WEIGHT;
    dist[source_id] = 0.0f;

    nid_t num_updates;
    do {
        num_updates = 0;

        for (nid_t v = 0; v < g.num_nodes; v++) {
            for (wnode_t nei : g.get_neighbors(v)) {
                weight_t prop_dist = dist[nei.v] + nei.w;
                if (prop_dist < dist[v]) {
                    dist[v] = prop_dist;
                    num_updates++;
                }
            }
        }
    } while (num_updates != 0);

    *ret_dist = dist;
}

/*****************************************************************************
 ***** Epoch Kernels *********************************************************
 *****************************************************************************/

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
void epoch_sssp_pull_cpu_one_to_one(
        const CSRWGraph &g, weight_t *dist, 
        const nid_t start_id, const nid_t end_id, 
        const int tid, const int num_threads, nid_t &updated
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

/******************************************************************************
 ***** Helper Functions *******************************************************
 ******************************************************************************/

/** Identifier for epoch kernels. */
enum class SSSPCPU {
    one_to_one, undefined
};

/** List of kernels available (no good iterator for enum classes). */
std::vector<SSSPCPU> sssp_cpu_kernels = {
    SSSPCPU::one_to_one
};

std::vector<SSSPCPU> get_kernels(UNUSED SSSPCPU unused) {
    // Using hack to overload function by return type.
    return sssp_cpu_kernels;
}

/** 
 * Convert epoch kernel ID to its representation name (not as human-readable).
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_repr(SSSPCPU ker) {
    switch (ker) {
        case SSSPCPU::one_to_one: return "sssp_cpu_onetoone";
        case SSSPCPU::undefined:
        default:                  return "";
    }
}

/** 
 * Convert epoch kernel ID to its human-readable name. 
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_string(SSSPCPU ker) {
    switch (ker) {
        case SSSPCPU::one_to_one: return "SSSP CPU one-to-one";
        case SSSPCPU::undefined:
        default:                  return "undefined SSSP CPU kernel";
    }
}

/**
 * Convert epoch kernel ID to kernel function pointer.
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel function pointer.
 */
sssp_cpu_epoch_func get_kernel(SSSPCPU ker) {
    switch (ker) {
        case SSSPCPU::one_to_one: return epoch_sssp_pull_cpu_one_to_one;
        case SSSPCPU::undefined:
        default:                  return nullptr;
    }
}

std::ostream &operator<<(std::ostream &os, SSSPCPU ker) {
    os << to_string(ker);
    return os;
}

#endif // SRC_KERNELS_CPU__KERNEL_SSSP_CUH
