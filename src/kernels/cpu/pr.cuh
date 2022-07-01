/**
 * CPU implementations of SSSP pull kernels.
 */

#ifndef SRC_KERNELS_CPU__KERNEL_PR_CUH
#define SRC_KERNELS_CPU__KERNEL_PR_CUH

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
 * Runs PR kernel on CPU in parallel. Synchronization occurs in serial.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch kernel.
 *   - init_score    <- initial score array.
 *   - ret_score     <- pointer to the address of the return score array.
 * Returns:
 *   Execution time in milliseconds.
 */
double pr_pull_cpu(
        const CSRWGraph &g, pr_cpu_epoch_func epoch_kernel, 
        const weight_t *init_score, weight_t ** const ret_score
) {
    // Setup computed scores.
    weight_t *score = new weight_t[g.num_nodes];
    weight_t init_score = 1.0f / g.num_nodes;
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        score[i] = init_score;

    nid_t updated = 1;

    // Start kernel!
    Timer timer; timer.Start();
    while (updated != 0) {
        updated = 0;

        #pragma omp parallel
        {
            (*epoch_kernel)(g, score, 0, g.num_nodes, omp_get_thread_num(), 
                    omp_get_num_threads(), updated);
        }
        // Implicit OMP BARRIER here (see "implicit barrier at end of 
        // parallel region").
    }
    timer.Stop();

    // Assign output.
    *ret_score = score;

    return timer.Millisecs();
}

/**
 * Run SSSP kernel on CPU in serial.
 * Parameters:
 *   - g         <- graph.
 *   - source_id <- initial point. 
 *   - ret_score  <- pointer to the address of the return score array.
 */
void pr_dijkstras_cpu_serial(
        const CSRWGraph &g, nid_t source_id, weight_t **ret_score
) {
    // Setup.
    weight_t *score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        score[i] = INF_WEIGHT;
    score[source_id] = 0.0f;

    using wnode_pair_t = std::pair<weight_t, nid_t>;
    std::priority_queue<wnode_pair_t, std::vector<wnode_pair_t>, 
        std::greater<wnode_pair_t>> frontier;
    frontier.push(std::make_pair(0.0f, source_id));

    while (not frontier.empty()) {
        weight_t weight = frontier.top().first;
        nid_t    nid    = frontier.top().second;
        frontier.pop();

        // If this is the latest score.
        if (weight == score[nid]) {
            for (wnode_t nei : g.get_neighbors(nid)) {
                weight_t new_weight = weight + nei.w;

                // If possible, update weight.
                if (new_weight < score[nei.v]) {
                    score[nei.v] = new_weight;
                    frontier.push(std::make_pair(new_weight, nei.v));
                }
            }
        }
    }

    *ret_score = score;
}

void pr_pull_cpu_serial(
        const CSRWGraph &g, nid_t source_id, weight_t **ret_score
) {
    // Setup.
    weight_t *score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        score[i] = INF_WEIGHT;
    score[source_id] = 0.0f;

    nid_t num_updates;
    do {
        num_updates = 0;

        for (nid_t v = 0; v < g.num_nodes; v++) {
            for (wnode_t nei : g.get_neighbors(v)) {
                weight_t prop_score = score[nei.v] + nei.w;
                if (prop_score < score[v]) {
                    score[v] = prop_score;
                    num_updates++;
                }
            }
        }
    } while (num_updates != 0);

    *ret_score = score;
}

const float kDamp = 0.85;

/*****************************************************************************
 ***** Epoch Kernels *********************************************************
 *****************************************************************************/

/**
 * Runs SSSP pull on CPU for one epoch.
 * Parameters:
 *   - g           <- graph.
 *   - score        <- input scores and output scores computed this 
 *                    epoch.
 *   - start_id    <- starting node id.
 *   - end_id      <- ending node id (exclusive).
 *   - tid         <- processor id.
 *   - num_threads <- number of processors.
 *   - updated     <- global counter of number of nodes updated.
 */
void epoch_pr_pull_cpu_one_to_one(
        const CSRWGraph &g, weight_t *score, 
        const nid_t start_id, const nid_t end_id, 
        const int tid, const int num_threads, nid_t &updated
) {

    float epsilon=0.00000005;
    nid_t local_updated = 0;
    weight_t base_score = (1.0f - kDamp) / g.num_nodes;

    // Propagate, reduce, and apply.
    for (int nid = start_id + tid; nid < end_id; nid += num_threads) {
	weight_t incoming_total=0;

        // Find shortest candidate score.
        for (wnode_t nei : g.get_neighbors(nid)) {
	    incoming_total+=score[nei.v]/g.get_degree(nei.v);
        }
	weight_t new_score=base_score + kDamp * incoming_total;

        // Update score if applicable
        if (abs(new_score-score[nid])>epsilon) {
            score[nid] = new_score;
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
enum class PRCPU {
    one_to_one, undefined
};

/** List of kernels available (no good iterator for enum classes). */
std::vector<PRCPU> pr_cpu_kernels = {
    PRCPU::one_to_one
};

std::vector<PRCPU> get_kernels(UNUSED PRCPU unused) {
    // Using hack to overload function by return type.
    return pr_cpu_kernels;
}

/** 
 * Convert epoch kernel ID to its representation name (not as human-readable).
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_repr(PRCPU ker) {
    switch (ker) {
        case PRCPU::one_to_one: return "pr_cpu_onetoone";
        case PRCPU::undefined:
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
std::string to_string(PRCPU ker) {
    switch (ker) {
        case PRCPU::one_to_one: return "PR CPU one-to-one";
        case PRCPU::undefined:
        default:                  return "undefined PR CPU kernel";
    }
}

/**
 * Convert epoch kernel ID to kernel function pointer.
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel function pointer.
 */
pr_cpu_epoch_func get_kernel(PRCPU ker) {
    switch (ker) {
        case PRCPU::one_to_one: return epoch_pr_pull_cpu_one_to_one;
        case PRCPU::undefined:
        default:                  return nullptr;
    }
}

std::ostream &operator<<(std::ostream &os, PRCPU ker) {
    os << to_string(ker);
    return os;
}

#endif // SRC_KERNELS_CPU__KERNEL_PR_CUH
