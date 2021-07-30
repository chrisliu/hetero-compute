/**
 * CPU implementations of BFS kernels.
 */

#ifndef SRC_KERNELS_CPU__BFS_H
#define SRC_KERNELS_CPU__BFS_H

#include "../../graph.h"
#include "../../window.h"

/*****************************************************************************
 ***** BFS Epoch Kernels *****************************************************
 *****************************************************************************/

/**
 * Runs BFS push on CPU for one epoch.
 * Parameters:
 *   - g        <- graph.
 *   - parents  <- node parent list.
 *   - frontier <- BFS frontier.
 *   - updated  <- number of nodes updated.
 */
void epoch_bfs_push_one_to_one(
        const CSRUWGraph &g, nid_t *parents,
        SlidingWindow<nid_t> &frontier, nid_t &updated
) {
    #pragma omp parallel 
    {
        LocalWindow<nid_t> local_frontier(frontier);
        nid_t local_updated = 0;
    
        #pragma omp for nowait
        for (auto q_elem = frontier.begin(); q_elem < frontier.end(); 
                q_elem++
        ) {
            nid_t nid = *q_elem;
            for (nid_t nei : g.get_neighbors(nid)) {
                nid_t cur_parent = parents[nei];

                // If parent is not set and hasn't been set between
                // instructions, update.
                if (cur_parent == INVALID_NODE and 
                        __sync_val_compare_and_swap(&parents[nei], cur_parent, nid)
                ) {
                    local_frontier.push_back(nei);
                    local_updated++;
                }
            }
        }

        // Update global frontier.
        local_frontier.flush();

        // Push update count.
        #pragma omp atomic
        updated += local_updated;
    }
}

/**
 * Runs BFS pull on CPU for one epoch for a particular range of nodes.
 * Parameters:
 *   - g        <- graph.
 *   - parents  <- node parent list.
 *   - start_id <- starting node id.
 *   - end_id   <- ending node id.
 *   - updated  <- number of nodes updated.
 */
void epoch_bfs_pull_one_to_one(
        const CSRUWGraph &g, nid_t *parents,
        const nid_t start_id, const nid_t end_id,
        nid_t &updated
) {
    #pragma omp parallel
    {
        nid_t local_updated = 0;

        #pragma omp for nowait
        for (nid_t nid = start_id; nid < end_id; nid++) {
            // If current node hasn't been explored.
            if (parents[nid] == INVALID_NODE) {
                for (nid_t nei : g.get_neighbors(nid)) {
                    // If parent has been explored, it's valid.
                    if (parents[nei] != INVALID_NODE) {
                        parents[nid] = nei;
                        local_updated++;
                        break; // Early exit.
                    }
                }
            }
        }

        #pragma omp atomic
        updated += local_updated;
    }
}

#endif // SRC_KERNELS_CPU__BFS_H
