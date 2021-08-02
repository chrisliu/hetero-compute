/**
 * CPU implementations of BFS kernels.
 */

#ifndef SRC_KERNELS_CPU__BFS_H
#define SRC_KERNELS_CPU__BFS_H

#include "../../bitmap.cuh"
#include "../../graph.h"
#include "../../util.h"
#include "../../window.h"

/** Forward decl. */
void epoch_bfs_push_one_to_one(const CSRUWGraph &g, nid_t * const parents,
        SlidingWindow<nid_t> &frontier, nid_t &num_edges);
void epoch_bfs_pull_one_to_one(const CSRUWGraph &g, nid_t * const parents,
        Bitmap::Bitmap * const next_frontier,
        const nid_t start_id, const nid_t end_id,
        nid_t &num_nodes);
void conv_bitmap_to_window(
        const Bitmap::Bitmap * const bitmap, SlidingWindow<nid_t> &window, 
        const nid_t num_nodes);

/*****************************************************************************
 ***** BFS Kernels ***********************************************************
 *****************************************************************************/

/**
 * Runs Scott Beamer's Direction Optimizing BFS kernel on CPU in parallel.
 * Parameters:
 *   - g           <- graph.
 *   - source_id   <- starting node id.
 *   - ret_parents <- pointer to the address of the return parents array.
 *   - alpha       <- alpha parameter that determines PUSH->PULL.
 *   - beta        <- beta parameter that determines PULL->PUSH.
 * Returns:
 *   Execution time in milliseconds.
 */
double bfs_do_cpu(
        const CSRUWGraph &g, const nid_t source_id, nid_t ** const ret_parents,
        const int alpha = 15, const int beta = 18
) {
    // Set parents array.
    nid_t *parents = new nid_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++) 
        parents[i] = INVALID_NODE;
    parents[source_id] = source_id;

    // Set push & pull frontiers.
    nid_t iters = 0;
    SlidingWindow<nid_t> push_frontier(g.num_nodes);
    push_frontier.push_back(source_id); push_frontier.slide_window();
    Bitmap::Bitmap *pull_frontier = new Bitmap::Bitmap;
    Bitmap::constructor(pull_frontier, g.num_nodes);

    nid_t edges_to_check = g.num_edges;
    nid_t num_edges = g.get_degree(source_id);

    Timer timer; timer.Start();
    while (not push_frontier.empty()) {
        if (num_edges > edges_to_check / alpha) {
            nid_t num_nodes = push_frontier.size();
            nid_t prev_num_nodes;
            do {
                prev_num_nodes = num_nodes;
                Bitmap::reset(pull_frontier);
                epoch_bfs_pull_one_to_one(g, parents, pull_frontier,
                        0, g.num_nodes, num_nodes);                
                std::cout << "(Pull) Iter " << iters << " num nodes: " <<
                    num_nodes << std::endl;
            } while (num_nodes > prev_num_nodes 
                    or num_nodes > g.num_nodes / beta);
            num_edges = 1;
            conv_bitmap_to_window(pull_frontier, push_frontier, g.num_nodes);
        } else {
            edges_to_check -= num_edges;
            epoch_bfs_push_one_to_one(g, parents, push_frontier, num_edges);        
            push_frontier.slide_window();
            std::cout << "(Push) Iter " << iters << " num edges: " << num_edges 
                << std::endl;
        }
        iters++;
    }
    timer.Stop();

    // Assign output.
    *ret_parents = parents;

    // Free memory.
    Bitmap::destructor(&pull_frontier);

    // Setup parents array.
    return timer.Millisecs();
}


/*****************************************************************************
 ***** BFS Epoch Kernels *****************************************************
 *****************************************************************************/

/**
 * Runs BFS push on CPU for one epoch.
 * Parameters:
 *   - g         <- graph.
 *   - parents   <- node parent list.
 *   - frontier  <- BFS frontier.
 *   - num_edges <-  number of edges goung out of the next frontier.
 */
void epoch_bfs_push_one_to_one(
        const CSRUWGraph &g, nid_t * const parents,
        SlidingWindow<nid_t> &frontier, nid_t &num_edges
) {
    num_edges = 0;
    #pragma omp parallel 
    {
        LocalWindow<nid_t> local_frontier(frontier);
        nid_t local_num_edges = 0;
    
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
                    local_num_edges += g.get_degree(nei);
                }
            }
        }

        // Update global frontier.
        local_frontier.flush();

        // Push update count.
        #pragma omp atomic
        num_edges += local_num_edges;
    }
}

/**
 * Runs BFS pull on CPU for one epoch for a particular range of nodes.
 * Parameters:
 *   - g             <- graph.
 *   - parents       <- node parent list.
 *   - next_frontier <- nodes in the next frontier.
 *   - start_id      <- starting node id.
 *   - end_id        <- ending node id.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
void epoch_bfs_pull_one_to_one(
        const CSRUWGraph &g, nid_t * const parents,
        Bitmap::Bitmap * const next_frontier,
        const nid_t start_id, const nid_t end_id,
        nid_t &num_nodes
) {
    num_nodes = 0;
    #pragma omp parallel
    {
        nid_t local_num_nodes = 0;

        #pragma omp for nowait
        for (nid_t nid = start_id; nid < end_id; nid++) {
            // If current node hasn't been explored.
            if (parents[nid] == INVALID_NODE) {
                for (nid_t nei : g.get_neighbors(nid)) {
                    // If parent has been explored, it's valid.
                    if (parents[nei] != INVALID_NODE) {
                        parents[nid] = nei;
                        Bitmap::set_bit(next_frontier, nid);
                        local_num_nodes++;
                        break; // Early exit.
                    }
                }
            }
        }

        #pragma omp atomic
        num_nodes += local_num_nodes;
    }
}

/** 
 * Convert bitmap frontier into sliding window frontier.
 * Parameters:
 *   - bitmap    <- input bitmap frontier.
 *   - frontier  <- output bitmap frontier.
 *   - num_nodes <- number of nodes in graph.
 */
void conv_bitmap_to_window(
        const Bitmap::Bitmap * const bitmap, SlidingWindow<nid_t> &window, 
        const nid_t num_nodes
) {
    #pragma omp parallel
    {
        LocalWindow<nid_t> local_window(window);
        
        #pragma omp for nowait
        for (int nid = 0; nid < num_nodes; nid++)
            if (Bitmap::get_bit(bitmap, nid))
                local_window.push_back(nid);

        local_window.flush();
    }

    // Implicit OMP barrier at end of of OMP parallel construct.
    window.slide_window();
}

#endif // SRC_KERNELS_CPU__BFS_H
