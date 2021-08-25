/**
 * CPU implementations of BFS kernels.
 */

#ifndef SRC_KERNELS_CPU__BFS_CUH
#define SRC_KERNELS_CPU__BFS_CUH

#include <algorithm>
#include <vector>

#include "../kernel_types.cuh"
#include "../../bitmap.cuh"
#include "../../graph.cuh"
#include "../../util.h"
#include "../../window.h"

/** Forward decl. */
void epoch_bfs_push_cpu_by_node(const CSRUWGraph &g, nid_t * const parents,
        SlidingWindow<nid_t> &frontier, nid_t &num_edges);
void epoch_bfs_pull_cpu(const CSRUWGraph &g, nid_t * const parents,
        const nid_t start_id, const nid_t end_id,
        const Bitmap::Bitmap * const frontier,
        Bitmap::Bitmap * const next_frontier,
        nid_t &num_nodes);
void conv_window_to_bitmap(
        const SlidingWindow<nid_t> &window, Bitmap::Bitmap * const frontier);
void conv_bitmap_to_window(
        const Bitmap::Bitmap * const bitmap, SlidingWindow<nid_t> &window, 
        const nid_t num_nodes);
void reset_parents(nid_t * const parents, const nid_t num_nodes, 
        const nid_t source_id);

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
template <int alpha = 15, int beta = 18>
double bfs_do_cpu(
        const CSRUWGraph &g, const nid_t source_id, nid_t ** const ret_parents
) {
    // Set parents array.
    nid_t *parents = new nid_t[g.num_nodes];
    reset_parents(parents, g.num_nodes, source_id);

    // Set push & pull frontiers.
    SlidingWindow<nid_t> push_frontier(g.num_nodes);
    push_frontier.push_back(source_id); push_frontier.slide_window();

    Bitmap::Bitmap *pull_frontier      = Bitmap::constructor(g.num_nodes);
    Bitmap::Bitmap *pull_next_frontier = Bitmap::constructor(g.num_nodes);

    nid_t edges_to_check = g.num_edges;
    nid_t num_edges = g.get_degree(source_id);

    nid_t iters = 0;
    Timer timer; timer.Start();
    while (not push_frontier.empty()) {
        // If PULL.
        if (num_edges > edges_to_check / alpha) {
            nid_t prev_num_nodes;
            nid_t num_nodes = push_frontier.size();
            conv_window_to_bitmap(push_frontier, pull_frontier);

            // Run pull epochs.
            do {
                prev_num_nodes = num_nodes;
                num_nodes = 0;

                epoch_bfs_pull_cpu(g, parents, 0, g.num_nodes,
                        pull_frontier, pull_next_frontier, num_nodes);               
                /*std::cout << "(Pull) Iter " << iters << " num nodes: " <<*/
                    /*num_nodes << std::endl;*/

                std::swap(pull_frontier, pull_next_frontier);
                Bitmap::reset(pull_next_frontier);
                iters++;
            } while (num_nodes > prev_num_nodes
                    or num_nodes > g.num_nodes / beta);

            num_edges = 1;
            conv_bitmap_to_window(pull_frontier, push_frontier, g.num_nodes);
        // If PUSH.
        } else {
            edges_to_check -= num_edges;
            num_edges = 0;
            epoch_bfs_push_cpu_by_node(g, parents, push_frontier, num_edges);       
            push_frontier.slide_window();
            /*std::cout << "(Push) Iter " << iters << " num edges: " << num_edges*/
                /*<< std::endl;*/
            iters++;
        }
    }
    timer.Stop();

    // Assign output.
    *ret_parents = parents;

    // Free memory.
    Bitmap::destructor(&pull_frontier);
    Bitmap::destructor(&pull_next_frontier);

    return timer.Millisecs();
}

/**
 * Runs BFS push kernel on CPU in parallel.
 * Parameters:
 *   - g           <- graph.
 *   - source_id   <- starting node id.
 *   - ret_parents <- pointer to the address of the return parents array.
 * Returns:
 *    Execution time in milliseconds.
 */
template <bfs_cpu_push_epoch_func push_epoch_func>
double bfs_push_cpu(
        const CSRUWGraph &g, const nid_t source_id, nid_t ** const ret_parents
) {
    // Set parents array.
    nid_t *parents = new nid_t[g.num_nodes];
    reset_parents(parents, g.num_nodes, source_id);

    // Set push frontier.
    SlidingWindow<nid_t> frontier(g.num_nodes);
    frontier.push_back(source_id); frontier.slide_window();

    nid_t num_edges;

    Timer timer; timer.Start();
    while (not frontier.empty()) {
        num_edges = 0;
        (*push_epoch_func)(g, parents, frontier, num_edges);
        frontier.slide_window();
    }
    timer.Stop();

    // Assign output.
    *ret_parents = parents;

    return timer.Millisecs();
}

/**
 * Runs BFS pull kernel on CPU in parallel.
 * Parameters:
 *   - g           <- graph.
 *   - source_id   <- starting node id.
 *   - ret_parents <- pointer to the address of the return parents array.
 * Returns:
 *    Execution time in milliseconds.
 */
double bfs_pull_cpu(
        const CSRUWGraph &g, const nid_t source_id, nid_t ** const ret_parents
) {
    // Set parents array.
    nid_t *parents = new nid_t[g.num_nodes];
    reset_parents(parents, g.num_nodes, source_id);

    // Set pull frontier.
    Bitmap::Bitmap *frontier = Bitmap::constructor(g.num_nodes);
    Bitmap::set_bit(frontier, source_id); // Set source node.

    Bitmap::Bitmap *next_frontier = Bitmap::constructor(g.num_nodes);

    nid_t num_nodes;
    
    Timer timer; timer.Start();
    do {
        num_nodes = 0;
        epoch_bfs_pull_cpu(g, parents, 0, g.num_nodes, frontier, 
                next_frontier, num_nodes);       

        std::swap(frontier, next_frontier);
        Bitmap::reset(next_frontier);
    } while (num_nodes != 0);
    timer.Stop();

    // Assign output.
    *ret_parents = parents;

    // Free memory.
    Bitmap::destructor(&frontier);
    Bitmap::destructor(&next_frontier);
    
    return timer.Millisecs();
}

/*****************************************************************************
 ***** BFS Epoch Kernels *****************************************************
 *****************************************************************************/

/**
 * Runs BFS push on CPU for one epoch.
 * Parallel over all nodes.
 *
 * Parameters:
 *   - g         <- graph.
 *   - parents   <- node parent list.
 *   - frontier  <- BFS frontier.
 *   - num_edges <-number of edges goung out of the next frontier.
 */
void epoch_bfs_push_cpu_by_node(
        const CSRUWGraph &g, nid_t * const parents,
        SlidingWindow<nid_t> &frontier, nid_t &num_edges
) {
    #pragma omp parallel
    {
        LocalWindow<nid_t> local_frontier(frontier);
        nid_t local_num_edges = 0;
    
        #pragma omp for nowait
        for (const nid_t nid : frontier) {
            for (nid_t nei : g.get_neighbors(nid)) {
                nid_t cur_parent = parents[nei];

                // If parent is not set and hasn't been set between
                // instructions, update.
                if (cur_parent == INVALID_NODE 
                        and __sync_val_compare_and_swap(&parents[nei], cur_parent, nid)
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
 * Runs BFS push on CPU for one epoch.
 * Parallel over all edges.
 *
 * Parameters:
 *   - g         <- graph.
 *   - parents   <- node parent list.
 *   - frontier  <- BFS frontier.
 *   - num_edges <-number of edges goung out of the next frontier.
 */
void epoch_bfs_push_cpu_by_edge(
        const CSRUWGraph &g, nid_t * const parents,
        SlidingWindow<nid_t> &frontier, nid_t &num_edges
) {
    #pragma omp parallel
    {
        LocalWindow<nid_t> local_frontier(frontier);
        nid_t local_num_edges = 0;
    
        for (const nid_t nid : frontier) {
            #pragma omp for nowait
            for (nid_t nei : g.get_neighbors(nid)) {
                nid_t cur_parent = parents[nei];

                // If parent is not set and hasn't been set between
                // instructions, update.
                if (cur_parent == INVALID_NODE 
                        and __sync_val_compare_and_swap(&parents[nei], cur_parent, nid)
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
 * 
 * Note: Bitmap is not set atomically as long as OMP schedule blocks are
 *       aligned to the size of a bitmap's internal data type.
 *
 * Parameters:
 *   - g             <- graph.
 *   - parents       <- node parent list.
 *   - frontier      <- current frontier.
 *   - next_frontier <- nodes in the next frontier.
 *   - start_id      <- starting node id.
 *   - end_id        <- ending node id.
 *   - num_nodes     <- number of nodes in the next frontier.
 */
void epoch_bfs_pull_cpu(
        const CSRUWGraph &g, nid_t * const parents,
        const nid_t start_id, const nid_t end_id,
        const Bitmap::Bitmap * const frontier,
        Bitmap::Bitmap * const next_frontier,
        nid_t &num_nodes
) {
    #pragma omp parallel
    {
        nid_t local_num_nodes = 0;

        #pragma omp for nowait schedule(dynamic, 1024)
        for (nid_t u = start_id; u < end_id; u++) {
            // If current node hasn't been explored.
            if (parents[u] == INVALID_NODE) {
                for (nid_t nei : g.get_neighbors(u)) {
                    // If parent has been explored, it's valid.
                    if (Bitmap::get_bit(frontier, nei)) {
                        parents[u] = nei;
                        Bitmap::set_bit(next_frontier, u);
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

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

/** Convert sliding window frontier into bitmap frontier.
 * Parameters:
 *   - window <- input sliding window frontier.
 *   - bitmap <- output bitmap frontier.
 */
void conv_window_to_bitmap(
        const SlidingWindow<nid_t> &window, Bitmap::Bitmap * const bitmap
) {
    #pragma omp parallel for
    for (const nid_t u : window)
        Bitmap::set_bit_atomic(bitmap, u);
}

/** 
 * Convert bitmap frontier into sliding window frontier.
 * Parameters:
 *   - bitmap    <- input bitmap frontier.
 *   - frontier  <- output sliding window frontier.
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


/**
 * Reset parents array. Set source ID's parent to itself.
 * Parameters:
 *   - parents   <- node parent list.
 *   - num_nodes <- number of nodes in the graph.
 *   - source_id <- starting node id.
 */
__inline__
void reset_parents(nid_t * const parents, const nid_t num_nodes, 
        const nid_t source_id
) {
    #pragma omp parallel for
    for (nid_t i = 0; i < num_nodes; i++)
        parents[i] = INVALID_NODE;
    parents[source_id] = source_id;
}

/******************************************************************************
 ***** Helper Functions *******************************************************
 ******************************************************************************/

/** Identifier for epoch kernels. */
enum class BFSCPUPush {
    by_node, by_edge, undefined
};
enum class BFSCPUPull {
    pull, undefined
};

/** List of kernels available (no good iterator for enum classes). */
std::vector<BFSCPUPush> bfs_cpu_push_kernels = {
    BFSCPUPush::by_node, BFSCPUPush::by_edge
};

std::vector<BFSCPUPull> bfs_cpu_pull_kernels = {
    BFSCPUPull::pull
};

std::vector<BFSCPUPush> get_kernels(UNUSED BFSCPUPush unused) {
    // Using hack to overload function by return type.
    return bfs_cpu_push_kernels;
}

std::vector<BFSCPUPull> get_kernels(UNUSED BFSCPUPull unused) {
    // Using hack to overload function by return type.
    return bfs_cpu_pull_kernels;
}

/** 
 * Convert epoch kernel ID to its representation name (not as human-readable).
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_repr(BFSCPUPush ker) {
    switch (ker) {
        case BFSCPUPush::by_node:   return "bfs_cpu_push_by_node";
        case BFSCPUPush::by_edge:   return "bfs_cpu_push_by_edge";
        case BFSCPUPush::undefined:
        default:                    return "";
    }
}

/** 
 * Convert epoch kernel ID to its representation name (not as human-readable).
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_repr(BFSCPUPull ker) {
    switch (ker) {
        case BFSCPUPull::pull:      return "bfs_cpu_pull";
        case BFSCPUPull::undefined:
        default:                    return "";
    }
}

/** 
 * Convert epoch kernel ID to its human-readable name. 
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_string(BFSCPUPush ker) {
    switch (ker) {
        case BFSCPUPush::by_node:   return "BFS CPU push by edge";
        case BFSCPUPush::by_edge:   return "BFS CPU push by node";
        case BFSCPUPush::undefined:
        default:                    return "undefined BFS CPU push kernel";
    }
}

/** 
 * Convert epoch kernel ID to its human-readable name. 
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel name.
 */
std::string to_string(BFSCPUPull ker) {
    switch (ker) {
        case BFSCPUPull::pull:      return "BFS CPU pull";
        case BFSCPUPull::undefined:
        default:                    return "undefined BFS CPU pull kernel";
    }
}

/**
 * Convert epoch kernel ID to kernel function pointer.
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel function pointer.
 */
bfs_cpu_push_epoch_func get_kernel(BFSCPUPush ker) {
    switch (ker) {
        case BFSCPUPush::by_node:   return epoch_bfs_push_cpu_by_node;
        case BFSCPUPush::by_edge:   return epoch_bfs_push_cpu_by_edge;
        case BFSCPUPush::undefined:
        default:                    return nullptr;
    }
}

/**
 * Convert epoch kernel ID to kernel function pointer.
 * Parameters:
 *   - ker <- kernel ID.
 * Returns:
 *   kernel function pointer.
 */
bfs_cpu_pull_epoch_func get_kernel(BFSCPUPull ker) {
    switch (ker) {
        case BFSCPUPull::pull:      return epoch_bfs_pull_cpu;
        case BFSCPUPull::undefined:
        default:                    return nullptr;
    }
}

std::ostream &operator<<(std::ostream &os, BFSCPUPush ker) {
    os << to_string(ker);
    return os;
}

std::ostream &operator<<(std::ostream &os, BFSCPUPull ker) {
    os << to_string(ker);
    return os;
}

#endif // SRC_KERNELS_CPU__BFS_CUH
