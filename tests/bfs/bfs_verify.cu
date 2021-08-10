// Maximum number of errors to print out.
#define MAX_PRINT_ERRORS 10

#include <cstdlib>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>

#include "../../src/graph.h"
#include "../../src/kernels/cpu/bfs.cuh"
#include "../../src/kernels/gpu/bfs.cuh"

/** Forward decl. */
void verify(const CSRUWGraph &g, 
        const nid_t * const depths, const nid_t source_id,
        std::string kernel_name, std::function<nid_t *(void)> get_parents);
bool verify_parents(const nid_t * const depths, const nid_t * const parents,
        const nid_t num_nodes, const nid_t source_id);
nid_t *compute_proper_depth(const CSRUWGraph &g, const nid_t source_id);
void reset_parents(nid_t * const parents, const nid_t num_nodes, 
        const nid_t source_id);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.sg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load graph.
    CSRUWGraph g = load_graph_from_file<CSRUWGraph>(argv[1]);
    std::cout << "Loaded graph." << std::endl;

    SourcePicker<CSRUWGraph> sp(&g);
    nid_t source_id = sp.next_vertex();

    // Compute expected depths.
    nid_t *depths = compute_proper_depth(g, source_id);
    std::cout << "Computed oracle depths." << std::endl;

    /*// Check BFS CPU push kernel.*/
    /*verify(g, depths, source_id, "BFS CPU push", */
            /*[&](){*/
                /*nid_t *parents = nullptr;*/
                /*bfs_push_cpu(g, source_id, &parents);*/
                /*return parents;*/
            /*}*/
    /*);*/

    /*// Check BFS CPU pull kernel.*/
    /*verify(g, depths, source_id, "BFS CPU push",*/
            /*[&](){*/
                /*nid_t *parents = nullptr;*/
                /*bfs_pull_cpu(g, source_id, &parents);*/
                /*return parents;*/
            /*}*/
    /*);*/

    /*// Check BFS DO kernel.*/
    /*verify(g, depths, source_id, "BFS CPU DO",*/
            /*[&](){*/
                /*nid_t *parents = nullptr;*/
                /*bfs_do_cpu(g, source_id, &parents);*/
                /*return parents;*/
            /*}*/
    /*);*/

    // Check BFS GPU one-to-one kernel.
    verify(g, depths, source_id, "BFS GPU one-to-one",
            [&](){
                nid_t *parents = nullptr;
                bfs_gpu(g, source_id, &parents, epoch_bfs_pull_gpu_one_to_one);
                return parents;
            }
    );

    delete[] depths;

    return EXIT_SUCCESS;
}

void verify(const CSRUWGraph &g, 
        const nid_t * const depths, const nid_t source_id,
        std::string kernel_name, std::function<nid_t *(void)> get_parents
) {
    nid_t *parents = get_parents();

    std::cout << "Verifying " << kernel_name << " kernel ..." << std::endl;
    if(verify_parents(depths, parents, g.num_nodes, source_id))
        std::cout << " > Verification success!" << std::endl;
}

bool verify_parents(const nid_t * const depths, const nid_t * const parents,
        const nid_t num_nodes, const nid_t source_id
) {
    bool is_correct = true;
    nid_t error_count = 0;

    for (nid_t u = 0; u < num_nodes; u++) {
        // If is unexplored node, make sure ut's actually supposed to be 
        // unexplored.
        if (parents[u] == INVALID_NODE) {
            if (depths[u] != INVALID_NODE) {
                if (error_count < MAX_PRINT_ERRORS)
                    std::cout << " > " << u << ": "
                        << parents[u] << " != " << depths[u] << std::endl;
                is_correct = false;
                error_count++;
            }
        // If source node, check parent equals utself.
        } else if (u == source_id) {
            if (parents[u] != source_id) {
                if (error_count < MAX_PRINT_ERRORS)
                    std::cout << " > " << u << ": "
                        << parents[u] << " != " << source_id << std::endl;
                is_correct = false;
                error_count++;
            }
        // If is explored node, make sure it's at the correct depth.
        } else {
            if (depths[u] != depths[parents[u]] + 1) {
                if (error_count < MAX_PRINT_ERRORS)
                    std::cout << " > " << u << ": "
                        << depths[u] << " != " << depths[parents[u]] << " + 1"
                        << std::endl;
                is_correct = false;
                error_count++;
            }
        }
    }

    // Print extra error count if any.
    if (error_count >= MAX_PRINT_ERRORS) {
        nid_t more_error_count = error_count - MAX_PRINT_ERRORS;
        std::cout << " > ... " << more_error_count << " more error"
            << (more_error_count != 1 ? "s" : "") << "!" << std::endl;
    }

    return is_correct;
}

/**
 * Compute and return BFS frontier depths.
 * Source node's depth is 0. Source node's neighbors' depth is 0 + 1, etc.
 * Remaining nodes that don't belong to the source node's component is 
 * set to INVALID_NODE.
 * 
 * Parameters:
 *   - g         <- graph.
 *   - source_id <- source node id.
 * Returns:
 *    BFS frontier depth for each node (heap allocated).
 */
nid_t *compute_proper_depth(const CSRUWGraph &g, const nid_t source_id) {
    // Initialize depths array.
    nid_t *depths = new nid_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        depths[i] = INVALID_NODE;
    depths[source_id] = 0;

    // Compute depths.
    std::deque<nid_t> frontier = { source_id };
    while (not frontier.empty()) {
        nid_t u = frontier.front(); frontier.pop_front();

        for (nid_t v : g.get_neighbors(u)) {
            if (depths[v] == INVALID_NODE) {
                depths[v] = depths[u] + 1;
                frontier.push_back(v);
            }
        }
    }

    return depths;
}

