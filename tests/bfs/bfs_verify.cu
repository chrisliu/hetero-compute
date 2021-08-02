#include <cstdlib>
#include <iostream>

#include "../../src/graph.h"
#include "../../src/kernels/cpu/bfs.cuh"

/** Forward decl. */
void reset_parents(nid_t *parents, nid_t num_nodes, nid_t source_id);

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

    nid_t *parents = nullptr;

    double exec_time = bfs_do_cpu(g, source_id, &parents);
    delete[] parents;
    std::cout << "Kernel completed in " << exec_time << " ms." << std::endl;

    /*nid_t *parents = new nid_t[g.num_nodes];*/
    /*reset_parents(parents, g.num_nodes, source_id);*/
    /*nid_t updated = 0;*/

    /*int iter = 0;*/
    /*do {*/
        /*updated = 0;*/
        /*epoch_bfs_push_one_to_one(g, parents, frontier, updated);*/
        /*std::cout << "Iter " << iter << ": " << updated << std::endl;*/
        /*frontier.slide_window();*/
        /*iter++;*/
    /*} while (updated != 0);*/

    /*delete[] parents;*/

    return EXIT_SUCCESS;
}

/**
 * Reset parents array. Set source ID's parent to itself.
 * Parameters:
 *   - parents   <- node parent list.
 *   - num_nodes <- number of nodes in the graph.
 *   - source_id <- starting node id.
 */
__inline__
void reset_parents(nid_t *parents, nid_t num_nodes, nid_t source_id) {
    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
        parents[i] = INVALID_NODE;
    parents[source_id] = source_id;
}
