#include <cstdlib>
#include <iostream>

#include "../../src/graph.h"
#include "../../src/kernels/cpu/sssp_pull.h"
#include "../../src/kernels/gpu/sssp_pull.cuh"

/** Forward decl. */
bool verify(const weight_t *oracle_dist, const weight_t *dist, 
        const nid_t num_nodes);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.wsg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load graph.
    CSRWGraph g = load_graph_from_file(argv[1]);
    std::cout << "Loaded graph." << std::endl;

    const nid_t source_id = 0; // Arbitrarily set the highest degree node as
                               // the source node.

    // Compute oracle distance.
    weight_t *oracle_dist = nullptr;
    sssp_pull_cpu_serial(g, source_id, &oracle_dist);
    std::cout << "Computed oracle distance." << std::endl;

    // Generate init distances.
    weight_t *init_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_dist[i] = INF_WEIGHT;
    init_dist[source_id] = 0.0f;

    // Check SSSP CPU kernel.
    {
        weight_t *dist = nullptr;
        sssp_pull_cpu(g, epoch_sssp_pull_cpu, init_dist, &dist);

        std::cout << "Verifying SSSP CPU kernel ..." << std::endl;
        bool success = verify(oracle_dist, dist, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] dist;
    }

    // Check SSSP GPU naive kernel.
    {
        weight_t *dist = nullptr;
        sssp_pull_gpu(g, epoch_sssp_pull_gpu_naive, init_dist, &dist);

        std::cout << "Verifying SSSP GPU naive kernel ..." << std::endl;
        bool success = verify(oracle_dist, dist, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] dist;
    }

    // Check SSSP GPU warp min kernel.
    {
        weight_t *dist = nullptr;
        sssp_pull_gpu(g, epoch_sssp_pull_gpu_warp_min, init_dist, &dist);

        std::cout << "Verifying SSSP GPU warp min kernel ..." << std::endl;
        bool success = verify(oracle_dist, dist, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] dist;
    }

    // Check SSSP GPU block min kernel.
    {
        weight_t *dist = nullptr;
        sssp_pull_gpu(g, epoch_sssp_pull_gpu_block_min, init_dist, &dist);

        std::cout << "Verifying SSSP GPU block min kernel ..." << std::endl;
        bool success = verify(oracle_dist, dist, g.num_nodes);
        std::cout << " > Verification " << (success ? "succeeded" : "failed")
            << "!" << std::endl;

        delete[] dist;
    }

    return EXIT_SUCCESS;
}

/**
 * Verifies that the computed distances is the same as the oracle distance.
 * Parameters:
 *   - oracle_dist <- correct distance.
 *   - dist        <- computed distance (distance to check).
 *   - num_nodes   <- number of nodes in the graph.
 * Returns:
 *   true if the computed distance is correct.
 */
bool verify(const weight_t *oracle_dist, const weight_t *dist,
        const nid_t num_nodes
) {
    bool is_correct = true;

    for (nid_t nid = 0; nid < num_nodes; nid++) {
        if (oracle_dist[nid] != dist[nid]) {
            std::cout << " > " << nid << ": " 
                << dist[nid] << " != " << oracle_dist[nid] << std::endl;
            is_correct = false;
        }
    }

    return is_correct;
}
