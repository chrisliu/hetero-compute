/**
 * Implementations of CPU and GPU SSSP pull with data race.
 */

#include <cstdlib>
#include <cstring> // memcmp

#include "gapbs.h"
#include "util.h"

/** Kernels */
#include "sssp_pull_cpu.h"
#include "sssp_pull_gpu.cuh"

/******************************************************************************
 ***** Main Function **********************************************************
 ******************************************************************************/

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);

    // Run SSSP.
    weight_t *cpu_distances = nullptr;
    sssp_pull_cpu(ordered_g, &cpu_distances);
    weight_t *gpu_distances = nullptr;
    sssp_pull_gpu(ordered_g, &gpu_distances);

    bool is_dist_same = !std::memcmp(cpu_distances, gpu_distances, 
            g.num_nodes() * sizeof(weight_t));
    std::cout << "CPU distances " << (is_dist_same ? "==" : "!=")
        << " GPU distances" << std::endl;

    if (cli.scale() <= 4) {
        std::cout << "node neighbors" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++) {
            std::cout << " > node " << i << std::endl;
            for (auto &out_nei : ordered_g.out_neigh(i)) {
                std::cout << "    > node " << out_nei.v << ": " << out_nei.w
                    << std::endl;
            }
        }

        std::cout << "node: distance" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++)
            std::cout << " > " << i << ": " << cpu_distances[i] << std::endl;
    }

    // Free memory.
    delete[] cpu_distances;
    delete[] gpu_distances;

    return EXIT_SUCCESS;
}
