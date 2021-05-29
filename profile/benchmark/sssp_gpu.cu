#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../../src/gapbs.h"
#include "../../src/benchmark.cuh"

#define DEPTH 3

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);

    // Run warp min kernel.
    {
        SSSPGPUBenchmark bench(&ordered_g, sssp_pull_gpu_warp_min);
        
        tree_res_t tree_res_warp_min = bench.tree_microbenchmark(DEPTH);

        std::cout << tree_res_warp_min;
        
        // Write out results.
        /*std::fstream ofs("results.yaml", std::ofstream::out);*/
        /*ofs << tree_res_warp_min;*/
        /*ofs.close();*/
    }

    // Run naive kernel.
    {
        SSSPGPUBenchmark bench(&ordered_g, sssp_pull_gpu_naive);
        tree_res_t tree_res_naive = bench.tree_microbenchmark(DEPTH);
        
        std::cout << tree_res_naive;
    }

    weight_t *ret_dist = nullptr;
    sssp_pull_gpu(ordered_g, sssp_pull_gpu_warp_min, &ret_dist); delete[] ret_dist;
    sssp_pull_gpu(ordered_g, sssp_pull_gpu_naive, &ret_dist); delete[] ret_dist;

    return EXIT_SUCCESS;
}
