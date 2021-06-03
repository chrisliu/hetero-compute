#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/benchmarks/gpu_benchmark.cuh"
#include "../../src/kernels/cpu/sssp_pull.h"
#include "../../src/kernels/gpu/sssp_pull.cuh"

#define DEPTH 3

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.wsg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load in graph.
    CSRWGraph g;
    std::cout << "Loading graph ..." << std::endl;

    std::ifstream ifs(argv[1], std::ifstream::in | std::ifstream::binary);
    Timer timer; timer.Start(); // Start timing.
    ifs >> g;
    timer.Stop(); // Stop timing.
    ifs.close();

    std::cout << " > Loaded in " << timer.Millisecs() << " ms." << std::endl;

    // TODO: reuse benchmark object.

    // Run warp min kernel.
    {
        SSSPGPUBenchmark bench(&g, sssp_pull_gpu_warp_min);
        
        tree_res_t res = bench.tree_microbenchmark(DEPTH);

        std::cout << res;
    }

    // Run naive kernel.
    {
        SSSPGPUBenchmark bench(&g, sssp_pull_gpu_naive);
        
        tree_res_t res = bench.tree_microbenchmark(DEPTH);

        std::cout << res;
    }

    weight_t *ret_dist = nullptr;
    sssp_pull_cpu(g, &ret_dist); delete[] ret_dist;
    sssp_pull_gpu(g, sssp_pull_gpu_naive, &ret_dist); delete[] ret_dist;
    sssp_pull_gpu(g, sssp_pull_gpu_warp_min, &ret_dist); delete[] ret_dist;

    return EXIT_SUCCESS;
}
