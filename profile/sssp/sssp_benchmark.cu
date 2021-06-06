#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/benchmarks/gpu_benchmark.cuh"
#include "../../src/benchmarks/cpu_benchmark.h"
#include "../../src/kernels/cpu/sssp_pull.h"
#include "../../src/kernels/gpu/sssp_pull.cuh"

// Only get results for a particular depth (not the whole tree).
#define ONLY_LAYER
// Print results to std::cout.
#define PRINT_RESULTS
// Save results to YAML files.
#define SAVE_RESULTS
// Current/Up to (inclusive) this depth.
#define DEPTH 4

#ifdef SAVE_RESULTS
template <typename ResT>
__inline__
void save_results(std::string filename, ResT &result) {
    std::ofstream ofs(filename, std::ofstream::out);
    ofs << result;
    ofs.close();
}

#endif // SAVE_RESULTS

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

    // Run CPU epoch kernel.
    {
        SSSPCPUTreeBenchmark bench(&g, epoch_sssp_pull_cpu);

#ifdef ONLY_LAYER
        auto res = bench.layer_microbenchmark(DEPTH);
#else
        auto res = bench.tree_microbenchmark(DEPTH);
#endif // ONLY_LAYER
        
#ifdef PRINT_RESULTS
        std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
        save_results("sssp_cpu.yaml", res);
#endif // SAVE_RESULTS
    }

    // Run naive epoch kernel.
    {
        SSSPGPUTreeBenchmark bench(&g, epoch_sssp_pull_gpu_naive);
        
#ifdef ONLY_LAYER
        auto res = bench.layer_microbenchmark(DEPTH);
#else
        auto res = bench.tree_microbenchmark(DEPTH);
#endif // ONLY_LAYER

#ifdef PRINT_RESULTS
        std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
        save_results("sssp_gpu_naive.yaml", res);
#endif // SAVE_RESULTS
    }

    // Run warp min epoch kernel.
    {
        SSSPGPUTreeBenchmark bench(&g, epoch_sssp_pull_gpu_warp_min);
        
#ifdef ONLY_LAYER
        auto res = bench.layer_microbenchmark(DEPTH);
#else
        auto res = bench.tree_microbenchmark(DEPTH);
#endif // ONLY_LAYER

#ifdef PRINT_RESULTS
        std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
        save_results("sssp_gpu_warp_min.yaml", res);
#endif // SAVE_RESULTS
    }

    weight_t *ret_dist = nullptr;

    // Run CPU kernel. 
    {
        segment_res_t res = sssp_pull_cpu(g, epoch_sssp_pull_cpu, &ret_dist);

        std::cout << res;

        delete[] ret_dist;
    }

    // Run naive kernel.
    {
        segment_res_t res = sssp_pull_gpu(g, epoch_sssp_pull_gpu_naive, 
                &ret_dist);

        std::cout << res;

        delete[] ret_dist;
    }

    // Run warp min kernel.
    {
        segment_res_t res = sssp_pull_gpu(g, epoch_sssp_pull_gpu_warp_min, 
                &ret_dist);

        std::cout << res;

        delete[] ret_dist;
    }

    return EXIT_SUCCESS;
}
