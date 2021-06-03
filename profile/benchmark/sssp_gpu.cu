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

#define DEPTH 3

__inline__
void save_results(std::string filename, tree_res_t &result) {
    std::ofstream ofs(filename, std::ofstream::out);
    ofs << result;
    ofs.close();
}

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

    /*// Run CPU epoch kernel.*/
    /*{*/
        /*SSSPCPUTreeBenchmark bench(&g, epoch_sssp_pull_cpu);*/

        /*tree_res_t res = bench.tree_microbenchmark(DEPTH);*/
        
        /*[>std::cout << res;<]*/
        /*save_results("sssp_cpu.yaml", res);*/
    /*}*/

    /*// Run naive epoch kernel.*/
    /*{*/
        /*SSSPGPUTreeBenchmark bench(&g, epoch_sssp_pull_gpu_naive);*/
        
        /*tree_res_t res = bench.tree_microbenchmark(DEPTH);*/

        /*[>std::cout << res;<]*/
        /*save_results("sssp_gpu_naive.yaml", res);*/
    /*}*/

    /*// Run warp min epoch kernel.*/
    /*{*/
        /*SSSPGPUTreeBenchmark bench(&g, epoch_sssp_pull_gpu_warp_min);*/
        
        /*tree_res_t res = bench.tree_microbenchmark(DEPTH);*/

        /*[>std::cout << res;<]*/
        /*save_results("sssp_gpu_warp_min.yaml", res);*/
    /*}*/

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
