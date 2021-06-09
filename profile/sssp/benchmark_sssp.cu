#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/benchmarks/gpu_benchmark.cuh"
#include "../../src/benchmarks/cpu_benchmark.h"
#include "../../src/benchmarks/heterogeneous_benchmark.cuh"
#include "../../src/kernels/cpu/sssp_pull.h"
#include "../../src/kernels/gpu/sssp_pull.cuh"

// Only get results for a particular depth (not the whole tree).
#define ONLY_LAYER
// Print results to std::cout.
#define PRINT_RESULTS
// Save results to YAML files.
#define SAVE_RESULTS
// Current/Up to (inclusive) this depth.
#define DEPTH 6

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

    // Run block min epoch kernel.
    {
        int thread_count = 64;
        SSSPGPUTreeBenchmark bench(&g, epoch_sssp_pull_gpu_block_min,
                64 * (1024 / thread_count), thread_count);
        
#ifdef ONLY_LAYER
        auto res = bench.layer_microbenchmark(DEPTH);
#else
        auto res = bench.tree_microbenchmark(DEPTH);
#endif // ONLY_LAYER

#ifdef PRINT_RESULTS
        std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
        save_results("sssp_gpu_block_min.yaml", res);
#endif // SAVE_RESULTS
    }

    weight_t *ret_dist  = nullptr;
    weight_t *init_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_dist[i] = INF_WEIGHT;
    init_dist[0] = 0; // Arbitrarily set highest degree node to source.

    // Run CPU kernel.
    {
        std::cout << "SSSP CPU:" << std::endl;
        segment_res_t res = benchmark_sssp_cpu(g, epoch_sssp_pull_cpu,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU naive kernel.
    {
        std::cout << "SSSP GPU naive:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_naive,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU warp min kernel.
    {
        std::cout << "SSSP GPU warp min:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_warp_min,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    // Run GPU block min kernel.
    {
        std::cout << "SSSP GPU block min:" << std::endl;
        segment_res_t res = benchmark_sssp_gpu(g, epoch_sssp_pull_gpu_block_min,
                init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
   }

    // Run heterogeneous kernel.
    {
        nid_t                      middle     = g.num_nodes / 16 * 2;
        /*graph_range_t              cpu_range  = { middle, g.num_nodes };*/
        /*std::vector<graph_range_t> gpu_ranges = { { 0, middle } };*/
        graph_range_t              cpu_range  = { middle - 250000, middle};
        std::vector<graph_range_t> gpu_ranges = { { 0, middle - 250000},
                                                  { middle, g.num_nodes } };
        std::cout << "SSSP heterogeneous:" << std::endl;
        segment_res_t res = benchmark_sssp_heterogeneous(g,
                epoch_sssp_pull_cpu, epoch_sssp_pull_gpu_warp_min,
                cpu_range, gpu_ranges, init_dist, &ret_dist);
        std::cout << res;
        delete[] ret_dist;
    }

    delete[] init_dist;

    return EXIT_SUCCESS;
}
