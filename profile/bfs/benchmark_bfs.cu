#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/benchmarks/cpu_benchmark.cuh"
#include "../../src/benchmarks/heterogeneous_benchmark.cuh"
#include "../../src/benchmarks/gpu_benchmark.cuh"
#include "../../src/kernels/cpu/bfs.cuh"
#include "../../src/kernels/gpu/bfs.cuh"

/*****************************************************************************
 ***** Config ****************************************************************
 *****************************************************************************/

// Only get results for a particular depth (not the whole tree).
#define ONLY_LAYER
// Print results to std::cout.
#define PRINT_RESULTS
// Save results to YAML files.
#define SAVE_RESULTS
// Run epoch kernels.
/*#define RUN_EPOCH_KERNELS*/
// Run full kernels.
#define RUN_FULL_KERNELS

#ifdef ONLY_LAYER
// Number of segments (NOT depth).
#define SEGMENTS 24
#else
// Current/Up to (inclusive) this depth.
#define DEPTH 6
#endif // ONLY_LAYER

#define NUM_BLOCKS 256

#define DEVCPU Device::intel_i7_9700K
#define DEVGPU Device::nvidia_quadro_rtx_4000

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

/**
 * Save results to a file.
 * Parameters:
 *   - filename <- filename.
 *   - result <- result type (must have an operator<< implemented).
 */
#ifdef SAVE_RESULTS
template <typename ResT>
__inline__
void save_results(std::string filename, ResT &result) {
    std::ofstream ofs(filename, std::ofstream::out);
    ofs << result;
    ofs.close();
}
#endif // SAVE_RESULTS

/**
 * Generates the YAML filename for an arbitrary kernel.
 * Parameters:
 *   - ker  <- kernel ID.
 *   - args <- zero or more arguments to append to the identifer.
 * Returns:
 *   kernel identifier.
 */
template <typename IdT, typename ...OptArgsT>
__inline__
std::string get_filename(IdT ker, OptArgsT ...args) {
    std::stringstream ss;
    ss << to_repr(ker);
    if (sizeof...(args) > 0)
        volatile int unused[] = { (ss << "_" << args, 0)... };
    ss << ".yaml";
    return ss.str();
}

/**
 * Generates the human-readable kernel name for an arbitrary kernel.
 * Parameters:
 *   - ker  <- kernel ID.
 *   - args <- zero or more arguments to append to the identifer.
 * Returns:
 *   kernel identifier.
 */
template <typename IdT, typename ...OptArgsT>
__inline__
std::string get_kernel_name(IdT ker, OptArgsT ...args) {
    std::stringstream ss;
    ss << to_string(ker);
    if (sizeof...(args) > 0)
        volatile int unused[] = { (ss << " " << args, 0)... };
    return ss.str();
}

/**
 * Runs a tree benchmark from an arbitrary device and kernel.
 * Parameters:
 *   - BenchmarkT <- tree benchmark class.
 *   - g          <- graph.
 *   - dev        <- device ID.
 *   - ker        <- kernel ID.
 *   - args       <- optional arguments for the identifier.
 */
template <class BenchmarkT, typename IdT, typename ...OptArgsT,
         typename = typename std::enable_if<
            std::is_base_of<TreeBenchmark, BenchmarkT>::value>>
void run_treebenchmark(CSRWGraph &g, Device dev, IdT ker, OptArgsT ...args) {
    // Run benchmark.
    // TOOD: this is a hacky way to pass arguments into the benchmark function.
    BenchmarkT bench(&g, get_kernel(ker), args...);

#ifdef ONLY_LAYER
    auto res = bench.layer_microbenchmark(SEGMENTS);
#else 
    auto res = bench.tree_microbenchmark(DEPTH);
#endif  // ONLY_LAYER

    // Configure metadata.
    res.device_name = to_string(dev);
    res.kernel_name = get_kernel_name(ker, args...);

    // Output results appropriately.
#ifdef PRINT_RESULTS
    std::cout << res;
#endif // PRINT_RESULTS

#ifdef SAVE_RESULTS
    save_results(get_filename(ker, args...), res);
#endif // SAVE_RESULTS
}

/*****************************************************************************
 ***** Main ******************************************************************
 *****************************************************************************/

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [graph.sg]" << std::endl;
        return EXIT_FAILURE;
    }

    // Load in graph.
    CSRUWGraph g;
    std::cout << "Loading graph ..." << std::endl;

    std::ifstream ifs(argv[1], std::ifstream::in | std::ifstream::binary);
    Timer timer; timer.Start(); // Start timing.
    ifs >> g;
    timer.Stop(); // Stop timing.
    ifs.close();

    std::cout << " > Loaded in " << timer.Millisecs() << " ms." << std::endl;

/*#ifdef RUN_EPOCH_KERNELS*/
    /*// Run CPU benchmarks.*/
    /*run_treebenchmark<SSSPCPUTreeBenchmark>(g, DEVCPU,*/
            /*SSSPCPU::one_to_one);*/

    /*// Run GPU benchmarks.*/
    /*run_treebenchmark<SSSPGPUTreeBenchmark>(g, DEVGPU,*/
            /*SSSPGPU::one_to_one, NUM_BLOCKS, 1024);*/
    /*run_treebenchmark<SSSPGPUTreeBenchmark>(g, DEVGPU,*/
            /*SSSPGPU::warp_min, NUM_BLOCKS, 1024);*/

    /*std::vector<int> thread_counts = {64, 128, 256, 512, 1024};*/
    /*for (int thread_count : thread_counts)*/
        /*run_treebenchmark<SSSPGPUTreeBenchmark>(g,*/
                /*DEVGPU, SSSPGPU::block_min,*/
                /*NUM_BLOCKS * (1024 / thread_count), thread_count);*/
/*#endif // RUN_EPOCH_KERNELS*/

    /*enable_all_peer_access();*/

    // Full kernel runs.
#ifdef RUN_FULL_KERNELS
    SourcePicker<CSRUWGraph> sp(&g);

    /*// Run CPU push kernel.*/
    /*sp.reset();*/
    /*{*/
        /*std::cout << "BFS CPU push:" << std::endl;*/
        /*segment_res_t res = benchmark_bfs_cpu(g, bfs_push_cpu, sp);*/
        /*std::cout << res;*/
    /*}*/

    /*// Run CPU pull kernel.*/
    /*sp.reset();*/
    /*{*/
        /*std::cout << "BFS CPU pull:" << std::endl;*/
        /*segment_res_t res = benchmark_bfs_cpu(g, bfs_pull_cpu, sp);*/
        /*std::cout << res;*/
    /*}*/

    /*// Run CPU DO kernel.*/
    /*sp.reset();*/
    /*{*/
        /*std::cout << "BFS CPU DO:" << std::endl;*/
        /*segment_res_t res = benchmark_bfs_cpu(g, bfs_do_cpu, sp);*/
        /*std::cout << res;*/
    /*}*/

    // Run GPU one-to-one kernel.
    sp.reset();
    {
        std::cout << "BFS GPU one-to-one:" << std::endl;
        segment_res_t res = benchmark_bfs_gpu(g,
                epoch_bfs_pull_gpu_one_to_one, sp, NUM_BLOCKS, 1024);
        std::cout << res;
    }

    // Run GPU warp kernel.
    sp.reset();
    {
        std::cout << "BFS GPU warp:" << std::endl;
        segment_res_t res = benchmark_bfs_gpu(g,
                epoch_bfs_pull_gpu_warp, sp, NUM_BLOCKS, 1024);
        std::cout << res;
    }

    // Run GPU sync warp kernel.
    sp.reset();
    {
        std::cout << "BFS GPU sync warp (sync_iters = 1):" 
            << std::endl;
        segment_res_t res = benchmark_bfs_gpu(g,
                epoch_bfs_sync_pull_gpu_warp<1>, sp, NUM_BLOCKS, 1024);
        std::cout << res;
    }

    sp.reset();
    {
        std::cout << "BFS GPU sync warp (sync_iters = 2):" 
            << std::endl;
        segment_res_t res = benchmark_bfs_gpu(g,
                epoch_bfs_sync_pull_gpu_warp<2>, sp, NUM_BLOCKS, 1024);
        std::cout << res;
    }

    sp.reset();
    {
        std::cout << "BFS GPU sync warp (sync_iters = 3):" 
            << std::endl;
        segment_res_t res = benchmark_bfs_gpu(g,
                epoch_bfs_sync_pull_gpu_warp<3>, sp, NUM_BLOCKS, 1024);
        std::cout << res;
    }

    // Run heterogeneous kernel.
    /*sp.reset();*/
    /*{*/
        /*// Load in schedule.*/
        /*std::cout << "SSSP heterogeneous:" << std::endl;*/
        /*segment_res_t res = benchmark_sssp_heterogeneous(g, sp);*/
        /*std::cout << res;*/
    /*}*/
#endif // RUN_FULL_KERNELS

    return EXIT_SUCCESS;
}
