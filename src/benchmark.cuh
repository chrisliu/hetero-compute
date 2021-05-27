/**
 * Generic micro benchmark suite. 
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

// Number of warmup rounds.
#define BENCHMARK_WARMUP_ITERS 5
// Number of rounds to average.
#define BENCHMARK_TIME_ITERS 5

#include <iomanip>
#include <ostream>
#include <random>

#include "gapbs.h"
#include "kernels/sssp_pull_gpu.cuh"

/******************************************************************************
 ***** Results Data Structures ************************************************
 ******************************************************************************/

/**
 * Results for a particular segment of nodes.
 */
typedef struct {
    nid_t  start_id;   // Starting node id.
    nid_t  end_id;     // Ending node id.
    float  avg_degree; // Average degree of segment.
    nid_t  num_edges;  // Total number of edges.
    double millisecs;  // Execution time.  
    double teps;       // TEPS.
} segment_res_t;

/**
 * Results for a particular decomposition of nodes.
 */
typedef struct {
    segment_res_t *segments;    // Benchmarked segments for this layer.
    nid_t  num_segments; // Number of segments this layer.
} layer_res_t;

/**
 * Results for a tree decomposition of nodes.
 */
typedef struct {
    layer_res_t *layers; // Benchmarked layers for this tree.
    int         num_layers;  // Number of layers in this tree.
} tree_res_t;

// YAML generators.
std::ostream& operator<<(std::ostream& os, const tree_res_t res);

/******************************************************************************
 ***** Microbenchmark Base Classes ********************************************
 ******************************************************************************/

/**
 * Tree based microbenchmark.
 * 
 * For a graph g, its nodes are subdivided by 2^{depth}. Each segment's 
 * performance is subsequently evaluated.
 *
 * Implementations should implement the benchmark_segment.
 */ 
class TreeBenchmark {
public:
    TreeBenchmark(const wgraph_t *g_);
    tree_res_t  tree_microbenchmark(const int depth);
    layer_res_t layer_microbenchmark(const int depth);

protected:
    const wgraph_t *g; // CPU graph.

    // Benchmark segment of nodes of range [start, end).
    virtual segment_res_t benchmark_segment(const nid_t start, 
            const nid_t end) = 0;
};

/******************************************************************************
 ***** Specialized Microbenchmark Classes *************************************
 ******************************************************************************/

/**
 * Tree based microbenchmark for GPU implementations.
 */
class SSSPGPUBenchmark : public TreeBenchmark {
public:
    // TODO: pass in specific kernel.
    SSSPGPUBenchmark(const wgraph_t *g_, const int block_count_ = 8,
            const int thread_count_ = 1024); 
    ~SSSPGPUBenchmark();

    // TODO: dynamically modify block and thread count.
    // TODO: dynamically modify kernel.

protected:
    nid_t      *index;        // (CPU) graph indices (used to calculate degree).
    weight_t   *init_dist;    // (CPU) initial distances. 
    nid_t      *cu_index;     // (GPU) graph indices.
    cu_wnode_t *cu_neighbors; // (GPU) graph neighbors and weights.
    weight_t   *cu_dist;      // (GPU) distances.
    nid_t      *cu_updated;   // (GPU) update counter.

    int        block_count;   // Number of blocks to launch kernel.
    int        thread_count;  // Number of threads to launch kernel.

    segment_res_t benchmark_segment(const int start, const int end);
};

/******************************************************************************
 ***** Data Structure Implementations *****************************************
 ******************************************************************************/

/**
 * Writer for list of layer results. Emits valid YAML.
 */
std::ostream& operator<<(std::ostream& os, const tree_res_t res) {
    os << "results:" << std::endl;
    for (int depth = 0; depth < res.num_layers; depth++) {
        os << "  - depth: " << depth << std::endl
           << "    segments:" << std::endl;
        for (int seg = 0; seg < res.layers[depth].num_segments; seg++) {
            segment_res_t cur_seg = res.layers[depth].segments[seg];
            os << std::setprecision(10)
               << "      - start: " << cur_seg.start_id << std::endl
               << "        end: " << cur_seg.end_id << std::endl
               << "        avg_deg: " << cur_seg.avg_degree << std::endl
               << "        num_edges: " << cur_seg.num_edges << std::endl
               << "        millis: " << cur_seg.millisecs << std::endl
               << "        teps: " << cur_seg.teps << std::endl;
        }
    }
    return os;
}

/******************************************************************************
 ***** Microbenchmark Base Classes' Default Implementations *******************
 ******************************************************************************/

TreeBenchmark::TreeBenchmark(const wgraph_t *g_)
    : g(g_)
{}

/**
 * Performs tree microbenchmark on kernel.
 * For each depth, the graph's nodes will be divided into 2^{depth} segments.
 * Parameters:
 *   - depth <- 0 to tree depth (inclusive).
 * Returns:
 *   list of benchmark results (layer_res_t) ordered by increasing depth.
 */
tree_res_t TreeBenchmark::tree_microbenchmark(const int depth) {
    tree_res_t results = { nullptr, depth + 1 };
    results.layers = new layer_res_t[depth + 1]; // 2^0, 2^1, ... , 2^{depth}.

    // Warmup caches.
    for (int iter = 0; iter < BENCHMARK_WARMUP_ITERS; iter++)
        layer_microbenchmark(0);

    // Actual runs.
    for (int i = 0; i <= depth; i++)
        results.layers[i] = layer_microbenchmark(i);

    return results;
}

/**
 * Performs tree microbenchmark on kernel.
 * The graph's nodes will be divided into 2^{depth} segments.
 * Parameters:
 *   - depth <- tree depth to test.
 * Returns:
 *   results of benchmark for a particular layer.
 */
layer_res_t TreeBenchmark::layer_microbenchmark(const int depth) {
    nid_t num_segments = 1 << depth;

    // Init results.
    layer_res_t results = { nullptr, num_segments };
    results.segments = new segment_res_t[num_segments]; 

    // g->num_nodes() is always divisible by 2.
    nid_t segment_size = g->num_nodes() / num_segments;

    for (int s = 0; s < g->num_nodes() / segment_size; s++)
        results.segments[s] = benchmark_segment(s * segment_size, 
                (s + 1) * segment_size);        

    return results;
}

/******************************************************************************
 ***** Specialized Microbenchmark Classes' Implementations ********************
 ******************************************************************************/

SSSPGPUBenchmark::SSSPGPUBenchmark(const wgraph_t *g_,
        const int block_count_, const int thread_count_)
    : TreeBenchmark(g_)
    , block_count(block_count_)
    , thread_count(thread_count_)
{
    // Initialize GPU copy of graph.
    cu_wnode_t *neighbors = nullptr;
    wgraph_to_cugraph(*g, &index, &neighbors);

    nid_t index_size     = g->num_nodes() * sizeof(nid_t);
    nid_t neighbors_size = 2 * g->num_edges() * sizeof(cu_wnode_t);
    cudaMalloc((void **) &cu_index, index_size);
    cudaMalloc((void **) &cu_neighbors, neighbors_size);
    cudaMemcpy(cu_index, index, index_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cu_neighbors, neighbors, neighbors_size, cudaMemcpyHostToDevice);

    delete[] neighbors;

    // Initialize update counter.
    cudaMalloc((void **) &cu_updated, sizeof(nid_t));

    // Initialize distances.
    init_dist = new weight_t[g->num_nodes()];
    unsigned init_seed = 1024; // TODO: make this random?
    #pragma omp parallel
    {
        std::mt19937_64 gen(init_seed + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = omp_get_thread_num(); i < g->num_nodes();
                i += omp_get_num_threads()
        )
            init_dist[i] = dist(gen);          
    }

    cudaMalloc((void **) &cu_dist, g->num_nodes() * sizeof(weight_t));
}

SSSPGPUBenchmark::~SSSPGPUBenchmark() {
    delete[] index;
    delete[] init_dist;
    cudaFree(cu_index);
    cudaFree(cu_neighbors);
    cudaFree(cu_dist);
    cudaFree(cu_updated);
}

/**
 * Performs benchmark on a single slice of nodes of range [start, end).
 * Parameters:
 *   - start <- starting node ID.
 *   - end   <- ending node ID (exclusive).
 * Returns:
 *   result segment data structure.
 */
segment_res_t SSSPGPUBenchmark::benchmark_segment(const nid_t start,
        const nid_t end
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start;
    result.end_id     = end;
    result.num_edges  = index[end] - index[start];
    result.avg_degree = static_cast<float>(result.num_edges) / (end - start);

    // Setup kernel.
    cudaMemcpy(cu_dist, init_dist, g->num_nodes() * sizeof(weight_t),
            cudaMemcpyHostToDevice);
    cudaMemset(cu_updated, 0, sizeof(nid_t));

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    double total_time = 0.0;
    Timer timer; 

    for (int iter = 0; iter < BENCHMARK_TIME_ITERS; iter++) {
        timer.Start();
        sssp_pull_gpu_impl<<<block_count, thread_count>>>(cu_index, cu_neighbors,
                start, end, cu_dist, cu_updated);
        timer.Stop();
        total_time += timer.Millisecs();
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_TIME_ITERS;
    result.teps      = result.num_edges / (result.millisecs * 1000);

    return result;
}

#endif // BENCHMARK_H
