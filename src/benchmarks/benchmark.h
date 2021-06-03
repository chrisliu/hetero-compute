/**
 * Generic benchmark suites.. 
 */

#ifndef SRC_BENCHMARKS__BENCHMARK_H
#define SRC_BENCHMARKS__BENCHMARK_H

// Number of warmup rounds.
#define BENCHMARK_WARMUP_ITERS 5
// Number of rounds to average.
#define BENCHMARK_TIME_ITERS 5

#include <iomanip>
#include <omp.h>
#include <ostream>
#include <random>

#include "../graph.h"

/******************************************************************************
 ***** Results Data Structures ************************************************
 ******************************************************************************/

/**
 * Results for a particular segment of nodes.
 */
typedef struct {
    nid_t  start_id;   // Starting node id.
    nid_t  end_id;     // Ending node id (exclusive).
    float  avg_degree; // Average degree of segment.
    double num_edges;  // Total number of edges traversed.
    double millisecs;  // Execution time.  
    double gteps;      // GTEPS.
} segment_res_t;

/**
 * Results for a particular decomposition of nodes.
 */
typedef struct {
    segment_res_t *segments;    // Benchmarked segments for this layer.
    nid_t         num_segments; // Number of segments this layer.
} layer_res_t;

/**
 * Results for a tree decomposition of nodes.
 */
typedef struct {
    layer_res_t *layers; // Benchmarked layers for this tree.
    int         num_layers;  // Number of layers in this tree.
} tree_res_t;

// YAML generators.
std::ostream& operator<<(std::ostream &os, const segment_res_t &res);
std::ostream& operator<<(std::ostream &os, const tree_res_t &res);

/******************************************************************************
 ***** Microbenchmark Classes *************************************************
 ******************************************************************************/

/**
 * Base cass for a tree based microbenchmark.
 * 
 * For a graph g, its nodes are subdivided by 2^{depth}. Each segment's 
 * performance is subsequently evaluated.
 *
 * Implementations should implement the benchmark_segment.
 */ 
class TreeBenchmark {
public:
    TreeBenchmark(const CSRWGraph *g_);
    tree_res_t  tree_microbenchmark(const int depth);
    layer_res_t layer_microbenchmark(const int depth);

protected:
    const CSRWGraph *g; // CPU graph.

    // Benchmark segment of nodes of range [start, end).
    virtual segment_res_t benchmark_segment(const nid_t start_id, 
            const nid_t end_id) = 0;
};

/**
 * SSSP tree based benchmark.
 * Initializes a random distance vector to test on.
 */
class SSSPTreeBenchmark : public TreeBenchmark {
public:
    SSSPTreeBenchmark(const CSRWGraph *g_);
    ~SSSPTreeBenchmark();

protected:
    weight_t *init_dist; // Initial distances.
};

/******************************************************************************
 ***** Data Structure Implementations *****************************************
 ******************************************************************************/

/**
 * Writer for single segment result. Emits valid YAML.
 */
std::ostream& operator<<(std::ostream &os, const segment_res_t &res) {
    os << std::setprecision(10)
       << "results:" << std::endl
       << "  - start: " << res.start_id << std::endl
       << "    end: " << res.end_id << std::endl
       << "    avg_deg: " << res.avg_degree << std::endl
       << "    num_edges: " << res.num_edges << std::endl
       << "    millis: " << res.millisecs << std::endl
       << "    gteps: " << res.gteps << std::endl;

    return os;
}

/**
 * Writer for list of layer results. Emits valid YAML.
 */
std::ostream& operator<<(std::ostream &os, const tree_res_t &res) {
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
               << "        gteps: " << cur_seg.gteps << std::endl;
        }
    }
    return os;
}

/******************************************************************************
 ***** Microbenchmark Classes' Default Implementations ************************
 ******************************************************************************/

TreeBenchmark::TreeBenchmark(const CSRWGraph *g_)
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

    // g->num_nodes is always divisible by 2.
    nid_t segment_size = g->num_nodes / num_segments;

    for (int s = 0; s < num_segments; s++)
        results.segments[s] = benchmark_segment(s * segment_size, 
                (s + 1) * segment_size);        

    return results;
}

SSSPTreeBenchmark::SSSPTreeBenchmark(const CSRWGraph *g_)
    : TreeBenchmark(g_)
{
    // Initialize distances.
    init_dist = new weight_t[g->num_nodes];
    unsigned init_seed = 1024; // TODO: make this random?
    #pragma omp parallel
    {
        std::mt19937_64 gen(init_seed + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = omp_get_thread_num(); i < g->num_nodes;
                i += omp_get_num_threads()
        )

        init_dist[i] = dist(gen);          
    }
}

SSSPTreeBenchmark::~SSSPTreeBenchmark() {
    delete[] init_dist;
}

#endif // SRC_BENCHMARKS__BENCHMARK_H
