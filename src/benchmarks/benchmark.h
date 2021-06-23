/**
 * Generic benchmark suites.. 
 */

#ifndef SRC_BENCHMARKS__BENCHMARK_H
#define SRC_BENCHMARKS__BENCHMARK_H

// Number of warmup rounds.
#include <string>
#define BENCHMARK_WARMUP_ITERS 5
// Number of rounds to average.
#define BENCHMARK_TIME_ITERS 5
// Output precision.
#define BENCHMARK_PRECISION 10

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
struct segment_res_t {
    nid_t       start_id;         // Starting node id.
    nid_t       end_id;           // Ending node id (exclusive).
    float       avg_degree;       // Average degree of segment.
    offset_t    num_edges;        // Total number of edges in this subgraph. 
    double      millisecs;        // Execution time.  
    double      gteps;            // GTEPS.
    std::string device_name = ""; // Device name.
    std::string kernel_name = ""; // Kernel name.
};

/**
 * Results for a particular decomposition of nodes.
 */
struct layer_res_t {
    segment_res_t *segments;      // Benchmarked segments for this layer.
    nid_t         num_segments;   // Number of segments this layer.
    std::string device_name = ""; // Device name.
    std::string kernel_name = ""; // Kernel name.
};

/**
 * Results for a tree decomposition of nodes.
 */
struct tree_res_t {
    layer_res_t *layers;          // Benchmarked layers for this tree.
    int         num_layers;       // Number of layers in this tree.
    std::string device_name = ""; // Device name.
    std::string kernel_name = ""; // Kernel name.
};

// YAML generators.
std::ostream &operator<<(std::ostream &os, const segment_res_t &res);
std::ostream &operator<<(std::ostream &os, const layer_res_t &res);
std::ostream &operator<<(std::ostream &os, const tree_res_t &res);

/******************************************************************************
 ***** Microbenchmark Classes *************************************************
 ******************************************************************************/

/**
 * Base class for a tree based microbenchmark.
 * 
 * Given a graph g, the ordered list of nodes (by descending degree) will be
 * divided into <num_segments> groups. Each segment will roughly have 
 * an average degree of (# of edges) / @num_segments.
 *
 * Implementations should implement the benchmark_segment.
 */ 
class TreeBenchmark {
public:
    TreeBenchmark(const CSRWGraph *g_);
    tree_res_t  tree_microbenchmark(const int depth);
    layer_res_t layer_microbenchmark(const nid_t num_segments);

protected:
    const CSRWGraph *g; // CPU graph.

    // Benchmark segment of nodes of range [start, end).
    virtual segment_res_t benchmark_segment(const nid_t start_id, 
            const nid_t end_id) = 0;
    // Generates node ranges for each segment. 
    virtual nid_t *compute_ranges(const nid_t num_segments) const;
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
    // Write kernel configuration.
    os << "config:" << std::endl
       << "  device: " << res.device_name << std::endl
       << "  kernel: " << res.kernel_name << std::endl;
    
    // Write kernel results.
    os << std::setprecision(BENCHMARK_PRECISION)
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
 * Writer for a layer of results. Emits valid YAML.
 */
std::ostream &operator<<(std::ostream &os, const layer_res_t &res) {
    // Write kernel configuration.
    os << "config:" << std::endl
       << "  device: " << res.device_name << std::endl
       << "  kernel: " << res.kernel_name << std::endl;
    
    // Write kernel results.
    os << std::setprecision(BENCHMARK_PRECISION)
       << "results:" << std::endl
       << "  - segments: " << std::endl;
    
    for (int seg = 0; seg < res.num_segments; seg++) {
        segment_res_t cur_seg = res.segments[seg];        
        os << "      - start: " << cur_seg.start_id << std::endl
           << "        end: " << cur_seg.end_id << std::endl
           << "        avg_deg: " << cur_seg.avg_degree << std::endl
           << "        num_edges: " << cur_seg.num_edges << std::endl
           << "        millis: " << cur_seg.millisecs << std::endl
           << "        gteps: " << cur_seg.gteps << std::endl;
    }

    return os;
}

/**
 * Writer for list of layer results. Emits valid YAML.
 */
std::ostream& operator<<(std::ostream &os, const tree_res_t &res) {
    // Write kernel configuration.
    os << "config:" << std::endl
       << "  device: " << res.device_name << std::endl
       << "  kernel: " << res.kernel_name << std::endl;
    
    // Write kernel results.
    os << "results:" << std::endl;
    for (int depth = 0; depth < res.num_layers; depth++) {
        os << "  - depth: " << depth << std::endl
           << "    segments:" << std::endl;
        for (int seg = 0; seg < res.layers[depth].num_segments; seg++) {
            segment_res_t cur_seg = res.layers[depth].segments[seg];
            os << std::setprecision(BENCHMARK_PRECISION)
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
    tree_res_t results;
    results.num_layers = depth + 1;
    results.layers     = new layer_res_t[depth + 1]; // 2^0, ... , 2^{depth}.

    // Warmup caches.
    for (int iter = 0; iter < BENCHMARK_WARMUP_ITERS; iter++)
        layer_microbenchmark(1);

    // Actual runs.
    for (int i = 0; i <= depth; i++)
        results.layers[i] = layer_microbenchmark(1 << i);

    return results;
}

/**
 * Performs tree microbenchmark on kernel.
 * The graph's nodes will be divided into ^{depth} segments.
 * Parameters:
 *   - num_segments <- number of segments.
 * Returns:
 *   results of benchmark for a particular layer.
 */
layer_res_t TreeBenchmark::layer_microbenchmark(const nid_t num_segments) {
    // Init results.
    layer_res_t results;
    results.num_segments = num_segments;
    results.segments     = new segment_res_t[num_segments]; 

    nid_t *seg_ranges = compute_ranges(num_segments);

    for (int s = 0; s < num_segments; s++)
        results.segments[s] = benchmark_segment(seg_ranges[s], 
                seg_ranges[s + 1]);        

    delete[] seg_ranges;

    return results;
}

/**
 * Computes the starting and ending node IDs for each segments such that
 * the average degree of each segment is roughly (# of edges) / @num_segments.
 * Parameters:
 *   - num_segments <- number of segments.
 * Returns:
 *   List of length @num_segments + 1. For each segment i, the segment's range
 *   is defined as [range[i], range[i + 1]). Memory is dynamically allocated so 
 *   it must be freed to prevent memory leaks.
 */
nid_t *TreeBenchmark::compute_ranges(const nid_t num_segments) const {
    nid_t *seg_ranges = new nid_t[num_segments + 1];    
    seg_ranges[0] = 0;

    offset_t avg_deg = g->num_edges / num_segments;

    nid_t end_id   = 0;
    int   seg_id   = 0;
    int   seg_deg  = 0;

    while (end_id != g->num_nodes) {
        seg_deg += g->get_degree(end_id);

        // If segment exceeds average degree, save it and move on to next.
        if (seg_deg >= avg_deg) {
            seg_ranges[seg_id + 1] = end_id;
            seg_deg = 0; // Reset segment degree.
            seg_id++;
        }

        end_id++;
    }

    // If last segment hasn't been saved yet (almost guaranteed to happen).
    if (seg_id != num_segments)
        seg_ranges[seg_id + 1] = end_id;

    return seg_ranges;
}

SSSPTreeBenchmark::~SSSPTreeBenchmark() {
    delete[] init_dist;
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

#endif // SRC_BENCHMARKS__BENCHMARK_H
