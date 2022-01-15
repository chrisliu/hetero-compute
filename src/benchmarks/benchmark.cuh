/**
 * Generic benchmark suites.. 
 */

#ifndef SRC_BENCHMARKS__BENCHMARK_H
#define SRC_BENCHMARKS__BENCHMARK_H

// Number of warmup rounds.
#define BENCHMARK_WARMUP_ITERS 5
// Number of rounds to average for segments.
#define BENCHMARK_SEGMENT_TIME_ITERS 5
// Number of rounds to average for full kernel runs.
#define BENCHMARK_FULL_TIME_ITERS 64
// Output precision.
#define BENCHMARK_PRECISION 10

#include <algorithm>
#include <iomanip>
#include <omp.h>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#include "../bitmap.cuh"
#include "../graph.cuh"
/*#include "../kernels/cpu/bfs.cuh"*/

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
    float       min_degree;       // Minimum degree of segment.
    float       max_degree;       // Maximum degree of segment.
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
 * Base class for a tree-based benchmark.
 * 
 * Given a graph g, the ordered list of nodes (by descending degree) will be
 * divided into <num_segments> groups. Each segment will roughly have 
 * an average degree of (# of edges) / @num_segments.
 *
 * Implementations should implement the benchmark_segment.
 */ 
template <typename GraphT>
class TreeBenchmark {
public:
    TreeBenchmark(const GraphT *g_);
    virtual tree_res_t  tree_microbenchmark(const int depth);
    virtual layer_res_t layer_microbenchmark(const nid_t num_segments);

protected:
    const GraphT *g; // CPU graph.

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
class SSSPTreeBenchmark : public TreeBenchmark<CSRWGraph> {
public:
    SSSPTreeBenchmark(const CSRWGraph *g_);
    ~SSSPTreeBenchmark();

protected:
    weight_t *init_dist; // Initial distances.
};

/**
 * BFS tree-based benchmark.
 *
 * Precomputes parents array and frontiers for each level.
 */
/*class BFSTreeBenchmark : public TreeBenchmark<CSRUWGraph> {*/
/*public:*/
    /*BFSTreeBenchmark(const CSRUWGraph *g_);*/
    /*~BFSTreeBenchmark();*/

    /*bool  set_epoch(nid_t epoch);*/
    /*nid_t get_epoch() const;*/
    /*nid_t num_epochs() const;*/

/*protected:*/
    /*std::vector<nid_t *>          parents_arr;*/
    /*std::vector<Bitmap::Bitmap *> frontiers;*/
    /*nid_t                         cur_epoch;*/

    /*nid_t          *get_parents() const;*/
    /*Bitmap::Bitmap *get_frontier() const;*/

    /*nid_t *compute_ranges(const nid_t num_segments) const override;*/
/*};*/

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
       << "    min_deg: " << res.min_degree << std::endl
       << "    max_deg: " << res.max_degree << std::endl
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
           << "        min_deg: " << cur_seg.min_degree << std::endl
           << "        max_deg: " << cur_seg.max_degree << std::endl
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
               << "        min_deg: " << cur_seg.min_degree << std::endl
               << "        max_deg: " << cur_seg.max_degree << std::endl
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

template <typename GraphT>
TreeBenchmark<GraphT>::TreeBenchmark(const GraphT *g_)
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
template <typename GraphT>
tree_res_t TreeBenchmark<GraphT>::tree_microbenchmark(const int depth) {
    tree_res_t results;
    results.num_layers = depth + 1;
    results.layers     = new layer_res_t[depth + 1]; // 2^0, ... , 2^{depth}.

    // Warmup caches.
    int warmup_iters = BENCHMARK_WARMUP_ITERS / BENCHMARK_SEGMENT_TIME_ITERS;
    for (int iter = 0; iter < warmup_iters; iter++)
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
template <typename GraphT>
layer_res_t TreeBenchmark<GraphT>::layer_microbenchmark(const nid_t num_segments) {
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
 * Wrapper for compute_equal_edge_ranges.
 */
template <typename GraphT>
inline
nid_t *TreeBenchmark<GraphT>::compute_ranges(const nid_t num_segments) const {
    return compute_equal_edge_ranges(*g, num_segments);
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

/*BFSTreeBenchmark::BFSTreeBenchmark(const CSRUWGraph *g_)*/
    /*: TreeBenchmark(g_)*/
    /*, cur_epoch(0)*/
/*{*/
    /*// Run BFS CPU pull to get expected parents and frontier bitmap.*/
    /*SourcePicker<CSRUWGraph> sp(g);*/
    /*nid_t source_id = sp.next_vertex();*/

    /*nid_t *parents = new nid_t[g->num_nodes];*/
    /*#pragma omp parallel for*/
    /*for (int i = 0; i < g->num_nodes; i++)*/
        /*parents[i] = INVALID_NODE;*/
    /*parents[source_id] = source_id;*/

    /*Bitmap::Bitmap *frontier = Bitmap::constructor(g->num_nodes);*/
    /*Bitmap::set_bit(frontier, source_id);*/

    /*Bitmap::Bitmap *next_frontier = Bitmap::constructor(g->num_nodes);*/

    /*nid_t num_nodes;*/

    /*do {*/
        /*// Update parents and frontier arrays.*/
        /*nid_t *cpy_parents = new nid_t[g->num_nodes];*/
        /*std::copy(parents, parents + g->num_nodes, cpy_parents);*/
        /*parents_arr.push_back(cpy_parents);*/

        /*Bitmap::Bitmap *cpy_frontier = Bitmap::constructor(g->num_nodes);*/
        /*Bitmap::copy(frontier, cpy_frontier);*/
        /*frontiers.push_back(cpy_frontier);*/

        /*// Run BFS pull as normal.*/
        /*num_nodes = 0;*/
        /*epoch_bfs_pull_cpu(*g, parents, 0, g->num_nodes, frontier,*/
                /*next_frontier, num_nodes);*/
        /*std::swap(frontier, next_frontier);*/
        /*Bitmap::reset(next_frontier);*/
    /*} while(num_nodes != 0);*/

    /*// Free memory.*/
    /*delete[] parents;*/
    /*Bitmap::destructor(&frontier);*/
    /*Bitmap::destructor(&next_frontier);*/
/*}*/

/*BFSTreeBenchmark::~BFSTreeBenchmark() {*/
    /*for (int i = 0; i < parents_arr.size(); i++) {*/
        /*delete[] parents_arr[i];*/
        /*Bitmap::destructor(&frontiers[i]);*/
    /*}*/
/*}*/

/**
 * Set current BFS epoch to benchmark.
 * Parameters:
 *   - epoch <- epoch to set.
 * Returns:
 *   true if successful; false if otherwise.
 */
/*bool BFSTreeBenchmark::set_epoch(nid_t epoch) {*/
    /*if (epoch < 0 or epoch >= num_epochs())*/
        /*return false;*/
    
    /*cur_epoch = epoch;*/
    /*return true;*/
/*}*/

/**
 * Get current BFS epoch that's being benchmarked.
 * Returns:
 *   Current BFS epoch.
 */
/*nid_t BFSTreeBenchmark::get_epoch() const {*/
    /*return cur_epoch;*/
/*}*/

/**
 * Get total number of epochs to benchmark.
 * Returns:
 *   Number of epochs available for benchmarking.
 */
/*nid_t BFSTreeBenchmark::num_epochs() const {*/
    /*return parents_arr.size();*/
/*}*/

/*inline*/
/*nid_t *BFSTreeBenchmark::get_parents() const {*/
    /*return parents_arr[cur_epoch];*/
/*}*/

/*inline*/
/*Bitmap::Bitmap *BFSTreeBenchmark::get_frontier() const {*/
    /*return frontiers[cur_epoch];*/
/*}*/

/**
 * Modify default compute ranges to align range start and end to bitmap's
 * internal data type size (i.e., num bits).
 */
/*inline*/
/*nid_t *BFSTreeBenchmark::compute_ranges(const nid_t num_segments) const {*/
    /*return compute_equal_edge_ranges(*g, num_segments, Bitmap::data_size);*/
/*}*/

#endif // SRC_BENCHMARKS__BENCHMARK_H
