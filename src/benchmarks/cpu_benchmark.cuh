/**
 * Benchamrks for CPU implementations.
 */

#ifndef SRC_BENCHMARKS__CPU_BENCHMARK_CUH
#define SRC_BENCHMARKS__CPU_BENCHMARK_CUH

#include <algorithm>
#include <iostream>
#include <omp.h>

#include "benchmark.cuh"
#include "../graph.cuh"
#include "../util.h"
#include "../kernels/kernel_types.cuh"
/*#include "../kernels/cpu/bfs.cuh"*/
#include "../kernels/cpu/sssp.cuh"
#include "../kernels/cpu/pr.cuh"

/*****************************************************************************
 ***** Benchmarks ************************************************************
 *****************************************************************************/

class SSSPCPUTreeBenchmark : public SSSPTreeBenchmark {
public:
    SSSPCPUTreeBenchmark(
            const CSRWGraph *g_, sssp_cpu_epoch_func epoch_kernel_);

protected:
    sssp_cpu_epoch_func epoch_kernel; // CPU epoch kernel.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};



class PRCPUTreeBenchmark : public PRTreeBenchmark {
public:
    PRCPUTreeBenchmark(
	const CSRWGraph *g_, pr_cpu_epoch_func epoch_kernel_);

protected:
    pr_cpu_epoch_func epoch_kernel; // CPU epoch kernel.

    segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);
};

/*class BFSCPUPushTreeBenchmark : public BFSTreeBenchmark {*/
/*public:*/
    /*BFSCPUPushTreeBenchmark(const CSRUWGraph *g, */
            /*bfs_cpu_push_epoch_func epoch_kernel_);*/
    /*tree_res_t tree_microbenchmark(const int depth) override;*/
    /*layer_res_t layer_microbenchmark(const nid_t num_segments) override;*/

/*protected:*/
    /*bfs_cpu_push_epoch_func epoch_kernel;*/

    /*segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);*/
/*};*/


/*class BFSCPUPullTreeBenchmark : public BFSTreeBenchmark {*/
/*public:*/
    /*BFSCPUPullTreeBenchmark(const CSRUWGraph *g, */
            /*bfs_cpu_pull_epoch_func epoch_kernel_);*/

/*protected:*/
    /*bfs_cpu_pull_epoch_func epoch_kernel;*/

    /*segment_res_t benchmark_segment(const nid_t start_id, const nid_t end_id);*/
/*};*/

/*****************************************************************************
 ***** Tree Benchmark Implementations ****************************************
 *****************************************************************************/

SSSPCPUTreeBenchmark::SSSPCPUTreeBenchmark(
        const CSRWGraph *g_, sssp_cpu_epoch_func epoch_kernel_)
    : SSSPTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
{}

segment_res_t SSSPCPUTreeBenchmark::benchmark_segment(
        const nid_t start_id, const nid_t end_id
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start_id;
    result.end_id     = end_id;
    result.num_edges  = g->index[end_id] - g->index[start_id];
    result.avg_degree = static_cast<float>(result.num_edges) 
        / (end_id - start_id);
    
    // Compute min and max degree.
    float min_degree, max_degree;
    min_degree = max_degree = static_cast<float>(
            g->index[start_id + 1] - g->index[start_id]);
    #pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)
    for (int nid = start_id + 1; nid < end_id; nid++) {
        float ndeg = static_cast<float>(g->index[nid + 1] - g->index[nid]);
        min_degree = min(min_degree, ndeg);
        max_degree = max(max_degree, ndeg);
    }
    result.min_degree = min_degree;
    result.max_degree = max_degree;

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    Timer timer;
    double total_time = 0.0;

    for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {
        // Setup kernel.
        weight_t *dist = new weight_t[g->num_nodes];
        #pragma omp parallel for
        for (int i = 0; i < g->num_nodes; i++)
            dist[i] = init_dist[i];
        nid_t updated = 0;

        // Run kernel.
        timer.Start();
        #pragma omp parallel
        {
            (*epoch_kernel)(*g, dist, start_id, end_id, omp_get_thread_num(),
                    omp_get_num_threads(), updated);
        }
        timer.Stop();

        // Save time.
        total_time += timer.Millisecs();
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;
}


PRCPUTreeBenchmark::PRCPUTreeBenchmark(
      const CSRWGraph *g_, pr_cpu_epoch_func epoch_kernel_)
    : PRTreeBenchmark(g_)
    , epoch_kernel(epoch_kernel_)
{}

segment_res_t PRCPUTreeBenchmark::benchmark_segment(
	const nid_t start_id, const nid_t end_id
    ) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = start_id;
    result.end_id     = end_id;
    result.num_edges  = g->index[end_id] - g->index[start_id];
    result.avg_degree = static_cast<float>(result.num_edges) 
	/ (end_id - start_id);

    // Compute min and max degree.
    float min_degree, max_degree;
    min_degree = max_degree = static_cast<float>(
        g->index[start_id + 1] - g->index[start_id]);
    #pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)
    for (int nid = start_id + 1; nid < end_id; nid++) {
        float ndeg = static_cast<float>(g->index[nid + 1] - g->index[nid]);
	min_degree = min(min_degree, ndeg);
	max_degree = max(max_degree, ndeg);
    }
    result.min_degree = min_degree;
    result.max_degree = max_degree;

    // Time kernel (avg of BENCHMARK_TIME_ITERS).
    Timer timer;
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {
        // Setup kernel.
	weight_t *score = new weight_t[g->num_nodes];
        #pragma omp parallel for
	for (int i = 0; i < g->num_nodes; i++)
	    score[i] = 1.0f/g->num_nodes;//init_score[i];
	nid_t updated = 0;

	// Run kernel.
	timer.Start();
        #pragma omp parallel
	{
	    (*epoch_kernel)(*g, score, start_id, end_id, omp_get_thread_num(),
	    	omp_get_num_threads(), updated);
	}
	timer.Stop();

	// Save time.
	total_time += timer.Millisecs();
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.
    return result;
}

/*BFSCPUPushTreeBenchmark::BFSCPUPushTreeBenchmark(const CSRUWGraph *g,*/
        /*bfs_cpu_push_epoch_func epoch_kernel_)*/
    /*: BFSTreeBenchmark(g)*/
    /*, epoch_kernel(epoch_kernel_)*/
/*{}*/

/*tree_res_t BFSCPUPushTreeBenchmark::tree_microbenchmark(const int depth) {*/
    /*if (depth != 0)*/
        /*std::cerr << "[Warning] BFS CPU push cannot be divided into segments. "*/
            /*<< "Depth can only be 0, not " << depth << "." << std::endl;*/
    
    /*tree_res_t results;*/
    /*results.num_layers = 1;*/
    /*results.layers = new layer_res_t[1];*/

    /*// Warmup caches.*/
    /*int warmup_iters = BENCHMARK_WARMUP_ITERS / BENCHMARK_SEGMENT_TIME_ITERS;*/
    /*for (int iter = 0; iter < warmup_iters; iter++)*/
        /*layer_microbenchmark(1);*/

    /*// Actual runs.*/
    /*results.layers[0] = layer_microbenchmark(1);*/

    /*return results;*/
/*}*/

/*layer_res_t BFSCPUPushTreeBenchmark::layer_microbenchmark(*/
        /*const nid_t num_segments*/
/*) {*/
    /*if (num_segments != 1)*/
        /*std::cerr << "[Warning] BFS CPU push cannot be divided into segments. "*/
            /*<< "Num segments can only be 1, not " << num_segments*/
            /*<< "." << std::endl;*/
    
    /*layer_res_t results;*/
    /*results.num_segments = 1;*/
    /*results.segments     = new segment_res_t[1];*/

    /*results.segments[0] = benchmark_segment(0, g->num_nodes);*/

    /*return results;*/
/*}*/

/*segment_res_t BFSCPUPushTreeBenchmark::benchmark_segment(*/
        /*UNUSED const nid_t start_id, UNUSED const nid_t end_id*/
/*) {*/
    /*segment_res_t result;*/
    /*result.start_id   = 0;*/
    /*result.end_id     = g->num_nodes;*/
    /*result.num_edges  = g->num_edges;*/
    /*result.avg_degree = static_cast<float>(result.num_edges) / g->num_nodes;*/

    /*// Compute min and max degree.*/
    /*float min_degree, max_degree;*/
    /*min_degree = max_degree = static_cast<float>(g->index[1] - g->index[0]);*/
    /*#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)*/
    /*for (int nid = start_id + 1; nid < end_id; nid++) {*/
        /*float ndeg = static_cast<float>(g->index[nid + 1] - g->index[nid]);*/
        /*min_degree = min(min_degree, ndeg);*/
        /*max_degree = max(max_degree, ndeg);*/
    /*}*/
    /*result.min_degree = min_degree;*/
    /*result.max_degree = max_degree;*/

    /*SlidingWindow<nid_t> frontier(g->num_nodes);*/
    /*nid_t *parents = new nid_t[g->num_nodes];*/
    /*nid_t num_edges;*/

    /*Timer timer;*/
    /*double total_time = 0.0;*/

    /*for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {*/
        /*// Setup froniter  and parents.*/
        /*frontier.reset();*/
        /*conv_bitmap_to_window(get_frontier(), frontier, g->num_nodes);                */
        /*std::copy(get_parents(), get_parents() + g->num_nodes, parents);*/
        /*num_edges = 0;*/

        /*// Run kernel.*/
        /*timer.Start();*/
        /*(*epoch_kernel)(*g, parents, frontier, num_edges);*/
        /*timer.Stop();*/

        /*frontier.slide_window();*/
        /*total_time += timer.Millisecs();*/
    /*}*/

    /*// Save results.*/
    /*result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;*/
    /*result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;*/

    /*// Free memory.*/
    /*delete[] parents;*/

    /*return result;*/
/*}*/

/*BFSCPUPullTreeBenchmark::BFSCPUPullTreeBenchmark(const CSRUWGraph *g,*/
        /*bfs_cpu_pull_epoch_func epoch_kernel_)*/
    /*: BFSTreeBenchmark(g)*/
    /*, epoch_kernel(epoch_kernel_)*/
/*{}*/

/*segment_res_t BFSCPUPullTreeBenchmark::benchmark_segment(*/
        /*const nid_t start_id, const nid_t end_id*/
/*) {*/
    /*segment_res_t result;*/
    /*result.start_id   = start_id;*/
    /*result.end_id     = end_id;*/
    /*result.num_edges  = g->index[end_id] - g->index[start_id];*/
    /*result.avg_degree = static_cast<float>(result.num_edges) / g->num_nodes;*/

    /*// Compute min and max degree.*/
    /*float min_degree, max_degree;*/
    /*min_degree = max_degree = static_cast<float>(*/
            /*g->index[start_id + 1] - g->index[start_id]);*/
    /*#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)*/
    /*for (int nid = start_id + 1; nid < end_id; nid++) {*/
        /*float ndeg = static_cast<float>(g->index[nid + 1] - g->index[nid]);*/
        /*min_degree = min(min_degree, ndeg);*/
        /*max_degree = max(max_degree, ndeg);*/
    /*}*/
    /*result.min_degree = min_degree;*/
    /*result.max_degree = max_degree;*/


    /*Bitmap::Bitmap *frontier      = Bitmap::constructor(g->num_nodes);*/
    /*Bitmap::Bitmap *next_frontier = Bitmap::constructor(g->num_nodes);*/
    /*nid_t          *parents       = new nid_t[g->num_nodes];*/
    /*nid_t          num_nodes      = 0;*/

    /*Timer timer;*/
    /*double total_time = 0.0;*/

    /*for (int iter = 0; iter < BENCHMARK_SEGMENT_TIME_ITERS; iter++) {*/
        /*// Setup frontier and parents.*/
        /*Bitmap::copy(get_frontier(), frontier);*/
        /*Bitmap::reset(next_frontier);*/
        /*std::copy(get_parents(), get_parents() + g->num_nodes, parents);*/
        /*num_nodes = 0;*/

        /*// Run kernel.*/
        /*timer.Start();*/
        /*(*epoch_kernel)(*g, parents, start_id, end_id,*/
                /*frontier, next_frontier, num_nodes);*/
        /*timer.Stop();*/

        /*total_time += timer.Millisecs();*/
    /*}*/

    /*//Save results.*/
    /*result.millisecs = total_time / BENCHMARK_SEGMENT_TIME_ITERS;*/
    /*result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;*/
    
    /*// Free memory.*/
    /*Bitmap::destructor(&frontier);*/
    /*Bitmap::destructor(&next_frontier);*/
    /*delete[] parents;*/

    /*return result;*/
/*}*/

/**
 * Benchmarks a full SSSP CPU run.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch_kernel.
 *   - init_dist    <- initial distance array.
 *   - ret_dist     <- pointer to the address of the return distance array.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_sssp_cpu(
        const CSRWGraph &g, sssp_cpu_epoch_func epoch_kernel,
        SourcePicker<CSRWGraph> &sp
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    /*// Compute min and max degree.*/
    /*float min_degree, max_degree;*/
    /*min_degree = max_degree = static_cast<float>(g.index[1] - g.index[0]);*/
    /*#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)*/
    /*for (int nid = 1; nid < g.num_nodes; nid++) {*/
        /*float ndeg = static_cast<float>(g.index[nid + 1] - g.index[nid]);*/
        /*min_degree = min(min_degree, ndeg);*/
        /*max_degree = max(max_degree, ndeg);*/
    /*}*/
    result.min_degree = 0;
    result.max_degree = 0;

    // Define initial and return distances.
    weight_t *init_dist = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
        init_dist[i] = INF_WEIGHT;
    weight_t *ret_dist = nullptr;

    // Run kernel!
    nid_t previous_source = 0;
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
        nid_t cur_source	= sp.next_vertex();
        init_dist[previous_source] = INF_WEIGHT;
        init_dist[cur_source]      = 0;
        previous_source = cur_source;

        total_time += sssp_pull_cpu(g, epoch_kernel, init_dist, &ret_dist);
        delete[] ret_dist;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;    
}

/**
 * Benchmarks a full PR CPU run.
 * Parameters:
 *   - g            <- graph.
 *   - epoch_kernel <- cpu epoch_kernel.
 *   - init_score   <- initial score array.
 *   - ret_score     <- pointer to the address of the return score array.
 * Returns:
 *   Execution results.
 */
segment_res_t benchmark_pr_cpu(
		const CSRWGraph &g, pr_cpu_epoch_func epoch_kernel,
		SourcePicker<CSRWGraph> &sp
) {
    // Initialize results and calculate segment properties.
    segment_res_t result;
    result.start_id   = 0;
    result.end_id     = g.num_nodes;
    result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;
    result.num_edges  = g.num_edges;

    /*// Compute min and max degree.*/
    result.min_degree = 0;
    result.max_degree = 0;

    // Define initial and return scores.
    weight_t *init_score = new weight_t[g.num_nodes];
    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes; i++)
	init_score[i] = 1.0f/g.num_nodes;
    weight_t *ret_score = nullptr;

    // Run kernel!
    double total_time = 0.0;
    for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {
	total_time += pr_pull_cpu(g, epoch_kernel, init_score, &ret_score);
	delete[] ret_score;
    }

    // Save results.
    result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;
    result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;
    // TODO: divided by 2 is a conservative estimate.

    return result;    
}

/*segment_res_t benchmark_bfs_cpu(*/
        /*const CSRUWGraph &g, bfs_cpu_kernel kernel, SourcePicker<CSRUWGraph> &sp*/
/*) {*/
    /*// Initialize results and calculate segment properties.*/
    /*segment_res_t result;*/
    /*result.start_id   = 0;*/
    /*result.end_id     = g.num_nodes;*/
    /*result.avg_degree = static_cast<float>(g.num_edges) / g.num_nodes;*/
    /*result.num_edges  = g.num_edges;*/

    /*[>// Compute min and max degree.<]*/
    /*[>float min_degree, max_degree;<]*/
    /*[>min_degree = max_degree = static_cast<float>(g.index[1] - g.index[0]);<]*/
    /*[>#pragma omp parallel for reduction(min:min_degree) reduction(max:max_degree)<]*/
    /*[>for (int nid = 1; nid < g.num_nodes; nid++) {<]*/
        /*[>float ndeg = static_cast<float>(g.index[nid + 1] - g.index[nid]);<]*/
        /*[>min_degree = min(min_degree, ndeg);<]*/
        /*[>max_degree = max(max_degree, ndeg);<]*/
    /*[>}<]*/
    /*result.min_degree = 0;*/
    /*result.max_degree = 0;*/

    /*nid_t *parents = nullptr;*/

    /*// Run kernel!*/
    /*double total_time = 0.0;*/
    /*for (int iter = 0; iter < BENCHMARK_FULL_TIME_ITERS; iter++) {*/
        /*nid_t cur_source = sp.next_vertex();*/

        /*total_time += kernel(g, cur_source, &parents);*/
        
        /*delete[] parents;*/
    /*}*/

    /*// Save results.*/
    /*result.millisecs = total_time / BENCHMARK_FULL_TIME_ITERS;*/
    /*result.gteps     = result.num_edges / (result.millisecs / 1000) / 1e9 / 2;*/
    /*// TODO: divided by 2 is a conservative estimate.*/

    /*return result;    */
/*}*/

#endif // SRC_BENCHMARKS__CPU_BENCHMARK_CUH
