/**
 * Implementations of CPU and GPU SSSP pull with data race.
 */

#include <omp.h> 


#include "gapbs.h"
#include "util.h"

// Forward decl.
void kernel_sssp_pull_cpu(const wgraph_t &g, weight_t *dist, int tid, 
        int num_threads, int &updated);
void sssp_pull_cpu(const wgraph_t &g, weight_t **ret_dist);

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);
    //wgraph_t ordered_g = b.MakeGraph();

    // Run SSSP.
    weight_t *distances = nullptr;
    sssp_pull_cpu(ordered_g, &distances);

    if (cli.scale() <= 4) {
        std::cout << "node neighbors" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++) {
            std::cout << " > node " << i << std::endl;
            for (auto &out_nei : ordered_g.out_neigh(i)) {
            //for (auto &out_nei : ordered_g.in_neigh(i)) {
                std::cout << "    > node " << out_nei.v << ": " << out_nei.w
                    << std::endl;
            }
        }

        std::cout << "node: distance" << std::endl;
        for (int i = 0; i < ordered_g.num_nodes(); i++)
            std::cout << " > " << i << ": " << distances[i] << std::endl;
    }

    //WeightedWriter w(ordered_g);
    //w.WriteGraph("graph.wel");

    //WeightedReader r("graph.wel");

    return EXIT_SUCCESS;
}

/******************************************************************************
 ***** Kernels ****************************************************************
 ******************************************************************************/

/**
 * Runs SSSP kernel on CPU. Synchronization occurs in serial.
 * Parameters:
 *   - g        <- graph.
 *   - ret_dist <- pointer to the address of the return distance array.
 */
void sssp_pull_cpu(const wgraph_t &g, weight_t **ret_dist) {
    weight_t *dist = new weight_t[g.num_nodes()];

    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++)
        dist[i] = MAX_WEIGHT;

    // Arbitrary: Set lowest degree node as source.
    dist[0] = 0;

    // Start kernel.
    std::cout << "Starting kernel ..." << std::endl;
    Timer timer; timer.Start();

    int updated = 1;

    while (updated != 0) {
        updated = 0;

        #pragma omp parallel
        {
            kernel_sssp_pull_cpu(g, dist, omp_get_thread_num(), 
                    omp_get_num_threads(), updated);
        }

        // Implicit OMP BARRIER here (see "implicit barrier at end of parallel 
        // region").
    }

    timer.Stop();
    std::cout << "Kernel completed in: " << timer.Millisecs() << " ms."
        << std::endl;

    // Assign output.
    *ret_dist = dist;
}

/******************************************************************************
 ***** Epoch Kernels **********************************************************
 ******************************************************************************/

/**
 * Runs SSSP pull on CPU for one epoch.
 * Parameters:
 *   - g           <- graph.
 *   - dist        <- input distance and output distances computed by this 
 *                    epoch.
 *   - tid         <- processor id.
 *   - num_threads <- number of processors.
 *   - updated     <- global counter of number of nodes updated.
 */
void kernel_sssp_pull_cpu(const wgraph_t &g, weight_t *dist, int tid,
        int num_threads, int &updated
) {
    int local_updated = 0;

    // Propagate, reduce, and apply.
    for (int nid = tid; nid < g.num_nodes(); nid += num_threads) {
        for (wnode_t nei : g.in_neigh(nid)) {
            weight_t new_dist = dist[nid];

            // If neighbor has been initialized.
            if (dist[nei.v] != MAX_WEIGHT) {
                weight_t prop_dist = dist[nei.v] + nei.w;
                new_dist = std::min(prop_dist, new_dist);
            }

            // Update distance if applicable.
            if (new_dist != dist[nid]) {
                dist[nid] = new_dist;
                local_updated++;
            }
        }
    }

    // Push update count.
    #pragma omp atomic
    updated += local_updated;
}
