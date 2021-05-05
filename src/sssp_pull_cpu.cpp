#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <omp.h>

#include "gapbs.h"
#include "util.h"

//#define DEBUG_ON
//#define PROFILE_ON

/**
 * Based on Princeton's LLAMAS SSSP pull algorithm using the CSRGraph 
 * implementation from GAPBS.
 */

// Shared var: number of updated nodes.
int num_changed = 1;

void kernel_sssp_pull(wgraph_t &g, int *prop_dist, int *tmp_dist, int tid, 
        int num_threads
) {
#ifdef DEBUG_ON
    int iters = 0; // Iteration count.
    if (tid == 0) {
        std::cout << "Config" << std::endl;
        std::cout << "  - num threads: " << num_threads << std::endl;
    }
#endif // DEBUG_ON

    // While changes still need to be propagated.
    while (true) {
        #pragma omp barrier
        
        if (num_changed == 0) { break; }

        #pragma omp barrier
        // Reset.
        if (tid == 0) { 
#ifdef DEBUG_ON
            std::cout << "  - num_changed: " << num_changed << std::endl;
#endif // DEBUG_ON
            num_changed = 0; 
#ifdef DEBUG_ON // DEBUG_ON
            std::cout << "iteration: " << iters << std::endl;            
            iters++;
#endif // END DEBUG_ON
        }

        // Propagate and reduce.
        for (int nid = tid; nid < g.num_nodes(); nid += num_threads) {
            weight_t cur_dist = prop_dist[nid];

            for (wnode_t wnode : g.in_neigh(nid)) {
                if (prop_dist[wnode.v] != MAX_WEIGHT) { 
                    weight_t new_dist = prop_dist[wnode.v] + wnode.w;
                    tmp_dist[nid] = std::min(tmp_dist[nid], new_dist);
                }
            }
        }

        #pragma omp barrier

        // Apply phase.
        for (int nid = tid; nid < g.num_nodes(); nid += num_threads) {
            if (tmp_dist[nid] != prop_dist[nid]) {
                #pragma omp atomic
                num_changed++;

                prop_dist[nid] = tmp_dist[nid];
            }
        }
    }
}

void sssp_pull(wgraph_t &g, int **ret_dist) {
    // Initialize distance arrays.
    weight_t *dist = (weight_t *) malloc(g.num_nodes() * sizeof(weight_t));
    weight_t *tmp  = (weight_t *) malloc(g.num_nodes() * sizeof(weight_t));

    #pragma omp parallel for
    for (int i = 0; i < g.num_nodes(); i++) {
        dist[i] = MAX_WEIGHT;
        tmp[i]  = MAX_WEIGHT;
    }

    // Arbitrary: Set lowest degree node as source. 
    dist[0] = 0; tmp[0] = 0;

    // Start kernel.
    Timer timer;
    std::cout << "Starting kernel." << std::endl;
    timer.Start();

    //omp_set_num_threads(4);
    #pragma omp parallel
    {
#ifndef PROFILE_ON // !PROFILE_ON
        kernel_sssp_pull(g, dist, tmp, omp_get_thread_num(),
                omp_get_num_threads());
#endif // END PROFILE_ON
    }

    timer.Stop();
    std::cout << "Kernel completed in: " << timer.Millisecs() << " ms." 
        << std::endl;

    // Assign output.
    *ret_dist = dist;
}

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);

    // Run SSSP.
    int *distances = nullptr;
    sssp_pull(ordered_g, &distances);


    if (cli.scale() <= 4) {
        std::cout << "node: distance" << std::endl;
        for (int i = 0; i < g.num_nodes(); i++)
            std::cout << i << ": " << distances[i] << std::endl;
    }
    
    return EXIT_SUCCESS;
}
