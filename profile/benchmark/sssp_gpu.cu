#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../../src/gapbs.h"
#include "../../src/benchmark.cuh"

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build ordered graph (by descending degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);

    // Run benchmark.
    SSSPGPUBenchmark bench(&ordered_g);
    
    /*layer_res_t res = bench.layer_microbenchmark(3); */
    /*for (size_t i = 0; i < res.num_segments; i++) {*/
        /*std::cout << "Segment " << (i + 1) << std::endl;*/
        /*std::cout << " > Average degree:   " << res.segments[i].avg_degree */
            /*<< std::endl;*/
        /*std::cout << " > # of edges:       " << res.segments[i].num_edges*/
            /*<< std::endl;*/
        /*std::cout << " > Computation time: " << res.segments[i].millisecs*/
            /*<< std::endl;*/
        /*std::cout << " > TEPS:             " << res.segments[i].teps*/
            /*<< std::endl;*/
    /*}    */

    // Run tree decomposition benchmark.
    tree_res_t tree_res = bench.tree_microbenchmark(3);
    
    // Write out results.
    std::fstream ofs("results.yaml", std::ofstream::out);
    ofs << tree_res;
    ofs.close();

    return EXIT_SUCCESS;
}
