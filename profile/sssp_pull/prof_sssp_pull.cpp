#include <cstdlib>

#include "../../src/gapbs.h"
#include "../../src/algorithms/sssp_pull.h"

int main(int argc, char *argv[]) {
    // Obtain command line configs.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Build graph.
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t ordered_g = b.RelabelByDegree(g);

    // Run kernel.
    int *distances = nullptr;
    sssp_pull(ordered_g, &distances);

    return EXIT_SUCCESS;
}
