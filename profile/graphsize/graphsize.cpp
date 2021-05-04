#include <cstdlib>
#include <iostream>

#include "../../src/gapbs.h"

int main(int argc, char *argv[]) {

    // Obtain command line configs. 
    CLConvert cli(argc, argv, "");
    if (not cli.ParseArgs())
        return EXIT_FAILURE;

    // Build ordered graph (by descendnig degree).
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();
    wgraph_t sorted_g = b.RelabelByDegree(g);

    // Write out file.
    WeightedWriter ww(sorted_g);
    ww.WriteGraph(cli.out_filename(), cli.out_sg());

    return EXIT_SUCCESS;
}
