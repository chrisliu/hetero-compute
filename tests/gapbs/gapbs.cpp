#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "../../src/gapbs.h"

TEST_CASE("weighted graphs are properly deconstructed", "[gapbs]") {
    // Initialize graph.
    int argc = 5;
    char *argv[] = {
       (char *) "gapbs",
       (char *) "-g",
       (char *) "10",
       (char *) "-k", 
       (char *) "8",
       NULL
    };
    CLBase cli(argc, argv);

    REQUIRE( cli.ParseArgs() );
    REQUIRE( cli.scale() == 10 );
    REQUIRE( cli.degree() == 8 );

    // Create reference graph.
    WeightedBuilder b(cli);
    wgraph_t g = b.MakeGraph();

    // Generate CUDA graph.
    nid_t      *index     = nullptr;
    cu_wnode_t *neighbors = nullptr;
    wgraph_to_cugraph(g, &index, &neighbors);

    // Check neighbors.
    for (nid_t nid = 0; nid < g.num_nodes(); nid++) {
        auto actual = g.out_neigh(nid).begin(); // Iterator to first neighbor.
        for (nid_t i = index[nid]; i < index[nid + 1]; i++) {
            REQUIRE( neighbors[i].v == actual->v );
            REQUIRE( neighbors[i].w == actual->w );
            actual++; // Increment neighbor iterator.
        }
    }
}
