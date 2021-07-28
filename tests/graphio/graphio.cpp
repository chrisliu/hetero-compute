#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/gapbs/gapbs.h"

const std::string wgraph_fname = "graph.wsg";
const std::string uwgraph_fname = "graph.sg";

int main() {
    int argc = 5;
    char *argv[] = {
       (char *) "graphio",
       (char *) "-g",
       (char *) "16",
       (char *) "-k", 
       (char *) "16",
       NULL
    };
    CLBase cli(argc, argv);

    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    GapbsWGraph gapbs_g = make_wgraph(cli);
    
    // Create weighted graph.
    {
        std::ofstream ofs(wgraph_fname, 
                std::ofstream::out | std::ofstream::binary);
        ofs << gapbs_g;
        ofs.close();
    }

    // Create unweighted graph.
    {
        GapbsGraph uwg(gapbs_g);
        std::ofstream ofs(uwgraph_fname,
                std::ofstream::out | std::ofstream::binary);
        ofs << uwg;
        ofs.close();
    }

    // Get weighted graph.
    CSRWGraph g;
    {
        std::ifstream ifs(wgraph_fname, 
                std::ifstream::in | std::ifstream::binary);
        Timer timer; timer.Start();
        ifs >> g;
        timer.Stop();
        ifs.close();

        std::cout << "Weighted graph load time:   " 
            << timer.Millisecs() << " ms." << std::endl;
    }

    // Get unweighted graph.
    CSRUWGraph uwg;
    {
        std::ifstream ifs(uwgraph_fname, 
                std::ifstream::in | std::ifstream::binary);
        Timer timer; timer.Start();
        ifs >> uwg;
        timer.Stop();
        ifs.close();

        std::cout << "Unweighted graph load time: " 
            << timer.Millisecs() << " ms." << std::endl;
    }

    // Testing code.
    assert( g.num_nodes == gapbs_g.num_nodes() );
    assert( uwg.num_nodes == gapbs_g.num_nodes() );
    for (nid_t nid = 0; nid < g.num_nodes; nid++) {
        assert( g.get_degree(nid) == gapbs_g.in_degree(nid) );
        assert( uwg.get_degree(nid) == gapbs_g.in_degree(nid) );

        for (int v = 0; v < g.get_degree(nid); v++) {
            GapbsWNode gapbs_node = *(gapbs_g.in_neigh(nid).begin() + v);

            // Verify neighbor node ID and weight.
            wnode_t node = *(g.get_neighbors(nid).begin() + v);
            assert( node.v == gapbs_node.v );
            assert( node.w == gapbs_node.w );

            // Verify neighbor node ID.
            nid_t nei_id = *(uwg.get_neighbors(nid).begin() + v);
            assert( nei_id == gapbs_node.v );
        }
    }

    std::cout << "All tests passed!" << std::endl;

    // Remove generated files.
    std::remove(wgraph_fname.c_str());
    std::remove(uwgraph_fname.c_str());

    return EXIT_SUCCESS;
}

/** BUGGY */

//#include <fstream>
//#define CATCH_CONFIG_MAIN
//#include <catch2/catch.hpp>

//#include "../../src/graph.h"
//#include "../../src/gapbs.h"

//TEST_CASE( "serializing a GAPBS graph and deserializing into custom graph",
        //"[gapbs, graph]"
//) {
    //// Initialize graph.
    //int argc = 5;
    //char *argv[] = {
       //(char *) "graphio",
       //(char *) "-g",
       //(char *) "10",
       //(char *) "-k", 
       //(char *) "8",
       //NULL
    //};
    //CLBase cli(argc, argv);

    //REQUIRE( cli.ParseArgs() );
    //REQUIRE( cli.scale() == 10 );
    //REQUIRE( cli.degree() == 8 );

    //gapbs_wgraph_t gapbs_g = make_graph(cli); 

    //SECTION( "serialize GAPBS graph" ) {
        //REQUIRE_NOTHROW([&](){
            //std::ofstream ofs("graph.wsg", 
                    //std::ofstream::out | std::ofstream::binary);
            //ofs << gapbs_g;
            //ofs.close();
        //}());
    //}

    //CSRWGraph g;

    //SECTION( "deserialize graph" ) {
        //REQUIRE_NOTHROW([&](){
            //std::ifstream ifs("graph.wsg", 
                    //std::ifstream::in | std::ifstream::binary);
            //ifs >> g;
            //ifs.close();
        //}());
    //}

    ////SECTION( "graphs are identical" ) {
        ////REQUIRE( g.num_nodes == gapbs_g.num_nodes() ); 

        ////for (nid_t nid = 0; nid < g.num_nodes; nid++) {
            ////REQUIRE( g.get_degree(nid) == gapbs_g.in_degree(nid) );            

            ////for (int v = 0; v < g.get_degree(nid); v++) {
                ////gapbs_wnode_t gapbs_node = *(gapbs_g.in_neigh(nid).begin() + v);
                ////wnode_t node = *(g.get_neighbors(nid).begin() + v);

                ////REQUIRE( node.v == gapbs_node.v );
                ////REQUIRE( node.w == gapbs_node.w );
            ////}
        ////}
    ////}
//}
