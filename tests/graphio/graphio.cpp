#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../../src/graph.h"
#include "../../src/util.h"
#include "../../src/gapbs/gapbs.h"

int main() {
    int argc = 5;
    char *argv[] = {
       (char *) "graphio",
       (char *) "-g",
       (char *) "23",
       (char *) "-k", 
       (char *) "16",
       NULL
    };
    CLBase cli(argc, argv);

    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    gapbs_wgraph_t gapbs_g = make_graph(cli);
    
    std::ofstream ofs("graph.wsg", std::ofstream::out | std::ofstream::binary);
    ofs << gapbs_g;
    ofs.close();

    CSRWGraph g;
    std::ifstream ifs("graph.wsg", std::ifstream::in | std::ifstream::binary);
    Timer timer; timer.Start();
    ifs >> g;
    timer.Stop();
    ifs.close();

    std::cout << "Graph load time: " << timer.Millisecs() << " ms." 
        << std::endl;

    assert( g.num_nodes == gapbs_g.num_nodes() );
    for (nid_t nid = 0; nid < g.num_nodes; nid++) {
        assert( g.get_degree(nid) == gapbs_g.in_degree(nid) );

        for (int v = 0; v < g.get_degree(nid); v++) {
            gapbs_wnode_t gapbs_node = *(gapbs_g.in_neigh(nid).begin() + v);
            wnode_t       node       = *(g.get_neighbors(nid).begin() + v);

            assert( node.v == gapbs_node.v );
            assert( node.w == gapbs_node.w );
        }
    }

    std::cout << "All tests passed!" << std::endl;

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
