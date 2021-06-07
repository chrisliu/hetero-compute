/*
 * Given graph parameters emit a graph binary file.
 */

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "../../src/gapbs/gapbs.h"

std::string gen_filename(CLBase &cli);

int main(int argc, char *argv[]) {
    // Grab command line args.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Generate graph.
    gapbs_wgraph_t g = make_graph(cli);
    
    // Emit graph.
    std::ofstream ofs(gen_filename(cli), 
            std::ofstream::out | std::ofstream::binary);
    ofs << g;
    ofs.close();
    
    return EXIT_SUCCESS;
}

std::string gen_filename(CLBase &cli) {
    std::stringstream ss;
    ss << "graph_scale" << cli.scale() << "_degree" << cli.degree() << ".wsg";
    return ss.str();
}
