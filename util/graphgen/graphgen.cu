/*
 * Given graph parameters emit a graph binary file.
 */

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

#include "../../src/gapbs/gapbs.cuh"
#include "../../src/util.h"

#define WEIGHTED_ON
#define UNWEIGHTED_ON

std::string gen_filename(CLBase &cli, std::string ext);

int main(int argc, char *argv[]) {
    // Grab command line args.
    CLBase cli(argc, argv);
    if (not cli.ParseArgs()) { return EXIT_FAILURE; }

    // Generate graph.
    GapbsWGraph g = make_wgraph(cli);
    
    // Emit graph.
#ifdef WEIGHTED_ON
    {
        std::ofstream ofs(gen_filename(cli, ".wsg"), 
                std::ofstream::out | std::ofstream::binary);
        Timer t; t.Start();
        ofs << g;
        ofs.close();
        t.Stop();
        std::cout << "Weighted File I/O Time:     " 
            << std::fixed << std::setprecision(5) << t.Seconds() << std::endl;
    }
#endif // WEIGHTED_ON

#ifdef UNWEIGHTED_ON
    {
        Timer t; t.Start();
        GapbsGraph uwg(g);
        t.Stop();
        std::cout << "Unweighted Conversion Time: "
            << std::fixed << std::setprecision(5) << t.Seconds() << std::endl;
        t.Start();
        std::ofstream ofs(gen_filename(cli, ".sg"),
                std::ofstream::out | std::ofstream::binary);
        ofs << uwg;        
        t.Stop();
        std::cout << "Unweighted File I/O Time:   " 
            << std::fixed << std::setprecision(5) << t.Seconds() << std::endl;
    }
#endif // UNWEIGHTED_ON

    return EXIT_SUCCESS;
}

std::string gen_filename(CLBase &cli, std::string ext) {
    std::stringstream ss;
    ss << "graph_scale" << cli.scale() << "_degree" << cli.degree() << ext;
    return ss.str();
}
