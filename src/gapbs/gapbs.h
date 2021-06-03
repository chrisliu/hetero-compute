/**
 * GAPBS header and helper functions. 
 */

#ifndef SRC_GAPBS__GAPBS_H
#define SRC_GAPBS__GAPBS_H

#include <ostream>

// Custom CSR graph (for types).
#include "../graph.h"

// Modified implementations of GAPBS.
#include "builder.h" 
#include "../../deps/gapbs/src/graph.h"
// Original implmeentations of GAPBS.
#include "../../deps/gapbs/src/command_line.h"

using gapbs_wnode_t  = NodeWeight<nid_t, weight_t>;
using gapbs_wgraph_t = CSRGraph<nid_t, gapbs_wnode_t>;

using WeightedBuilder = BuilderBase<nid_t, gapbs_wnode_t, weight_t>;

/**
 * Serialize GAPBS graph.
 * Layout (contiguous; no delimiters):
 *   - Number of nodes.
 *   - Number of directed edges (size of neighbors list).
 *   - Index list [size = (# of nodes + 1) * sizeof(nid_t)].
 *   - Neighbors list [size = (# of directed edges) * sizeof(wnode_t)];
 */
std::ostream &operator<<(std::ostream &os, gapbs_wgraph_t &g) {
    nid_t    num_nodes = g.num_nodes();
    offset_t num_edges = g.num_edges_directed();

    // Write out metadata.
    os.write(reinterpret_cast<char *>(&num_nodes), sizeof(nid_t));
    os.write(reinterpret_cast<char *>(&num_edges), sizeof(offset_t));

    // Write out index list.
    os.write(reinterpret_cast<char *>(g.VertexOffsets(false).data()), 
            (num_nodes + 1) * sizeof(offset_t));

    // Write out neighbors.
    // Layout of gapbs_wnode_t is identical to wnode_t defined in src/graph.h.
    os.write(reinterpret_cast<char *>(g.in_neigh(0).begin()),
            num_edges * sizeof(gapbs_wnode_t));
    
    return os;
}

/**
 * Generates ordered graph given command line arguments.
 * Parameters:
 *   - cli <- command line arguments.
 * Returns:
 *   ordered graph based on specification.
 */
gapbs_wgraph_t make_graph(CLBase &cli) {
    WeightedBuilder b(cli);
    gapbs_wgraph_t g = b.MakeGraph();
    return b.RelabelByDegree(g);
}

#endif // SRC_GAPBS__GAPBS_H
