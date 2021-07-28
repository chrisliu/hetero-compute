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
// Original implmeentations of GAPBS.
#include "../../deps/gapbs/src/command_line.h"
#include "../../deps/gapbs/src/graph.h"

using GapbsWNode  = NodeWeight<nid_t, weight_t>;
using GapbsWGraph = CSRGraph<nid_t, GapbsWNode>;
using WeightedBuilder = BuilderBase<nid_t, GapbsWNode, weight_t>;

/** Gapbs Unweighted Graph. */
class GapbsGraph {
public:
    GapbsGraph(GapbsWGraph &other)
        : num_nodes(other.num_nodes())
        , num_edges_directed(other.num_edges_directed())
    {
        // Copy values.
        index     = new offset_t[other.num_nodes() + 1];
        neighbors = new nid_t[other.num_edges_directed()];

        // Generate prefix sum.
        index[0] = 0;
        for (int i = 0; i < other.num_nodes(); i++)
            index[i + 1] = index[i] + other.in_degree(i);

        // Populate neighbor list with just neighbor IDs.
        #pragma omp parallel for
        for (int i = 0; i < other.num_nodes(); i++) {
            offset_t offset    = index[i];
            int      nei_count = 0;
            for (GapbsWNode &wnode : other.in_neigh(i))
                neighbors[offset + nei_count++] = wnode.v;
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const GapbsGraph g);

private:
    offset_t *index;
    nid_t    *neighbors;
    nid_t    num_nodes;
    offset_t num_edges_directed;
};

/**
 * Serialize GAPBS unweighted graph.
 * Layout (contiguous; no delimiters):
 *   - Number of nodes.
 *   - Number of directed edges (size of neighbors list).
 *   - Index list [size = (# of nodes + 1) * sizeof(nid_t)].
 *   - Neighbors list [size = (# of directed edges) * sizeof(wnode_t)];
 */
std::ostream &operator<<(std::ostream &os, const GapbsGraph g) {
    nid_t num_nodes    = g.num_nodes;
    offset_t num_edges = g.num_edges_directed;
    
    // Write out metadata.
    os.write(reinterpret_cast<char *>(&num_nodes), sizeof(nid_t));
    os.write(reinterpret_cast<char *>(&num_edges), sizeof(offset_t));

    // Write out index list.
    os.write(reinterpret_cast<char *>(g.index), 
            (num_nodes + 1) * sizeof(offset_t));

    // Write out neighbors.
    os.write(reinterpret_cast<char *>(g.neighbors), 
            num_edges * sizeof(GapbsWNode));
    
    return os;
}


/**
 * Serialize GAPBS weighted graph.
 * Layout (contiguous; no delimiters):
 *   - Number of nodes.
 *   - Number of directed edges (size of neighbors list).
 *   - Index list [size = (# of nodes + 1) * sizeof(nid_t)].
 *   - Neighbors list [size = (# of directed edges) * sizeof(wnode_t)];
 */
std::ostream &operator<<(std::ostream &os, const GapbsWGraph &g) {
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
            num_edges * sizeof(GapbsWNode));
    
    return os;
}

/**
 * Generates ordered weighted graph given command line arguments.
 * Parameters:
 *   - cli <- command line arguments.
 * Returns:
 *   ordered weighted graph based on specification.
 */
GapbsWGraph make_wgraph(CLBase &cli) {
    WeightedBuilder b(cli);
    GapbsWGraph g = b.MakeGraph();
    return b.RelabelByDegree(g);
}

#endif // SRC_GAPBS__GAPBS_H
