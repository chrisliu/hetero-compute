/**
 * Implementation of a CSR graph that is compatible with both CPU and GPU
 * kernels.
 * 
 * Why not use GAPBS's graph?
 *   > GAPBS's graph's index list are just pointers to elements in the neighbor
 *     list. It is possible to translate it to the GPU's neighbor list but
 *     it's an extra step.
 */

#ifndef SRC__GRAPH_H
#define SRC__GRAPH_H

#include <cstdint>
#include <istream>
#include <limits>

/** Type info. */
using nid_t    = std::int32_t;     // Node ID type.
using weight_t = float;            // Edge weight type.
using offset_t = std::int64_t;     // Edge ID type. Represents number of edges
                                   // (Equivalent to GAPBS SGOffset).

// Max weight is 90% of theoretical max.
const weight_t MAX_WEIGHT = 0.9 * std::numeric_limits<weight_t>::max();

/** Directed edge struct (which neighbor node ID and what weight). */
struct wnode_t {
    nid_t    v;                    // Destination node ID.
    weight_t w;                    // Edge weight.
};

/** Implementation of undirected CSR graph. */
// TODO: separate implementation from header (to allow multiple objects using
//       this header to be compiled independently and linked).
class CSRWGraph {
private:
    /** Sugar for iterators (see gapbs/src/graph.h). */
    class Neighborhood {
    public:
        Neighborhood(nid_t nid_, offset_t *index_, wnode_t *neighbors_) 
            : nid(nid_), index(index_), neighbors(neighbors_) {}

        using iterator = wnode_t *;
        iterator begin() { return &neighbors[index[nid]]; }
        iterator end() { return &neighbors[index[nid + 1]]; }

    private:
        nid_t nid;
        offset_t *index;
        wnode_t *neighbors;
    };

public:
    offset_t *index     = nullptr; // Points to a give node's neighbors.
    wnode_t  *neighbors = nullptr; // Neighbor list (neighbor nid and weight).
    nid_t    num_nodes  = 0;       // Number of nodes.
    offset_t num_edges  = 0;       // Number of edges.

    offset_t get_degree(nid_t nid) {
        return index[nid + 1] - index[nid];
    }

    Neighborhood get_neighbors(nid_t nid) const {
        return Neighborhood(nid, index, neighbors);
    }
};

/**
 * Deserializes GAPBS generated graph into new data structure.
 */
std::istream& operator>>(std::istream &is, CSRWGraph &g) {
    // Read in metadata.
    is.read(reinterpret_cast<char *>(&g.num_nodes), sizeof(nid_t));
    is.read(reinterpret_cast<char *>(&g.num_edges), sizeof(offset_t));

    // Read in index list.
    if (g.index == nullptr) { g.index = new offset_t[g.num_nodes + 1]; }
    is.read(reinterpret_cast<char *>(g.index), 
            (g.num_nodes + 1) * sizeof(offset_t));

    // Read in neighbor list.
    if (g.neighbors == nullptr) { g.neighbors = new wnode_t[g.num_edges]; }
    is.read(reinterpret_cast<char *>(g.neighbors), 
            g.num_edges * sizeof(wnode_t));

    return is;
}

#endif // SRC__GRAPH_H
