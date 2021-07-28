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
#include <fstream>
#include <istream>
#include <limits>
#include <random>

/** Type info. */
using nid_t    = std::int32_t;     // Node ID type.
using weight_t = float;            // Edge weight type.
using offset_t = std::int64_t;     // Edge ID type. Represents number of edges
                                   // (Equivalent to GAPBS SGOffset).

// Infinite weight.
const weight_t INF_WEIGHT = std::numeric_limits<weight_t>::infinity();

/** Directed edge struct (which neighbor node ID and what weight). */
struct wnode_t {
    nid_t    v;                    // Destination node ID.
    weight_t w;                    // Edge weight.
};

/** Implementation of unweighted CSR graph. */
class CSRUWGraph {
private:
    /** Sugar for iterators (see gapbs/src/graph.h). */
    class Neighborhood {
    public:
        Neighborhood(nid_t nid_, offset_t *index_, nid_t *neighbors_) 
            : nid(nid_), index(index_), neighbors(neighbors_) {}

        using iterator = nid_t *;
        iterator begin() { return &neighbors[index[nid]]; }
        iterator end() { return &neighbors[index[nid + 1]]; }

    private:
        nid_t    nid;
        offset_t *index;
        nid_t    *neighbors;
    };

public:
    offset_t *index     = nullptr; // Points to a give node's neighbors.
    nid_t    *neighbors = nullptr; // Neighbor list (neighbor nid).
    nid_t    num_nodes  = 0;       // Number of nodes.
    offset_t num_edges  = 0;       // Number of edges.

    offset_t get_degree(nid_t nid) const {
        return index[nid + 1] - index[nid];
    }

    Neighborhood get_neighbors(nid_t nid) const {
        return Neighborhood(nid, index, neighbors);
    }

    bool has_self_cycle(nid_t nid) const {
        for (const auto &nei : get_neighbors(nid))
            if (nei == nid) return true;
        return false;
    }
};

/** Implementation of weighted CSR graph. */
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
        nid_t    nid;
        offset_t *index;
        wnode_t  *neighbors;
    };

public:
    offset_t *index     = nullptr; // Points to a give node's neighbors.
    wnode_t  *neighbors = nullptr; // Neighbor list (neighbor nid and weight).
    nid_t    num_nodes  = 0;       // Number of nodes.
    offset_t num_edges  = 0;       // Number of edges.

    offset_t get_degree(nid_t nid) const {
        return index[nid + 1] - index[nid];
    }

    Neighborhood get_neighbors(nid_t nid) const {
        return Neighborhood(nid, index, neighbors);
    }

    bool has_self_cycle(nid_t nid) const {
        for (const auto &nei : get_neighbors(nid))
            if (nei.v == nid) return true;
        return false;
    }
};

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

template <typename GraphT>
class SourcePicker {
public:
    // TODO: randomly generate seed.
    SourcePicker(const GraphT * const g_, nid_t const_source_ = -1) 
        : g(g_), udist(0, g->num_nodes - 1), const_source(const_source_)
    {
        rng = std::mt19937(rand_seed);
    }

    nid_t next_vertex() {
        if (const_source >= 0) return const_source;

        nid_t source;
        do {
            source = udist(rng);
        } while (g->get_degree(source) == 0 or 
                (g->get_degree(source) == 1 and g->has_self_cycle(source)));
        return source;
    }

    void reset() {
        rng = std::mt19937(rand_seed);
    }

private:
    const GraphT * const                 g;
    nid_t                                const_source; // Always pick same 
                                                       // vertex.
    std::mt19937                         rng;
    std::uniform_int_distribution<nid_t> udist;

    const int rand_seed = 37525;
};

/**
 * Computes the starting and ending node IDs for each segments such that
 * the average degree of each segment is roughly (# of edges) / @num_segments.
 * Parameters:
 *   - num_segments <- number of segments.
 * Returns:
 *   List of length @num_segments + 1. For each segment i, the segment's range
 *   is defined as [range[i], range[i + 1]). Memory is dynamically allocated so 
 *   it must be freed to prevent memory leaks.
 */
nid_t *compute_equal_edge_ranges(const CSRWGraph &g, const nid_t num_segments) {
    nid_t *seg_ranges = new nid_t[num_segments + 1];    
    seg_ranges[0] = 0;

    offset_t avg_deg = g.num_edges / num_segments;

    nid_t end_id   = 0;
    int   seg_id   = 0;
    int   seg_deg  = 0;

    while (end_id != g.num_nodes) {
        seg_deg += g.get_degree(end_id);

        // If segment exceeds average degree, save it and move on to next.
        if (seg_deg >= avg_deg) {
            seg_ranges[seg_id + 1] = end_id;
            seg_deg = 0; // Reset segment degree.
            seg_id++;
        }

        end_id++;
    }

    // If last segment hasn't been saved yet (almost guaranteed to happen).
    if (seg_id != num_segments)
        seg_ranges[seg_id + 1] = end_id;

    return seg_ranges;
}

/*****************************************************************************
 ***** Graph Input Functions *************************************************
 *****************************************************************************/

/**
 * Deserializes GAPBS generated unweighted graph into new data structure.
 */
std::istream& operator>>(std::istream &is, CSRUWGraph &g) {
    // Read in metadata.
    is.read(reinterpret_cast<char *>(&g.num_nodes), sizeof(nid_t));
    is.read(reinterpret_cast<char *>(&g.num_edges), sizeof(offset_t));

    // Read in index list.
    if (g.index == nullptr) { g.index = new offset_t[g.num_nodes + 1]; }
    is.read(reinterpret_cast<char *>(g.index), 
            (g.num_nodes + 1) * sizeof(offset_t));

    // Read in neighbor list.
    if (g.neighbors == nullptr) { g.neighbors = new nid_t[g.num_edges]; }
    is.read(reinterpret_cast<char *>(g.neighbors), 
            g.num_edges * sizeof(wnode_t));

    return is;
}

/**
 * Deserializes GAPBS generated weighted graph into new data structure.
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

/**
 * Wrapper function that deserializes graph from filename.
 */
__inline__
CSRWGraph load_graph_from_file(char *filename) {
    CSRWGraph g;
    std::ifstream ifs(filename, std::ifstream::in | std::ifstream::binary);
    ifs >> g;
    ifs.close();
    return g;
}

#endif // SRC__GRAPH_H
