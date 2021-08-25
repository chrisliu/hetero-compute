/**
 * Implementation of a CSR graph that is compatible with both CPU and GPU
 * kernels.
 * 
 * Why not use GAPBS's graph?
 *   > GAPBS's graph's index list are just pointers to elements in the neighbor
 *     list. It is possible to translate it to the GPU's neighbor list but
 *     it's an extra step.
 */

#ifndef SRC__GRAPH_CUH
#define SRC__GRAPH_CUH

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <istream>
#include <limits>
#include <random>
#include <type_traits>

#include "cuda.cuh"
#include "util.h"

/** Type info. */
using nid_t    = std::int32_t;     // Node ID type.
using weight_t = float;            // Edge weight type.
using offset_t = std::int64_t;     // Edge ID type. Represents number of edges
                                   // (Equivalent to GAPBS SGOffset).

// Infinite weight.
const weight_t INF_WEIGHT = std::numeric_limits<weight_t>::infinity();

// Invalid node.
const nid_t INVALID_NODE = -1;

/** Directed edge struct (which neighbor node ID and what weight). */
struct wnode_t {
    nid_t    v;                    // Destination node ID.
    weight_t w;                    // Edge weight.
};

template <typename NeighborT>
class GenericCSRGraph {
private:
    /** Sugar for iterators (see gapbs/src/graph.h). */
    class Neighborhood {
    public:
        Neighborhood(nid_t nid_, offset_t *index_, NeighborT *neighbors_) 
            : nid(nid_), index(index_), neighbors(neighbors_) {}

        using iterator = NeighborT *;
        iterator begin() { return &neighbors[index[nid]]; }
        iterator end() { return &neighbors[index[nid + 1]]; }

    private:
        nid_t     nid;
        offset_t  *index;
        NeighborT *neighbors;
    };

public:
    offset_t  *index     = nullptr; // Points to a give node's neighbors.
    NeighborT *neighbors = nullptr; // Neighbor list (neighbor nid).
    nid_t     num_nodes  = 0;       // Number of nodes.
    offset_t  num_edges  = 0;       // Number of edges.

    ~GenericCSRGraph() {
        delete[] index;
        delete[] neighbors;
    }

    offset_t get_degree(nid_t nid) const {
        return index[nid + 1] - index[nid];
    }

    Neighborhood get_neighbors(nid_t nid) const {
        return Neighborhood(nid, index, neighbors);
    }
    
    bool has_self_cycle(nid_t nid) const {
        for (NeighborT nei : get_neighbors(nid))
            if (nei == nid) return true;
        return false;
    }
};

/**
 * Specialized has_self_cycle for weighted node types.
 */
template <>
bool GenericCSRGraph<wnode_t>::has_self_cycle(nid_t nid) const {
    for (wnode_t nei : get_neighbors(nid))
        if (nei.v == nid) return true;
    return false;
}

/** Define weighted & unweighted graph types. */
using CSRWGraph  = GenericCSRGraph<wnode_t>;
using CSRUWGraph = GenericCSRGraph<nid_t>;

/*****************************************************************************
 ***** Helper Functions ******************************************************
 *****************************************************************************/

template <typename GraphT,
          typename = typename std::enable_if<
              is_templated_instance<GraphT, GenericCSRGraph>::value>>
class SourcePicker {
public:
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
        } while (g->get_degree(source) == 0
                 or (g->get_degree(source) == 1 and g->has_self_cycle(source)));

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
 *   - g            <- weighted or unweighted graph.
 *   - num_segments <- number of segments.
 *   - chunk_size   <- each chunk is divisible by @chunk_size (e.g., 
 *                     chunk_size of 4 works for range [0, 16) but not [0, 9)).
 * Returns:
 *   List of length @num_segments + 1. For each segment i, the segment's range
 *   is defined as [range[i], range[i + 1]). Memory is dynamically allocated so 
 *   it must be freed to prevent memory leaks.
 */
template <typename GraphT,
          typename = typename std::enable_if<
              is_templated_instance<GraphT, GenericCSRGraph>::value>>
nid_t *compute_equal_edge_ranges(const GraphT &g, const nid_t num_segments,
        nid_t chunk_size = 1) {
    nid_t *seg_ranges = new nid_t[num_segments + 1];    
    seg_ranges[0] = 0;

    offset_t avg_deg = g.num_edges / num_segments;

    nid_t    end_id   = 0;
    nid_t    seg_id   = 0;
    offset_t seg_deg  = 0;

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

    // Align segments to chunk size.
    for (int i = 1; i < num_segments; i++) {
        nid_t diff = seg_ranges[i] % chunk_size;

        if (diff != 0) {
            // Align to closest mark.
            if (diff < chunk_size / 2)
                seg_ranges[i] = seg_ranges[i] / chunk_size * chunk_size;
            else
                seg_ranges[i] = (seg_ranges[i] / chunk_size + 1) * chunk_size;
        }
    }

    return seg_ranges;
}

/*****************************************************************************
 ***** Graph Input Functions *************************************************
 *****************************************************************************/

template <typename NeighborT>
std::istream& operator>>(std::istream &is, GenericCSRGraph<NeighborT> &g) {
    // Read in metadata.
    is.read(reinterpret_cast<char *>(&g.num_nodes), sizeof(nid_t));
    is.read(reinterpret_cast<char *>(&g.num_edges), sizeof(offset_t));

    // Read in index list.
    if (g.index == nullptr) { g.index = new offset_t[g.num_nodes + 1]; }
    is.read(reinterpret_cast<char *>(g.index), 
            (g.num_nodes + 1) * sizeof(offset_t));

    // Read in neighbor list.
    if (g.neighbors == nullptr) { g.neighbors = new NeighborT[g.num_edges]; }
    is.read(reinterpret_cast<char *>(g.neighbors), 
            g.num_edges * sizeof(NeighborT));

    return is;
}

/**
 * Wrapper function that deserializes graph from filename.
 */
template <typename GraphT,
          typename = typename std::enable_if<
              is_templated_instance<GraphT, GenericCSRGraph>::value>>
inline
GraphT load_graph_from_file(char *filename) {
    GraphT g;
    std::ifstream ifs(filename, std::ifstream::in | std::ifstream::binary);
    ifs >> g;
    ifs.close();
    return g;
}

/**
 * Copy specified subgraph to cu_index and cu_neighbors.
 * Parameters:
 *   - g               <- graph.
 *   - cu_index_to     <- pointer to target cuda index array.
 *   - cu_neighbors_to <- pointer to traget cuda neighbors array.
 *   - start_id        <- starting node id.
 *   - end_id          <- ending node id (exclusive).
 */
template <typename NeighborT>
inline
void copy_subgraph_to_device(const GenericCSRGraph<NeighborT> &g, 
        offset_t **cu_index_to, NeighborT **cu_neighbors_to, 
        const nid_t start_id, const nid_t end_id
) {
    // Local copies of cu_index and cu_neighbors.
    offset_t  *cu_index     = nullptr;
    NeighborT *cu_neighbors = nullptr;

    // Get number of nodes and edges.
    size_t index_size     = (end_id - start_id + 1) * sizeof(offset_t);
    size_t neighbors_size = (g.index[end_id] - g.index[start_id])
                                * sizeof(NeighborT);

    // Make a down-shifted copy of the index. Index values in the graph will no 
    // longer point to the same location in memory.
    offset_t *subgraph_index = new offset_t[end_id - start_id + 1];
    #pragma omp parallel for
    for (int i = start_id; i <= end_id; i++)
        subgraph_index[i - start_id] = g.index[i] - g.index[start_id];

    // Allocate memory and copy subgraph over.
    CUDA_ERRCHK(cudaMalloc((void **) &cu_index, index_size));
    CUDA_ERRCHK(cudaMalloc((void **) &cu_neighbors, neighbors_size));
    CUDA_ERRCHK(cudaMemcpy(cu_index, subgraph_index, 
                index_size, cudaMemcpyHostToDevice));
    CUDA_ERRCHK(cudaMemcpy(cu_neighbors, &g.neighbors[g.index[start_id]],
                neighbors_size, cudaMemcpyHostToDevice));

    // Delete main memory copy of subgraph index.
    delete[] subgraph_index;

    // Assign output.
    *cu_index_to     = cu_index;
    *cu_neighbors_to = cu_neighbors;
}


#endif // SRC__GRAPH_CUH
