/**
 * GAPBS header and helper functions. 
 */

#ifndef GAPBS_H
#define GAPBS_H

#include <cstdint>
#include <limits>

// Modified implementations of GAPBS.
#include "gapbs/builder.h" 
#include "gapbs/graph.h"
// Original implmeentations of GAPBS.
#include "../deps/gapbs/src/command_line.h"
#include "../deps/gapbs/src/reader.h"
#include "../deps/gapbs/src/writer.h"

using nid_t    = std::int32_t;
using weight_t = float;

using wnode_t  = NodeWeight<nid_t, weight_t>;
using wgraph_t = CSRGraph<nid_t, wnode_t>;

using WeightedBuilder = BuilderBase<nid_t, wnode_t, weight_t>;
using WeightedWriter  = WriterBase<nid_t, wnode_t>; 
using WeightedReader  = Reader<nid_t, wnode_t, weight_t>;

const weight_t MAX_WEIGHT = std::numeric_limits<weight_t>::max() - 1;

typedef struct {
    nid_t    v; // Destination node ID.
    weight_t w; // Edge weight.
} cu_wnode_t;

/**
 * Converts a weighted graph to its CSR primatives.
 * Parameters:
 *   - g         <- graph.
 *   - index     <- range of neighbors and weights for each node. 
 *   - neighbors <- neighbors of a node and their respective weights.
 */
void wgraph_to_cugraph(const wgraph_t &g, nid_t **index, 
        cu_wnode_t **neighbors
) {
    // Alloc arrays.
    *index     = new nid_t[g.num_nodes()];
    *neighbors = new cu_wnode_t[2 * g.num_edges()];

    // Copy index.
    wnode_t *start = g.out_index_[0]; // First element in the neighbor list.
    (*index)[0] = 0;
    #pragma omp parallel for
    for (nid_t i = 1; i < g.num_nodes() + 1; i++)
        (*index)[i] = g.out_index_[i] - start;

    // Copy neighbor ids and weights.
    #pragma omp parallel for
    for (nid_t nid = 0; nid < g.num_nodes(); nid++) {
        auto wnode = g.out_index_[nid]; // Iterator to first neighbor.
        for (nid_t nei = (*index)[nid]; nei < (*index)[nid + 1]; nei++) {
            (*neighbors)[nei].v = wnode->v; 
            (*neighbors)[nei].w = wnode->w;
            wnode++; // Increment neighbor iterator.
        }
    }
}

#endif // GAPBS_H
