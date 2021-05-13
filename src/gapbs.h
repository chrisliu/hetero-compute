#ifndef GAPBS_H
#define GAPBS_H

/**
 * Include files from Scott's benchmark.
 */

#include <cstdint>
#include <limits>

#include "gapbs/builder.h" // Custom implementation of builder.
#include "../deps/gapbs/src/command_line.h"
#include "../deps/gapbs/src/graph.h"
#include "../deps/gapbs/src/reader.h"
#include "../deps/gapbs/src/writer.h"

using nid_t = std::int32_t;
using weight_t = float;

using wnode_t = NodeWeight<nid_t, weight_t>;
using wgraph_t = CSRGraph<nid_t, wnode_t>;

using WeightedBuilder = BuilderBase<nid_t, wnode_t, weight_t>;
using WeightedWriter = WriterBase<nid_t, wnode_t>; 
using WeightedReader = Reader<nid_t, wnode_t, weight_t>;

const weight_t MAX_WEIGHT = std::numeric_limits<weight_t>::max();

#endif // GAPBS_H
