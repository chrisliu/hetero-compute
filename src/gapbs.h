/**
 * Include files from Scott's benchmark.
 */

#include <cstdint>

#include "../deps/gapbs/src/builder.h"
#include "../deps/gapbs/src/command_line.h"
#include "../deps/gapbs/src/graph.h"
#include "../deps/gapbs/src/writer.h"

using nid_t = std::int32_t;
using weight_t = std::int32_t;

using wnode_t = NodeWeight<nid_t, weight_t>;
using wgraph_t = CSRGraph<nid_t, wnode_t>;

using WeightedBuilder = BuilderBase<nid_t, wnode_t, weight_t>;
using WeightedWriter = WriterBase<nid_t, wnode_t>; 
