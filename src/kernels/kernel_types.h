/**
 * Define type signatures of epoch kernels.
 */

#ifndef SRC_KERNELS__KERNEL_TYPES_H
#define SRC_KERNELS__KERNEL_TYPES_H

#include "../graph.h"

/******************************************************************************
 ***** Data Structures ********************************************************
 ******************************************************************************/

/**
 * Defines the segment a kernel is supposed to process.
 * A kernel should process the range [start_id, end_id).
 */
struct graph_range_t {
    nid_t start_id; // Starting node id.
    nid_t end_id;   // Ending node id (exclusive).
};

/******************************************************************************
 ***** Kernel Types ***********************************************************
 ******************************************************************************/

/** SSSP epoch kernels. */
/** CPU */
typedef void (*sssp_cpu_epoch_func)(const CSRWGraph &, weight_t *, 
        const nid_t, const nid_t, const int, const int, nid_t &);

/** GPU */
typedef void (*sssp_gpu_epoch_func)(const offset_t *, 
        const wnode_t *, const nid_t, const nid_t, weight_t *, nid_t *);

#endif // SRC_KERNELS__KERNEL_TYPES_H
