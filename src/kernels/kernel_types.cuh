/**
 * Define type signatures of epoch kernels.
 */

#ifndef SRC_KERNELS__KERNEL_TYPES_CUH
#define SRC_KERNELS__KERNEL_TYPES_CUH

#include "../bitmap.cuh"
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

/** SSSP */
/**   Epoch Kernels */
/**     CPU */
typedef void (*sssp_cpu_epoch_func)(const CSRWGraph &, weight_t *, 
        const nid_t, const nid_t, const int, const int, nid_t &);

/**     GPU */
typedef void (*sssp_gpu_epoch_func)(const offset_t *, 
        const wnode_t *, const nid_t, const nid_t, weight_t *, nid_t *);

/** BFS */
/**   Epoch Kernels */
/**     GPU */
typedef void (*bfs_gpu_epoch_func)(const offset_t * const, 
        const nid_t * const, nid_t * const, const nid_t, const nid_t, 
        const Bitmap::Bitmap * const, Bitmap::Bitmap * const, 
        nid_t * const);

/**   Kernels */
/**     CPU */
typedef double (bfs_cpu_kernel)(const CSRUWGraph &g, 
        const nid_t, nid_t ** const);

#endif // SRC_KERNELS__KERNEL_TYPES_CUH
