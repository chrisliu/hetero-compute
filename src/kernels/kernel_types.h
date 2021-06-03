/**
 * Define type signatures of epoch kernels.
 */

#ifndef SRC_KERNELS__KERNEL_TYPES_H
#define SRC_KERNELS__KERNEL_TYPES_H

#include "../graph.h"

/** SSSP epoch kernels. */
/** CPU */
typedef void (*sssp_cpu_epoch_func)(const CSRWGraph &, weight_t *, 
        const nid_t, const nid_t, const int, const int, nid_t &);

/** GPU */
typedef void (*sssp_gpu_epoch_func)(const offset_t *, 
        const wnode_t *, const nid_t, const nid_t, weight_t *, nid_t *);

#endif // SRC_KERNELS__KERNEL_TYPES_H
