/**
 * Bitmap C-style class compatible with CPUs & GPUs.
 * Implementation follows Scott Beamer's GAPBS bitmap class.
 */

#ifndef SRC__BITMAP_H
#define SRC__BITMAP_H

#include <algorithm>
#include <cstddef>

#include "cuda.cuh"

namespace Bitmap {

// Define bitmap internal data type and the number of bits contained in each 
// instance of data type.
// Using a 32-bit integer since it's compatible with CUDA atomics.
using data_t = int;
const std::size_t data_size = sizeof(data_t) * 8;

// Functions for offset computation. 
__host__ __device__ __inline__
std::size_t data_offset(std::size_t n) { return n / data_size; }

__host__ __device__ __inline__
std::size_t bit_offset(std::size_t n) { return n & (data_size - 1); }

// Bitmap data structure.
struct Bitmap {
    data_t      *buffer;
    bool        using_cuda;
    std::size_t size;
};

/** Forward decl. */
void reset(Bitmap * const bitmap);
void cu_cpu_reset(Bitmap * const cu_bitmap);

/*****************************************************************************
 ***** General/CPU Implementations *******************************************
 *****************************************************************************/

/**
 * Creates a dynamically-allocated CPU bitmap.
 * Parameters:
 *   - num_elements <- number of elements in the bitmap.
 *   - use_cuda     <- if true, allocate pinned host memory with CUDA.
 * Returns:
 *   Pointer to bitmap object.
 */
__host__ __inline__
Bitmap *constructor(const std::size_t num_elements, const bool use_cuda = false) {
    Bitmap *bitmap     = new Bitmap;
    bitmap->size       = (num_elements + data_size - 1) / data_size;
    bitmap->using_cuda = use_cuda;

    if (use_cuda) {
        CUDA_ERRCHK(cudaMallocHost((void **) &bitmap->buffer, 
                    bitmap->size * sizeof(data_t)));
    } else
        bitmap->buffer = new data_t[bitmap->size];

    reset(bitmap);
    return bitmap;
}

__host__ __inline__
void destructor(Bitmap ** const bitmap) {
    if ((*bitmap)->using_cuda) {
        CUDA_ERRCHK(cudaFreeHost((*bitmap)->buffer));
    } else
        delete[] (*bitmap)->buffer;

    delete *bitmap;
    *bitmap = nullptr;
}

__host__ __inline__
void set_bit_atomic(Bitmap * const bitmap, const std::size_t idx) {
    data_t bit = static_cast<data_t>(1) << bit_offset(idx);
    __sync_fetch_and_or(&bitmap->buffer[data_offset(idx)], bit);
}

__host__ __device__ __inline__
void set_bit(Bitmap * const bitmap, const std::size_t idx) {
    bitmap->buffer[data_offset(idx)] |= \
            static_cast<data_t>(1) << bit_offset(idx);
}

__host__ __device__ __inline__
bool get_bit(const Bitmap * const bitmap, const std::size_t idx) {
    return bitmap->buffer[data_offset(idx)] >> bit_offset(idx) \
            & static_cast<data_t>(1);    
}

__host__ __inline__
void reset(Bitmap * const bitmap) {
    std::fill(bitmap->buffer, bitmap->buffer + bitmap->size, 0);
}

/*****************************************************************************
 ***** GPU-specific Implementations ******************************************
 *****************************************************************************/

/**
 * Creates a dynamically-allocated CPU-copy of a GPU bitmap.
 * Bitmap object is in CPU memory. Bitmap buffer is in GPU memory.
 * Parameters:
 *   - num_elements <- number of elements in the bitmap.
 * Returns:
 *   Pointer ot CPU bitmap object with GPU buffer.
 */
__host__ __inline__
Bitmap *cu_cpu_constructor(const std::size_t num_elements) {
    Bitmap *cu_bitmap     = new Bitmap;
    cu_bitmap->size       = (num_elements + data_size - 1) / data_size;
    cu_bitmap->using_cuda = true;

    CUDA_ERRCHK(cudaMalloc((void **) &cu_bitmap->buffer,
                cu_bitmap->size * sizeof(data_t)));
    cu_cpu_reset(cu_bitmap);

    return cu_bitmap;
}

__host__ __inline__
void cu_cpu_destructor(Bitmap ** const cu_bitmap) {
    CUDA_ERRCHK(cudaFree((*cu_bitmap)->buffer));
    delete[] *cu_bitmap;
    *cu_bitmap = nullptr;
}

__host__ __inline__
void cu_cpu_reset(Bitmap * const cu_bitmap) {
    CUDA_ERRCHK(cudaMemset(cu_bitmap->buffer, 0, 
                cu_bitmap->size * sizeof(data_t)));
}

__host__ __inline__
void cu_cpu_set_bit(Bitmap * const cu_bitmap, const std::size_t idx) {
    data_t dat;
    CUDA_ERRCHK(cudaMemcpy(&dat, cu_bitmap->buffer + data_offset(idx),
                sizeof(data_t), cudaMemcpyDeviceToHost));
    dat |= static_cast<data_t>(1) << bit_offset(idx);
    CUDA_ERRCHK(cudaMemcpy(cu_bitmap->buffer + data_offset(idx), &dat,
                sizeof(data_t), cudaMemcpyHostToDevice));
}

/**
 * Converts cu_cpu_constructor bitmap into GPU-only bitmap object.
 * Parameters:
 *   - bitmap <- cu_constructor bitmap.
 * Returns:
 *   Bitmap entirely allocated in GPU memory.
 */
__host__ __inline__
Bitmap *cu_constructor(const Bitmap::Bitmap * const bitmap) {
    Bitmap::Bitmap *cu_bitmap = nullptr;
    CUDA_ERRCHK(cudaMalloc((void **) &cu_bitmap, sizeof(Bitmap)));
    CUDA_ERRCHK(cudaMemcpy(cu_bitmap, bitmap, sizeof(Bitmap), 
                cudaMemcpyHostToDevice));
    return cu_bitmap;
}

/**
 * Destructor for cu_constructor bitmap.
 * Parameters:
 *   - cu_bitmap <- cu_copy_constructor bitmap.
 */
__host__ __inline__
void cu_destructor(Bitmap::Bitmap ** const cu_bitmap) {
    CUDA_ERRCHK(cudaFree(*cu_bitmap));
    *cu_bitmap = nullptr;
}

__device__ __inline__
void cu_set_bit_atomic(Bitmap * const cu_bitmap, const std::size_t idx) {
    data_t bit = static_cast<data_t>(1) << bit_offset(idx);
    atomicOr(cu_bitmap->buffer + data_offset(idx), bit);    
}

}

#endif // SRC__BITMAP_H
