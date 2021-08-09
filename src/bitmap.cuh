/**
 * Bitmap C-style class compatible with CPUs & GPUs.
 * Implementation follows Scott Beamer's GAPBS bitmap class.
 */

#ifndef SRC__BITMAP_H
#define SRC__BITMAP_H

#include <algorithm>
#include <cstddef>

namespace Bitmap {

// Define bitmap internal data type and the number of bits contained in  each 
// intsance of data type.
using data_t = char;
const std::size_t data_size = sizeof(data_t) * 8;

// Functions for offset computation. 
__host__ __device__ __inline__
std::size_t data_offset(std::size_t n) { return n / data_size; }

__host__ __device__ __inline__
std::size_t bit_offset(std::size_t n) { return n & (data_size - 1); }

// Bitmap data structure.
struct Bitmap {
    data_t *bitmap_start;
    data_t *bitmap_end;
};

__host__
void constructor(Bitmap * const bitmap, const std::size_t num_elements) {
    std::size_t bitmap_size = (num_elements + data_size - 1) / data_size;

    bitmap->bitmap_start = new data_t[bitmap_size];
    bitmap->bitmap_end   = bitmap->bitmap_start + bitmap_size;
    std::fill(bitmap->bitmap_start, bitmap->bitmap_end, 0);
}

__host__
void destructor(Bitmap ** const bitmap) {
    delete[] (*bitmap)->bitmap_start;
    delete (*bitmap);
    *bitmap = nullptr;
}

__host__ __inline__
void set_bit_atomic(Bitmap * const bitmap, const std::size_t idx) {
    data_t bit = static_cast<data_t>(1) << bit_offset(idx);
    __sync_fetch_and_or(&bitmap->bitmap_start[data_offset(idx)], bit);
}

__host__ __device__ __inline__
void set_bit(Bitmap * const bitmap, const std::size_t idx) {
    bitmap->bitmap_start[data_offset(idx)] |= \
            static_cast<data_t>(1) << bit_offset(idx);
}

__host__ __device__ __inline__
bool get_bit(const Bitmap * const bitmap, const std::size_t idx) {
    return bitmap->bitmap_start[data_offset(idx)] >> bit_offset(idx) \
            & static_cast<data_t>(1);    
}

__host__ __inline__
void reset(Bitmap * const bitmap) {
    std::fill(bitmap->bitmap_start, bitmap->bitmap_end, 0);
}

}

/** TODO: Impl CUDA variants of Bitmap. */
/*__host__*/
/*void cuBitmap_constructor(Bitmap ** const cu_bitmap, const std::size_t num_elements) {*/

/*}*/

/*__host__*/
/*void cuBitmap_destructor(Bitmap ** const cu_bitmap) {*/

/*}*/

/*__host__*/
/*void cuBitmap_reset(Bitmap *cu_bitmap) {*/

/*}*/

#endif // SRC__BITMAP_H
