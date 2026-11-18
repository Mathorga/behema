#ifndef __BHM_MACROS__
#define __BHM_MACROS__

// Translate an id wrapping it to the provided size (pacman effect).
// WARNING: Only works with signed types and does not show errors otherwise.
// [i] is the given index.
// [n] is the size over which to wrap.
#define BHM_WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

// Rotates the lowest [size] bits of [mask] right by [shift] positions: bits shifted out on the right re-enter on the left of the [size] bits.
// Bits above [size] are cleared.
// Example with size = 5 and shift = 2: ---edcba -> ---baedc.
// WARNING: [shift] must be smaller than [size], and [size] must be smaller than the number of bits in [mask] (64 for bhm_nh_mask_t).
// [mask] is the mask to rotate.
// [shift] is the number of positions to rotate by.
// [size] is the number of bits to rotate within.
#define BHM_ROTR_MASK(mask, shift, size) ((shift) == 0 ? (mask) : ((((mask) >> (shift)) | ((mask) << ((size) - (shift)))) & ((0x01ULL << (size)) - 1)))

// Computes the diameter of a square neighborhood given its radius.
#define BHM_NH_DIAM_2D(r) (2 * (r) + 1)

// Computes the number of neighbors in a square neighborhood given its diameter.
#define BHM_NH_COUNT_2D(d) ((d) * (d) - 1)

// Translates bidimensional indexes to a monodimensional one.
// |i| is the row index.
// |j| is the column index.
// |m| is the number of columns (length of the rows).
#define BHM_IDX2D(i, j, m) (((m) * (j)) + (i))

// Translates tridimensional indexes to a monodimensional one.
// |i| is the index in the first dimension.
// |j| is the index in the second dimension.
// |k| is the index in the third dimension.
// |m| is the size of the first dimension.
// |n| is the size of the second dimension.
#define BHM_IDX3D(i, j, k, m, n) (((m) * (n) * (k)) + ((m) * (j)) + (i))

#endif