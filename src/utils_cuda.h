#include <stdint.h>

/// Marsiglia's xorshift pseudo-random number generator with period 2^32-1.
__host__ __device__ uint32_t xorshf32(uint32_t state);