#include "utils_cuda.h"

// The state must be initialized to non-zero.
__host__ __device__ uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}