#include "utils_std.h"

// The state word must be initialized to non-zero.
uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}