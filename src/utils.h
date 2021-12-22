#include <stdint.h>

// Marsiglia's xorshift pseudo-random number generator with period 2^96-1.
unsigned long xorshf96(void);

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);