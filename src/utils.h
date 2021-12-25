#include <stdint.h>

#ifndef __PORTIA_UTILS__
#define __PORTIA_UTILS__

typedef uint8_t byte;

typedef enum {
    FALSE = 0,
    TRUE = 1
} bool;

// Marsiglia's xorshift pseudo-random number generator with period 2^96-1.
unsigned long xorshf96(void);

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

#endif