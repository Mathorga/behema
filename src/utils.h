#ifndef __PORTIA_UTILS__
#define __PORTIA_UTILS__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "cortex.h"

#ifdef __cplusplus
extern "C" {
#endif

// Marsiglia's xorshift pseudo-random number generator with period 2^96-1.
uint32_t xorshf32();

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

/// Dumps the cortex' content to a file.
/// The file is created if not already present, overwritten otherwise.
/// @param cortex The cortex to be written to file.
/// @param file_name The destination file to write the cortex to.
void c2d_to_file(cortex2d_t* cortex, char* file_name);

/// Reads the content from a file and initializes the provided cortex accordingly.
/// @param cortex The cortex to init from file.
/// @param file_name The file to read the cortex from.
void c2d_from_file(cortex2d_t* cortex, char* file_name);

#ifdef __cplusplus
}
#endif

#endif