#ifndef __PORTIA_UTILS__
#define __PORTIA_UTILS__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cortex.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DATA_OFFSET_OFFSET 0x000A
#define WIDTH_OFFSET 0x0012
#define HEIGHT_OFFSET 0x0016
#define BITS_PER_PIXEL_OFFSET 0x001C
#define HEADER_SIZE 14
#define INFO_HEADER_SIZE 40
#define NO_COMPRESION 0
#define MAX_NUMBER_OF_COLORS 0
#define ALL_COLORS_REQUIRED 0

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

void c2d_touch_from_map(cortex2d_t* cortex, char* map_file_name);

void c2d_inhexc_from_map(cortex2d_t* cortex, char* map_file_name);

#ifdef __cplusplus
}
#endif

#endif