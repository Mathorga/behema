#ifndef __behema_UTILS__
#define __behema_UTILS__

// This line **must** come **before** including <time.h> in order to
// bring in the POSIX functions such as `clock_gettime() from <time.h>`!
// #define _POSIX_C_SOURCE 199309L

#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "cortex.h"
#include "population.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Convert seconds to milliseconds
#define S_TO_MS(s) ((s) * 1e3)
/// Convert seconds to microseconds
#define S_TO_US(s) ((s) * 1e6)
/// Convert seconds to nanoseconds
#define S_TO_NS(s) ((s) * 1e9)

/// Convert nanoseconds to seconds
#define NS_TO_S(ns) ((ns) / 1e9)
/// Convert nanoseconds to milliseconds
#define NS_TO_MS(ns) ((ns) / 1e6)
/// Convert nanoseconds to microseconds
#define NS_TO_US(ns) ((ns) / 1e3)

// Structure for storing the
// image data
typedef struct pgm_content_t {
    char pgmType[3];
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t max_value;
} pgm_content_t;

// Maps a value to the specified output domain.
uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);
// Maps a value to the specified output domain while preserving decimal integrity.
uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

/// Get a time stamp in milliseconds.
uint64_t millis();

/// Get a time stamp in microseconds.
uint64_t micros();

/// Get a time stamp in nanoseconds.
uint64_t nanos();

/// Dumps the cortex' content to a file.
/// The file is created if not already present, overwritten otherwise.
/// @param cortex The cortex to be written to file.
/// @param file_name The destination file to write the cortex to.
void c2d_to_file(bhm_cortex2d_t* cortex, char* file_name);

/// Reads the content from a file and initializes the provided cortex accordingly.
/// @param cortex The cortex to init from file.
/// @param file_name The file to read the cortex from.
void c2d_from_file(bhm_cortex2d_t* cortex, char* file_name);

/// Dumps the population's content to a file.
/// The file is created if not already present, overwritten otherwise.
/// @param population The population to be written to file.
/// @param file_name The destination file to write the population to.
void p2d_to_file(bhm_population2d_t* population, char* file_name);

/// Reads the content from a file and initializes the provided population accordingly.
/// @param cortex The population to init from file.
/// @param file_name The file to read the population from.
void p2d_from_file(bhm_population2d_t* population, char* file_name);

/// @brief Sets touch for each neuron in the provided cortex by reading it from a pgm map file.
/// @param cortex The cortex to apply changes to.
/// @param map_file_name The path to the pgm map file to read.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_touch_from_map(bhm_cortex2d_t* cortex, char* map_file_name);

/// @brief Sets inhexc ratio for each neuron in the provided cortex by reading it from a pgm map file.
/// @param cortex The cortex to apply changes to.
/// @param map_file_name The path to the pgm map file to read.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_inhexc_from_map(bhm_cortex2d_t* cortex, char* map_file_name);

/// @brief Sets fire threshold for each neuron in the provided cortex by reading it from a pgm map file.
/// @param cortex The cortex to apply changes to.
/// @param map_file_name The path to the pgm map file to read.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_fthold_from_map(bhm_cortex2d_t* cortex, char* map_file_name);

#ifdef __cplusplus
}
#endif

#endif
