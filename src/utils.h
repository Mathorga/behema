#ifndef __behema_UTILS__
#define __behema_UTILS__

// This line **must** come **before** including <time.h> in order to
// bring in the POSIX functions such as `clock_gettime() from <time.h>`!
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L

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

// Structure for storing the image data
typedef struct pgm_content_t {
    char pgmType[3];
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t max_value;
} pgm_content_t;

/// @brief Splits the provided source string by the provided delimiter and isolates the first token obtained.
/// @param src_str The string to split.
/// @param dst_str The string in which the result will be stored.
/// @param delimiter The delimiter by which to split the source string.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t strsplit_first(char* src_str, char* dst_str, char* delimiter);

/// @brief Splits the provided source string by the provided delimiter and isolates the last token obtained.
/// @param src_str The string to split.
/// @param dst_str The string in which the result will be stored.
/// @param delimiter The delimiter by which to split the source string.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t strsplit_last(char* src_str, char* dst_str, char* delimiter);

/// @brief Splits the provided source string by the provided delimiter and isolates the token at the provided index.
/// @param src_str The string to split.
/// @param dst_str The string in which the result will be stored.
/// @param delimiter The delimiter by which to split the source string.
/// @param index The indedx of the token to retrieve.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t strsplit_nth(char* src_str, char* dst_str, char* delimiter, uint32_t index);

/// @brief Inserts the provided substring in the provided string at the provided index.
/// @param string The string in which to insert the value.
/// @param index The index at which to insert the substring.
/// @param substr The string to insert.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t strins(char* string, size_t index, char* substr);

/// @brief Replaces target with content in string.
/// @param string The string in which to replace a substring.
/// @param target The substring to replace from string.
/// @param content The string to replace target with.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t strrep(char* string, char* target, char* content);

/// @brief Maps a value to the specified output domain.
/// @param input The input value to be mapped.
/// @param input_start The lower bound of the input domain.
/// @param input_end The upper bound of the input domain.
/// @param output_start The lower bound of the output domain.
/// @param output_end The upper bound of the output domain.
/// @return The value mapped to the provided output domain.
uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

/// @brief Maps a value to the specified output domain while preserving decimal integrity.
/// @param input The input value to be mapped.
/// @param input_start The lower bound of the input domain.
/// @param input_end The upper bound of the input domain.
/// @param output_start The lower bound of the output domain.
/// @param output_end The upper bound of the output domain.
/// @return The value mapped to the provided output domain.
uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end);

/// @brief Computes and returns the current time in milliseconds since the epoch.
/// @return The current time in milliseconds since the epoch.
uint64_t millis();

/// @brief Computes and returns the current time in microseconds since the epoch.
/// @return The current time in microseconds since the epoch.
uint64_t micros();

/// @brief Computes and returns the current time in nanoseconds since the epoch.
/// @return The current time in nanoseconds since the epoch.
uint64_t nanos();

/// Dumps the cortex' content to a file.
/// The file is created if not already present, overwritten otherwise.
/// @param cortex The cortex to be written to file.
/// @param file_name The destination file to write the cortex to.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_to_file(bhm_cortex2d_t* cortex, const char* file_name);

/// Reads the content from a file and initializes the provided cortex accordingly.
/// @param cortex The cortex to init from file.
/// @param file_name The file to read the cortex from.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_from_file(bhm_cortex2d_t* cortex, const char* file_name);

/// Dumps the population's content to a file.
/// The file is created if not already present, overwritten otherwise.
/// @param population The population to be written to file.
/// @param file_name The destination file to write the population to.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t p2d_to_file(bhm_population2d_t* population, const char* file_name);

/// Reads the content from a file and initializes the provided population accordingly.
/// @param cortex The population to init from file.
/// @param file_name The file to read the population from.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t p2d_from_file(bhm_population2d_t* population, const char* file_name);

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
