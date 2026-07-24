/*
*****************************************************************
cortex.h

Copyright (C) 2026 Luka Micheletti
*****************************************************************
*/

#ifndef __BHM_INPUT__
#define __BHM_INPUT__

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "types.h"
#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Convenience data structure for input handling (cortex feeding).
typedef struct {
    bhm_cortex_size_t x0;
    bhm_cortex_size_t y0;
    bhm_cortex_size_t x1;
    bhm_cortex_size_t y1;

    // Value used to excite the target neurons.
    bhm_neuron_value_t exc_value;

    // Values to be mapped to pulse (input values).
    bhm_ticks_count_t* values;
} bhm_input2d_t;


/// @brief Initializes an input2d with the given values.
/// @param input The input to be 
/// @param x0 
/// @param y0 
/// @param x1 
/// @param y1 
/// @param exc_value 
/// @param pulse_mapping 
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t bhm_i2d_create(
    bhm_input2d_t** input,
    bhm_cortex_size_t x0,
    bhm_cortex_size_t y0,
    bhm_cortex_size_t x1,
    bhm_cortex_size_t y1,
    bhm_neuron_value_t exc_value,
    bhm_pulse_mapping_t pulse_mapping
);

/// @brief Destroys the given input2d and frees memory.
/// @param input The input to destroy.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t bhm_i2d_destroy(
    bhm_input2d_t* input
);

/// @brief Computes the mean value of an input2d's values.
/// @param input The input to compute the mean value from.
/// @param result Pointer to the result of the computation. The mean value will be stored here.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t i2d_mean(
    bhm_input2d_t* input,
    bhm_ticks_count_t* result
);

#ifdef __cplusplus
}
#endif

#endif
