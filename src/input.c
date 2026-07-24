#include "input.h"

bhm_error_code_t bhm_i2d_create(
    bhm_input2d_t** input,
    bhm_cortex_size_t x0,
    bhm_cortex_size_t y0,
    bhm_cortex_size_t x1,
    bhm_cortex_size_t y1,
    bhm_neuron_value_t exc_value,
    bhm_pulse_mapping_t pulse_mapping
) {
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0) {
        return BHM_ERROR_SIZE_WRONG;
    }

    // Allocate the input.
    (*input) = (bhm_input2d_t*) malloc(sizeof(bhm_input2d_t));
    if ((*input) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    (*input)->x0 = x0;
    (*input)->y0 = y0;
    (*input)->x1 = x1;
    (*input)->y1 = y1;
    (*input)->exc_value = exc_value;

    // Allocate values.
    (*input)->values = (bhm_ticks_count_t*) malloc((x1 - x0) * (y1 - y0) * sizeof(bhm_ticks_count_t));
    if ((*input)->values == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t i2d_mean(
    bhm_input2d_t* input,
    bhm_ticks_count_t* result
) {
    // Compute the input size beforehand.
    bhm_cortex_size_t input_width = input->x1 - input->x0;
    bhm_cortex_size_t input_height = input->y1 - input->y0;
    bhm_cortex_size_t input_size = input_width * input_height;

    // Compute the sum of the values.
    bhm_ticks_count_t total = 0;
    for (bhm_cortex_size_t i = 0; i < input_size; i++) {
        total += input->values[i];
    }

    // Store the mean value in the provided pointer.
    (*result) = (bhm_ticks_count_t) (total / input_size);

    return BHM_ERROR_NONE;
}

bhm_error_code_t bhm_i2d_destroy(
    bhm_input2d_t* input
) {
    // Free values.
    free(input->values);

    // Free input.
    free(input);

    return BHM_ERROR_NONE;
}