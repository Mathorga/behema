#include "cortex.h"

// The state word must be initialized to non-zero.
uint32_t xorshf32(
    uint32_t state
) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


// ########################################## Initialization functions ##########################################

bhm_error_code_t i2d_init(
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

bhm_error_code_t o2d_init(
    bhm_output2d_t** output,
    bhm_cortex_size_t x0,
    bhm_cortex_size_t y0,
    bhm_cortex_size_t x1,
    bhm_cortex_size_t y1
) {
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0) {
        return BHM_ERROR_SIZE_WRONG;
    }
    // Allocate the output.
    (*output) = (bhm_output2d_t*) malloc(sizeof(bhm_output2d_t));
    if ((*output) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    (*output)->x0 = x0;
    (*output)->y0 = y0;
    (*output)->x1 = x1;
    (*output)->y1 = y1;

    // Allocate values.
    (*output)->values = (bhm_ticks_count_t*) malloc((x1 - x0) * (y1 - y0) * sizeof(bhm_ticks_count_t));
    if ((*output)->values == NULL) {
        printf("ERROR_ALLOCATING_VALUES\n");
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_init(
    bhm_cortex2d_t** cortex,
    bhm_cortex_size_t width,
    bhm_cortex_size_t height,
    bhm_nh_radius_t nh_radius
) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(bhm_nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return BHM_ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (bhm_cortex2d_t*) malloc(sizeof(bhm_cortex2d_t));
    if ((*cortex) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->evol_step = BHM_DEFAULT_EVOL_STEP;
    (*cortex)->pulse_window = BHM_DEFAULT_PULSE_WINDOW;

    (*cortex)->nh_radius = nh_radius;
    (*cortex)->fire_threshold = BHM_DEFAULT_THRESHOLD;
    (*cortex)->recovery_value = BHM_DEFAULT_RECOVERY_VALUE;
    (*cortex)->exc_value = BHM_DEFAULT_EXC_VALUE;
    (*cortex)->decay_value = BHM_DEFAULT_DECAY_RATE;
    (*cortex)->rand_state = (bhm_rand_state_t) time(NULL);
    (*cortex)->syngen_chance = BHM_DEFAULT_SYNGEN_CHANCE;
    (*cortex)->synstr_chance = BHM_DEFAULT_SYNSTR_CHANCE;
    (*cortex)->max_tot_strength = BHM_DEFAULT_MAX_TOT_STRENGTH;
    (*cortex)->max_syn_count = BHM_DEFAULT_MAX_TOUCH * NH_COUNT_2D(NH_DIAM_2D(nh_radius));
    (*cortex)->inhexc_range = BHM_DEFAULT_INHEXC_RANGE;

    (*cortex)->sample_window = BHM_DEFAULT_SAMPLE_WINDOW;
    (*cortex)->pulse_mapping = BHM_PULSE_MAPPING_LINEAR;

    // Allocate neurons.
    (*cortex)->neurons = (bhm_neuron_t*) malloc((*cortex)->width * (*cortex)->height * sizeof(bhm_neuron_t));
    if ((*cortex)->neurons == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (bhm_cortex_size_t y = 0; y < (*cortex)->height; y++) {
        for (bhm_cortex_size_t x = 0; x < (*cortex)->width; x++) {
            bhm_neuron_t* neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];

            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;

            // The starting random state should be different for each neuron, otherwise repeting patterns occur.
            // Also the starting state should never be 0, so an arbitrary integer is added to every state.
            neuron->rand_state = 31 + x * y;
            neuron->pulse_mask = 0x00U;
            neuron->pulse = 0x00U;
            neuron->value = BHM_DEFAULT_STARTING_VALUE;
            neuron->max_syn_count = (*cortex)->max_syn_count;
            neuron->syn_count = 0x00U;
            neuron->tot_syn_strength = 0x00U;
            neuron->inhexc_ratio = BHM_DEFAULT_INHEXC_RATIO;
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_rand_init(
    bhm_cortex2d_t** cortex,
    bhm_cortex_size_t width,
    bhm_cortex_size_t height,
    bhm_nh_radius_t nh_radius
) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(bhm_nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return BHM_ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (bhm_cortex2d_t*) malloc(sizeof(bhm_cortex2d_t));
    if ((*cortex) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->rand_state = (bhm_rand_state_t) time(NULL);
    (*cortex)->evol_step = (*cortex)->rand_state % BHM_EVOL_STEP_NEVER;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->pulse_window = (*cortex)->rand_state % BHM_MAX_PULSE_WINDOW;

    (*cortex)->nh_radius = nh_radius;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->fire_threshold = (*cortex)->rand_state % BHM_MAX_THRESHOLD;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->recovery_value = ((*cortex)->rand_state % BHM_MAX_RECOVERY_VALUE) - BHM_MAX_RECOVERY_VALUE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->exc_value = (*cortex)->rand_state % BHM_MAX_EXC_VALUE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->decay_value = (*cortex)->rand_state % BHM_MAX_DECAY_RATE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->syngen_chance = (*cortex)->rand_state % BHM_MAX_SYNGEN_CHANCE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->synstr_chance = (*cortex)->rand_state % BHM_MAX_SYNSTR_CHANCE;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->max_tot_strength = (*cortex)->rand_state % BHM_MAX_MAX_TOT_STRENGTH;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->max_syn_count = (*cortex)->rand_state % ((bhm_syn_count_t) (BHM_MAX_MAX_TOUCH * NH_COUNT_2D(NH_DIAM_2D(nh_radius))));
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->inhexc_range = (*cortex)->rand_state % BHM_MAX_INHEXC_RANGE;

    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    (*cortex)->sample_window = (*cortex)->rand_state % BHM_MAX_SAMPLE_WINDOW;
    (*cortex)->rand_state = xorshf32((*cortex)->rand_state);
    // There are 4 possible pulse mappings, so pick one and assign it.
    int pulse_mapping = (*cortex)->rand_state % 4 + 0x100000;
    (*cortex)->pulse_mapping = pulse_mapping;

    // Allocate neurons.
    (*cortex)->neurons = (bhm_neuron_t*) malloc((*cortex)->width * (*cortex)->height * sizeof(bhm_neuron_t));
    if ((*cortex)->neurons == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (bhm_cortex_size_t y = 0; y < (*cortex)->height; y++) {
        for (bhm_cortex_size_t x = 0; x < (*cortex)->width; x++) {
            bhm_neuron_t* neuron = &(*cortex)->neurons[IDX2D(x, y, (*cortex)->width)];

            neuron->synac_mask = 0x00U;
            neuron->synex_mask = 0x00U;
            neuron->synstr_mask_a = 0x00U;
            neuron->synstr_mask_b = 0x00U;
            neuron->synstr_mask_c = 0x00U;

            // The starting random state should be different for each neuron, otherwise repeting patterns occur.
            // Also the starting state should never be 0, so an arbitrary integer is added to every state.
            neuron->rand_state = 31 + x * y;
            neuron->pulse_mask = 0x00U;
            neuron->pulse = 0x00U;
            neuron->value = BHM_DEFAULT_STARTING_VALUE;
            neuron->rand_state = xorshf32(neuron->rand_state);
            neuron->max_syn_count = neuron->rand_state % (*cortex)->max_syn_count;
            neuron->syn_count = 0x00U;
            neuron->tot_syn_strength = 0x00U;
            neuron->rand_state = xorshf32(neuron->rand_state);
            neuron->inhexc_ratio = neuron->rand_state % (*cortex)->inhexc_range;
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t i2d_destroy(
    bhm_input2d_t* input
) {
    // Free values.
    free(input->values);

    // Free input.
    free(input);

    return BHM_ERROR_NONE;
}

bhm_error_code_t o2d_destroy(
    bhm_output2d_t* output
) {
    // Free values.
    free(output->values);

    // Free output.
    free(output);

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_destroy(
    bhm_cortex2d_t* cortex
) {
    printf("%p %p\n", (void*) cortex, (void*) cortex->neurons);
    // Free neurons.
    free(cortex->neurons);

    printf("GIANFREDA\n");
    // Free cortex.
    free(cortex);

    printf("GIANFREDONIO\n");

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_copy(
    bhm_cortex2d_t* to,
    bhm_cortex2d_t* from
) {
    to->width = from->width;
    to->height = from->height;
    to->ticks_count = from->ticks_count;
    to->evols_count = from->evols_count;
    to->evol_step = from->evol_step;
    to->pulse_window = from->pulse_window;

    to->nh_radius = from->nh_radius;
    to->fire_threshold = from->fire_threshold;
    to->recovery_value = from->recovery_value;
    to->exc_value = from->exc_value;
    to->decay_value = from->decay_value;
    to->syngen_chance = from->syngen_chance;
    to->synstr_chance = from->synstr_chance;
    to->max_tot_strength = from->max_tot_strength;
    to->max_syn_count = from->max_syn_count;
    to->inhexc_range = from->inhexc_range;

    to->sample_window = from->sample_window;
    to->pulse_mapping = from->pulse_mapping;

    for (bhm_cortex_size_t y = 0; y < from->height; y++) {
        for (bhm_cortex_size_t x = 0; x < from->width; x++) {
            to->neurons[IDX2D(x, y, from->width)] = from->neurons[IDX2D(x, y, from->width)];
        }
    }

    return BHM_ERROR_NONE;
}


// ################################################## Setter functions ###################################################

bhm_error_code_t c2d_set_nhradius(
    bhm_cortex2d_t* cortex,
    bhm_nh_radius_t radius
) {
    // Make sure the provided radius is valid.
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(bhm_nh_mask_t) * 8) {
        return BHM_ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->nh_radius = radius;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_nhmask(
    bhm_cortex2d_t* cortex,
    bhm_nh_mask_t mask
) {
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = mask;
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_evol_step(
    bhm_cortex2d_t* cortex,
    bhm_evol_step_t evol_step
) {
    cortex->evol_step = evol_step;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_pulse_window(
    bhm_cortex2d_t* cortex,
    bhm_ticks_count_t window
) {
    // The given window size must be between 0 and the pulse mask size (in bits).
    if (window >= 0x00u && window < (sizeof(bhm_pulse_mask_t) * 8)) {
        cortex->pulse_window = window;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_sample_window(
    bhm_cortex2d_t* cortex,
    bhm_ticks_count_t sample_window
) {
    cortex->sample_window = sample_window;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_fire_threshold(
    bhm_cortex2d_t* cortex,
    bhm_neuron_value_t threshold
) {
    cortex->fire_threshold = threshold;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_syngen_chance(
    bhm_cortex2d_t* cortex,
    bhm_chance_t syngen_chance
) {
    // TODO Check for max value.
    cortex->syngen_chance = syngen_chance;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_synstr_chance(
    bhm_cortex2d_t* cortex,
    bhm_chance_t synstr_chance
) {
    // TODO Check for max value.
    cortex->synstr_chance = synstr_chance;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_max_syn_count(
    bhm_cortex2d_t* cortex,
    bhm_syn_count_t syn_count
) {
    cortex->max_syn_count = syn_count;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_max_touch(
    bhm_cortex2d_t* cortex,
    float touch
) {
    // Only set touch if a valid value is provided.
    if (touch <= 1 && touch >= 0) {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_pulse_mapping(
    bhm_cortex2d_t* cortex,
    bhm_pulse_mapping_t pulse_mapping
) {
    cortex->pulse_mapping = pulse_mapping;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_inhexc_range(
    bhm_cortex2d_t* cortex,
    bhm_chance_t inhexc_range
) {
    cortex->inhexc_range = inhexc_range;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_inhexc_ratio(
    bhm_cortex2d_t* cortex,
    bhm_chance_t inhexc_ratio
) {
    if (inhexc_ratio <= cortex->inhexc_range) {
        for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
            for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].inhexc_ratio = inhexc_ratio;
            }
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_syn_disable(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t x0,
    bhm_cortex_size_t y0,
    bhm_cortex_size_t x1,
    bhm_cortex_size_t y1
) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height) {
        for (bhm_cortex_size_t y = y0; y < y1; y++) {
            for (bhm_cortex_size_t x = x0; x < x1; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].max_syn_count = 0x00U;
            }
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_mutate_shape(
    bhm_cortex2d_t *cortex,
    bhm_chance_t mut_chance
) {
    bhm_cortex_size_t new_width = cortex->width;
    bhm_cortex_size_t new_height = cortex->height;

    // Mutate the cortex width.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance) {
        // Decide whether to increase or decrease the cortex width.
        new_width += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate the cortex height.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance) {
        // Decide whether to increase or decrease the cortex height.
        new_height += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    if (new_width != cortex->width || new_height != cortex->height) {
        // Resize neurons.
        cortex->neurons = (bhm_neuron_t*) realloc(cortex->neurons, new_width * new_height * sizeof(bhm_neuron_t));
        if (cortex->neurons == NULL) {
            return BHM_ERROR_FAILED_ALLOC;
        }

        // TODO Handle neurons' properties.
        // Loop

        // Store updated cortex shape.
        cortex->width = new_width;
        cortex->height = new_height;
    }
    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_mutate(
    bhm_cortex2d_t *cortex,
    bhm_chance_t mut_chance
) {
    // Start by mutating the network itself, then go on to single neurons.

    // TODO Mutate the cortex shape.
    // bhm_error_code_t error = c2d_mutate_shape(cortex, mut_chance);
    // if (error != BHM_ERROR_NONE) {
    //     return error;
    // }

    // Mutate pulse window.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance) {
        // Decide whether to increase or decrease the pulse window.
        cortex->pulse_window += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate syngen chance.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance) {
        // Decide whether to increase or decrease the syngen chance.
        cortex->syngen_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate synstr chance.
    cortex->rand_state = xorshf32(cortex->rand_state);
    if (cortex->rand_state > mut_chance) {
        // Decide whether to increase or decrease the synstr chance.
        cortex->synstr_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate neurons.
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            n2d_mutate(&(cortex->neurons[IDX2D(x, y, cortex->width)]), mut_chance);
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t n2d_mutate(
    bhm_neuron_t* neuron,
    bhm_chance_t mut_chance
) {
    // Mutate max syn count.
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance) {
        // Decide whether to increase or decrease the max syn count.
        neuron->max_syn_count += neuron->rand_state % 2 == 0 ? 1 : -1;
    }

    // Mutate inhexc ratio.
    neuron->rand_state = xorshf32(neuron->rand_state);
    if (neuron->rand_state > mut_chance) {
        // Decide whether to increase or decrease the inhexc ratio.
        neuron->inhexc_ratio += neuron->rand_state % 2 == 0 ? 1 : -1;
    }

    return BHM_ERROR_NONE;
}


// ########################################## Getter functions ##################################################

bhm_error_code_t c2d_to_string(
    bhm_cortex2d_t* cortex,
    char* result
) {
    int string_length = 0;

    // Header.
    string_length += sprintf(result + string_length, "\ncortex(\n");

    // Data.
    string_length += sprintf(result + string_length, "\twidth:%d\n", cortex->width);
    string_length += sprintf(result + string_length, "\theight:%d\n", cortex->height);
    string_length += sprintf(result + string_length, "\tnh_radius:%d\n", cortex->nh_radius);
    string_length += sprintf(result + string_length, "\tpulse_window:%d\n", cortex->pulse_window);
    string_length += sprintf(result + string_length, "\tsample_window:%d\n", cortex->sample_window);

    // Footer.
    string_length += sprintf(result + string_length, ")\n");

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

bhm_error_code_t o2d_mean(
    bhm_output2d_t* output,
    bhm_ticks_count_t* result
) {
    // Compute the output size beforehand.
    bhm_cortex_size_t output_width = output->x1 - output->x0;
    bhm_cortex_size_t output_height = output->y1 - output->y0;
    bhm_cortex_size_t output_size = output_width * output_height;

    // Compute the sum of the values.
    bhm_ticks_count_t total = 0;
    for (bhm_cortex_size_t i = 0; i < output_size; i++) {
        total += output->values[i];
    }

    // Store the mean value in the provided pointer.
    (*result) = (bhm_ticks_count_t) (total / output_size);

    return BHM_ERROR_NONE;
}
