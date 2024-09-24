#include "cortex.h"

// The state word must be initialized to non-zero.
uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


// ########################################## Initialization functions ##########################################

bhm_error_code_t i2d_init(input2d_t** input, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t exc_value, pulse_mapping_t pulse_mapping) {
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0) {
        return BHM_ERROR_SIZE_WRONG;
    }

    // Allocate the input.
    (*input) = (input2d_t*) malloc(sizeof(input2d_t));
    if ((*input) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    (*input)->x0 = x0;
    (*input)->y0 = y0;
    (*input)->x1 = x1;
    (*input)->y1 = y1;
    (*input)->exc_value = exc_value;

    // Allocate values.
    (*input)->values = (ticks_count_t*) malloc((x1 - x0) * (y1 - y0) * sizeof(ticks_count_t));
    if ((*input)->values == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t o2d_init(output2d_t** output, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1) {
    // Make sure the provided size is correct.
    if (x1 <= x0 || y1 <= y0) {
        return BHM_ERROR_SIZE_WRONG;
    }
    // Allocate the output.
    (*output) = (output2d_t*) malloc(sizeof(output2d_t));
    if ((*output) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    (*output)->x0 = x0;
    (*output)->y0 = y0;
    (*output)->x1 = x1;
    (*output)->y1 = y1;

    // Allocate values.
    (*output)->values = (ticks_count_t*) malloc((x1 - x0) * (y1 - y0) * sizeof(ticks_count_t));
    if ((*output)->values == NULL) {
        printf("ERROR_ALLOCATING_VALUES\n");
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_init(cortex2d_t** cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return BHM_ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if ((*cortex) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->evol_step = DEFAULT_EVOL_STEP;
    (*cortex)->pulse_window = DEFAULT_PULSE_WINDOW;

    (*cortex)->nh_radius = nh_radius;
    (*cortex)->fire_threshold = DEFAULT_THRESHOLD;
    (*cortex)->recovery_value = DEFAULT_RECOVERY_VALUE;
    (*cortex)->exc_value = DEFAULT_EXC_VALUE;
    (*cortex)->decay_value = DEFAULT_DECAY_RATE;
    (*cortex)->rand_state = (rand_state_t) time(NULL);
    (*cortex)->syngen_chance = DEFAULT_SYNGEN_CHANCE;
    (*cortex)->synstr_chance = DEFAULT_SYNSTR_CHANCE;
    (*cortex)->max_tot_strength = DEFAULT_MAX_TOT_STRENGTH;
    (*cortex)->max_syn_count = DEFAULT_MAX_TOUCH * NH_COUNT_2D(NH_DIAM_2D(nh_radius));
    (*cortex)->inhexc_range = DEFAULT_INHEXC_RANGE;

    (*cortex)->sample_window = DEFAULT_SAMPLE_WINDOW;
    (*cortex)->pulse_mapping = PULSE_MAPPING_LINEAR;

    // Allocate neurons.
    (*cortex)->neurons = (neuron_t*) malloc((*cortex)->width * (*cortex)->height * sizeof(neuron_t));
    if ((*cortex)->neurons == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (cortex_size_t y = 0; y < (*cortex)->height; y++) {
        for (cortex_size_t x = 0; x < (*cortex)->width; x++) {
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synac_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synex_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_a = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_b = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_c = 0x00U;

            // The starting random state should be different for each neuron, otherwise repeting patterns occur.
            // Also the starting state should not be 0, so an arbitrary integer is added to every state.
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].rand_state = 31 + x * y;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].value = DEFAULT_STARTING_VALUE;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].max_syn_count = (*cortex)->max_syn_count;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].syn_count = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].tot_syn_strength = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].inhexc_ratio = DEFAULT_INHEXC_RATIO;
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t i2d_destroy(input2d_t* input) {
    // Free values.
    free(input->values);

    // Free input.
    free(input);

    return BHM_ERROR_NONE;
}

bhm_error_code_t o2d_destroy(output2d_t* output) {
    // Free values.
    free(output->values);

    // Free output.
    free(output);

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_destroy(cortex2d_t* cortex) {
    // Free neurons.
    free(cortex->neurons);

    // Free cortex.
    free(cortex);

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_copy(cortex2d_t* to, cortex2d_t* from) {
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

    for (cortex_size_t y = 0; y < from->height; y++) {
        for (cortex_size_t x = 0; x < from->width; x++) {
            to->neurons[IDX2D(x, y, from->width)] = from->neurons[IDX2D(x, y, from->width)];
        }
    }

    return BHM_ERROR_NONE;
}


// ################################################## Setter functions ###################################################

bhm_error_code_t c2d_set_nhradius(cortex2d_t* cortex, nh_radius_t radius) {
    // Make sure the provided radius is valid.
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(nh_mask_t) * 8) {
        return BHM_ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->nh_radius = radius;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_nhmask(cortex2d_t* cortex, nh_mask_t mask) {
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = mask;
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_evol_step(cortex2d_t* cortex, evol_step_t evol_step) {
    cortex->evol_step = evol_step;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_pulse_window(cortex2d_t* cortex, ticks_count_t window) {
    // The given window size must be between 0 and the pulse mask size (in bits).
    if (window >= 0x00u && window < (sizeof(pulse_mask_t) * 8)) {
        cortex->pulse_window = window;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_sample_window(cortex2d_t* cortex, ticks_count_t sample_window) {
    cortex->sample_window = sample_window;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_fire_threshold(cortex2d_t* cortex, neuron_value_t threshold) {
    cortex->fire_threshold = threshold;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_syngen_chance(cortex2d_t* cortex, chance_t syngen_chance) {
    // TODO Check for max value.
    cortex->syngen_chance = syngen_chance;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_synstr_chance(cortex2d_t* cortex, chance_t synstr_chance) {
    // TODO Check for max value.
    cortex->synstr_chance = synstr_chance;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_max_syn_count(cortex2d_t* cortex, syn_count_t syn_count) {
    cortex->max_syn_count = syn_count;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_max_touch(cortex2d_t* cortex, float touch) {
    // Only set touch if a valid value is provided.
    if (touch <= 1 && touch >= 0) {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_pulse_mapping(cortex2d_t* cortex, pulse_mapping_t pulse_mapping) {
    cortex->pulse_mapping = pulse_mapping;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_inhexc_range(cortex2d_t* cortex, chance_t inhexc_range) {
    cortex->inhexc_range = inhexc_range;

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_set_inhexc_ratio(cortex2d_t* cortex, chance_t inhexc_ratio) {
    if (inhexc_ratio <= cortex->inhexc_range) {
        for (cortex_size_t y = 0; y < cortex->height; y++) {
            for (cortex_size_t x = 0; x < cortex->width; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].inhexc_ratio = inhexc_ratio;
            }
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_syn_disable(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height) {
        for (cortex_size_t y = y0; y < y1; y++) {
            for (cortex_size_t x = x0; x < x1; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].max_syn_count = 0x00U;
            }
        }
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_mutate(cortex2d_t *cortex, chance_t mut_chance) {
    // Start by mutating the network itself, then go on to single neurons.
    // TODO Mutate the cortex shape.

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
        // Decide whether to increase or decrease the syngen chance.
        cortex->synstr_chance += cortex->rand_state % 2 == 0 ? 1 : -1;
    }

    // TODO Mutate neurons.

    return BHM_ERROR_NONE;
}


// ########################################## Getter functions ##################################################

bhm_error_code_t c2d_to_string(cortex2d_t* cortex, char* target) {
    int string_length = 0;

    // Header.
    string_length += sprintf(target + string_length, "\ncortex(\n");

    // Data.
    string_length += sprintf(target + string_length, "\twidth:%d\n", cortex->width);
    string_length += sprintf(target + string_length, "\theight:%d\n", cortex->height);
    string_length += sprintf(target + string_length, "\tnh_radius:%d\n", cortex->nh_radius);
    string_length += sprintf(target + string_length, "\tpulse_window:%d\n", cortex->pulse_window);
    string_length += sprintf(target + string_length, "\tsample_window:%d\n", cortex->sample_window);

    // Footer.
    string_length += sprintf(target + string_length, ")\n");

    return BHM_ERROR_NONE;
}

bhm_error_code_t o2d_mean(output2d_t* output, ticks_count_t* target) {
    // Compute the output size beforehand.
    cortex_size_t output_width = output->x1 - output->x0;
    cortex_size_t output_height = output->y1 - output->y0;
    cortex_size_t output_size = output_width * output_height;

    // Compute the sum of the values.
    ticks_count_t total = 0;
    for (cortex_size_t i = 0; i < output_size; i++) {
        total += output->values[i];
    }

    // Store the mean value in the provided pointer.
    (*target) = (ticks_count_t) (total / output_size);

    return BHM_ERROR_NONE;
}
