#include "cortex.h"

error_code_t c2d_init(cortex2d_t** cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex.
    (*cortex) = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if ((*cortex) == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->rand_state = 0x01U;
    (*cortex)->evols_count = 0x00U;
    (*cortex)->evol_step = DEFAULT_EVOL_STEP;
    (*cortex)->pulse_window = DEFAULT_PULSE_WINDOW;

    (*cortex)->nh_radius = nh_radius;
    (*cortex)->fire_threshold = DEFAULT_THRESHOLD;
    (*cortex)->recovery_value = DEFAULT_RECOVERY_VALUE;
    (*cortex)->exc_value = DEFAULT_EXC_VALUE;
    (*cortex)->decay_value = DEFAULT_DECAY_RATE;
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
        return ERROR_FAILED_ALLOC;
    }

    // Setup neurons' properties.
    for (cortex_size_t y = 0; y < (*cortex)->height; y++) {
        for (cortex_size_t x = 0; x < (*cortex)->width; x++) {
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synac_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synex_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_a = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_b = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].synstr_mask_c = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse_mask = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].pulse = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].value = DEFAULT_STARTING_VALUE;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].max_syn_count = (*cortex)->max_syn_count;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].syn_count = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].tot_syn_strength = 0x00U;
            (*cortex)->neurons[IDX2D(x, y, (*cortex)->width)].inhexc_ratio = DEFAULT_INHEXC_RATIO;
        }
    }

    return ERROR_NONE;
}

error_code_t c2d_destroy(cortex2d_t* cortex) {
    // Free neurons.
    free(cortex->neurons);

    // Free cortex.
    free(cortex);

    return ERROR_NONE;
}

error_code_t c2d_copy(cortex2d_t* to, cortex2d_t* from) {
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

    return ERROR_NONE;
}