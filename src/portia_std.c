#include "portia_std.h"

error_code_t c2d_init(cortex2d_t* cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    if (SQNH_COUNT(SQNH_DIAM(nh_radius)) > sizeof(nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->width = width;
    cortex->height = height;
    cortex->ticks_count = 0x00;
    cortex->evol_step = DEFAULT_EVOL_STEP;
    cortex->pulse_window = DEFAULT_PULSE_WINDOW;

    cortex->nh_radius = nh_radius;
    cortex->fire_threshold = DEFAULT_THRESHOLD;
    cortex->recovery_value = DEFAULT_RECOVERY_VALUE;
    cortex->charge_value = DEFAULT_EXCITING_VALUE;
    cortex->decay_value = DEFAULT_DECAY_RATE;
    cortex->syngen_pulses_count = DEFAULT_SYNGEN_BEAT * DEFAULT_PULSE_WINDOW;
    cortex->max_syn_count = DEFAULT_MAX_TOUCH * SQNH_COUNT(SQNH_DIAM(nh_radius));
    cortex->inhexc_ratio = DEFAULT_INHEXC_RATIO;

    cortex->sample_window = DEFAULT_SAMPLE_WINDOW;
    cortex->pulse_mapping = PULSE_MAPPING_FPROP;

    cortex->neurons = (neuron_t*) malloc(cortex->width * cortex->height * sizeof(neuron_t));

    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = DEFAULT_NH_MASK;
            cortex->neurons[IDX2D(x, y, cortex->width)].synex_mask = ~DEFAULT_NH_MASK;
            cortex->neurons[IDX2D(x, y, cortex->width)].value = DEFAULT_STARTING_VALUE;
            cortex->neurons[IDX2D(x, y, cortex->width)].syn_count = 0x00u;
            cortex->neurons[IDX2D(x, y, cortex->width)].pulse_mask = DEFAULT_PULSE_MASK;
            cortex->neurons[IDX2D(x, y, cortex->width)].pulse = 0x00u;
        }
    }

    return NO_ERROR;
}

cortex2d_t* c2d_copy(cortex2d_t* other) {
    cortex2d_t* cortex = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    cortex->width = other->width;
    cortex->height = other->height;
    cortex->ticks_count = other->ticks_count;
    cortex->evol_step = other->evol_step;
    cortex->pulse_window = other->pulse_window;

    cortex->nh_radius = other->nh_radius;
    cortex->fire_threshold = other->fire_threshold;
    cortex->recovery_value = other->recovery_value;
    cortex->charge_value = other->charge_value;
    cortex->decay_value = other->decay_value;
    cortex->syngen_pulses_count = other->syngen_pulses_count;
    cortex->max_syn_count = other->max_syn_count;
    cortex->inhexc_ratio = other->inhexc_ratio;

    cortex->sample_window = other->sample_window;
    cortex->pulse_mapping = other->pulse_mapping;

    cortex->neurons = (neuron_t*) malloc(cortex->width * cortex->height * sizeof(neuron_t));

    for (cortex_size_t y = 0; y < other->height; y++) {
        for (cortex_size_t x = 0; x < other->width; x++) {
            cortex->neurons[IDX2D(x, y, other->width)] = other->neurons[IDX2D(x, y, other->width)];
        }
    }

    return cortex;
}

error_code_t c2d_set_nhradius(cortex2d_t* cortex, nh_radius_t radius) {
    // Make sure the provided radius is valid.
    if (radius <= 0 || SQNH_COUNT(SQNH_DIAM(radius)) > sizeof(nh_mask_t) * 8) {
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->nh_radius = radius;

    return NO_ERROR;
}

void c2d_set_nhmask(cortex2d_t* cortex, nh_mask_t mask) {
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            cortex->neurons[IDX2D(x, y, cortex->width)].synac_mask = mask;
        }
    }
}

void c2d_set_evol_step(cortex2d_t* cortex, evol_step_t evol_step) {
    cortex->evol_step = evol_step;
}

void c2d_set_pulse_window(cortex2d_t* cortex, pulses_count_t window) {
    if (window >= 0x00u && window < 0x3Fu) {
        cortex->pulse_window = window;
    }
}

void c2d_set_sample_window(cortex2d_t* cortex, ticks_count_t sample_window) {
    cortex->sample_window = sample_window;
}

void c2d_set_fire_threshold(cortex2d_t* cortex, neuron_threshold_t threshold) {
    cortex->fire_threshold = threshold;
}

void c2d_set_max_touch(cortex2d_t* cortex, float touch) {
    // Only set touch if a valid value is provided.
    if (touch <= 1 && touch >= 0) {
        cortex->max_syn_count = touch * SQNH_COUNT(SQNH_DIAM(cortex->nh_radius));
    }
}

void c2d_set_syngen_pulses_count(cortex2d_t* cortex, pulses_count_t pulses_count) {
    cortex->syngen_pulses_count = pulses_count;
}

void c2d_set_syngen_beat(cortex2d_t* cortex, float beat) {
    // Only set beat if a valid value is provided.
    if (beat <= 1.0F && beat >= 0.0F) {
        cortex->syngen_pulses_count = beat * cortex->pulse_window;
    }
}

void c2d_set_pulse_mapping(cortex2d_t* cortex, pulse_mapping_t pulse_mapping) {
    cortex->pulse_mapping = pulse_mapping;
}

void c2d_set_inhexc_ratio(cortex2d_t* cortex, ticks_count_t inhexc_ratio) {
    cortex->inhexc_ratio = inhexc_ratio;
}


void c2d_feed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t* values) {
    if (starting_index + count < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i].value += values[i];
        }
    }
}

void c2d_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t value) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 < cortex->width && y1 < cortex->height) {
        for (cortex_size_t y = y0; y < y1; y++) {
            for (cortex_size_t x = x0; x < x1; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].value += value;
            }
        }
    }
}

void c2d_sample_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, ticks_count_t sample_step, ticks_count_t* inputs, neuron_value_t value) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 < cortex->width && y1 < cortex->height) {
        #pragma omp parallel for
        for (cortex_size_t y = y0; y < y1; y++) {
            for (cortex_size_t x = x0; x < x1; x++) {
                ticks_count_t current_input = inputs[IDX2D(x - x0, y - y0, x1 - x0)];
                if (pulse_map(cortex->sample_window, sample_step, current_input, cortex->pulse_mapping)) {
                    cortex->neurons[IDX2D(x, y, cortex->width)].value += value;
                }
            }
        }
    }
}

void c2d_dfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t value) {
    if (starting_index + count < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i].value += value;
        }
    }
}

void c2d_rfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t max_value) {
    if (starting_index + count < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i].value += rand() % max_value;
        }
    }
}

void c2d_sfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t value, cortex_size_t spread) {
    if ((starting_index + count) * spread < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i * spread].value += value;
        }
    }
}

void c2d_rsfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t max_value, cortex_size_t spread) {
    if ((starting_index + count) * spread < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i * spread].value += rand() % max_value;
        }
    }
}

void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex) {
    uint32_t random;

    #pragma omp parallel for
    for (cortex_size_t y = 0; y < prev_cortex->height; y++) {
        for (cortex_size_t x = 0; x < prev_cortex->width; x++) {
            // Retrieve the involved neurons.
            neuron_t prev_neuron = prev_cortex->neurons[IDX2D(x, y, prev_cortex->width)];
            neuron_t* next_neuron = &(next_cortex->neurons[IDX2D(x, y, prev_cortex->width)]);

            // Copy prev neuron values to the new one.
            *next_neuron = prev_neuron;

            next_neuron->syn_count = 0x00u;

            /* Compute the neighborhood diameter:
                   d = 7
              <------------->
               r = 3
              <----->
              +-|-|-|-|-|-|-+
              |             |
              |             |
              |      @      |
              |             |
              |             |
              +-|-|-|-|-|-|-+
            */
            cortex_size_t nh_diameter = SQNH_DIAM(prev_cortex->nh_radius);

            nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
            nh_mask_t prev_exc_mask = prev_neuron.synex_mask;

            // Increment the current neuron value by reading its connected neighbors.
            for (nh_radius_t j = 0; j < nh_diameter; j++) {
                for (nh_radius_t i = 0; i < nh_diameter; i++) {
                    // Exclude the central neuron from the list of neighbors.
                    if (!(j == prev_cortex->nh_radius && i == prev_cortex->nh_radius)) {
                        // Fetch the current neighbor.
                        neuron_t neighbor = prev_cortex->neurons[IDX2D(WRAP(x + (i - prev_cortex->nh_radius), prev_cortex->width),
                                                                      WRAP(y + (j - prev_cortex->nh_radius), prev_cortex->height),
                                                                      prev_cortex->width)];

                        // Check if the last bit of the mask is 1 or 0: 1 = active synapse, 0 = inactive synapse.
                        if (prev_ac_mask & 0x01U) {
                            if (neighbor.value > prev_cortex->fire_threshold) {
                                next_neuron->value += prev_exc_mask & 0x01U ? DEFAULT_EXCITING_VALUE : DEFAULT_INHIBITING_VALUE;
                            }
                            next_neuron->syn_count++;
                        }

                        random = xorshf96();

                        // Perform evolution phase if allowed.
                        // evol_step is incremented by 1 to account for edge cases and human readable behavior:
                        // 0x0000 -> 0 + 1 = 1, so the cortex evolves at every tick, meaning that there are no free ticks between evolutions.
                        // 0xFFFF -> 65535 + 1 = 65536, so the cortex never evolves, meaning that there is an infinite amount of ticks between evolutions.
                        if ((prev_cortex->ticks_count % (((evol_step_t) prev_cortex->evol_step) + 1)) == 0 &&
                            random % 10000 < 10) {
                            if (prev_ac_mask & 0x01U &&
                                neighbor.pulse < prev_cortex->syngen_pulses_count) {
                                // Delete synapse.
                                nh_mask_t mask = ~(prev_neuron.synac_mask);
                                mask |= (0x01UL << IDX2D(i, j, nh_diameter));
                                next_neuron->synac_mask = ~mask;
                            } else if (!(prev_ac_mask & 0x01U) &&
                                       neighbor.pulse > prev_cortex->syngen_pulses_count &&
                                       prev_neuron.syn_count < prev_cortex->max_syn_count) {
                                // Add synapse.
                                next_neuron->synac_mask |= (0x01UL << IDX2D(i, j, nh_diameter));

                                // Define whether the new synapse is excitatory or inhibitory.
                                if (random % prev_cortex->inhexc_ratio == 0) {
                                    // Inhibitory.
                                    nh_mask_t mask = ~(prev_neuron.synex_mask);
                                    mask |= (0x01UL << IDX2D(i, j, nh_diameter));
                                    next_neuron->synex_mask = ~mask;
                                } else {
                                    // Excitatory.
                                    next_neuron->synex_mask |= (0x01UL << IDX2D(i, j, nh_diameter));
                                }
                            }
                        }
                    }

                    // Shift the mask to check for the next neighbor.
                    prev_ac_mask >>= 0x01U;
                    prev_exc_mask >>= 0x01U;
                }
            }

            // Push to equilibrium by decaying to zero, both from above and below.
            if (prev_neuron.value > 0x00) {
                next_neuron->value -= DEFAULT_DECAY_RATE;
            } else if (prev_neuron.value < 0x00) {
                next_neuron->value += DEFAULT_DECAY_RATE;
            }

            // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
            if (prev_neuron.value > prev_cortex->fire_threshold) {
                // Fired at the previous step.
                next_neuron->value = DEFAULT_RECOVERY_VALUE;

                // Store pulse.
                next_neuron->pulse_mask |= 0x01;
                next_neuron->pulse++;
            }

            if ((prev_neuron.pulse_mask >> prev_cortex->pulse_window) & 0x01) {
                // Decrease pulse if the oldest recorded pulse is active.
                next_neuron->pulse--;
            }

            next_neuron->pulse_mask <<= 0x01;
        }
    }

    next_cortex->ticks_count++;
}

bool_t pulse_map(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping) {
    bool_t result = FALSE;

    // Make sure the provided input correctly lies inside the provided window.
    if (input >= 0 && input < sample_window) {
        switch (pulse_mapping) {
            case PULSE_MAPPING_LINEAR:
                result = pulse_map_linear(sample_window, sample_step, input);
                break;
            case PULSE_MAPPING_FPROP:
                result = pulse_map_fprop(sample_window, sample_step, input);
                break;
            case PULSE_MAPPING_RPROP:
                result = pulse_map_rprop(sample_window, sample_step, input);
                break;
            default:
                break;
        }
    }

    return result;
}

bool_t pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    // sample_window = 10;
    // x = sample_window - input;
    // |@| | | | | | | | | | -> x = 10;
    // |@| | | | | | | | |@| -> x = 9;
    // |@| | | | | | | |@| | -> x = 8;
    // |@| | | | | | |@| | | -> x = 7;
    // |@| | | | | |@| | | | -> x = 6;
    // |@| | | | |@| | | | | -> x = 5;
    // |@| | | |@| | | |@| | -> x = 4;
    // |@| | |@| | |@| | |@| -> x = 3;
    // |@| |@| |@| |@| |@| | -> x = 2;
    // |@|@|@|@|@|@|@|@|@|@| -> x = 1;
    return sample_step % (sample_window - input) == 0;
}

bool_t pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    bool_t result = FALSE;
    ticks_count_t upper = sample_window - 1;

    // sample_window = 10;
    // upper = sample_window - 1 = 9;
    // | | | | | | | | | | | -> x = 0;
    // |@| | | | | | | | |@| -> x = 1;
    // |@| | | |@| | | |@| | -> x = 2;
    // |@| | |@| | |@| | |@| -> x = 3;
    // |@| |@| |@| |@| |@| | -> x = 4;
    // | |@| |@| |@| |@| |@| -> x = 5;
    // | |@|@| |@|@| |@|@| | -> x = 6;
    // | |@|@|@| |@|@|@| |@| -> x = 7;
    // | |@|@|@|@|@|@|@|@| | -> x = 8;
    // |@|@|@|@|@|@|@|@|@|@| -> x = 9;
    if (input < sample_window / 2) {
        if (input > 0 && sample_step % (upper / input) == 0) {
            result = TRUE;
        }
    } else {
        if (input >= upper || sample_step % (upper / (upper - input)) != 0) {
            result = TRUE;
        }
    }

    return result;
}

bool_t pulse_map_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    bool_t result = FALSE;
    double upper = sample_window - 1;
    double d_input = input;

    // sample_window = 10;
    // upper = sample_window - 1 = 9;
    // | | | | | | | | | | | -> x = 0;
    // |@| | | | | | | | |@| -> x = 1;
    // |@| | | | |@| | | | | -> x = 2;
    // |@| | |@| | |@| | |@| -> x = 3;
    // |@| |@| |@| |@| |@| | -> x = 4;
    // | |@| |@| |@| |@| |@| -> x = 5;
    // | |@|@| |@|@| |@|@| | -> x = 6;
    // | |@|@|@|@| |@|@|@|@| -> x = 7;
    // | |@|@|@|@|@|@|@|@| | -> x = 8;
    // |@|@|@|@|@|@|@|@|@|@| -> x = 9;
    if ((double) input < ((double) sample_window) / 2) {
        if (input > 0 && sample_step % (ticks_count_t) round(upper / d_input) == 0) {
            result = TRUE;
        }
    } else {
        if (input >= upper || sample_step % (ticks_count_t) round(upper / (upper - d_input)) != 0) {
            result = TRUE;
        }
    }

    return result;
}