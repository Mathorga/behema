#include "portia_std.h"

// The state word must be initialized to non-zero.
uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

error_code_t i2d_init(input2d_t* input, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t exc_value, pulse_mapping_t pulse_mapping) {
    input->x0 = x0;
    input->y0 = y0;
    input->x1 = x1;
    input->y1 = y1;
    input->exc_value = exc_value;
    input->pulse_mapping = pulse_mapping;

    // Allocate values.
    input->values = (ticks_count_t*) malloc((x1 - x0) * (y1 - y0) * sizeof(ticks_count_t));

    return ERROR_NONE;
}

error_code_t c2d_init(cortex2d_t** cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    // Allocate the cortex
    (*cortex) = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if ((*cortex) == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Setup cortex properties.
    (*cortex)->width = width;
    (*cortex)->height = height;
    (*cortex)->ticks_count = 0x00U;
    (*cortex)->rand_state = 0x01;
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

error_code_t c2d_set_nhradius(cortex2d_t* cortex, nh_radius_t radius) {
    // Make sure the provided radius is valid.
    if (radius <= 0 || NH_COUNT_2D(NH_DIAM_2D(radius)) > sizeof(nh_mask_t) * 8) {
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    cortex->nh_radius = radius;

    return ERROR_NONE;
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

void c2d_set_pulse_window(cortex2d_t* cortex, spikes_count_t window) {
    // The given window size must be between 0 and the pulse mask size (in bits).
    if (window >= 0x00u && window < (sizeof(pulse_mask_t) * 8)) {
        cortex->pulse_window = window;
    }
}

void c2d_set_sample_window(cortex2d_t* cortex, ticks_count_t sample_window) {
    cortex->sample_window = sample_window;
}

void c2d_set_fire_threshold(cortex2d_t* cortex, neuron_value_t threshold) {
    cortex->fire_threshold = threshold;
}

void c2d_set_max_syn_count(cortex2d_t* cortex, syn_count_t syn_count) {
    cortex->max_syn_count = syn_count;
}

void c2d_set_max_touch(cortex2d_t* cortex, float touch) {
    // Only set touch if a valid value is provided.
    if (touch <= 1 && touch >= 0) {
        cortex->max_syn_count = touch * NH_COUNT_2D(NH_DIAM_2D(cortex->nh_radius));
    }
}

void c2d_set_pulse_mapping(cortex2d_t* cortex, pulse_mapping_t pulse_mapping) {
    cortex->pulse_mapping = pulse_mapping;
}

void c2d_set_inhexc_range(cortex2d_t* cortex, chance_t inhexc_range) {
    cortex->inhexc_range = inhexc_range;
}

void c2d_set_inhexc_ratio(cortex2d_t* cortex, chance_t inhexc_ratio) {
    if (inhexc_ratio <= cortex->inhexc_range) {
        for (cortex_size_t y = 0; y < cortex->height; y++) {
            for (cortex_size_t x = 0; x < cortex->width; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].inhexc_ratio = inhexc_ratio;
            }
        }
    }
}

void c2d_syn_disable(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height) {
        for (cortex_size_t y = y0; y < y1; y++) {
            for (cortex_size_t x = x0; x < x1; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].max_syn_count = 0x00U;
            }
        }
    }
}


void c2d_feed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t* values) {
    if (starting_index + count < cortex->width * cortex->height) {
        // Loop through count.
        for (cortex_size_t i = starting_index; i < starting_index + count; i++) {
            cortex->neurons[i].value += values[i];
        }
    }
}

void c2d_feed2d(cortex2d_t* cortex, input2d_t* input) {
    #pragma omp parallel for collapse(2)
    for (cortex_size_t y = input->y0; y < input->y1; y++) {
        for (cortex_size_t x = input->x0; x < input->x1; x++) {
            if (pulse_map(cortex->sample_window,
                          cortex->ticks_count % cortex->sample_window,
                          input->values[IDX2D(x - input->x0, y - input->y0, input->x1 - input->x0)],
                          cortex->pulse_mapping)) {
                cortex->neurons[IDX2D(x, y, cortex->width)].value += input->exc_value;
            }
        }
    }
}

void c2d_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t value) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height) {
        for (cortex_size_t y = y0; y < y1; y++) {
            for (cortex_size_t x = x0; x < x1; x++) {
                cortex->neurons[IDX2D(x, y, cortex->width)].value += value;
            }
        }
    }
}

void c2d_sample_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, ticks_count_t sample_step, ticks_count_t* inputs, neuron_value_t value) {
    // Make sure the provided values are within the cortex size.
    if (x0 >= 0 && y0 >= 0 && x1 <= cortex->width && y1 <= cortex->height) {
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
            cortex->rand_state = xorshf32(cortex->rand_state);
            cortex->neurons[i].value += cortex->rand_state % max_value;
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
            cortex->rand_state = xorshf32(cortex->rand_state);
            cortex->neurons[i * spread].value += cortex->rand_state % max_value;
        }
    }
}

void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex) {
    #pragma omp parallel for collapse(2)
    for (cortex_size_t y = 0; y < prev_cortex->height; y++) {
        for (cortex_size_t x = 0; x < prev_cortex->width; x++) {
            // Retrieve the involved neurons.
            cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);
            neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
            neuron_t* next_neuron = &(next_cortex->neurons[neuron_index]);

            // Copy prev neuron values to the new one.
            *next_neuron = prev_neuron;

            // next_neuron->syn_count = 0x00u;

            /* Compute the neighborhood diameter:
                   d = 7
              <------------->
               r = 3
              <----->
              +-|-|-|-|-|-|-+
              |             |
              |             |
              |      X      |
              |             |
              |             |
              +-|-|-|-|-|-|-+
            */
            cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);

            nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
            nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
            nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
            nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
            nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;

            // Defines whether to evolve or not.
            // evol_step is incremented by 1 to account for edge cases and human readable behavior:
            // 0x0000 -> 0 + 1 = 1, so the cortex evolves at every tick, meaning that there are no free ticks between evolutions.
            // 0xFFFF -> 65535 + 1 = 65536, so the cortex never evolves, meaning that there is an infinite amount of ticks between evolutions.
            bool_t evolve = (prev_cortex->ticks_count % (((evol_step_t) prev_cortex->evol_step) + 1)) == 0;

            // Increment the current neuron value by reading its connected neighbors.
            for (nh_radius_t j = 0; j < nh_diameter; j++) {
                for (nh_radius_t i = 0; i < nh_diameter; i++) {
                    cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
                    cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);

                    // Exclude the central neuron from the list of neighbors.
                    if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                        (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height)) {
                        // The index of the current neighbor in the current neuron's neighborhood.
                        cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                        cortex_size_t neighbor_index = IDX2D(WRAP(neighbor_x, prev_cortex->width),
                                                             WRAP(neighbor_y, prev_cortex->height),
                                                             prev_cortex->width);

                        // Fetch the current neighbor.
                        neuron_t neighbor = prev_cortex->neurons[neighbor_index];

                        // Compute the current synapse strength.
                        syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) |
                                                      ((prev_str_mask_b & 0x01U) << 0x01U) |
                                                      ((prev_str_mask_c & 0x01U) << 0x02U);

                        // Pick a random number for each neighbor, capped to the max uint16 value.
                        next_cortex->rand_state = xorshf32(next_cortex->rand_state);
                        chance_t random = next_cortex->rand_state % 0xFFFFU;

                        // Inverse of the current neighbor's pulse, useful when computing depression probability (synapse deletion and weakening).
                        spikes_count_t pulse_diff = prev_cortex->pulse_window - neighbor.pulse;

                        // Check if the last bit of the mask is 1 or 0: 1 = active synapse, 0 = inactive synapse.
                        if (prev_ac_mask & 0x01U) {
                            neuron_value_t neighbor_influence = (prev_exc_mask & 0x01U ? prev_cortex->exc_value : -prev_cortex->exc_value) * ((syn_strength / 4) + 1);
                            if (neighbor.value > prev_cortex->fire_threshold) {
                                if (next_neuron->value + neighbor_influence < prev_cortex->recovery_value) {
                                    next_neuron->value = prev_cortex->recovery_value;
                                } else {
                                    next_neuron->value += neighbor_influence;
                                }
                            }
                        }

                        // Perform the evolution phase if allowed.
                        if (evolve) {
                            // Structural plasticity: create or destroy a synapse.
                            if (prev_ac_mask & 0x01U &&
                                // Only 0-strength synapses can be deleted.
                                syn_strength <= 0x00U &&
                                // TODO Make sure there's no overflow.
                                random < (prev_cortex->syngen_chance / POS_CHANCE_MULTIPLIER) * pulse_diff) {
                                // Delete synapse.
                                next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);

                                next_neuron->syn_count--;
                            } else if (!(prev_ac_mask & 0x01U) &&
                                       prev_neuron.syn_count < next_neuron->max_syn_count &&
                                       // TODO Make sure there's no overflow.
                                       random < prev_cortex->syngen_chance * neighbor.pulse) {
                                // Add synapse.
                                next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);

                                // Set the new synapse's strength to 0.
                                next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                                next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                                next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);

                                // Define whether the new synapse is excitatory or inhibitory.
                                if (random % next_cortex->inhexc_range < next_neuron->inhexc_ratio) {
                                    // Inhibitory.
                                    next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                                } else {
                                    // Excitatory.
                                    next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                                }

                                next_neuron->syn_count++;
                            }

                            // Functional plasticity: strengthen or weaken a synapse.
                            if (prev_ac_mask & 0x01U) {
                                if (syn_strength < MAX_SYN_STRENGTH &&
                                    prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                                    (
                                        // Random component.
                                        random < prev_cortex->synstr_chance / (syn_strength + 1) ||
                                        // TODO Make sure there's no overflow.
                                        random < prev_cortex->synstr_chance * neighbor.pulse ||
                                        // Neighbor fired right before the current neuron.
                                        (prev_neuron.pulse_mask & 0x01U && neighbor.pulse_mask >> 0x01U & 0x01U)
                                    )) {
                                    syn_strength++;
                                    next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                    next_neuron->tot_syn_strength++;
                                } else if (syn_strength > 0x00U &&
                                           // Random component.
                                           random < prev_cortex->synstr_chance / (syn_strength + 1) &&
                                           // TODO Make sure there's no overflow.
                                           random < (prev_cortex->synstr_chance / POS_CHANCE_MULTIPLIER) * pulse_diff &&
                                           // Neighbor fired right after the current neuron.
                                           (prev_neuron.pulse_mask >> 0x01U & 0x01U && neighbor.pulse_mask & 0x01U)) {
                                    syn_strength--;
                                    next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                    next_neuron->tot_syn_strength--;
                                }
                            }

                            // Increment evolutions count.
                            next_cortex->evols_count++;
                        }
                    }

                    // Shift the masks to check for the next neighbor.
                    prev_ac_mask >>= 0x01U;
                    prev_exc_mask >>= 0x01U;
                    prev_str_mask_a >>= 0x01U;
                    prev_str_mask_b >>= 0x01U;
                    prev_str_mask_c >>= 0x01U;
                }
            }

            // Push to equilibrium by decaying to zero, both from above and below.
            if (prev_neuron.value > 0x00) {
                next_neuron->value -= next_cortex->decay_value;
            } else if (prev_neuron.value < 0x00) {
                next_neuron->value += next_cortex->decay_value;
            }

            if ((prev_neuron.pulse_mask >> prev_cortex->pulse_window) & 0x01U) {
                // Decrease pulse if the oldest recorded pulse is active.
                next_neuron->pulse--;
            }

            next_neuron->pulse_mask <<= 0x01U;

            // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
            if (prev_neuron.value > prev_cortex->fire_threshold + prev_neuron.pulse) {
                // Fired at the previous step.
                next_neuron->value = next_cortex->recovery_value;

                // Store pulse.
                next_neuron->pulse_mask |= 0x01U;
                next_neuron->pulse++;
            }
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
            case PULSE_MAPPING_DFPROP:
                result = pulse_map_dfprop(sample_window, sample_step, input);
                break;
            default:
                break;
        }
    }

    return result;
}

bool_t pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    // sample_window = 10;
    // x = input;
    // |@| | | | | | | | | | -> x = 0;
    // |@| | | | | | | | |@| -> x = 1;
    // |@| | | | | | | |@| | -> x = 2;
    // |@| | | | | | |@| | | -> x = 3;
    // |@| | | | | |@| | | | -> x = 4;
    // |@| | | | |@| | | | | -> x = 5;
    // |@| | | |@| | | |@| | -> x = 6;
    // |@| | |@| | |@| | |@| -> x = 7;
    // |@| |@| |@| |@| |@| | -> x = 8;
    // |@|@|@|@|@|@|@|@|@|@| -> x = 9;
    return sample_step % (sample_window - input) == 0;
}

bool_t pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    bool_t result = FALSE;
    ticks_count_t upper = sample_window - 1;

    // sample_window = 10;
    // upper = sample_window - 1 = 9;
    // x = input;
    // |@| | | | | | | | | | -> x = 0;
    // |@| | | | | | | | |@| -> x = 1;
    // |@| | | |@| | | |@| | -> x = 2;
    // |@| | |@| | |@| | |@| -> x = 3;
    // |@| |@| |@| |@| |@| | -> x = 4;
    // | |@| |@| |@| |@| |@| -> x = 5;
    // | |@|@| |@|@| |@|@| | -> x = 6;
    // | |@|@|@| |@|@|@| |@| -> x = 7;
    // | |@|@|@|@|@|@|@|@| | -> x = 8;
    // | |@|@|@|@|@|@|@|@|@| -> x = 9;
    if (input < sample_window / 2) {
        if ((sample_step <= 0) ||
            (input > 0 && sample_step % (upper / input) == 0)) {
            result = TRUE;
        }
    } else {
        if ((sample_step >= 0) &&
            (input >= upper || sample_step % (upper / (upper - input)) != 0)) {
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
    // |@| | | | | | | | | | -> x = 0;
    // |@| | | | | | | | |@| -> x = 1;
    // |@| | | | |@| | | | | -> x = 2;
    // |@| | |@| | |@| | |@| -> x = 3;
    // |@| |@| |@| |@| |@| | -> x = 4;
    // | |@| |@| |@| |@| |@| -> x = 5;
    // | |@|@| |@|@| |@|@| | -> x = 6;
    // | |@|@|@|@| |@|@|@|@| -> x = 7;
    // | |@|@|@|@|@|@|@|@| | -> x = 8;
    // | |@|@|@|@|@|@|@|@|@| -> x = 9;
    if ((double) input < ((double) sample_window) / 2) {
        if ((sample_step <= 0) ||
            (input > 0 && sample_step % (ticks_count_t) round(upper / d_input) == 0)) {
            result = TRUE;
        }
    } else {
        if ((sample_step >= 0) &&
            (input >= upper || sample_step % (ticks_count_t) round(upper / (upper - d_input)) != 0)) {
            result = TRUE;
        }
    }

    return result;
}

bool_t pulse_map_dfprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    // TODO
    return FALSE;
}