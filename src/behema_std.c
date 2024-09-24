#include "behema_std.h"

void c2d_feed2d(cortex2d_t* cortex, input2d_t* input) {
    #pragma omp parallel for collapse(2)
    for (cortex_size_t y = input->y0; y < input->y1; y++) {
        for (cortex_size_t x = input->x0; x < input->x1; x++) {
            // Check whether the current input neuron should be excited or not.
            bhm_bool_t excite = value_to_pulse(
                cortex->sample_window,
                cortex->ticks_count % cortex->sample_window,
                input->values[
                    IDX2D(
                        x - input->x0,
                        y - input->y0,
                        input->x1 - input->x0
                    )
                ],
                cortex->pulse_mapping
            );

            if (excite) {
                cortex->neurons[IDX2D(x, y, cortex->width)].value += input->exc_value;
            }
        }
    }
}

void c2d_read2d(cortex2d_t* cortex, output2d_t* output) {
    #pragma omp parallel for collapse(2)
    for (cortex_size_t y = output->y0; y < output->y1; y++) {
        for (cortex_size_t x = output->x0; x < output->x1; x++) {
            output->values[
                IDX2D(
                    x - output->x0,
                    y - output->y0,
                    output->x1 - output->x0
                )
            ] = cortex->neurons[
                IDX2D(
                    x,
                    y,
                    cortex->width
                )
            ].pulse;
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
            bhm_bool_t evolve = (prev_cortex->ticks_count % (((evol_step_t) prev_cortex->evol_step) + 1)) == 0;

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
                        next_neuron->rand_state = xorshf32(next_neuron->rand_state);
                        chance_t random = next_neuron->rand_state % 0xFFFFU;

                        // Inverse of the current synapse strength, useful when computing depression probability (synapse deletion and weakening).
                        syn_strength_t strength_diff = MAX_SYN_STRENGTH - syn_strength;

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
                            if (!(prev_ac_mask & 0x01U) &&
                                prev_neuron.syn_count < next_neuron->max_syn_count &&
                                // Frequency component.
                                random < prev_cortex->syngen_chance * (chance_t) neighbor.pulse) {
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
                            } else if (prev_ac_mask & 0x01U &&
                                       // Only 0-strength synapses can be deleted.
                                       syn_strength <= 0x00U &&
                                       // Frequency component.
                                       random < prev_cortex->syngen_chance / (neighbor.pulse + 1)) {
                                // Delete synapse.
                                next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);

                                next_neuron->syn_count--;
                            }

                            // Functional plasticity: strengthen or weaken a synapse.
                            if (prev_ac_mask & 0x01U) {
                                if (syn_strength < MAX_SYN_STRENGTH &&
                                    prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                                    random < prev_cortex->synstr_chance * (chance_t) neighbor.pulse * (chance_t) strength_diff) {
                                    syn_strength++;
                                    next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                    next_neuron->tot_syn_strength++;
                                } else if (syn_strength > 0x00U &&
                                           random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1)) {
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


// ########################################## Input mapping functions ##########################################

bhm_bool_t value_to_pulse(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping) {
    bhm_bool_t result = BHM_FALSE;

    // Make sure the provided input correctly lies inside the provided window.
    if (input < sample_window) {
        switch (pulse_mapping) {
            case PULSE_MAPPING_LINEAR:
                result = value_to_pulse_linear(sample_window, sample_step, input);
                break;
            case PULSE_MAPPING_FPROP:
                result = value_to_pulse_fprop(sample_window, sample_step, input);
                break;
            case PULSE_MAPPING_RPROP:
                result = value_to_pulse_rprop(sample_window, sample_step, input);
                break;
            case PULSE_MAPPING_DFPROP:
                result = value_to_pulse_dfprop(sample_window, sample_step, input);
                break;
            default:
                break;
        }
    }

    return result;
}

bhm_bool_t value_to_pulse_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
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

bhm_bool_t value_to_pulse_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    bhm_bool_t result = BHM_FALSE;
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
            result = BHM_TRUE;
        }
    } else {
        if (input >= upper || sample_step % (upper / (upper - input)) != 0) {
            result = BHM_TRUE;
        }
    }

    return result;
}

bhm_bool_t value_to_pulse_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    bhm_bool_t result = BHM_FALSE;
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
            result = BHM_TRUE;
        }
    } else {
        if (input >= upper || sample_step % (ticks_count_t) round(upper / (upper - d_input)) != 0) {
            result = BHM_TRUE;
        }
    }

    return result;
}

bhm_bool_t value_to_pulse_dfprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
    // TODO
    return BHM_FALSE;
}