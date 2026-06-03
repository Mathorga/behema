#include "behema_std.h"

void c2d_feed2d(
    bhm_cortex2d_t* cortex,
    bhm_input2d_t* input,
    bhm_ticks_count_t ticks_count
) {
    #pragma omp parallel for collapse(2)
    for (bhm_cortex_size_t y = input->y0; y < input->y1; y++) {
        for (bhm_cortex_size_t x = input->x0; x < input->x1; x++) {
            // Check whether the current input neuron should be excited or not.
            bhm_bool_t excite = value_to_pulse(
                cortex->sample_window,
                ticks_count % cortex->sample_window,
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
                cortex->n_values[IDX2D(x, y, cortex->width)] += input->exc_value;
            }
        }
    }
}

void c2d_read2d(bhm_cortex2d_t* cortex, bhm_output2d_t* output) {
    #pragma omp parallel for collapse(2)
    for (bhm_cortex_size_t y = output->y0; y < output->y1; y++) {
        for (bhm_cortex_size_t x = output->x0; x < output->x1; x++) {
            output->values[
                IDX2D(
                    x - output->x0,
                    y - output->y0,
                    output->x1 - output->x0
                )
            ] = cortex->n_pulses[
                IDX2D(
                    x,
                    y,
                    cortex->width
                )
            ];
        }
    }
}

void c2d_tick(
    bhm_cortex2d_t* prev_cortex,
    bhm_cortex2d_t* next_cortex,
    bhm_bool_t evolve
) {
    #pragma omp parallel for collapse(2)
    for (bhm_cortex_size_t y = 0; y < prev_cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < prev_cortex->width; x++) {
            // Retrieve the involved neuron index.
            bhm_cortex_size_t neuron_index = IDX2D(x, y, prev_cortex->width);

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
            bhm_cortex_size_t nh_diameter = NH_DIAM_2D(prev_cortex->nh_radius);

            bhm_nh_mask_t synac_mask = prev_cortex->n_synac_masks[neuron_index];
            bhm_nh_mask_t synex_mask = prev_cortex->n_synex_masks[neuron_index];
            bhm_nh_mask_t synstr_mask_a = prev_cortex->n_synstr_masks_a[neuron_index];
            bhm_nh_mask_t synstr_mask_b = prev_cortex->n_synstr_masks_b[neuron_index];
            bhm_nh_mask_t synstr_mask_c = prev_cortex->n_synstr_masks_c[neuron_index];

            // Increment the current neuron value by reading its connected neighbors.
            for (bhm_nh_radius_t j = 0; j < nh_diameter; j++) {
                for (bhm_nh_radius_t i = 0; i < nh_diameter; i++) {
                    bhm_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
                    bhm_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);

                    // Exclude the central neuron from the list of neighbors.
                    if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                        (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height)) {
                        // The index of the current neighbor in the current neuron's neighborhood.
                        bhm_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                        bhm_cortex_size_t neighbor_index = IDX2D(
                            WRAP(neighbor_x, prev_cortex->width),
                            WRAP(neighbor_y, prev_cortex->height),
                            prev_cortex->width
                        );

                        // Compute the current synapse strength.
                        bhm_syn_strength_t syn_strength = (
                            (synstr_mask_a & 0x01U) |
                            ((synstr_mask_b & 0x01U) << 0x01U) |
                            ((synstr_mask_c & 0x01U) << 0x02U)
                        );

                        // Inverse of the current synapse strength, useful when computing depression probability (synapse deletion and weakening).
                        bhm_syn_strength_t strength_diff = BHM_MAX_SYN_STRENGTH - syn_strength;

                        // Check if the last bit of the mask is 1 or 0: 1 = active synapse, 0 = inactive synapse.
                        if (synac_mask & 0x01U) {
                            bhm_neuron_value_t neighbor_influence = (synex_mask & 0x01U ? prev_cortex->exc_value : -prev_cortex->exc_value) * ((syn_strength / 4) + 1);
                            if (prev_cortex->n_values[neighbor_index] > prev_cortex->fire_threshold) {
                                if (next_cortex->n_values[neuron_index] + neighbor_influence < prev_cortex->recovery_value) {
                                    next_cortex->n_values[neuron_index] = prev_cortex->recovery_value;
                                } else {
                                    next_cortex->n_values[neuron_index] += neighbor_influence;
                                }
                            }
                        }

                        // Perform the evolution phase if allowed.
                        if (evolve) {
                            // Pick a random number for each neighbor, capped to the max uint16 value.
                            next_cortex->n_l_rand_states[neuron_index] = xorshf32(prev_cortex->n_l_rand_states[neuron_index]);
                            bhm_chance_t random = next_cortex->n_l_rand_states[neuron_index] % 0xFFFFU;

                            // Structural plasticity: create or destroy a synapse.
                            if (
                                !(synac_mask & 0x01U) &&
                                prev_cortex->n_syn_counts[neuron_index] < next_cortex->n_max_syn_counts[neuron_index] &&
                                // Frequency component.
                                random < prev_cortex->syngen_chance * (bhm_chance_t) prev_cortex->n_pulses[neighbor_index]
                            ) {
                                // Add synapse.
                                next_cortex->n_synac_masks[neuron_index] |= (0x01UL << neighbor_nh_index);

                                // Set the new synapse's strength to 0.
                                next_cortex->n_synstr_masks_a[neuron_index] &= ~(0x01UL << neighbor_nh_index);
                                next_cortex->n_synstr_masks_b[neuron_index] &= ~(0x01UL << neighbor_nh_index);
                                next_cortex->n_synstr_masks_c[neuron_index] &= ~(0x01UL << neighbor_nh_index);

                                // Define whether the new synapse is excitatory or inhibitory.
                                if (random % next_cortex->inhexc_range < next_cortex->n_inhexc_ratios[neuron_index]) {
                                    // Inhibitory.
                                    next_cortex->n_synex_masks[neuron_index] &= ~(0x01UL << neighbor_nh_index);
                                } else {
                                    // Excitatory.
                                    next_cortex->n_synex_masks[neuron_index] |= (0x01UL << neighbor_nh_index);
                                }

                                next_cortex->n_syn_counts[neuron_index]++;
                            } else if (
                                synac_mask & 0x01U &&
                                // Only 0-strength synapses can be deleted.
                                syn_strength <= 0x00U &&
                                // Frequency component.
                                random < prev_cortex->syngen_chance / (prev_cortex->n_pulses[neighbor_index] + 1)
                            ) {
                                // Delete synapse.
                                next_cortex->n_synac_masks[neuron_index] &= ~(0x01UL << neighbor_nh_index);

                                next_cortex->n_syn_counts[neuron_index]--;
                            }

                            // Functional plasticity: strengthen or weaken a synapse.
                            if (synac_mask & 0x01U) {
                                if (
                                    syn_strength < BHM_MAX_SYN_STRENGTH &&
                                    prev_cortex->n_tot_syn_strengths[neuron_index] < prev_cortex->max_tot_strength &&
                                    random < prev_cortex->synstr_chance * (bhm_chance_t) prev_cortex->n_pulses[neighbor_index] * (bhm_chance_t) strength_diff
                                ) {
                                    syn_strength++;
                                    next_cortex->n_synstr_masks_a[neuron_index] = (synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_cortex->n_synstr_masks_b[neuron_index] = (synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_cortex->n_synstr_masks_c[neuron_index] = (synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                    next_cortex->n_tot_syn_strengths[neuron_index]++;
                                } else if (
                                    syn_strength > 0x00U &&
                                    random < prev_cortex->synstr_chance / (prev_cortex->n_pulses[neighbor_index] + syn_strength + 1)
                                ) {
                                    syn_strength--;
                                    next_cortex->n_synstr_masks_a[neuron_index] = (synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                    next_cortex->n_synstr_masks_b[neuron_index] = (synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                    next_cortex->n_synstr_masks_c[neuron_index] = (synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                    next_cortex->n_tot_syn_strengths[neuron_index]--;
                                }
                            }
                        }
                    }

                    // Shift the masks to check for the next neighbor.
                    synac_mask >>= 0x01U;
                    synex_mask >>= 0x01U;
                    synstr_mask_a >>= 0x01U;
                    synstr_mask_b >>= 0x01U;
                    synstr_mask_c >>= 0x01U;
                }
            }

            // Push to equilibrium by decaying to zero, both from above and below.
            if (prev_cortex->n_values[neuron_index] > 0x00) {
                next_cortex->n_values[neuron_index] -= next_cortex->decay_value;
            } else if (prev_cortex->n_values[neuron_index] < 0x00) {
                next_cortex->n_values[neuron_index] += next_cortex->decay_value;
            }

            if ((prev_cortex->n_pulse_masks[neuron_index] >> prev_cortex->pulse_window) & 0x01U) {
                // Decrease pulse if the oldest recorded pulse is active.
                next_cortex->n_pulses[neuron_index]--;
            }

            next_cortex->n_pulse_masks[neuron_index] <<= 0x01U;

            // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
            if (prev_cortex->n_values[neuron_index] > prev_cortex->fire_threshold + prev_cortex->n_pulses[neuron_index]) {
                printf("FIRE!\n");
                // Fired at the previous step.
                next_cortex->n_values[neuron_index] = next_cortex->recovery_value;

                // Store pulse.
                next_cortex->n_pulse_masks[neuron_index] |= 0x01U;
                next_cortex->n_pulses[neuron_index]++;
            }
        }
    }
}


// ########################################## Input mapping functions ##########################################

bhm_bool_t value_to_pulse(
    bhm_ticks_count_t sample_window,
    bhm_ticks_count_t sample_step,
    bhm_ticks_count_t input,
    bhm_pulse_mapping_t pulse_mapping
) {
    bhm_bool_t result = BHM_FALSE;

    // Make sure the provided input correctly lies inside the provided window.
    if (input < sample_window) {
        switch (pulse_mapping) {
            case BHM_PULSE_MAPPING_LINEAR:
                result = value_to_pulse_linear(sample_window, sample_step, input);
                break;
            case BHM_PULSE_MAPPING_FPROP:
                result = value_to_pulse_fprop(sample_window, sample_step, input);
                break;
            case BHM_PULSE_MAPPING_RPROP:
                result = value_to_pulse_rprop(sample_window, sample_step, input);
                break;
            case BHM_PULSE_MAPPING_DFPROP:
                result = value_to_pulse_dfprop(sample_window, sample_step, input);
                break;
            default:
                break;
        }
    }

    return result;
}

bhm_bool_t value_to_pulse_linear(
    bhm_ticks_count_t sample_window,
    bhm_ticks_count_t sample_step,
    bhm_ticks_count_t input
) {
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

bhm_bool_t value_to_pulse_fprop(
    bhm_ticks_count_t sample_window,
    bhm_ticks_count_t sample_step,
    bhm_ticks_count_t input
) {
    bhm_bool_t result = BHM_FALSE;
    bhm_ticks_count_t upper = sample_window - 1;

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

bhm_bool_t value_to_pulse_rprop(
    bhm_ticks_count_t sample_window,
    bhm_ticks_count_t sample_step,
    bhm_ticks_count_t input
) {
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
            (input > 0 && sample_step % (bhm_ticks_count_t) round(upper / d_input) == 0)) {
            result = BHM_TRUE;
        }
    } else {
        if (input >= upper || sample_step % (bhm_ticks_count_t) round(upper / (upper - d_input)) != 0) {
            result = BHM_TRUE;
        }
    }

    return result;
}

bhm_bool_t value_to_pulse_dfprop(
    bhm_ticks_count_t sample_window,
    bhm_ticks_count_t sample_step,
    bhm_ticks_count_t input
) {
    // TODO
    return BHM_FALSE;
}