#include "behema_std.h"

void crx2d_feed2d(
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
                    BHM_IDX2D(
                        x - input->x0,
                        y - input->y0,
                        input->x1 - input->x0
                    )
                ],
                cortex->pulse_mapping
            );

            if (excite) {
                cortex->neurons[BHM_IDX2D(x, y, cortex->width)].value += input->exc_value;
            }
        }
    }
}

void crx2d_read2d(bhm_cortex2d_t* cortex, bhm_output2d_t* output) {
    #pragma omp parallel for collapse(2)
    for (bhm_cortex_size_t y = output->y0; y < output->y1; y++) {
        for (bhm_cortex_size_t x = output->x0; x < output->x1; x++) {
            output->values[
                BHM_IDX2D(
                    x - output->x0,
                    y - output->y0,
                    output->x1 - output->x0
                )
            ] = cortex->neurons[
                BHM_IDX2D(
                    x,
                    y,
                    cortex->width
                )
            ].pulse;
        }
    }
}

void crx2d_tick(
    bhm_cortex2d_t* prev_cortex,
    bhm_cortex2d_t* next_cortex,
    bhm_bool_t evolve
) {
    #pragma omp parallel for collapse(2)
    for (bhm_cortex_size_t y = 0; y < prev_cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < prev_cortex->width; x++) {
            // Retrieve the involved neurons.
            bhm_cortex_size_t neuron_index = BHM_IDX2D(x, y, prev_cortex->width);
            bhm_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
            bhm_neuron_t* next_neuron = &(next_cortex->neurons[neuron_index]);

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
            bhm_cortex_size_t nh_diameter = BHM_NH_DIAM_2D(prev_cortex->nh_radius);

            // Number of cells in the neighborhood, central neuron included.
            bhm_cortex_size_t nh_cells = nh_diameter * nh_diameter;

            // Pick the neighborhood cell to start visiting neighbors from.
            // During evolution, neighbors compete for limited resources in the same tick (free synapse slots below max_syn_count, free
            // strength below max_tot_strength), and the ones visited first win. With a fixed starting cell, the same positions (top-left)
            // would always win, slowly biasing connectivity towards them (measured with tests/bias/bias.c).
            // Starting from a random cell gives every position the same chance of being visited first over time.
            // The upper 16 bits of the random state are used, since the lower 16 are already used as chance values during evolution.
            // Ticks without evolution keep the fixed starting cell (0): there is no competition there, and a random visiting order
            // makes memory accesses less predictable, which made ticks about 30% slower in benchmarks with default parameters.
            bhm_cortex_size_t nh_start = evolve ? (next_neuron->rand_state >> 0x10U) % nh_cells : 0;

            // Rotate the masks so that the starting cell's bits are the last ones.
            // The loop then only needs to read the last bit of each mask and shift the masks by one at the end of each iteration,
            // exactly as if the neighborhood started at nh_start. Rotating once here is cheaper than shifting by a different amount for each neighbor.
            bhm_nh_mask_t prev_ac_mask = BHM_ROTR_MASK(prev_neuron.synac_mask, nh_start, nh_cells);
            bhm_nh_mask_t prev_exc_mask = BHM_ROTR_MASK(prev_neuron.synex_mask, nh_start, nh_cells);
            bhm_nh_mask_t prev_str_mask_a = BHM_ROTR_MASK(prev_neuron.synstr_mask_a, nh_start, nh_cells);
            bhm_nh_mask_t prev_str_mask_b = BHM_ROTR_MASK(prev_neuron.synstr_mask_b, nh_start, nh_cells);
            bhm_nh_mask_t prev_str_mask_c = BHM_ROTR_MASK(prev_neuron.synstr_mask_c, nh_start, nh_cells);

            // Coordinates of the current cell inside the neighborhood.
            bhm_cortex_size_t i = nh_start % nh_diameter;
            bhm_cortex_size_t j = nh_start / nh_diameter;

            // Increment the current neuron value by reading its connected neighbors.
            // Every neighborhood cell is visited exactly once, starting from nh_start and wrapping around at the end of the neighborhood.
            for (bhm_cortex_size_t k = 0; k < nh_cells; k++) {
                // The index of the current neighbor in the current neuron's neighborhood (not rotated): used to edit next_neuron's masks.
                bhm_cortex_size_t neighbor_nh_index = BHM_IDX2D(i, j, nh_diameter);

                bhm_cortex_size_t neighbor_x = x + (i - prev_cortex->nh_radius);
                bhm_cortex_size_t neighbor_y = y + (j - prev_cortex->nh_radius);

                // Exclude the central neuron from the list of neighbors.
                if ((j != prev_cortex->nh_radius || i != prev_cortex->nh_radius) &&
                    (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < prev_cortex->width && neighbor_y < prev_cortex->height)) {
                    bhm_cortex_size_t neighbor_index = BHM_IDX2D(
                        BHM_WRAP(neighbor_x, prev_cortex->width),
                        BHM_WRAP(neighbor_y, prev_cortex->height),
                        prev_cortex->width
                    );

                    // Fetch the current neighbor.
                    bhm_neuron_t neighbor = prev_cortex->neurons[neighbor_index];

                    // Compute the current synapse strength.
                    bhm_syn_strength_t syn_strength = (prev_str_mask_a & 0x01U) |
                                                  ((prev_str_mask_b & 0x01U) << 0x01U) |
                                                  ((prev_str_mask_c & 0x01U) << 0x02U);

                    // Inverse of the current synapse strength, useful when computing depression probability (synapse deletion and weakening).
                    bhm_syn_strength_t strength_diff = BHM_MAX_SYN_STRENGTH - syn_strength;

                    // Check if the last bit of the mask is 1 or 0: 1 = active synapse, 0 = inactive synapse.
                    if (prev_ac_mask & 0x01U) {
                        bhm_neuron_value_t neighbor_influence = (prev_exc_mask & 0x01U ? prev_cortex->exc_value : -prev_cortex->exc_value) * ((syn_strength / 4) + 1);
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
                        // Pick a random number for each neighbor, capped to the max uint16 value.
                        next_neuron->rand_state = bhm_xorshf32(next_neuron->rand_state);

                        // Here capping to the max uint16 value is performed by discarding bytes in positions bigger than the biggest wanted value.
                        bhm_chance_t random = next_neuron->rand_state & 0xFFFFU;

                        // Structural plasticity: create or destroy a synapse.
                        if (
                            !(prev_ac_mask & 0x01U) &&
                            // Use the running count, so that the max holds even when multiple synapses are added in the same tick.
                            next_neuron->syn_count < next_neuron->max_syn_count &&
                            // Frequency component, scaled by the fraction of room left below max_syn_count (resource-limited growth):
                            // the closer the neuron is to its max, the less likely it is to grow a new synapse.
                            // Computed on 64 bits to avoid overflows. The check above guarantees max_syn_count > 0.
                            random < ((uint64_t) prev_cortex->syngen_chance *
                                      (uint64_t) neighbor.pulse *
                                      (uint64_t) (next_neuron->max_syn_count - next_neuron->syn_count)) /
                                     next_neuron->max_syn_count
                        ) {
                            // Add synapse.
                            next_neuron->synac_mask |= (0x01UL << neighbor_nh_index);

                            // Set the new synapse's strength to 0.
                            next_neuron->synstr_mask_a &= ~(0x01UL << neighbor_nh_index);
                            next_neuron->synstr_mask_b &= ~(0x01UL << neighbor_nh_index);
                            next_neuron->synstr_mask_c &= ~(0x01UL << neighbor_nh_index);

                            // Define whether the new synapse is excitatory or inhibitory.
                            if ((random % next_cortex->inhexc_range) < next_neuron->inhexc_ratio) {
                                // Inhibitory.
                                next_neuron->synex_mask &= ~(0x01UL << neighbor_nh_index);
                            } else {
                                // Excitatory.
                                next_neuron->synex_mask |= (0x01UL << neighbor_nh_index);
                            }

                            next_neuron->syn_count++;
                        } else if (
                            prev_ac_mask & 0x01U &&
                            // Only 0-strength synapses can be deleted.
                            syn_strength <= 0x00U &&
                            // Frequency component.
                            random < (prev_cortex->syngen_chance / (neighbor.pulse + 1))
                        ) {
                            // Delete synapse.
                            next_neuron->synac_mask &= ~(0x01UL << neighbor_nh_index);

                            next_neuron->syn_count--;
                        }

                        // Functional plasticity: strengthen or weaken a synapse.
                        // Only synapses that existed before this tick and were not deleted during it are affected.
                        if ((prev_ac_mask & 0x01U) && ((next_neuron->synac_mask >> neighbor_nh_index) & 0x01U)) {
                            if (
                                syn_strength < BHM_MAX_SYN_STRENGTH &&
                                // Use the running total, so that the max holds even when multiple synapses are strengthened in the same tick.
                                next_neuron->tot_syn_strength < prev_cortex->max_tot_strength &&
                                // Frequency component, scaled by the fraction of room left below max_tot_strength (resource-limited growth):
                                // the closer the neuron is to its max, the less likely it is to strengthen a synapse.
                                // Computed on 64 bits to avoid overflows. The check above guarantees max_tot_strength > 0.
                                random < ((uint64_t) prev_cortex->synstr_chance *
                                          (uint64_t) neighbor.pulse *
                                          (uint64_t) strength_diff *
                                          (uint64_t) (prev_cortex->max_tot_strength - next_neuron->tot_syn_strength)) /
                                         prev_cortex->max_tot_strength
                            ) {
                                syn_strength++;
                                next_neuron->synstr_mask_a = (next_neuron->synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                next_neuron->synstr_mask_b = (next_neuron->synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                next_neuron->synstr_mask_c = (next_neuron->synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                next_neuron->tot_syn_strength++;
                            } else if (
                                syn_strength > 0x00U &&
                                random < prev_cortex->synstr_chance / (neighbor.pulse + syn_strength + 1)
                            ) {
                                syn_strength--;
                                next_neuron->synstr_mask_a = (next_neuron->synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                                next_neuron->synstr_mask_b = (next_neuron->synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                                next_neuron->synstr_mask_c = (next_neuron->synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                                next_neuron->tot_syn_strength--;
                            }
                        }
                    }
                }

                // Shift the masks to check for the next neighbor.
                prev_ac_mask >>= 0x01U;
                prev_exc_mask >>= 0x01U;
                prev_str_mask_a >>= 0x01U;
                prev_str_mask_b >>= 0x01U;
                prev_str_mask_c >>= 0x01U;

                // Move to the next neighborhood cell: wrap to the next row at the end of each row, and back to the first row at the end of the neighborhood.
                i++;
                if (i >= nh_diameter) {
                    i = 0;
                    j++;
                    if (j >= nh_diameter) j = 0;
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
}

bhm_error_code_t ctx2d_tick(
    bhm_context2d_t* context
) {
    // Setup.
    bhm_cortex2d_t* prev_cortex = context->counts->ticks_count % 2 ? context->odd_cortex : context->even_cortex;
    bhm_cortex2d_t* next_cortex = context->counts->ticks_count % 2 ? context->even_cortex : context->odd_cortex;

    // #################### Step 1: Inputs feeding. ####################
    for (bhm_cortex_size_t i = 0; i < context->inputs_count; i++) {
        crx2d_feed2d(
            prev_cortex,
            context->inputs[i],
            context->counts->ticks_count
        );
    }

    // #################### Step 2: Cortex ticking. ####################

    // Defines whether to evolve or not.
    // evol_step is incremented by 1 to account for edge cases and human readable behavior:
    // 0x0000 -> 0 + 1 = 1, so the cortex evolves at EVERY tick, meaning that there are no free ticks between evolutions.
    // 0xFFFF -> 65535 + 1 = 65536, so the cortex NEVER evolves, meaning that there is an infinite amount of ticks between evolutions.
    bhm_bool_t evolve = (context->counts->ticks_count % (((bhm_evol_step_t) prev_cortex->evol_step) + 1)) == 0;
    crx2d_tick(
        prev_cortex,
        next_cortex,
        evolve
    );

    // Increment counters.
    context->counts->ticks_count++;
    if (evolve) context->counts->evols_count++;

    // #################### Step 3: Outputs reading. ####################
    for (bhm_cortex_size_t i = 0; i < context->outputs_count; i++) {
        crx2d_read2d(
            next_cortex,
            context->outputs[i]
        );
    }

    return BHM_ERROR_NONE;
}


// ########################################## Input mapping functions ##########################################

bhm_bool_t value_to_pulse(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input, bhm_pulse_mapping_t pulse_mapping) {
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

bhm_bool_t value_to_pulse_linear(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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

bhm_bool_t value_to_pulse_fprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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

bhm_bool_t value_to_pulse_rprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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

bhm_bool_t value_to_pulse_dfprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
    // TODO
    return BHM_FALSE;
}