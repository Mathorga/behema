#include "portia_cuda.h"

// The state must be initialized to non-zero.
__host__ __device__ uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


// Initialization functions.

error_code_t i2d_to_device(input2d_t* device_input, input2d_t* host_input) {
    cudaError_t cuda_error;

    // Allocate tmp input on the host.
    input2d_t* tmp_input = (input2d_t*) malloc(sizeof(input2d_t));
    if (tmp_input == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Copy host input to tmp input.
    (*tmp_input) = (*host_input);

    // Allocate values on the device.
    cuda_error = cudaMalloc((void**) &(tmp_input->values), (host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0) * sizeof(ticks_count_t));
    printf("\nHERE %d %d %d %d\n", host_input->x0, host_input->y0, host_input->x1, host_input->y1);
    cudaCheckError();
    if (cuda_error != cudaSuccess) {
        return ERROR_FAILED_ALLOC;
    }

    // Copy values to device.
    cudaMemcpy(
        tmp_input->values,
        host_input->values,
        ((host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0)) * sizeof(ticks_count_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Copy tmp input to device.
    cudaMemcpy(
        device_input,
        tmp_input,
        sizeof(input2d_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Cleanup.
    free(tmp_input);

    return ERROR_NONE;
}

error_code_t i2d_to_host(input2d_t* host_input, input2d_t* device_input) {
    // TODO
    return ERROR_NONE;
}

error_code_t c2d_to_device(cortex2d_t* device_cortex, cortex2d_t* host_cortex) {
    cudaError_t cuda_error;

    // Allocate tmp cortex on the host.
    cortex2d_t* tmp_cortex = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if (tmp_cortex == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Copy host cortex to tmp cortex.
    (*tmp_cortex) = (*host_cortex);

    // Allocate neurons on the device.
    cuda_error = cudaMalloc((void**) &(tmp_cortex->neurons), host_cortex->width * host_cortex->height * sizeof(neuron_t));
    cudaCheckError();
    if (cuda_error != cudaSuccess) {
        return ERROR_FAILED_ALLOC;
    }

    // Copy neurons to device.
    cudaMemcpy(
        tmp_cortex->neurons,
        host_cortex->neurons,
        host_cortex->width * host_cortex->height * sizeof(neuron_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Copy tmp cortex to device.
    cudaMemcpy(device_cortex, tmp_cortex, sizeof(cortex2d_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Cleanup.
    free(tmp_cortex);

    return ERROR_NONE;
}

error_code_t c2d_to_host(cortex2d_t* host_cortex, cortex2d_t* device_cortex) {
    // Allocate tmp cortex on the host.
    cortex2d_t* tmp_cortex = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if (tmp_cortex == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Copy tmp cortex to device.
    cudaMemcpy(tmp_cortex, device_cortex, sizeof(cortex2d_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Copy tmp cortex to host cortex.
    (*host_cortex) = (*tmp_cortex);

    // Allocate neurons on the host.
    host_cortex->neurons = (neuron_t*) malloc(tmp_cortex->width * tmp_cortex->height * sizeof(neuron_t));

    // Copy tmp cortex neurons (still on device) to host cortex.
    cudaMemcpy(host_cortex->neurons, tmp_cortex->neurons, tmp_cortex->width * tmp_cortex->height * sizeof(neuron_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Cleanup.
    free(tmp_cortex);

    return ERROR_NONE;
}

error_code_t i2d_device_destroy(input2d_t* input) {
    // TODO
    return ERROR_NONE;
}

error_code_t c2d_device_destroy(cortex2d_t* cortex) {
    // Allocate tmp cortex on the host.
    cortex2d_t* tmp_cortex = (cortex2d_t*) malloc(sizeof(cortex2d_t));
    if (tmp_cortex == NULL) {
        return ERROR_FAILED_ALLOC;
    }
    
    // Copy device cortex to host in order to free its neurons.
    cudaMemcpy(tmp_cortex, cortex, sizeof(cortex2d_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Free device neurons.
    cudaFree(tmp_cortex->neurons);
    cudaCheckError();

    // Free tmp cortex.
    free(tmp_cortex);

    // Finally free device cortex.
    cudaFree(cortex);
    cudaCheckError();

    return ERROR_NONE;
}


// Execution functions.

__global__ void c2d_feed2d(cortex2d_t* cortex, input2d_t* input) {
    cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= input->x1 - input->x0 || y >= input->y1 - input->y0) {
        return;
    }

    if (pulse_map(
            cortex->sample_window,
            cortex->ticks_count % cortex->sample_window,
            input->values[IDX2D(x, y, input->x1 - input->x0)],
            cortex->pulse_mapping
        )) {
        cortex->neurons[IDX2D(x + input->x0, y + input->y0, cortex->width)].value += input->exc_value;
    }
}

__global__ void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex) {
    cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= prev_cortex->width || y >= prev_cortex->height) {
        return;
    }

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
    bool evolve = (prev_cortex->ticks_count % (((evol_step_t) prev_cortex->evol_step) + 1)) == 0;

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
                        // TODO Make sure there's no overflow.
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
                            // TODO Make sure there's no overflow.
                            random < prev_cortex->synstr_chance * (chance_t) neighbor.pulse * (chance_t) strength_diff) {
                            syn_strength++;
                            next_neuron->synstr_mask_a = (prev_neuron.synstr_mask_a & ~(0x01UL << neighbor_nh_index)) | ((syn_strength & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_b = (prev_neuron.synstr_mask_b & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x01U) & 0x01U) << neighbor_nh_index);
                            next_neuron->synstr_mask_c = (prev_neuron.synstr_mask_c & ~(0x01UL << neighbor_nh_index)) | (((syn_strength >> 0x02U) & 0x01U) << neighbor_nh_index);

                            next_neuron->tot_syn_strength++;
                        } else if (syn_strength > 0x00U &&
                                    // TODO Make sure there's no overflow.
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

    next_cortex->ticks_count++;
}

__host__ __device__ bool_t pulse_map(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping) {
    bool_t result = FALSE;

    // Make sure the provided input correctly lies inside the provided window.
    if (input < sample_window) {
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

__host__ __device__ bool_t pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
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
    return (sample_step % (sample_window - input) == 0) ? TRUE : FALSE;
}

__host__ __device__ bool_t pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
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
            (sample_step % (upper / input) == 0)) {
            result = TRUE;
        }
    } else {
        if (input >= upper || sample_step % (upper / (upper - input)) != 0) {
            result = TRUE;
        }
    }

    return result;
}

__host__ __device__ bool_t pulse_map_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input) {
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
            sample_step % (ticks_count_t) round(upper / d_input) == 0) {
            result = TRUE;
        }
    } else {
        if (input >= upper || sample_step % (ticks_count_t) round(upper / (upper - d_input)) != 0) {
            result = TRUE;
        }
    }

    return result;
}