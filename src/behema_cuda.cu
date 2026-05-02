#include "behema_cuda.h"

// The state must be initialized to non-zero.
__host__ __device__ uint32_t cuda_xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// ########################################## Initialization functions ##########################################

dim3 c2d_get_grid_size(bhm_cortex2d_t* cortex, dim3 block_size) {
    // Cortex size may not be exactly divisible by BLOCK_SIZE, so an extra block is allocated when needed.
    return dim3(cortex->width / block_size.x + (cortex->width % block_size.x != 0 ? 1 : 0), cortex->height / block_size.y + (cortex->height % block_size.y ? 1 : 0));
}

dim3 c2d_get_block_size(bhm_cortex2d_t* cortex) {
    return dim3(BLOCK_SIZE_2D, BLOCK_SIZE_2D);
}

size_t c2d_get_shared_mem_size(bhm_cortex2d_t* cortex, dim3 block_size) {
    // Shared memory size is computed with extra space around it to act as ghost cells.
    // This ensures the neighborhood for every thread in the block is stored in shared memory.
    return (block_size.x + (2 * cortex->nh_radius)) * (block_size.y + (2 * cortex->nh_radius)) * sizeof(bhm_neuron_t);
}

bhm_error_code_t i2d_to_device(bhm_input2d_t* device_input, bhm_input2d_t* host_input) {
    cudaError_t cuda_error;

    // Allocate tmp input on the host.
    bhm_input2d_t* tmp_input = (bhm_input2d_t*) malloc(sizeof(bhm_input2d_t));
    if (tmp_input == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Copy host input to tmp input.
    (*tmp_input) = (*host_input);

    // Allocate values on the device.
    cuda_error = cudaMalloc((void**) &(tmp_input->values), (host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0) * sizeof(bhm_ticks_count_t));
    cudaCheckError();
    if (cuda_error != cudaSuccess) return BHM_ERROR_FAILED_ALLOC;

    // Copy values to device.
    cudaMemcpy(
        tmp_input->values,
        host_input->values,
        ((host_input->x1 - host_input->x0) * (host_input->y1 - host_input->y0)) * sizeof(bhm_ticks_count_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Copy tmp input to device.
    cudaMemcpy(
        device_input,
        tmp_input,
        sizeof(bhm_input2d_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Cleanup.
    free(tmp_input);

    return BHM_ERROR_NONE;
}

bhm_error_code_t i2d_to_host(bhm_input2d_t* host_input, bhm_input2d_t* device_input) {
    // TODO
    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_to_device(
    bhm_cortex2d_t* device_cortex,
    bhm_cortex2d_t* host_cortex
) {
    cudaError_t cuda_error;

    // Allocate tmp cortex on the host.
    bhm_cortex2d_t* tmp_cortex = (bhm_cortex2d_t*) malloc(sizeof(bhm_cortex2d_t));
    if (tmp_cortex == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Copy host cortex to tmp cortex.
    (*tmp_cortex) = (*host_cortex);

    // Allocate neurons on the device.
    cuda_error = cudaMalloc((void**) &(tmp_cortex->neurons), host_cortex->width * host_cortex->height * sizeof(bhm_neuron_t));
    cudaCheckError();
    if (cuda_error != cudaSuccess) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Copy neurons to device.
    cudaMemcpy(
        tmp_cortex->neurons,
        host_cortex->neurons,
        host_cortex->width * host_cortex->height * sizeof(bhm_neuron_t),
        cudaMemcpyHostToDevice
    );
    cudaCheckError();

    // Copy tmp cortex to device.
    cudaMemcpy(device_cortex, tmp_cortex, sizeof(bhm_cortex2d_t), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Cleanup.
    free(tmp_cortex);

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_to_host(
    bhm_cortex2d_t* host_cortex,
    bhm_cortex2d_t* device_cortex
) {
    // Allocate tmp cortex on the host.
    bhm_cortex2d_t* tmp_cortex = (bhm_cortex2d_t*) malloc(sizeof(bhm_cortex2d_t));
    if (tmp_cortex == NULL) return BHM_ERROR_FAILED_ALLOC;

    // Copy tmp cortex to host.
    cudaMemcpy(tmp_cortex, device_cortex, sizeof(bhm_cortex2d_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Copy tmp cortex to host cortex.
    (*host_cortex) = (*tmp_cortex);

    // Allocate neurons on the host.
    host_cortex->neurons = (bhm_neuron_t*) malloc(tmp_cortex->width * tmp_cortex->height * sizeof(bhm_neuron_t));

    // Copy tmp cortex neurons (still on device) to host cortex.
    cudaMemcpy(host_cortex->neurons, tmp_cortex->neurons, tmp_cortex->width * tmp_cortex->height * sizeof(bhm_neuron_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Cleanup.
    free(tmp_cortex);

    return BHM_ERROR_NONE;
}

bhm_error_code_t i2d_device_destroy(bhm_input2d_t* input) {
    // TODO
    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_device_destroy(bhm_cortex2d_t* cortex) {
    // Allocate tmp cortex on the host.
    bhm_cortex2d_t* tmp_cortex = (bhm_cortex2d_t*) malloc(sizeof(bhm_cortex2d_t));
    if (tmp_cortex == NULL) return BHM_ERROR_FAILED_ALLOC;
    
    // Copy device cortex to host in order to free its neurons.
    cudaMemcpy(tmp_cortex, cortex, sizeof(bhm_cortex2d_t), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Free device neurons.
    cudaFree(tmp_cortex->neurons);
    cudaCheckError();

    // Free tmp cortex.
    free(tmp_cortex);

    // Finally free device cortex.
    cudaFree(cortex);
    cudaCheckError();

    return BHM_ERROR_NONE;
}


// ########################################## Execution functions ##########################################

__global__ void c2d_feed2d(
    bhm_cortex2d_t* cortex,
    bhm_input2d_t* input,
    bhm_ticks_count_t ticks_count
) {
    bhm_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    bhm_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= input->x1 - input->x0 || y >= input->y1 - input->y0) {
        return;
    }

    // Check whether the current input neuron should be excited or not.
    bhm_bool_t excite = value_to_pulse(
        cortex->sample_window,
        ticks_count % cortex->sample_window,
        input->values[
            IDX2D(
                x,
                y,
                input->x1 - input->x0
            )
        ],
        cortex->pulse_mapping
    );

    if (excite) {
        cortex->neurons[IDX2D(x + input->x0, y + input->y0, cortex->width)].value += input->exc_value;
    }
}

__global__ void c2d_read2d(bhm_cortex2d_t* cortex, bhm_output2d_t* output) {
    bhm_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    bhm_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Avoid accessing unallocated memory.
    if (x >= output->x1 - output->x0 || y >= output->y1 - output->y0) {
        return;
    }

    // TODO.
}

__global__ void c2d_tick(
    bhm_cortex2d_t* prev_cortex,
    bhm_cortex2d_t* next_cortex,
    bool evolve
) {
    bhm_cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    bhm_cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

    // Fetch cortex size once.
    bhm_cortex_size_t cortex_width = prev_cortex->width;
    bhm_cortex_size_t cortex_height = prev_cortex->height;

    // Avoid accessing unallocated memory.
    if (x >= cortex_width || y >= cortex_height) return;

    // Retrieve the involved neurons.
    bhm_cortex_size_t neuron_index = IDX2D(x, y, cortex_width);

    // Compute the neighborhood diameter:
    // d = 7
    // <------------->
    // r = 3
    // <----->
    // +-|-|-|-|-|-|-+
    // |             |
    // |             |
    // |      X      |
    // |             |
    // |             |
    // +-|-|-|-|-|-|-+
    bhm_cortex_size_t nh_radius = prev_cortex->nh_radius;
    bhm_cortex_size_t nh_diameter = NH_DIAM_2D(nh_radius);

    // bhm_neuron_t prev_neuron = local_neurons[local_index];
    bhm_neuron_t prev_neuron = prev_cortex->neurons[neuron_index];
    bhm_neuron_t* next_neuron = &(next_cortex->neurons[neuron_index]);

    bhm_nh_mask_t prev_ac_mask = prev_neuron.synac_mask;
    bhm_nh_mask_t prev_exc_mask = prev_neuron.synex_mask;
    bhm_nh_mask_t prev_str_mask_a = prev_neuron.synstr_mask_a;
    bhm_nh_mask_t prev_str_mask_b = prev_neuron.synstr_mask_b;
    bhm_nh_mask_t prev_str_mask_c = prev_neuron.synstr_mask_c;

    // Copy the current neuron's rand state from global memory to thread-local.
    bhm_rand_state_t rand_state = prev_neuron.rand_state;

    // Copy prev neuron values to the new one.
    // *next_neuron = prev_neuron;
    next_neuron->value = prev_neuron.value;
    next_neuron->pulse = prev_neuron.pulse;
    next_neuron->pulse_mask = prev_neuron.pulse_mask;
    next_neuron->rand_state = prev_neuron.rand_state;

    if (evolve) {
        next_neuron->synac_mask = prev_ac_mask;
        next_neuron->synex_mask = prev_exc_mask;
        next_neuron->synstr_mask_a = prev_str_mask_a;
        next_neuron->synstr_mask_b = prev_str_mask_b;
        next_neuron->synstr_mask_c = prev_str_mask_c;
    }

    // Increment the current neuron value by reading its connected neighbors.
    for (bhm_nh_radius_t j = 0; j < nh_diameter; j++) {
        for (bhm_nh_radius_t i = 0; i < nh_diameter; i++) {
            bhm_cortex_size_t neighbor_x = x + (i - nh_radius);
            bhm_cortex_size_t neighbor_y = y + (j - nh_radius);

            // bhm_cortex_size_t neighbor_shared_mem_x = shared_mem_x + (i - nh_radius);
            // bhm_cortex_size_t neighbor_shared_mem_y = shared_mem_y + (j - nh_radius);

            // Exclude the central neuron from the list of neighbors.
            if ((j != nh_radius || i != nh_radius) &&
                (neighbor_x >= 0 && neighbor_y >= 0 && neighbor_x < cortex_width && neighbor_y < cortex_height)) {
                // The index of the current neighbor in the current neuron's neighborhood.
                bhm_cortex_size_t neighbor_nh_index = IDX2D(i, j, nh_diameter);
                bhm_cortex_size_t neighbor_index = IDX2D(
                    WRAP(neighbor_x, cortex_width),
                    WRAP(neighbor_y, cortex_height),
                    cortex_width
                );

                // Fetch the current neighbor.
                bhm_neuron_t neighbor = prev_cortex->neurons[neighbor_index];
                // bhm_neuron_t neighbor = local_neurons[IDX2D(neighbor_shared_mem_x, neighbor_shared_mem_y, shared_mem_width)];

                // Compute the current synapse strength.
                bhm_syn_strength_t syn_strength = (
                    (prev_str_mask_a & 0x01U) |
                    ((prev_str_mask_b & 0x01U) << 0x01U) |
                    ((prev_str_mask_c & 0x01U) << 0x02U)
                );

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
                    rand_state = cuda_xorshf32(rand_state);
                    bhm_chance_t random = rand_state % 0xFFFFU;

                    // Structural plasticity: create or destroy a synapse.
                    if (!(prev_ac_mask & 0x01U) &&
                        prev_neuron.syn_count < next_neuron->max_syn_count &&
                        // Frequency component.
                        // TODO Make sure there's no overflow.
                        random < prev_cortex->syngen_chance * (bhm_chance_t) neighbor.pulse) {
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
                        if (syn_strength < BHM_MAX_SYN_STRENGTH &&
                            prev_neuron.tot_syn_strength < prev_cortex->max_tot_strength &&
                            // TODO Make sure there's no overflow.
                            random < prev_cortex->synstr_chance * (bhm_chance_t) neighbor.pulse * (bhm_chance_t) strength_diff) {
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

    // Copy the current neuron's rand state from thread-local memory to global.
    next_neuron->rand_state = rand_state;

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

__host__ __device__ bhm_bool_t value_to_pulse(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input, bhm_pulse_mapping_t pulse_mapping) {
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
            default:
                break;
        }
    }

    return result;
}

__host__ __device__ bhm_bool_t value_to_pulse_linear(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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
    return (sample_step % (sample_window - input) == 0) ? BHM_TRUE : BHM_FALSE;
}

__host__ __device__ bhm_bool_t value_to_pulse_fprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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
            (sample_step % (upper / input) == 0)) {
            result = BHM_TRUE;
        }
    } else {
        if (input >= upper || sample_step % (upper / (upper - input)) != 0) {
            result = BHM_TRUE;
        }
    }

    return result;
}

__host__ __device__ bhm_bool_t value_to_pulse_rprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input) {
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
            sample_step % (bhm_ticks_count_t) round(upper / d_input) == 0) {
            result = BHM_TRUE;
        }
    } else {
        if (input >= upper || sample_step % (bhm_ticks_count_t) round(upper / (upper - d_input)) != 0) {
            result = BHM_TRUE;
        }
    }

    return result;
}