#include "portia_cuda.h"

// The state word must be initialized to non-zero.
__host__ __device__ uint32_t xorshf32(uint32_t state) {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}


error_code_t i2d_init(input2d_t* input, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t exc_value, pulse_mapping_t pulse_mapping) {
    // TODO.
    return ERROR_NONE;
}

error_code_t c2d_init(cortex2d_t** cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    if (NH_COUNT_2D(NH_DIAM_2D(nh_radius)) > sizeof(nh_mask_t) * 8) {
        // The provided radius makes for too many neighbors, which will end up in overflows, resulting in unexpected behavior during syngen.
        return ERROR_NH_RADIUS_TOO_BIG;
    }

    cudaError_t error;

    // Allocate the cortex
    error = cudaMalloc(cortex, sizeof(cortex2d_t));
    if (error != cudaSuccess) {
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
    error = cudaMalloc(&(*cortex)->neurons, (*cortex)->width * (*cortex)->height * sizeof(neuron_t));
    if (error != cudaSuccess) {
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

error_code_t c2d_to_device(cortex2d_t* host_cortex, cortex2d_t* device_cortex) {
    // TODO.
    return ERROR_NONE;
}

error_code_t c2d_to_host(cortex2d_t* device_cortex, cortex2d_t* host_cortex) {
    // TODO.
    return ERROR_NONE;
}


__global__ void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex) {
    cortex_size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    cortex_size_t y = threadIdx.y + blockIdx.y * blockDim.y;

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