# API Reference

All functions return `bhm_error_t` telling the outcome of the function call.

## Data Type Aliases

  * `bhm_byte_t` [`uint8_t`]

  * `bhm_neuron_value_t` [`int16_t`]

  * `bhm_nh_mask_t` [`uint64_t`];

    A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3). Using 16 bytes the radius can be up to 5 (120 neighbors).

  * `bhm_nh_radius_t` [`int8_t`]

  * `bhm_syn_count_t` [`uint8_t`]

  * `bhm_syn_strength_t` [`uint8_t`]

  * `bhm_ticks_count_t` [`uint16_t`]

  * `bhm_evol_step_t` [`uint32_t`]

  * `bhm_pulse_mask_t` [`uint64_t`]

  * `bhm_chance_t` [`uint32_t`]

  * `bhm_rand_state_t` [`uint32_t`]

  * `bhm_cortex_size_t` [`int32_t`]

## Cortex2D

The Cortex2D (`bhm_cortex_2d`) is a 2D grid of neurons. If you're familiar with cellular automata, the concept will be fairly easy to grasp, but, much like in a chess board, each neuron in the grid has neighbors.

### Members

  * width [`bhm_cortex_size_t`]

    Width of the cortex of neurons.

  * height [`bhm_cortex_size_t`]

    Height of the cortex of neurons.

  * ticks_count [`bhm_ticks_count_t`]

    Ticks performed since cortex creation.

  * evols_count [`bhm_ticks_count_t`]

  * evol_step [`bhm_ticks_count_t`]

  * pulse_window [`bhm_ticks_count_t`]

  * nh_radius [`bhm_nh_radius_t`]

  * fire_threshold [`bhm_neuron_value_t`]

  * recovery_value [`bhm_neuron_value_t`]

  * exc_value [`bhm_neuron_value_t`]

  * decay_value [`bhm_neuron_value_t`]

  * rand_state [`bhm_rand_state_t`]

    The random state is used to generate consistent random numbers across the lifespan of a cortex, therefore should NEVER be manually changed.

    Embedding the rand state allows for completely deterministic and reproducible results.

  * syngen_chance [`bhm_chance_t`]

    Chance  out of 0xFFFFU) of synapse generation or deletion (structural plasticity).

  * synstr_chance [`bhm_chance_t`]

    Chance (out of 0xFFFFU) of synapse strengthening or weakening (functional plasticity).

  * max_tot_strength [`bhm_syn_strength_t`]

    Max strength available for a single neuron, meaning the strength of all the synapses coming to each neuron cannot be more than this.

  * max_syn_count [`bhm_syn_count_t`]

    Maximum number of synapses between a neuron and its neighbors.

  * inhexc_range [`bhm_chance_t`]

    Maximum range for inhexc chance: single neurons' inhexc ratio will vary between 0 and inhexc_range. 0 means all excitatory, inhexc_range means all inhibitory.

  * sample_window [`bhm_ticks_count_t`]

    Length of the window used to sample inputs.

  * pulse_mapping [`bhm_pulse_mapping_t`]

  * neurons [`bhm_neuron_t*`]

    Array of neurons. Neurons are stored in a contiguous array in heap space.

### Common Functions

  #### c2d_init

  Initializes the provided cortex' internal components with default values.

  * Returns `bhm_error_code_t`

    The code for the occurred error, [BHM_ERROR_NONE] if none.

  * Params

    * cortex [`bhm_cortex2d_t*`]

      The cortex to initialize. Should be a valid pointer, not NULL.

    * width [`bhm_cortex_size_t`]

      The cortex width.

    * height [`bhm_cortex_size_t`]

      The cortex height.

    * nh_radius [`bhm_nh_radius_t`]

      The neighborhood radius for each individual cortex neuron.

  * Example
    ```
    // Define a new cortex pointer.
    bhm_cortex2d_t* cortex;

    bhm_error_code_t error;

    // Allocate the cortex.
    error = c2d_alloc(&cortex);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }

    // Initialize its values.
    error = c2d_init(cortex, 512, 256, 2);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }
    ```

  #### c2d_rand_init

  Initializes the provided cortex' internal components with random values.

  * Returns `bhm_error_code_t`

    The code for the occurred error, [BHM_ERROR_NONE] if none.

  * Params

    * cortex [`bhm_cortex2d_t*`]

      The cortex to initialize. Should be a valid pointer, not NULL.

    * width [`bhm_cortex_size_t`]

      The cortex width.

    * height [`bhm_cortex_size_t`]

      The cortex height.

    * nh_radius [`bhm_nh_radius_t`]

      The neighborhood radius for each individual cortex neuron.

  * Example
    ```
    // Define a new cortex pointer.
    bhm_cortex2d_t* cortex;

    bhm_error_code_t error;

    // Allocate the cortex.
    error = c2d_alloc(&cortex);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }

    // Initialize its values.
    error = c2d_rand_init(cortex, 512, 256, 2);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }
    ```

  #### c2d_create

  Allocates and initializes a new cortex with default values.

  Works as a shorthand for `c2d_alloc` + `c2d_init`.

  * Returns `bhm_error_code_t`

    The code for the occurred error, [BHM_ERROR_NONE] if none.

  * Params

    * cortex [`bhm_cortex2d_t**`]

      The pointer to the allocate the cortex to. Should be a valid pointer, not NULL.

    * width [`bhm_cortex_size_t`]

      The cortex width.

    * height [`bhm_cortex_size_t`]

      The cortex height.

    * nh_radius [`bhm_nh_radius_t`]

      The neighborhood radius for each individual cortex neuron.

  * Example
    ```
    // Define a new cortex pointer.
    bhm_cortex2d_t* cortex;

    bhm_error_code_t error;

    // Allocate the cortex.
    error = c2d_create(&cortex, 512, 256, 2);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }
    ```

  #### c2d_destroy

  Destroys the given cortex2d and frees memory for it and its neurons.

  * Returns `bhm_error_code_t`

    The code for the occurred error, [BHM_ERROR_NONE] if none.

  * Params

    * cortex [`bhm_cortex2d_t*`]

      The cortex to destroy. Should be a valid pointer, not NULL.

  * Example
    ```
    // Define a new cortex pointer.
    bhm_cortex2d_t* cortex;

    bhm_error_code_t error;

    // Allocate the cortex.
    error = c2d_create(&even_cortex, 512, 256, 2);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }

    // [...] Work with the cortex. [...]

    // Destroy the cortex when not needed anymore.
    error = c2d_destroy(cortex);
    if (error != BHM_ERROR_NONE) {
        printf("An error occurred!\n");
        return 1;
    }
    ```

/// @brief Returns a cortex with the same properties as the given one.
/// @param to The destination cortex.
/// @param from The source cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_copy(
    bhm_cortex2d_t* to,
    bhm_cortex2d_t* from
);

/// @brief Sets the neighborhood radius for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_nhradius(
    bhm_cortex2d_t* cortex,
    bhm_nh_radius_t radius
);

/// @brief Sets the neighborhood mask for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_nhmask(
    bhm_cortex2d_t* cortex,
    bhm_nh_mask_t mask
);

/// @brief Sets the evolution step for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_evol_step(
    bhm_cortex2d_t* cortex,
    bhm_evol_step_t evol_step
);

/// @brief Sets the pulse window width for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_pulse_window(
    bhm_cortex2d_t* cortex,
    bhm_ticks_count_t window
);

/// @brief Sets the sample window for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_sample_window(
    bhm_cortex2d_t* cortex,
    bhm_ticks_count_t sample_window
);

/// @brief Sets the fire threshold for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_fire_threshold(
    bhm_cortex2d_t* cortex,
    bhm_neuron_value_t threshold
);

/// @brief Sets the syngen chance for the cortex. Syngen chance defines the probability for synapse generation and deletion.
/// @param syngen_chance The chance to apply (must be between 0x0000U and 0xFFFFU).
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_syngen_chance(
    bhm_cortex2d_t* cortex,
    bhm_chance_t syngen_chance
);

/// @brief Sets the synstr chance for the cortex. Synstr chance defines the probability for synapse strengthening and weakening.
/// @param synstr_chance The chance to apply (must be between 0x0000U and 0xFFFFU).
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_synstr_chance(
    bhm_cortex2d_t* cortex,
    bhm_chance_t synstr_chance
);

/// @brief Sets the maximum number of (input) synapses for the neurons of the cortex.
/// @param cortex The cortex to edit.
/// @param syn_count The max number of allowable synapses.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_max_syn_count(
    bhm_cortex2d_t* cortex,
    bhm_syn_count_t syn_count
);

/// @brief Sets the maximum allowable touch for each neuron in the network.
/// A neuron touch is defined as its synapses count divided by its total neighbors count.
/// @param touch The touch to assign the cortex. Only values between 0 and 1 are allowed.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_max_touch(
    bhm_cortex2d_t* cortex,
    float touch
);

/// @brief Sets the preferred input mapping for the given cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_pulse_mapping(
    bhm_cortex2d_t* cortex,
    bhm_pulse_mapping_t pulse_mapping
);

/// @brief Sets the range for excitatory to inhibitory ratios in single neurons.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_inhexc_range(
    bhm_cortex2d_t* cortex,
    bhm_chance_t inhexc_range
);

/// @brief Sets the proportion between excitatory and inhibitory generated synapses.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_inhexc_ratio(
    bhm_cortex2d_t* cortex,
    bhm_chance_t inhexc_ratio
);

/// @brief Disables self connections whithin the specified bounds.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_syn_disable(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t x0,
    bhm_cortex_size_t y0,
    bhm_cortex_size_t x1,
    bhm_cortex_size_t y1
);

/// @brief Randomly mutates the cortex shape.
/// @param cortex The cortex to edit.
/// @param mut_chance The probability of applying a mutation to the cortex shape.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_mutate_shape(
    bhm_cortex2d_t* cortex,
    bhm_chance_t mut_chance
);

/// @brief Randomly mutates the cortex.
/// @param cortex The cortex to edit.
/// @param mut_chance The probability of applying a mutation to any mutable property of the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_mutate(
    bhm_cortex2d_t* cortex,
    bhm_chance_t mut_chance
);

/// @brief Stores the string representation of the given cortex to the provided string [result].
/// @param cortex The cortex to inspect.
/// @param result The string to fill with cortex data.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_to_string(
    bhm_cortex2d_t* cortex,
    char* result
);

/// @brief Reads and returns the spiking state of the neurons in the cortex (whether each neuron is spiking or not).
/// @param cortex The cortex to read from.
/// @param result The array in which to store the spiking state to. The result is a flattened 2D array the size of [cortex].
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_get_spiking_state(
    bhm_cortex2d_t* cortex,
    bhm_bool_t* result
);

/// @brief Reads and returns the number of output synapses from each neurons in the provided cortex.
/// @param cortex The cortex to read from.
/// @param result The array in which to store the synapses count state to. The result is a flattened 2d array the size of [cortex].
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_get_synout_state(
    bhm_cortex2d_t* cortex,
    bhm_syn_count_t* result
);

/// @brief Adds a row of neurons at the provided index.
/// @param cortex The cortex to add a row to.
/// @param index The index at which to add the new row of neurons.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_add_row(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t index
);

/// @brief Adds a column of neurons at the provided index.
/// @param cortex The cortex to add a column to.
/// @param index The index at which to add the new column of neurons.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_add_column(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t index
);

/// @brief Removes the row of neurons at the provided index.
/// @param cortex The cortex to remove a row from.
/// @param index The index of the row to remove.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_remove_row(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t index
);

/// @brief Removes the column of neurons at the provided index.
/// @param cortex The cortex to remove a column from.
/// @param index The index of the column to remove.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_remove_column(
    bhm_cortex2d_t* cortex,
    bhm_cortex_size_t index
);

/// @brief Transposes the provided cortex' neurons. Width and height are switched as well.
/// @param cortex The cortex whose neurons to transpose.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_transpose(
    bhm_cortex2d_t* cortex
);

### Implementation-specific Functions

  #### CPU

  Refer to [STD API Reference](/STD_API_REF.md) for more details.

  #### GPU (CUDA)

  Refer to [CUDA API Reference](/CUDA_API_REF.md) for more details.

## Input2D

## Output2D

## Neuron