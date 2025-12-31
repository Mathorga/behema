# API Reference

All functions return `bhm_error_t` telling the outcome of the function call.

## Data Types

  * `bhm_byte_t` (`uint8_t`)

  * `bhm_neuron_value_t` (`int16_t`)

  * `bhm_nh_mask_t` (`uint64_t`);

    A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3). Using 16 bytes the radius can be up to 5 (120 neighbors).

  * `bhm_nh_radius_t` (`int8_t`)

  * `bhm_syn_count_t` (`uint8_t`)

  * `bhm_syn_strength_t` (`uint8_t`)

  * `bhm_ticks_count_t` (`uint16_t`)

  * `bhm_evol_step_t` (`uint32_t`)

  * `bhm_pulse_mask_t` (`uint64_t`)

  * `bhm_chance_t` (`uint32_t`)

  * `bhm_rand_state_t` (`uint32_t`)

  * `bhm_cortex_size_t` (`int32_t`)

## Cortex2D

The Cortex2D (`bhm_cortex_2d`) is simply a 2D grid of neurons. If you're familiar with cellular automata, the concept will be fairly easy to grasp, but, much like in a chess board, each neuron in the grid has neighbors.

### Members

  * width (`bhm_cortex_size_t`)

    Width of the cortex of neurons.

  * height (`bhm_cortex_size_t`)

    Height of the cortex of neurons.

  * ticks_count (`bhm_ticks_count_t`)

    Ticks performed since cortex creation.

  * evols_count (`bhm_ticks_count_t`)

  * evol_step (`bhm_ticks_count_t`)

  * pulse_window (`bhm_ticks_count_t`)

  * nh_radius (`bhm_nh_radius_t`)

  * fire_threshold (`bhm_neuron_value_t`)

  * recovery_value (`bhm_neuron_value_t`)

  * exc_value (`bhm_neuron_value_t`)

  * decay_value (`bhm_neuron_value_t`)

  * rand_state (`bhm_rand_state_t`)

    The random state is used to generate consistent random numbers across the lifespan of a cortex, therefore should NEVER be manually changed.

    Embedding the rand state allows for completely deterministic and reproducible results.

  * syngen_chance (`bhm_chance_t`)

    Chance (out of 0xFFFFU) of synapse generation or deletion (structural plasticity).

  * synstr_chance (`bhm_chance_t`)

    Chance (out of 0xFFFFU) of synapse strengthening or weakening (functional plasticity).

  * max_tot_strength (`bhm_syn_strength_t`)

    Max strength available for a single neuron, meaning the strength of all the synapses coming to each neuron cannot be more than this.

  * max_syn_count (`bhm_syn_count_t`)

    Maximum number of synapses between a neuron and its neighbors.

  * inhexc_range (`bhm_chance_t`)

    Maximum range for inhexc chance: single neurons' inhexc ratio will vary between 0 and inhexc_range. 0 means all excitatory, inhexc_range means all inhibitory.

  * sample_window (`bhm_ticks_count_t`)

    Length of the window used to sample inputs.

  * pulse_mapping (`bhm_pulse_mapping_t`)

  * neurons (`bhm_neuron_t*`)

    Array of neurons. Neurons are stored in a contiguous array in heap space.

## Input2D

## Output2D

## Neuron