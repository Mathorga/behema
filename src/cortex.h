/*
*****************************************************************
cortex.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __CORTEX__
#define __CORTEX__

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Translate an id wrapping it to the provided size (pacman effect).
// WARNING: Only works with signed types and does not show errors otherwise.
// [i] is the given index.
// [n] is the size over which to wrap.
#define WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

// Computes the diameter of a square neighborhood given its radius.
#define NH_DIAM_2D(r) (2 * (r) + 1)

// Computes the number of neighbors in a square neighborhood given its diameter.
#define NH_COUNT_2D(d) ((d) * (d) - 1)

// Translates bidimensional indexes to a monodimensional one.
// |i| is the row index.
// |j| is the column index.
// |m| is the number of columns (length of the rows).
#define IDX2D(i, j, m) (((m) * (j)) + (i))

// Translates tridimensional indexes to a monodimensional one.
// |i| is the index in the first dimension.
// |j| is the index in the second dimension.
// |k| is the index in the third dimension.
// |m| is the size of the first dimension.
// |n| is the size of the second dimension.
#define IDX3D(i, j, k, m, n) (((m) * (n) * (k)) + ((m) * (j)) + (i))

#define BHM_EVOL_STEP_NEVER 0x0000FFFFU

#define BHM_PULSE_WINDOW_LARGE 0x3FU
#define BHM_PULSE_WINDOW_MID 0x1FU
#define BHM_PULSE_WINDOW_SMALL 0x0AU

#define BHM_SAMPLE_WINDOW_LARGE 0x40U
#define BHM_SAMPLE_WINDOW_MID 0x20U
#define BHM_SAMPLE_WINDOW_SMALL 0x10U

#define BHM_MAX_SYN_STRENGTH 0x07U
#define BHM_MAX_CHANCE 0xFFFFU

// Completely arbitrary values used to define a sort of acceptable cortex right away.
#define BHM_DEFAULT_THRESHOLD 0x88U
#define BHM_DEFAULT_STARTING_VALUE 0x00U
#define BHM_DEFAULT_RECOVERY_VALUE -0x2A
#define BHM_DEFAULT_MAX_TOUCH 0.25F
#define BHM_DEFAULT_EXC_VALUE 0x20U
#define BHM_DEFAULT_DECAY_RATE 0x01U
#define BHM_DEFAULT_PULSE_WINDOW BHM_PULSE_WINDOW_LARGE
#define BHM_DEFAULT_EVOL_STEP 0x0000000AU
#define BHM_DEFAULT_INHEXC_RANGE 0x64U
#define BHM_DEFAULT_INHEXC_RATIO 0x06U
#define BHM_DEFAULT_SAMPLE_WINDOW BHM_SAMPLE_WINDOW_SMALL
#define BHM_DEFAULT_MAX_TOT_STRENGTH 0x20U
#define BHM_DEFAULT_SYNGEN_CHANCE 0x02A0U
#define BHM_DEFAULT_SYNSTR_CHANCE 0x00A0U

#define BHM_MAX_SYNGEN_CHANCE 0xFFFFU
#define BHM_MAX_SYNSTR_CHANCE 0xFFFFU

typedef uint8_t bhm_byte;

typedef int16_t bhm_neuron_value_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t bhm_nh_mask_t;
typedef int8_t bhm_nh_radius_t;
typedef uint8_t bhm_syn_count_t;
typedef uint8_t bhm_syn_strength_t;
typedef uint16_t bhm_ticks_count_t;
typedef uint32_t bhm_evol_step_t;
typedef uint64_t bhm_pulse_mask_t;
typedef uint32_t bhm_chance_t;
typedef uint32_t bhm_rand_state_t;

typedef int32_t bhm_cortex_size_t;

typedef enum {
    BHM_FALSE = 0,
    BHM_TRUE = 1
} bhm_bool_t;

typedef enum {
    // Values are forced to 32 bit integers by using big enough values: 100000 is 17 bits long, so 32 bits are automatically allocated.
    // Linear.
    BHM_PULSE_MAPPING_LINEAR = 0x100000,
    // Floored proportional.
    BHM_PULSE_MAPPING_FPROP = 0x100001,
    // Rounded proportional.
    BHM_PULSE_MAPPING_RPROP = 0x100002,
    // Double floored proportional.
    BHM_PULSE_MAPPING_DFPROP = 0x100003,
} bhm_pulse_mapping_t;

/// @brief Convenience data structure for input handling (cortex feeding).
typedef struct {
    bhm_cortex_size_t x0;
    bhm_cortex_size_t y0;
    bhm_cortex_size_t x1;
    bhm_cortex_size_t y1;

    // Value used to excite the target neurons.
    bhm_neuron_value_t exc_value;

    // Values to be mapped to pulse (input values).
    bhm_ticks_count_t* values;
} bhm_input2d_t;

/// @brief Convenience data structure for output handling (cortex reading).
typedef struct {
    bhm_cortex_size_t x0;
    bhm_cortex_size_t y0;
    bhm_cortex_size_t x1;
    bhm_cortex_size_t y1;

    // Values mapped from pulse (output values).
    bhm_ticks_count_t* values;
} bhm_output2d_t;

/// @brief Neuron definition data structure.
typedef struct {
    // Neighborhood connections pattern (SYNapses ACtivation state):
    // 1|1|0
    // 0|x|1 => 1100x1100
    // 1|0|0
    bhm_nh_mask_t synac_mask;
    // Neighborhood excitatory states pattern (SYNapses EXcitatory state), defines whether the synapses from the neighbors are excitatory (1) or inhibitory (0).
    // Only values corresponding to active synapses are used.
    bhm_nh_mask_t synex_mask;
    // Neighborhood synapses strength pattern (SYNapses STRength). Defines a 3 bit value defined as [cba].
    bhm_nh_mask_t synstr_mask_a;
    bhm_nh_mask_t synstr_mask_b;
    bhm_nh_mask_t synstr_mask_c;


    // Random state. The random state has to be consistent inside a single neuron in order to allow for parallel edits without any race condition.
    // The random state is used to generate consistent random numbers across the lifespan of a neuron, therefore should NEVER be manually changed.
    bhm_rand_state_t rand_state;


    // Activation history pattern:
    //           |<--pulse_window-->|
    // xxxxxxxxxx01001010001010001001--------> t
    //                              ^
    // Used to know the pulse frequency at a given moment (e.g. for syngen).
    bhm_pulse_mask_t pulse_mask;
    // Amount of activations in the cortex' pulse window.
    bhm_ticks_count_t pulse;


    // Current internal value.
    bhm_neuron_value_t value;
    // Maximum number of synapses to the neuron. Cannot be greater than the cortex' max_syn_count.
    //* Mutable.
    bhm_syn_count_t max_syn_count;
    // Amount of connected neighbors.
    bhm_syn_count_t syn_count;
    // Total amount of syn strength from input neurons.
    bhm_syn_strength_t tot_syn_strength;
    // Proportion between excitatory and inhibitory generated synapses. Can vary between 0 and cortex.inhexc_range.
    // inhexc_ratio = 0 -> all synapses are excitatory.
    // inhexc_ratio = cortex.inhexc_range -> all synapses are inhibitory.
    //* Mutable.
    bhm_chance_t inhexc_ratio;
} bhm_neuron_t;

/// @brief 2D cortex of neurons.
typedef struct {
    // Width of the cortex.
    //* Mutable.
    bhm_cortex_size_t width;
    // Height of the cortex.
    //* Mutable.
    bhm_cortex_size_t height;
    // Ticks performed since cortex creation.
    bhm_ticks_count_t ticks_count;
    // Evolutions performed since cortex creation.
    bhm_ticks_count_t evols_count;
    // Amount of ticks between each evolution.
    bhm_ticks_count_t evol_step;
    // Length of the window used to count pulses in the cortex' neurons.
    //* Mutable.
    bhm_ticks_count_t pulse_window;


    // Radius of each neuron's neighborhood.
    bhm_nh_radius_t nh_radius;
    bhm_neuron_value_t fire_threshold;
    bhm_neuron_value_t recovery_value;
    bhm_neuron_value_t exc_value;
    bhm_neuron_value_t decay_value;


    // Random state.
    // The random state is used to generate consistent random numbers across the lifespan of a cortex, therefore should NEVER be manually changed.
    // Embedding the rand state allows for completely deterministic and reproducible results.
    bhm_rand_state_t rand_state;


    // Chance (out of 0xFFFFU) of synapse generation or deletion (structural plasticity).
    //* Mutable.
    bhm_chance_t syngen_chance;
    // Chance (out of 0xFFFFU) of synapse strengthening or weakening (functional plasticity).
    //* Mutable.
    bhm_chance_t synstr_chance;


    // Max strength available for a single neuron, meaning the strength of all the synapses coming to each neuron cannot be more than this.
    bhm_syn_strength_t max_tot_strength;
    // Maximum number of synapses between a neuron and its neighbors.
    bhm_syn_count_t max_syn_count;
    // Maximum range for inhexc chance: single neurons' inhexc ratio will vary between 0 and inhexc_range. 0 means all excitatory, inhexc_range means all inhibitory.
    bhm_chance_t inhexc_range;


    // Length of the window used to sample inputs.
    bhm_ticks_count_t sample_window;
    bhm_pulse_mapping_t pulse_mapping;

    bhm_neuron_t* neurons;
} bhm_cortex2d_t;

/// @brief 3D cortex of neurons.
typedef struct {
    // Width of the cortex.
    bhm_cortex_size_t width;
    // Height of the cortex.
    bhm_cortex_size_t height;
    // Depth of the cortex.
    bhm_cortex_size_t depth;

    // TODO Other data.

    bhm_neuron_t* neurons;
} bhm_cortex3d_t;


/// Marsiglia's xorshift pseudo-random number generator with period 2^32-1.
uint32_t xorshf32(uint32_t state);


// ########################################## Initialization functions ##########################################

/// @brief Initializes an input2d with the given values.
/// @param input 
/// @param x0 
/// @param y0 
/// @param x1 
/// @param y1 
/// @param exc_value 
/// @param pulse_mapping 
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t i2d_init(bhm_input2d_t** input, bhm_cortex_size_t x0, bhm_cortex_size_t y0, bhm_cortex_size_t x1, bhm_cortex_size_t y1, bhm_neuron_value_t exc_value, bhm_pulse_mapping_t pulse_mapping);

/// @brief Initializes an output2d with the provided values.
/// @param output 
/// @param x0 
/// @param y0 
/// @param x1 
/// @param y1 
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t o2d_init(bhm_output2d_t** output, bhm_cortex_size_t x0, bhm_cortex_size_t y0, bhm_cortex_size_t x1, bhm_cortex_size_t y1);

/// @brief Initializes the given cortex with default values.
/// @param cortex The cortex to initialize.
/// @param width The width of the cortex.
/// @param height The height of the cortex.
/// @param nh_radius The neighborhood radius for each individual cortex neuron.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_init(bhm_cortex2d_t** cortex, bhm_cortex_size_t width, bhm_cortex_size_t height, bhm_nh_radius_t nh_radius);

/// @brief Destroys the given input2d and frees memory.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t i2d_destroy(bhm_input2d_t* input);

/// @brief Destroys the given output2d and frees memory.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t o2d_destroy(bhm_output2d_t* output);

/// @brief Destroys the given cortex2d and frees memory for it and its neurons.
/// @param cortex The cortex to destroy
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_destroy(bhm_cortex2d_t* cortex);

/// @brief Returns a cortex with the same properties as the given one.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_copy(bhm_cortex2d_t* to, bhm_cortex2d_t* from);


// ########################################## Setter functions ##################################################

/// @brief Sets the neighborhood radius for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_nhradius(bhm_cortex2d_t* cortex, bhm_nh_radius_t radius);

/// @brief Sets the neighborhood mask for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_nhmask(bhm_cortex2d_t* cortex, bhm_nh_mask_t mask);

/// @brief Sets the evolution step for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_evol_step(bhm_cortex2d_t* cortex, bhm_evol_step_t evol_step);

/// @brief Sets the pulse window width for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_pulse_window(bhm_cortex2d_t* cortex, bhm_ticks_count_t window);

/// @brief Sets the sample window for the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_sample_window(bhm_cortex2d_t* cortex, bhm_ticks_count_t sample_window);

/// @brief Sets the fire threshold for all neurons in the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_fire_threshold(bhm_cortex2d_t* cortex, bhm_neuron_value_t threshold);

/// @brief Sets the syngen chance for the cortex. Syngen chance defines the probability for synapse generation and deletion.
/// @param syngen_chance The chance to apply (must be between 0x0000U and 0xFFFFU).
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_syngen_chance(bhm_cortex2d_t* cortex, bhm_chance_t syngen_chance);

/// @brief Sets the synstr chance for the cortex. Synstr chance defines the probability for synapse strengthening and weakening.
/// @param synstr_chance The chance to apply (must be between 0x0000U and 0xFFFFU).
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_synstr_chance(bhm_cortex2d_t* cortex, bhm_chance_t synstr_chance);

/// @brief Sets the maximum number of (input) synapses for the neurons of the cortex.
/// @param cortex The cortex to edit.
/// @param syn_count The max number of allowable synapses.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_max_syn_count(bhm_cortex2d_t* cortex, bhm_syn_count_t syn_count);

/// @brief Sets the maximum allowable touch for each neuron in the network.
/// A neuron touch is defined as its synapses count divided by its total neighbors count.
/// @param touch The touch to assign the cortex. Only values between 0 and 1 are allowed.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_max_touch(bhm_cortex2d_t* cortex, float touch);

/// @brief Sets the preferred input mapping for the given cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_pulse_mapping(bhm_cortex2d_t* cortex, bhm_pulse_mapping_t pulse_mapping);

/// @brief Sets the range for excitatory to inhibitory ratios in single neurons.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_inhexc_range(bhm_cortex2d_t* cortex, bhm_chance_t inhexc_range);

/// @brief Sets the proportion between excitatory and inhibitory generated synapses.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_inhexc_ratio(bhm_cortex2d_t* cortex, bhm_chance_t inhexc_ratio);

/// @brief Sets whether the tick pass should wrap around the edges (pacman effect).
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_set_wrapped(bhm_cortex2d_t* cortex, bhm_bool_t wrapped);

/// @brief Disables self connections whithin the specified bounds.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_syn_disable(bhm_cortex2d_t* cortex, bhm_cortex_size_t x0, bhm_cortex_size_t y0, bhm_cortex_size_t x1, bhm_cortex_size_t y1);

/// @brief Randomly mutates the cortex.
/// @param cortex The cortex to edit.
/// @param mut_chance The probability of applying mutation to any mutable property of the cortex.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_mutate(bhm_cortex2d_t* cortex, bhm_chance_t mut_chance);


// ########################################## Getter functions ##################################################

/// @brief Stores the string representation of the given cortex to the provided string [target].
/// @param cortex The cortex to inspect.
/// @param target The string to fill with cortex data.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t c2d_to_string(bhm_cortex2d_t* cortex, char* target);

/// @brief Computes the mean value of an output2d values.
/// @param output The output to compute the mean value from.
/// @param target Pointer to the result of the computation. The mean value will be stored here.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
bhm_error_code_t o2d_mean(bhm_output2d_t* output, bhm_ticks_count_t* target);

#ifdef __cplusplus
}
#endif

#endif
