/*
*****************************************************************
cortex.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __cortex__
#define __cortex__

#include <stdint.h>

// Translate an id wrapping it to the provided size (pacman effect).
// WARNING: Only works with signed types and does not show errors otherwise.
// [i] is the given index.
// [n] is the size over which to wrap.
#define WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

// Computes the diameter of a square neighborhood given its radius.
#define SQNH_DIAM(r) (2 * (r) + 1)

// Computes the number of neighbors in a square neighborhood given its diameter.
#define SQNH_COUNT(d) ((d) * (d) - 1)

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

#define EVOL_STEP_NEVER 0x0000FFFFU

#define PULSE_WINDOW_LARGE 0x3FU
#define PULSE_WINDOW_MID 0x1FU
#define PULSE_WINDOW_SMALL 0x0AU

#define MAX_SYN_STRENGTH 0x07U

// Completely arbitrary values used to define a sort of acceptable cortex right away.
#define DEFAULT_THRESHOLD 0x88U
#define DEFAULT_STARTING_VALUE 0x00U
#define DEFAULT_RECOVERY_VALUE -0x22
#define DEFAULT_MAX_TOUCH 0.22F
#define DEFAULT_EXCITING_VALUE 0x14U
#define DEFAULT_INHIBITING_VALUE -0x06
#define DEFAULT_DECAY_RATE 0x01U
#define DEFAULT_SYNGEN_BEAT 0.05F
#define DEFAULT_SYNSTR_BEAT 0.5F
#define DEFAULT_PULSE_WINDOW 0x39U
#define DEFAULT_EVOL_STEP 0x0000000AU
#define DEFAULT_INHEXC_RATIO 0x0FU
#define DEFAULT_SAMPLE_WINDOW 0x0AU
#define DEFAULT_MAX_TOT_STRENGTH 0x20U
#define DEFAULT_SYNGEN_CHANCE 0x0A00U
#define DEFAULT_SYNDEL_CHANCE 0x0A00U
#define DEFAULT_SYNSTR_CHANCE 0x0F00U
#define DEFAULT_SYNWK_CHANCE 0x0A00U

typedef uint8_t byte;

typedef int16_t neuron_value_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t nh_mask_t;
typedef int8_t nh_radius_t;
typedef uint8_t syn_count_t;
typedef uint16_t syn_strength_t;
typedef uint16_t ticks_count_t;
typedef uint32_t evol_step_t;
typedef uint64_t pulse_mask_t;
typedef int8_t pulses_count_t;
typedef uint16_t chance_t;

typedef int32_t cortex_size_t;

typedef enum {
    FALSE = 0,
    TRUE = 1
} bool_t;

typedef enum {
    // Values are forced to 32 bit integers by using big enough values: 100000 is 17 bits long, so 32 bits are automatically allocated.
    // Linear.
    PULSE_MAPPING_LINEAR = 0x10000,
    // Floored proportional.
    PULSE_MAPPING_FPROP = 0x10001,
    // Rounded proportional.
    PULSE_MAPPING_RPROP = 0x10002,
} pulse_mapping_t;

typedef struct input2d {
    cortex_size_t x0;
    cortex_size_t y0;
    cortex_size_t x1;
    cortex_size_t y1;
    ticks_count_t sample_window;
    pulse_mapping_t pulse_mapping;
    ticks_count_t* values;
} input2d_t;

/// Neuron.
typedef struct {
    // Neighborhood connections pattern (SYNapses ACtivation state):
    // 1|1|0
    // 0|x|1 => 1100x1100
    // 1|0|0
    nh_mask_t synac_mask;
    // Neighborhood excitatory states pattern (SYNapses EXcitatory state), defines whether the synapses from the neighbors are excitatory (1) or inhibitory (0).
    // Only values corresponding to active synapses are used.
    nh_mask_t synex_mask;
    // Neighborhood synapses strength pattern (SYNapses STRength). Defines a 3 bit value defined as [cba].
    nh_mask_t synstr_mask_a;
    nh_mask_t synstr_mask_b;
    nh_mask_t synstr_mask_c;


    // Activation history pattern:
    //           |<--pulse_window-->|
    // xxxxxxxxxx01001010001010001001--------> t
    //                              ^
    // Used to know the pulse frequency in a given moment (e.g. for syngen).
    pulse_mask_t tick_pulse_mask;
    // Amount of activations in the cortex's tick_pulse window.
    pulses_count_t tick_pulse;
    // Activation history pattern.
    // Used to know the pulse frequency over long periods of time (e.g. for synaptic plasticity).
    // evol_pulse is just tick_pulse, but only updated every [evol_step] ticks (and with a different logic, but that's not crucial).
    pulse_mask_t evol_pulse_mask;
    // Amount of activations in the cortex's tick_pulse window.
    pulses_count_t evol_pulse;


    // Current internal value.
    neuron_value_t value;
    // Amount of connected neighbors.
    syn_count_t syn_count;
    // Total amount of syn strength from input neurons.
    syn_strength_t tot_syn_strength;
} neuron_t;

/// 2D cortex of neurons.
typedef struct {
    // Width of the cortex.
    cortex_size_t width;
    // Height of the cortex.
    cortex_size_t height;
    // Ticks performed since cortex creation.
    ticks_count_t ticks_count;
    // Evolutions performed since cortex creation.
    ticks_count_t evols_count;
    // Amount of ticks between each evolution.
    ticks_count_t evol_step;
    // Length of the window used to count pulses in the cortex' neurons.
    // TODO Switch "beat" and "pulse".
    pulses_count_t pulse_window;


    // Radius of each neuron's neighborhood.
    nh_radius_t nh_radius;
    neuron_value_t fire_threshold;
    neuron_value_t recovery_value;
    neuron_value_t exc_value;
    neuron_value_t inh_value;
    neuron_value_t decay_value;


    // Chance of synapse generation (out of 0xFFFFU).
    chance_t syngen_chance;
    // Chance of synapse deletion (out of 0xFFFFU).
    chance_t syndel_chance;
    // Chance of synapse strengthening (out of 0xFFFFU).
    chance_t synstr_chance;
    // Chance of synapse weakening (out of 0xFFFFU).
    chance_t synwk_chance;


    // Pulses count (tick_pulse) needed to generate (or delete, if possible) a synapse between neurons.
    pulses_count_t syngen_pulses_count;
    // Pulses count (evol_pulse) needed to strengthen an existing synapse between two neurons.
    pulses_count_t synstr_pulses_count;
    syn_strength_t max_tot_strength;
    // Maximum number of synapses between a neuron and its neighbors.
    syn_count_t max_syn_count;
    // Proportion between excitatory and inhibitory generated synapses (e.g. inhexc_ratio = 10 => 9 exc, 1 inh).
    ticks_count_t inhexc_ratio;


    ticks_count_t sample_window;
    pulse_mapping_t pulse_mapping;

    bool_t wrapped;

    neuron_t* neurons;
} cortex2d_t;

// TODO cortex3d_t

#endif
