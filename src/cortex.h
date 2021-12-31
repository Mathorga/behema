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

#define DEFAULT_THRESHOLD 0x88U
#define DEFAULT_STARTING_VALUE 0x00U
#define DEFAULT_RECOVERY_VALUE -0x22
#define DEFAULT_MAX_TOUCH 0.1F
#define DEFAULT_EXCITING_VALUE 0x20U
#define DEFAULT_INHIBITING_VALUE -0x20U
#define DEFAULT_DECAY_RATE 0x01U

#define DEFAULT_SYNGEN_BEAT 0.01F

// Default mask is 1010101010101010101010101010101010101010101010101010101010101010 (AAAAAAAAAAAAAAAA in hex), meaning 50% of neighbors are connected.
// #define DEFAULT_NH_MASK 0xAAAAAAAAAAAAAAAAu
#define DEFAULT_NH_MASK 0x0000000000000000U
#define DEFAULT_PULSE_MASK 0x0000000000000000U

#define DEFAULT_PULSE_WINDOW 0x0AU
#define DEFAULT_EVOL_STEP 0x00000001U
#define EVOL_STEP_NEVER 0x0000FFFFU

#define DEFAULT_INHEXC_RATIO 0x0AU
#define DEFAULT_SAMPLE_WINDOW 0x0AU

typedef int16_t neuron_value_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t nh_mask_t;
typedef int8_t nh_radius_t;
typedef uint8_t syn_count_t;
typedef uint16_t ticks_count_t;
typedef uint32_t evol_step_t;
typedef uint64_t pulse_mask_t;
typedef int8_t pulses_count_t;

typedef int32_t cortex_size_t;

typedef enum {
    // Values are forced to 32 bit integers by using big enough values: 100000 is 17 bits long, so 32 bits are automatically allocated.
    // Linear.
    PULSE_MAPPING_LINEAR = 0x10000,
    // Floored proportional.
    PULSE_MAPPING_FPROP = 0x10001,
    // Rounded proportional.
    PULSE_MAPPING_RPROP = 0x10002,
} pulse_mapping_t;

/// Neuron.
typedef struct {
    // Neighborhood connections pattern (SYNapse ACtivation state):
    // 1|1|0
    // 0|x|1 => 1100x1100
    // 1|0|0
    nh_mask_t synac_mask;

    // Neighborhood excitatory states pattern (SYNapse EXcitatory state), defines whether the synapses from the neighbors are excitatory (1) or inhibitory (0).
    // Only values corresponding to active synapses are used.
    nh_mask_t synex_mask;

    // Activation history pattern:
    //           |<--pulse_window-->|
    // xxxxxxxxxx01001010001010001001--------> t
    //                              ^
    pulse_mask_t pulse_mask;

    // Amount of activations in the cortex's pulse window.
    pulses_count_t pulse;

    // Current internal value.
    neuron_value_t value;

    // Amount of connected neighbors.
    syn_count_t syn_count;
} neuron_t;

/// 2D cortex of neurons.
typedef struct {
    cortex_size_t width;
    cortex_size_t height;
    ticks_count_t ticks_count;
    ticks_count_t evol_step;
    pulses_count_t pulse_window;


    nh_radius_t nh_radius;
    neuron_value_t fire_threshold;
    neuron_value_t recovery_value;
    neuron_value_t charge_value;
    neuron_value_t decay_value;
    pulses_count_t syngen_pulses_count;

    // Maximum number of synapses between a neuron and its neighbors.
    syn_count_t max_syn_count;

    // Proportion between excitatory and inhibitory generated synapses (e.g. inhexc_ratio = 10 => 9 exc, 1 inh).
    ticks_count_t inhexc_ratio;


    ticks_count_t sample_window;
    pulse_mapping_t pulse_mapping;

    neuron_t* neurons;
} cortex2d_t;

#endif
