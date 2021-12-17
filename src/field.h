/*
*****************************************************************
field.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __FIELD__
#define __FIELD__

#include <stdint.h>

// Translate an id wrapping it to the provided size (pacman effect).
// WARNING: Only works with signed types and does not show errors otherwise.
// [i] is the given index.
// [n] is the size over which to wrap.
#define WRAP(i, n) ((i) >= 0 ? ((i) % (n)) : ((n) + ((i) % (n))))

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

#define NEURON_DEFAULT_THRESHOLD 0x88u
#define NEURON_STARTING_VALUE 0x00u
#define NEURON_RECOVERY_VALUE -0x22
#define NEURON_MAX_TOUCH 0.5f
#define NEURON_CHARGE_RATE 0x20u
#define NEURON_DECAY_RATE 0x01u
#define NEURON_INFLUENCE_GAIN 0x0020u

// Should these two be merged?
#define NEURON_SYNDEL_THRESHOLD 0x0100u
#define NEURON_SYNGEN_THRESHOLD 0x0100u
#define NEURON_SYNGEN_PULSE 0.1f

// Default mask is 1010101010101010101010101010101010101010101010101010101010101010 (AAAAAAAAAAAAAAAA in hex), meaning 50% of neighbors are connected.
// #define NEURON_DEFAULT_NH_MASK 0xAAAAAAAAAAAAAAAAu
#define NEURON_DEFAULT_NH_MASK 0x0000000000000000u
#define NEURON_DEFAULT_PULSE_MASK 0x00000000u

#define F2D_DEFAULT_PULSE_WINDOW 0x1F;

typedef int16_t neuron_value_t;
typedef uint8_t neuron_threshold_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t nh_mask_t;
typedef int8_t nh_radius_t;
typedef uint32_t neuron_influence_t;
typedef uint8_t nh_count_t;
typedef uint16_t ticks_count_t;
typedef uint32_t evol_step_t;
typedef uint32_t pulse_mask_t;
typedef int8_t pulse_width_t;

typedef int32_t field_size_t;

/// Neuron.
typedef struct {
    // Neighborhood connections pattern:
    // 1|1|0
    // 0|x|1 => 0011x0011
    // 1|0|0
    nh_mask_t nh_mask;

    // Activation history pattern:
    //           |<--pulse_window-->|
    // xxxxxxxxxx01001010001010001001--------> t
    //                              ^
    pulse_mask_t pulse_mask;

    // Amount of activations in the current pulse window.
    pulse_width_t pulse;

    // Current internal value.
    neuron_value_t value;
    nh_count_t nh_count;
} neuron_t;

/// 2D Field of neurons.
typedef struct {
    field_size_t width;
    field_size_t height;
    nh_radius_t nh_radius;
    neuron_threshold_t fire_threshold;
    ticks_count_t ticks_count;
    pulse_width_t pulse_window;
    neuron_t* neurons;
} field2d_t;

#endif