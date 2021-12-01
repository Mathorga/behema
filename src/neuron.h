/*
*****************************************************************
neuron.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __NEURON__
#define __NEURON__

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

#define NEURON_DEFAULT_THRESHOLD 0xCCu
#define NEURON_STARTING_VALUE 0x00u
#define NEURON_DECAY_RATE 0x01u
#define NEURON_RECOVERY_VALUE -0x77

typedef uint8_t neighbors_count_t;
typedef int16_t neuron_value_t;

typedef int32_t field_size_t;

typedef struct {
    neighbors_count_t input_neighbors_count;
    uint8_t* input_neighbors;
    neuron_value_t value;
} neuron_t;

#endif