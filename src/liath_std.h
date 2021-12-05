/*
*****************************************************************
liath_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __LIATH_STD__
#define __LIATH_STD__

#include <stdint.h>
#include <stdlib.h>
// TODO Remove in release.
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "field.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

/// Initializes the given field with default values.
void field2d_init(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius);

/// Returns a field with the same properties as the given one.
field2d_t* field2d_copy(field2d_t* other);

/// Sets the neighborhood radius for all neurons in the field.
void field2d_set_nh_radius(field2d_t* field, nh_radius_t radius);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void field2d_feed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value);

/// Performs a full run cycle over the network braph.
void field2d_tick(field2d_t* prev_field, field2d_t* next_field);


// Learning functions:

/// Deletes all unused synapses.
void field2d_syndel(field2d_t* field);

/// Adds synapses to busy neurons (those that fire frequently).
void field2d_syngen(field2d_t* field);

/// Performs a full evolution cycle over the network braph.
void field2d_evolve(field2d_t* field);


#ifdef __cplusplus
}
#endif

#endif