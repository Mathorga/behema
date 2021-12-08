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
void f2d_init(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius);

/// Random init.
void f2d_rinit(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius);

/// Returns a field with the same properties as the given one.
field2d_t* f2d_copy(field2d_t* other);

/// Sets the neighborhood radius for all neurons in the field.
void f2d_set_nhradius(field2d_t* field, nh_radius_t radius);

/// Sets the neighborhood mask for all neurons in the field.
void f2d_set_nhmask(field2d_t* field, nh_mask_t mask);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void f2d_feed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value);

/// Random feed.
void f2d_rfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value);

/// Spread feed.
void f2d_sfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value, field_size_t spread);

/// Random spread feed.
void f2d_rsfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value, field_size_t spread);

/// Performs a full run cycle over the network field.
void f2d_tick(field2d_t* prev_field, field2d_t* next_field, ticks_count_t evol_step);


#ifdef __cplusplus
}
#endif

#endif
