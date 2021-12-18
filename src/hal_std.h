/*
*****************************************************************
hal_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __HAL_STD__
#define __HAL_STD__

#include <stdint.h>
#include <stdlib.h>
// TODO Remove in release.
#include <stdio.h>
#include <time.h>
#include <string.h>
#include "field.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

/// Initializes the given field with default values.
void f2d_init(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius);

/// Returns a field with the same properties as the given one.
field2d_t* f2d_copy(field2d_t* other);

/// Sets the neighborhood radius for all neurons in the field.
void f2d_set_nhradius(field2d_t* field, nh_radius_t radius);

/// Sets the neighborhood mask for all neurons in the field.
void f2d_set_nhmask(field2d_t* field, nh_mask_t mask);

/// Sets the fire threshold for all neurons in the field.
void f2d_set_fire_threshold(field2d_t* field, neuron_threshold_t threshold);

/// Sets the maximum allowable touch for each neuron in the network.
/// A neuron touch is defined as its synapses count divided by its total neighbors count.
/// @param touch The touch to assign the field. Only values between 0 and 1 are allowed.
void f2d_set_max_touch(field2d_t* field, float touch);

/// Sets the pulse needed to allow synapse generation.
/// A neuron beat is defined as its pulses count divided by the size of the field's pulse window (aka the max possible pulses count).
/// @param beat The beat to assign the field. Only values between 0 and 1 are allowed.
void f2d_set_syngen_beat(field2d_t* field, float beat);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void f2d_feed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t* values);

/// Externally feeds the neurons inside the rectangle described by the given parameters.
/// x0 and y0 cannot be less than 0, while x1 and y1 cannot be greater than the field's width and height respectively.
/// @param field The target field to feed.
/// @param x0 The starting x index of the target neurons square.
/// @param y0 The starting y index of the target neurons square.
/// @param x1 The ending x index of the target neurons square.
/// @param y1 The ending y index of the target neurons square.
/// @param value The value used to feed each input neuron.
void f2d_sqfeed(field2d_t* field, field_size_t x0, field_size_t y0, field_size_t x1, field_size_t y1, neuron_value_t value);

/// Default feed.
void f2d_dfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value);

/// Random feed.
void f2d_rfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value);

/// Spread feed.
void f2d_sfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value, field_size_t spread);

/// Random spread feed.
void f2d_rsfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value, field_size_t spread);

/// Performs a full run cycle over the network field.
void f2d_tick(field2d_t* prev_field, field2d_t* next_field);


#ifdef __cplusplus
}
#endif

#endif
