/*
*****************************************************************
portia_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __PORTIA_STD__
#define __PORTIA_STD__

#include <stdint.h>
#include <stdlib.h>
// TODO Remove in release.
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
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

/// Sets the evolution step for the field.
void f2d_set_evol_step(field2d_t* field, evol_step_t evol_step);

/// Sets the pulse window width for the field.
void f2d_set_pulse_window(field2d_t* field, pulses_count_t window);

/// Sets the sample window for the field.
void f2d_set_sample_window(field2d_t* field, ticks_count_t sample_window);

/// Sets the fire threshold for all neurons in the field.
void f2d_set_fire_threshold(field2d_t* field, neuron_threshold_t threshold);

/// Sets the maximum allowable touch for each neuron in the network.
/// A neuron touch is defined as its synapses count divided by its total neighbors count.
/// @param touch The touch to assign the field. Only values between 0 and 1 are allowed.
void f2d_set_max_touch(field2d_t* field, float touch);

/// Sets the minimum amount of pulses needed to create a synapse.
/// @param field The field to set the value to.
/// @param pulse The minimum amount of pulses count to generate a synapse.
void f2d_set_syngen_pulses_count(field2d_t* field, pulses_count_t pulses_count);

/// Sets the pulse needed to allow synapse generation.
/// A neuron beat is defined as its pulses count divided by the size of the field's pulse window (aka the max possible pulses count).
/// @param beat The beat to assign the field. Only values between 0 and 1 are allowed.
void f2d_set_syngen_beat(field2d_t* field, float beat);

/// Sets the preferred input mapping for the given field.
void f2d_set_pulse_mapping(field2d_t* field, pulse_mapping_t pulse_mapping);

/// Sets the proportion between excitatory and inhibitory generated synapses.
void f2d_set_inhexc_ratio(field2d_t* field, ticks_count_t inhexc_ratio);


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

/// Externally feeds the neurons inside the rectangle described by the given parameters.
/// The feeding is done by inputs mapping: inputs have to be values between 0 and field->sample_window.
/// @param field The target field to feed.
/// @param x0 The starting x index of the target neurons square.
/// @param y0 The starting y index of the target neurons square.
/// @param x1 The ending x index of the target neurons square.
/// @param y1 The ending y index of the target neurons square.
/// @param sample_step The current step in the sample window, must be between 0 and field->sample_window - 1.
/// @param inputs The input values mapped to the sample window.
/// @param value The value used to feed each input neuron.
void f2d_sample_sqfeed(field2d_t* field, field_size_t x0, field_size_t y0, field_size_t x1, field_size_t y1, ticks_count_t sample_step, ticks_count_t* inputs, neuron_value_t value);

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

/// Maps a value to a pulse pattern according to the specified input mapping.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
/// @param pulse_mapping The mapping algorithm to apply for mapping.
bool pulse_map(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping);

/// Computes a linear mapping for the given input and sample step.
/// Linear mapping always fire at least once, even if input is 0.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// This is computationally cheap if compared to rprop, but it provides a less even distribution. The difference can be seen on big windows.
/// This is to be preferred if a narrow window is being used or an even distribution is not critical.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// Provides a better distribution if compared to fprop, but is computationally more expensive. The difference can be seen on big windows.
/// This is to be preferred if a wide window is being used and an even distribution are critical, otherwise go for fprop.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool pulse_map_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);


#ifdef __cplusplus
}
#endif

#endif
