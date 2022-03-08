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
#include "cortex.h"
#include "error.h"
#include "utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialization functions:

error_code_t i2d_init(input2d_t* input, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, ticks_count_t sample_window);
error_code_t i2d_map(input2d_t* input, uint32_t src_min, uint32_t src_max, uint32_t* values);

/// Initializes the given cortex with default values.
error_code_t c2d_init(cortex2d_t* cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius);

/// Returns a cortex with the same properties as the given one.
error_code_t c2d_copy(cortex2d_t* to, cortex2d_t* from);

/// Sets the neighborhood radius for all neurons in the cortex.
error_code_t c2d_set_nhradius(cortex2d_t* cortex, nh_radius_t radius);

/// Sets the neighborhood mask for all neurons in the cortex.
void c2d_set_nhmask(cortex2d_t* cortex, nh_mask_t mask);

/// Sets the evolution step for the cortex.
void c2d_set_evol_step(cortex2d_t* cortex, evol_step_t evol_step);

/// Sets the pulse window width for the cortex.
void c2d_set_pulse_window(cortex2d_t* cortex, spikes_count_t window);

/// Sets the sample window for the cortex.
void c2d_set_sample_window(cortex2d_t* cortex, ticks_count_t sample_window);

/// Sets the fire threshold for all neurons in the cortex.
void c2d_set_fire_threshold(cortex2d_t* cortex, neuron_value_t threshold);

/// Sets the maximum number of (input) synapses for the neurons of the cortex.
/// @param cortex The cortex to edit.
/// @param syn_count The max number of allowable synapses.
void c2d_set_max_syn_count(cortex2d_t* cortex, syn_count_t syn_count);

/// Sets the maximum allowable touch for each neuron in the network.
/// A neuron touch is defined as its synapses count divided by its total neighbors count.
/// @param touch The touch to assign the cortex. Only values between 0 and 1 are allowed.
void c2d_set_max_touch(cortex2d_t* cortex, float touch);

/// Sets the preferred input mapping for the given cortex.
void c2d_set_pulse_mapping(cortex2d_t* cortex, pulse_mapping_t pulse_mapping);

/// Sets the range for excitatory to inhibitory ratios in single neurons.
void c2d_set_inhexc_range(cortex2d_t* cortex, chance_t inhexc_range);

/// Sets the proportion between excitatory and inhibitory generated synapses.
void c2d_set_inhexc_ratio(cortex2d_t* cortex, chance_t inhexc_ratio);

/// Sets whether the tick pass should wrap around the edges (pacman effect).
void c2d_set_wrapped(cortex2d_t* cortex, bool_t wrapped);

/// Disables self connections whithin the specified bounds.
void c2d_syn_disable(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1);


// Execution functions:

/// Feeds external spikes to the specified neurons.
void c2d_feed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t* values);

/// Externally feeds the neurons inside the rectangle described by the given parameters.
/// x0 and y0 cannot be less than 0, while x1 and y1 cannot be greater than the cortex's width and height respectively.
/// @param cortex The target cortex to feed.
/// @param x0 The starting x index of the target neurons square.
/// @param y0 The starting y index of the target neurons square.
/// @param x1 The ending x index of the target neurons square.
/// @param y1 The ending y index of the target neurons square.
/// @param value The value used to feed each input neuron.
void c2d_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t value);

/// Externally feeds the neurons inside the rectangle described by the given parameters.
/// The feeding is done by inputs mapping: inputs have to be values between 0 and cortex->sample_window.
/// @param cortex The target cortex to feed.
/// @param x0 The starting x index of the target neurons square.
/// @param y0 The starting y index of the target neurons square.
/// @param x1 The ending x index of the target neurons square.
/// @param y1 The ending y index of the target neurons square.
/// @param sample_step The current step in the sample window, must be between 0 and cortex->sample_window - 1.
/// @param inputs The input values mapped to the sample window.
/// @param value The value used to feed each input neuron.
void c2d_sample_sqfeed(cortex2d_t* cortex, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, ticks_count_t sample_step, ticks_count_t* inputs, neuron_value_t value);

/// Default feed.
void c2d_dfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t value);

/// Random feed.
void c2d_rfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t max_value);

/// Spread feed.
void c2d_sfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t value, cortex_size_t spread);

/// Random spread feed.
void c2d_rsfeed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t max_value, cortex_size_t spread);

/// Performs a full run cycle over the network cortex.
void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex);


// Mapping functions.

/// Maps a value to a pulse pattern according to the specified input mapping.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
/// @param pulse_mapping The mapping algorithm to apply for mapping.
bool_t pulse_map(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping);

/// Computes a linear mapping for the given input and sample step.
/// Linear mapping always fire at least once, even if input is 0.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool_t pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// This is computationally cheap if compared to rprop, but it provides a less even distribution. The difference can be seen on big windows.
/// This is to be preferred if a narrow window is being used or an even distribution is not critical.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool_t pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// Provides a better distribution if compared to fprop, but is computationally more expensive. The difference can be seen on big windows.
/// This is to be preferred if a wide window is being used and an even distribution are critical, otherwise go for fprop.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bool_t pulse_map_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);


#ifdef __cplusplus
}
#endif

#endif
