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

// Util functions:

/// Marsiglia's xorshift pseudo-random number generator with period 2^32-1.
uint32_t xorshf32();


// Execution functions:

/// Feeds external spikes to the specified neurons.
void c2d_feed(cortex2d_t* cortex, cortex_size_t starting_index, cortex_size_t count, neuron_value_t* values);

/// Feeds a cortex with the provided input2d.
/// @param cortex The cortex to feed.
/// @param input The input to feed the cortex.
void c2d_feed2d(cortex2d_t* cortex, input2d_t* input);

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
/// @param input The actual input to map to a pulse (must be in range 0..(sample_window - 1)).
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

bool_t pulse_map_dfprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);


#ifdef __cplusplus
}
#endif

#endif
