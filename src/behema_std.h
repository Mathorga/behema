/*
*****************************************************************
behema_std.h

Copyright (C) 2021 Luka Micheletti
*****************************************************************
*/

#ifndef __BEHEMA_STD__
#define __BEHEMA_STD__

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

// Execution functions:

/// Feeds a cortex with the provided input2d.
/// @param cortex The cortex to feed.
/// @param input The input to feed the cortex.
void c2d_feed2d(cortex2d_t* cortex, input2d_t* input);

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
