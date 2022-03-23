/*
*****************************************************************
portia_cuda.h

Copyright (C) 2022 Luka Micheletti
*****************************************************************
*/

#ifndef __PORTIA_CUDA__
#define __PORTIA_CUDA__

#include "cortex.h"
#include "error.h"
#include "utils.h"

// Util functions:

/// Marsiglia's xorshift pseudo-random number generator with period 2^96-1.
__device__ uint32_t xorshf32();


// Initialization functions:

/// Initializes the given input with the given values.
error_code_t i2d_init(input2d_t* input, cortex_size_t x0, cortex_size_t y0, cortex_size_t x1, cortex_size_t y1, neuron_value_t exc_value, pulse_mapping_t pulse_mapping);

/// Initializes the given cortex with default values.
error_code_t c2d_init(cortex2d_t** cortex, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius);

/// Returns a cortex with the same properties as the given one.
error_code_t c2d_copy(cortex2d_t* to, cortex2d_t* from);

/// Copies a cortex from host to device.
error_code_t c2d_to_device(cortex2d_t* host_cortex, cortex2d_t* device_cortex);

/// Copies a cortex from device to host.
error_code_t c2d_to_host(cortex2d_t* device_cortex, cortex2d_t* host_cortex);


// Execution functions:

/// Performs a full run cycle over the network cortex.
__global__ void c2d_tick(cortex2d_t* prev_cortex, cortex2d_t* next_cortex);


// Mapping functions.

/// Maps a value to a pulse pattern according to the specified input mapping.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
/// @param pulse_mapping The mapping algorithm to apply for mapping.
__host__ __device__ bool_t pulse_map(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input, pulse_mapping_t pulse_mapping);

/// Computes a linear mapping for the given input and sample step.
/// Linear mapping always fire at least once, even if input is 0.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
__host__ __device__ bool_t pulse_map_linear(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// This is computationally cheap if compared to rprop, but it provides a less even distribution. The difference can be seen on big windows.
/// This is to be preferred if a narrow window is being used or an even distribution is not critical.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
__host__ __device__ bool_t pulse_map_fprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

/// Computes a proportional mapping for the given input and sample step.
/// Provides a better distribution if compared to fprop, but is computationally more expensive. The difference can be seen on big windows.
/// This is to be preferred if a wide window is being used and an even distribution are critical, otherwise go for fprop.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
__host__ __device__ bool_t pulse_map_rprop(ticks_count_t sample_window, ticks_count_t sample_step, ticks_count_t input);

#endif