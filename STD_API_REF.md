# STD API Reference

All functions return `bhm_error_t` telling the outcome of the function call.

## Functions

  #### c2d_feed2d
/// @brief Feeds a cortex through the provided input2d. Input data should already be in the provided input2d by the time this function is called.
/// @param cortex The cortex to feed.
/// @param input The input to feed the cortex.
void c2d_feed2d(bhm_cortex2d_t* cortex, bhm_input2d_t* input);

  #### c2d_read2d
/// @brief Reads data from a cortex through the provided output2d. When the mapping is done, output data is stored in the provided output2d.
/// @param cortex The cortex to read values from.
/// @param output The output used to read data from the cortex.
void c2d_read2d(bhm_cortex2d_t* cortex, bhm_output2d_t* output);

  #### c2d_tick
/// @brief Performs a full run cycle over the provided cortex.
/// @param prev_cortex The cortex at its current state.
/// @param next_cortex The cortex that will be updated by the tick cycle.
/// @warning prev_cortex and next_cortex should contain the same data (aka be copies one of the other), otherwise this operation may lead to unexpected behavior.
void c2d_tick(bhm_cortex2d_t* prev_cortex, bhm_cortex2d_t* next_cortex);


// ########################################## Input mapping functions ##########################################

/// @brief Maps a value to a pulse pattern according to the specified pulse mapping.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..(sample_window - 1)).
/// @param pulse_mapping The mapping algorithm to apply for mapping.
bhm_bool_t value_to_pulse(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input, bhm_pulse_mapping_t pulse_mapping);

/// @brief Computes a linear mapping for the given input and sample step.
/// Linear mapping always fire at least once, even if input is 0.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bhm_bool_t value_to_pulse_linear(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input);

/// @brief Computes a proportional mapping for the given input and sample step.
/// This is computationally cheap if compared to rprop, but it provides a less even distribution. The difference can be seen on big windows.
/// This is to be preferred if a narrow window is being used or an even distribution is not critical.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bhm_bool_t value_to_pulse_fprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input);

/// @brief Computes a proportional mapping for the given input and sample step.
/// Provides a better distribution if compared to fprop, but is computationally more expensive. The difference can be seen on big windows.
/// This is to be preferred if a wide window is being used and an even distribution are critical, otherwise go for fprop.
/// @param sample_window The width of the sampling window.
/// @param sample_step The step to test inside the specified window (e.g. w=10 s=3 => | | | |X| | | | | | |).
/// @param input The actual input to map to a pulse (must be in range 0..sample_window).
bhm_bool_t value_to_pulse_rprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input);

bhm_bool_t value_to_pulse_dfprop(bhm_ticks_count_t sample_window, bhm_ticks_count_t sample_step, bhm_ticks_count_t input);