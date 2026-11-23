/*
*****************************************************************
tests.h

Copyright (C) 2026 Luka Micheletti
*****************************************************************
*/

#ifndef __BHM_TESTS__
#define __BHM_TESTS__

#include <stdio.h>

#include "behema.h"

// Checks the given condition and makes the calling test fail if the condition is false.
// The calling test is expected to return an int: 0 on success, 1 on failure.
#define BHM_ASSERT(condition) \
    do { \
        if (!(condition)) { \
            printf("    ASSERTION FAILED: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

// Same as BHM_ASSERT, but also prints the expected and actual values on failure.
#define BHM_ASSERT_EQ(actual, expected) \
    do { \
        long long _actual = (long long) (actual); \
        long long _expected = (long long) (expected); \
        if (_actual != _expected) { \
            printf("    ASSERTION FAILED: %s == %s, got %lld, expected %lld (%s:%d)\n", #actual, #expected, _actual, _expected, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)


// ##########################################
// Macros tests.
// ##########################################

/// @brief Checks that BHM_ROTR_MASK rotates masks correctly.
/// @return 0 if the test passed, 1 otherwise.
int test_rotr_mask();


// ##########################################
// Plasticity tests.
// ##########################################

/// @brief Checks that each neuron's syn_count matches the amount of its active synapses.
/// @return 0 if the test passed, 1 otherwise.
int test_syn_count_matches_synac_mask();

/// @brief Checks that each neuron's tot_syn_strength matches the sum of the strengths of its active synapses.
/// @return 0 if the test passed, 1 otherwise.
int test_tot_syn_strength_matches_masks();

/// @brief Checks that no neuron has more synapses than its max_syn_count.
/// @return 0 if the test passed, 1 otherwise.
int test_syn_count_within_max();

/// @brief Checks that no neuron has a tot_syn_strength above the cortex' max_tot_strength.
/// @return 0 if the test passed, 1 otherwise.
int test_tot_syn_strength_within_max();

/// @brief Checks all the bookkeeping invariants above on a run with reward-modulated plasticity enabled.
/// @return 0 if the test passed, 1 otherwise.
int test_modulation_keeps_bookkeeping();


// ##########################################
// Reward-modulated plasticity tests.
// ##########################################

/// @brief Checks that a positive modulation strengthens a synapse whose input neuron fired right before its output neuron.
/// @return 0 if the test passed, 1 otherwise.
int test_modulation_rewards_eligible_synapse();

/// @brief Checks that a negative modulation weakens a synapse whose input neuron fired right before its output neuron.
/// @return 0 if the test passed, 1 otherwise.
int test_modulation_punishes_eligible_synapse();

/// @brief Checks that synapses are left unchanged without modulation, with the wrong firing order, with simultaneous firing,
/// and when the coincidence is older than the eligibility window.
/// @return 0 if the test passed, 1 otherwise.
int test_modulation_ignores_non_eligible_synapses();

/// @brief Checks that bhm_ctx2d_add_modulation sums and clamps values, and that the modulation is reset after an evolution.
/// @return 0 if the test passed, 1 otherwise.
int test_modulation_add_clamps();

#endif
