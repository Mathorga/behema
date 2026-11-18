/*
*****************************************************************
bias.c

Diagnostic (not a pass/fail test): measures whether the fixed order of the neighborhood loop in the tick function
favors some neighborhood positions over others when neurons reach their max_syn_count or max_tot_strength caps.

The cortex and its input are symmetric, so without any order bias, synapses should be distributed evenly between
neighborhood positions visited early in the loop (top-left half) and late in the loop (bottom-right half).

Copyright (C) 2026 Luka Micheletti
*****************************************************************
*/

#include <stdio.h>
#include <math.h>

#include "behema.h"

#define BIAS_CORTEX_SIZE 128
#define BIAS_NH_RADIUS 2
#define BIAS_NH_DIAM BHM_NH_DIAM_2D(BIAS_NH_RADIUS)
#define BIAS_NH_CELLS (BIAS_NH_DIAM * BIAS_NH_DIAM)
#define BIAS_NH_CENTER (BIAS_NH_CELLS / 2)
#define BIAS_INPUT_SIZE 8
#define BIAS_TICKS_COUNT 5000

// Same value used in the plasticity tests: makes many synapses change strength in the same tick, so caps are often hit.
#define BIAS_STRESS_SYNSTR_CHANCE 0x0800U

typedef struct {
    const char* name;
    // Evolve at every tick with a high synstr chance.
    bhm_bool_t stressed;
    // Keep the default caps. If false, caps are raised so that they can never be reached (control run).
    bhm_bool_t capped;
} bias_config_t;

/// @brief Reads the strength of the synapse at the given neighborhood index.
static uint32_t syn_strength_at(
    bhm_neuron_t* neuron,
    uint32_t nh_index
) {
    return ((neuron->synstr_mask_a >> nh_index) & 0x01U) |
           (((neuron->synstr_mask_b >> nh_index) & 0x01U) << 0x01U) |
           (((neuron->synstr_mask_c >> nh_index) & 0x01U) << 0x02U);
}

/// @brief Computes the relative difference (in percent) between [first] and [second], with respect to their mean.
static double asymmetry(
    double first,
    double second
) {
    double mean = (first + second) / 2.0;
    return mean > 0.0 ? 100.0 * (first - second) / mean : 0.0;
}

/// @brief Raises all caps so that they can never be reached.
static void remove_caps(
    bhm_cortex2d_t* cortex
) {
    // A neuron can't have more than (BIAS_NH_CELLS - 1) synapses of strength BHM_MAX_SYN_STRENGTH each, which is well below 0xFF.
    cortex->max_tot_strength = 0xFFU;
    cortex->max_syn_count = BHM_NH_COUNT_2D(BIAS_NH_DIAM);

    // The tick reads max_syn_count from each neuron, not from the cortex.
    for (bhm_cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
        cortex->neurons[i].max_syn_count = BHM_NH_COUNT_2D(BIAS_NH_DIAM);
    }
}

static bhm_error_code_t run_config(
    bias_config_t config
) {
    bhm_error_code_t error;

    // Context init.
    bhm_context2d_t* context;
    error = bhm_ctx2d_create(&context, BIAS_CORTEX_SIZE, BIAS_CORTEX_SIZE, BIAS_NH_RADIUS);
    if (error != BHM_ERROR_NONE) return error;

    // Context setup.
    if (config.stressed) {
        bhm_ctx2d_set_evol_step(context, 0x00U);
        context->even_cortex->synstr_chance = BIAS_STRESS_SYNSTR_CHANCE;
        context->odd_cortex->synstr_chance = BIAS_STRESS_SYNSTR_CHANCE;
    }
    if (!config.capped) {
        remove_caps(context->even_cortex);
        remove_caps(context->odd_cortex);
    }

    // Input init: a square patch exactly in the center of the cortex, so that the whole setup is symmetric.
    bhm_input2d_t* input;
    error = bhm_i2d_create(
        &input,
        (BIAS_CORTEX_SIZE / 2) - (BIAS_INPUT_SIZE / 2),
        (BIAS_CORTEX_SIZE / 2) - (BIAS_INPUT_SIZE / 2),
        (BIAS_CORTEX_SIZE / 2) + (BIAS_INPUT_SIZE / 2),
        (BIAS_CORTEX_SIZE / 2) + (BIAS_INPUT_SIZE / 2),
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_LINEAR
    );
    if (error != BHM_ERROR_NONE) return error;

    for (bhm_cortex_size_t i = 0; i < BIAS_INPUT_SIZE * BIAS_INPUT_SIZE; i++) {
        input->values[i] = context->even_cortex->sample_window - 1;
    }

    error = bhm_ctx2d_add_input(context, input);
    if (error != BHM_ERROR_NONE) return error;

    // Run.
    for (uint32_t i = 0; i < BIAS_TICKS_COUNT; i++) {
        error = ctx2d_tick(context);
        if (error != BHM_ERROR_NONE) return error;
    }

    // Collect per-position statistics, only on neurons whose whole neighborhood lies inside the cortex.
    bhm_cortex2d_t* cortex = context->counts->ticks_count % 2 ? context->odd_cortex : context->even_cortex;
    uint64_t active_counts[BIAS_NH_CELLS] = {0};
    uint64_t strength_sums[BIAS_NH_CELLS] = {0};
    uint64_t neurons_count = 0;
    uint64_t at_syn_cap_count = 0;
    uint64_t at_strength_cap_count = 0;

    // Per-neuron difference between the synapses in the first and in the last half of the neighborhood, used to estimate noise.
    double diff_sum = 0.0;
    double diff_squares_sum = 0.0;

    for (bhm_cortex_size_t y = BIAS_NH_RADIUS; y < cortex->height - BIAS_NH_RADIUS; y++) {
        for (bhm_cortex_size_t x = BIAS_NH_RADIUS; x < cortex->width - BIAS_NH_RADIUS; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            neurons_count++;
            if (neuron->syn_count >= neuron->max_syn_count) at_syn_cap_count++;
            if (neuron->tot_syn_strength >= cortex->max_tot_strength) at_strength_cap_count++;

            int32_t diff = 0;
            for (uint32_t k = 0; k < BIAS_NH_CELLS; k++) {
                if (k == BIAS_NH_CENTER) continue;

                if ((neuron->synac_mask >> k) & 0x01U) {
                    active_counts[k]++;
                    strength_sums[k] += syn_strength_at(neuron, k);
                    diff += k < BIAS_NH_CENTER ? 1 : -1;
                }
            }
            diff_sum += diff;
            diff_squares_sum += diff * diff;
        }
    }

    // Print results.
    printf("\n########## %s ##########\n", config.name);
    printf("Neurons at max_syn_count: %.1f%%, at max_tot_strength: %.1f%%\n",
        100.0 * at_syn_cap_count / neurons_count,
        100.0 * at_strength_cap_count / neurons_count);

    printf("\nActive synapses per neighborhood position (%% of neurons):\n");
    for (uint32_t j = 0; j < BIAS_NH_DIAM; j++) {
        for (uint32_t i = 0; i < BIAS_NH_DIAM; i++) {
            uint32_t k = BHM_IDX2D(i, j, BIAS_NH_DIAM);
            if (k == BIAS_NH_CENTER) printf("      X");
            else printf(" %6.1f", 100.0 * active_counts[k] / neurons_count);
        }
        printf("\n");
    }

    printf("\nMean strength of active synapses per neighborhood position:\n");
    for (uint32_t j = 0; j < BIAS_NH_DIAM; j++) {
        for (uint32_t i = 0; i < BIAS_NH_DIAM; i++) {
            uint32_t k = BHM_IDX2D(i, j, BIAS_NH_DIAM);
            if (k == BIAS_NH_CENTER) printf("      X");
            else printf(" %6.2f", active_counts[k] > 0 ? (double) strength_sums[k] / active_counts[k] : 0.0);
        }
        printf("\n");
    }

    // Positions before the center are visited first in the loop, positions after the center are visited last.
    uint64_t first_active = 0, last_active = 0;
    uint64_t first_strength = 0, last_strength = 0;
    for (uint32_t k = 0; k < BIAS_NH_CELLS; k++) {
        if (k < BIAS_NH_CENTER) {
            first_active += active_counts[k];
            first_strength += strength_sums[k];
        } else if (k > BIAS_NH_CENTER) {
            last_active += active_counts[k];
            last_strength += strength_sums[k];
        }
    }

    printf("\nVisited first vs visited last (positive = first half favored):\n");
    printf("    synapses:            %+.2f%%\n", asymmetry(first_active, last_active));
    printf("    total strength:      %+.2f%%\n", asymmetry(first_strength, last_strength));
    printf("    mean strength:       %+.2f%%\n", asymmetry(
        first_active > 0 ? (double) first_strength / first_active : 0.0,
        last_active > 0 ? (double) last_strength / last_active : 0.0
    ));

    // How far the mean per-neuron difference is from 0, in standard errors. Above ~3 it's unlikely to be noise.
    // Neighboring neurons are correlated, so the real standard error is somewhat larger than this estimate.
    double diff_mean = diff_sum / neurons_count;
    double diff_variance = diff_squares_sum / neurons_count - diff_mean * diff_mean;
    double diff_std_error = sqrt(diff_variance / neurons_count);
    printf("\nPer-neuron synapses, first half minus last half: %+.3f (std error %.3f, %.1f std errors from 0)\n",
        diff_mean,
        diff_std_error,
        diff_std_error > 0.0 ? diff_mean / diff_std_error : 0.0);

    bhm_ctx2d_destroy(context);

    return BHM_ERROR_NONE;
}

int main(int argc, char **argv) {
    bias_config_t configs[] = {
        {"Default parameters", BHM_FALSE, BHM_TRUE},
        {"Stressed, capped", BHM_TRUE, BHM_TRUE},
    };
    uint32_t configs_count = sizeof(configs) / sizeof(bias_config_t);

    for (uint32_t i = 0; i < configs_count; i++) {
        bhm_error_code_t error = run_config(configs[i]);
        if (error != BHM_ERROR_NONE) {
            printf("There was an error running %s: %d\n", configs[i].name, error);
            return 1;
        }
    }

    return 0;
}
