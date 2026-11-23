/*
*****************************************************************
reward.c

Diagnostic (not a pass/fail test): measures how well reward-modulated plasticity credits the right synapses
when the reward comes some time after the activity that earned it.

Two input patches, A (left) and B (right), are placed symmetrically in the cortex. They are stimulated in turns:
each stimulation lasts [REWARD_STIM_TICKS] ticks, and [delay] ticks after it ends the cortex receives a reward if it
was A, or a punishment if it was B. Then nothing happens for [REWARD_GAP_TICKS] ticks, so that eligibility from one
episode doesn't leak into the next one.

If reward-modulated plasticity works at a given delay, synapses around A end up stronger than synapses around B.
Each delay is run with and without reward-modulated plasticity (control), on several seeds:
the effect is the difference between the two.

Copyright (C) 2026 Luka Micheletti
*****************************************************************
*/

#include <stdio.h>
#include <math.h>

#include "behema.h"

#define REWARD_CORTEX_WIDTH 48
#define REWARD_CORTEX_HEIGHT 24
#define REWARD_NH_RADIUS 2
#define REWARD_PATCH_SIZE 6
// Distance of each patch's center from the cortex' center.
#define REWARD_PATCH_OFFSET 12
// Margin around each patch included in its measurement region: activity (and eligible synapses) spreads around the patch.
#define REWARD_REGION_MARGIN 4

#define REWARD_STIM_TICKS 20
// Longer than the max eligibility window, so that episodes don't overlap.
#define REWARD_GAP_TICKS 80
// Episodes per run: A and B alternate, so half are rewarded and half are punished.
#define REWARD_EPISODES_COUNT 60
#define REWARD_SEEDS_COUNT 4

#define REWARD_MODSTR_CHANCE 0x4000U

// Unsupervised (Hebbian) strength changes are disabled, so that strength only changes through reward and punishment.
// With the default synstr chance, both regions quickly reach max_tot_strength and become self-sustaining (they keep firing after
// the stimulation ends), so at reward time both are equally eligible and rewards and punishments cancel out.
// Synapses are still created and deleted by structural plasticity.
#define REWARD_SYNSTR_CHANCE 0x0000U

/// @brief Applies the given seed to every neuron's random state, so that runs with different seeds evolve differently.
static void apply_seed(
    bhm_cortex2d_t* cortex,
    uint32_t seed
) {
    for (bhm_cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
        // Random states must never be 0: xorshift of a non-zero state is never 0.
        cortex->neurons[i].rand_state = bhm_xorshf32(cortex->neurons[i].rand_state + seed * 0x9E3779B9U);
        if (cortex->neurons[i].rand_state == 0x00U) cortex->neurons[i].rand_state = BHM_RAND_OFFSET;
    }
}

/// @brief Turns the given input fully on (fires at every tick) or fully off (never fires).
static void set_input(
    bhm_input2d_t* input,
    bhm_ticks_count_t sample_window,
    bhm_bool_t on
) {
    for (bhm_cortex_size_t i = 0; i < REWARD_PATCH_SIZE * REWARD_PATCH_SIZE; i++) {
        // Values equal to the sample window never fire, see value_to_pulse.
        input->values[i] = on ? sample_window - 1 : sample_window;
    }
}

/// @brief Computes the mean tot_syn_strength of the neurons in the region around the given input.
static double region_mean_strength(
    bhm_cortex2d_t* cortex,
    bhm_input2d_t* input
) {
    double sum = 0.0;
    uint32_t count = 0;

    for (bhm_cortex_size_t y = input->y0 - REWARD_REGION_MARGIN; y < input->y1 + REWARD_REGION_MARGIN; y++) {
        for (bhm_cortex_size_t x = input->x0 - REWARD_REGION_MARGIN; x < input->x1 + REWARD_REGION_MARGIN; x++) {
            if (x < 0 || y < 0 || x >= cortex->width || y >= cortex->height) continue;

            sum += cortex->neurons[BHM_IDX2D(x, y, cortex->width)].tot_syn_strength;
            count++;
        }
    }

    return sum / count;
}

/// @brief Runs all episodes with the given delay and returns the difference between the mean strength around A and around B.
static bhm_error_code_t run(
    bhm_ticks_count_t delay,
    bhm_bool_t modulated,
    uint32_t seed,
    double* result
) {
    bhm_error_code_t error;

    // Context init.
    bhm_context2d_t* context;
    error = bhm_ctx2d_create(&context, REWARD_CORTEX_WIDTH, REWARD_CORTEX_HEIGHT, REWARD_NH_RADIUS);
    if (error != BHM_ERROR_NONE) return error;

    // Evolve at every tick, so that the reward is applied exactly [delay] ticks after the stimulation ends.
    bhm_ctx2d_set_evol_step(context, 0x00U);
    error = bhm_ctx2d_set_elig_window(context, BHM_MAX_ELIG_WINDOW);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_set_modstr_chance(context, modulated ? REWARD_MODSTR_CHANCE : 0x00U);
    if (error != BHM_ERROR_NONE) return error;
    context->even_cortex->synstr_chance = REWARD_SYNSTR_CHANCE;
    context->odd_cortex->synstr_chance = REWARD_SYNSTR_CHANCE;
    apply_seed(context->even_cortex, seed);
    apply_seed(context->odd_cortex, seed);

    // Inputs init: A on the left and B on the right, symmetric around the cortex' center.
    bhm_input2d_t* inputs[2];
    for (uint32_t i = 0; i < 2; i++) {
        bhm_cortex_size_t center_x = (REWARD_CORTEX_WIDTH / 2) + (i == 0 ? -REWARD_PATCH_OFFSET : REWARD_PATCH_OFFSET);
        error = bhm_i2d_create(
            &(inputs[i]),
            center_x - (REWARD_PATCH_SIZE / 2),
            (REWARD_CORTEX_HEIGHT / 2) - (REWARD_PATCH_SIZE / 2),
            center_x + (REWARD_PATCH_SIZE / 2),
            (REWARD_CORTEX_HEIGHT / 2) + (REWARD_PATCH_SIZE / 2),
            BHM_DEFAULT_EXC_VALUE * 2,
            BHM_PULSE_MAPPING_LINEAR
        );
        if (error != BHM_ERROR_NONE) return error;

        set_input(inputs[i], context->even_cortex->sample_window, BHM_FALSE);
        error = bhm_ctx2d_add_input(context, inputs[i]);
        if (error != BHM_ERROR_NONE) return error;
    }

    // Run all episodes.
    for (uint32_t episode = 0; episode < REWARD_EPISODES_COUNT; episode++) {
        // Even episodes stimulate A (rewarded), odd episodes stimulate B (punished).
        uint32_t stimulated = episode % 2;
        uint32_t episode_ticks = REWARD_STIM_TICKS + delay + REWARD_GAP_TICKS;

        for (uint32_t t = 0; t < episode_ticks; t++) {
            set_input(inputs[stimulated], context->even_cortex->sample_window, t < REWARD_STIM_TICKS);

            // Deliver the reward or punishment [delay] ticks after the stimulation ended.
            if (t == REWARD_STIM_TICKS + delay) {
                bhm_ctx2d_add_modulation(context, stimulated == 0 ? BHM_MAX_MODULATION : -BHM_MAX_MODULATION);
            }

            error = ctx2d_tick(context);
            if (error != BHM_ERROR_NONE) return error;
        }
    }

    bhm_cortex2d_t* cortex = context->counts->ticks_count % 2 ? context->odd_cortex : context->even_cortex;
    *result = region_mean_strength(cortex, inputs[0]) - region_mean_strength(cortex, inputs[1]);

    // The context frees the inputs' values, but not the inputs themselves.
    bhm_ctx2d_destroy(context);
    free(inputs[0]);
    free(inputs[1]);

    return BHM_ERROR_NONE;
}

int main(int argc, char **argv) {
    bhm_ticks_count_t delays[] = {0, 5, 10, 20, 30, 40, 50, 55, 60, 65, 80, 120};
    uint32_t delays_count = sizeof(delays) / sizeof(bhm_ticks_count_t);

    printf("Eligibility window: %d ticks, %d seeds per delay\n", BHM_MAX_ELIG_WINDOW, REWARD_SEEDS_COUNT);
    printf("A - B = mean tot_syn_strength around A (rewarded) minus around B (punished), mean of all seeds (min..max)\n\n");
    printf(" delay | control A - B          | modulated A - B        | effect\n");
    printf("-------+------------------------+------------------------+--------\n");

    for (uint32_t d = 0; d < delays_count; d++) {
        double results[2][REWARD_SEEDS_COUNT];

        for (uint32_t modulated = 0; modulated < 2; modulated++) {
            for (uint32_t seed = 0; seed < REWARD_SEEDS_COUNT; seed++) {
                bhm_error_code_t error = run(delays[d], modulated, seed + 1, &(results[modulated][seed]));
                if (error != BHM_ERROR_NONE) {
                    printf("There was an error running delay %d: %d\n", delays[d], error);
                    return 1;
                }
            }
        }

        double means[2];
        printf(" %5d", delays[d]);
        for (uint32_t modulated = 0; modulated < 2; modulated++) {
            double sum = 0.0;
            double min = results[modulated][0];
            double max = results[modulated][0];
            for (uint32_t seed = 0; seed < REWARD_SEEDS_COUNT; seed++) {
                sum += results[modulated][seed];
                if (results[modulated][seed] < min) min = results[modulated][seed];
                if (results[modulated][seed] > max) max = results[modulated][seed];
            }
            means[modulated] = sum / REWARD_SEEDS_COUNT;
            printf(" | %+6.2f (%+6.2f..%+6.2f)", means[modulated], min, max);
        }
        printf(" | %+6.2f\n", means[1] - means[0]);
    }

    return 0;
}
