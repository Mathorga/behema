#include "tests.h"

// Random states whose bit 16 selects the mutation direction in bhm_mutate_value.
#define TEST_RAND_INCREASE 0x00000000U
#define TEST_RAND_DECREASE 0x00010000U

int test_mutate_value() {
    // Steps are about 1/8 of the value.
    BHM_ASSERT_EQ(bhm_mutate_value(800, 0, 0xFFFF, TEST_RAND_INCREASE), 900);
    BHM_ASSERT_EQ(bhm_mutate_value(800, 0, 0xFFFF, TEST_RAND_DECREASE), 700);

    // Small values still change, by at least 1.
    BHM_ASSERT_EQ(bhm_mutate_value(3, 0, 0xFFFF, TEST_RAND_INCREASE), 4);
    BHM_ASSERT_EQ(bhm_mutate_value(0, 0, 0xFFFF, TEST_RAND_INCREASE), 1);

    // Limits are respected, without wrapping around.
    BHM_ASSERT_EQ(bhm_mutate_value(0, 0, 0xFFFF, TEST_RAND_DECREASE), 0);
    BHM_ASSERT_EQ(bhm_mutate_value(2, 1, 0xFFFF, TEST_RAND_DECREASE), 1);
    BHM_ASSERT_EQ(bhm_mutate_value(62, 1, 63, TEST_RAND_INCREASE), 63);
    BHM_ASSERT_EQ(bhm_mutate_value(63, 1, 63, TEST_RAND_INCREASE), 63);

    return 0;
}

int test_mutation_happens_and_stays_in_range() {
    bhm_cortex2d_t* cortex;
    BHM_ASSERT_EQ(bhm_crx2d_create(&cortex, 8, 8, 2), BHM_ERROR_NONE);

    // With a 0 chance, nothing changes.
    bhm_chance_t syngen_chance = cortex->syngen_chance;
    for (uint32_t i = 0; i < 100; i++) {
        BHM_ASSERT_EQ(bhm_crx2d_mutate(cortex, 0x0000U, BHM_FALSE), BHM_ERROR_NONE);
    }
    BHM_ASSERT_EQ(cortex->syngen_chance, syngen_chance);

    // With a 50% chance, after many mutations every property has changed at some point, and all stayed in their valid ranges.
    bhm_bool_t syngen_changed = BHM_FALSE;
    bhm_bool_t neuron_changed = BHM_FALSE;
    for (uint32_t i = 0; i < 1000; i++) {
        BHM_ASSERT_EQ(bhm_crx2d_mutate(cortex, 0x8000U, BHM_FALSE), BHM_ERROR_NONE);

        if (cortex->syngen_chance != syngen_chance) syngen_changed = BHM_TRUE;
        if (cortex->neurons[0].max_syn_count != cortex->max_syn_count) neuron_changed = BHM_TRUE;

        BHM_ASSERT(cortex->pulse_window >= 0x01U && cortex->pulse_window < sizeof(bhm_pulse_mask_t) * 8);
        BHM_ASSERT(cortex->elig_window >= 0x01U && cortex->elig_window <= BHM_MAX_ELIG_WINDOW);
        BHM_ASSERT(cortex->syngen_chance <= BHM_MAX_SYNGEN_CHANCE);
        BHM_ASSERT(cortex->synstr_chance <= BHM_MAX_SYNSTR_CHANCE);
        BHM_ASSERT(cortex->modstr_chance <= BHM_MAX_MODSTR_CHANCE);
        BHM_ASSERT(cortex->max_tot_strength >= 0x01U);
        BHM_ASSERT(cortex->max_syn_count <= BHM_NH_COUNT_2D(BHM_NH_DIAM_2D(cortex->nh_radius)));
        BHM_ASSERT(cortex->neurons[0].max_syn_count <= sizeof(bhm_nh_mask_t) * 8);
        BHM_ASSERT(cortex->neurons[0].inhexc_ratio <= BHM_MAX_INHEXC_RANGE);
    }
    BHM_ASSERT(syngen_changed);
    BHM_ASSERT(neuron_changed);

    bhm_crx2d_destroy(cortex);

    return 0;
}

/// @brief Dummy evaluation function: the fitness is the cortex' syngen chance, so selection is predictable.
static bhm_error_code_t eval_syngen(
    bhm_cortex2d_t* cortex,
    bhm_cortex_fitness_t* fitness
) {
    *fitness = (bhm_cortex_fitness_t) cortex->syngen_chance;

    return BHM_ERROR_NONE;
}

int test_breed_inherits_from_parents() {
    bhm_population2d_t* population;
    BHM_ASSERT_EQ(p2d_init(&population, 4, 2, 0x0000U, eval_syngen), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(p2d_populate(population, 8, 8, 2), BHM_ERROR_NONE);

    // Give every cortex distinct values, so that each child property can be traced back to a parent.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        bhm_cortex2d_t* cortex = &(population->cortices[i]);
        cortex->syngen_chance = 0x0100U * (i + 1);
        cortex->modstr_chance = 0x0200U * (i + 1);
        cortex->elig_window = 0x04U * (i + 1);
        cortex->max_tot_strength = 0x10U + i;
        BHM_ASSERT_EQ(bhm_crx2d_set_max_syn_count(cortex, 0x02U + i), BHM_ERROR_NONE);
    }

    // The fittest two are cortices 3 and 2 (highest syngen chance).
    BHM_ASSERT_EQ(p2d_evaluate(population), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(p2d_select(population), BHM_ERROR_NONE);

    for (uint32_t i = 0; i < 20; i++) {
        bhm_cortex2d_t* child;
        BHM_ASSERT_EQ(p2d_breed(population, &child), BHM_ERROR_NONE);

        // Each property comes from one of the two selected parents.
        BHM_ASSERT(child->modstr_chance == 0x0600U || child->modstr_chance == 0x0800U);
        BHM_ASSERT(child->elig_window == 0x0CU || child->elig_window == 0x10U);
        BHM_ASSERT(child->max_tot_strength == 0x12U || child->max_tot_strength == 0x13U);

        // max_syn_count comes from a parent's max_syn_count (it used to come from max_tot_strength), and reaches the neurons.
        BHM_ASSERT(child->max_syn_count == 0x04U || child->max_syn_count == 0x05U);
        BHM_ASSERT(child->neurons[0].max_syn_count == 0x04U || child->neurons[0].max_syn_count == 0x05U);

        bhm_crx2d_destroy(child);
    }

    p2d_destroy(population);

    return 0;
}

int test_set_max_syn_count_reaches_neurons() {
    bhm_context2d_t* context;
    BHM_ASSERT_EQ(bhm_ctx2d_create(&context, 8, 8, 2), BHM_ERROR_NONE);

    BHM_ASSERT_EQ(bhm_ctx2d_set_max_syn_count(context, 0x0AU), BHM_ERROR_NONE);

    for (bhm_cortex_size_t i = 0; i < 8 * 8; i++) {
        BHM_ASSERT_EQ(context->even_cortex->neurons[i].max_syn_count, 0x0AU);
        BHM_ASSERT_EQ(context->odd_cortex->neurons[i].max_syn_count, 0x0AU);
    }

    bhm_ctx2d_destroy(context);

    return 0;
}
