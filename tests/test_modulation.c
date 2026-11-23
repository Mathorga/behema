#include "tests.h"

// Neuron receiving the synapse (output) and its neighbor sending it (input), in a 4x4 cortex with radius 1.
// The neighbor is on the left of the neuron, so its index in the neuron's 3x3 neighborhood is (0, 1) -> 3.
#define TEST_NEURON_X 1
#define TEST_NEURON_Y 1
#define TEST_NEIGHBOR_X 0
#define TEST_NEIGHBOR_Y 1
#define TEST_NEIGHBOR_NH_INDEX 3

/// @brief Reads the strength of the synapse at the given neighborhood index.
static uint32_t syn_strength_at(
    bhm_neuron_t* neuron,
    uint32_t nh_index
) {
    return ((neuron->synstr_mask_a >> nh_index) & 0x01U) |
           (((neuron->synstr_mask_b >> nh_index) & 0x01U) << 0x01U) |
           (((neuron->synstr_mask_c >> nh_index) & 0x01U) << 0x02U);
}

/// @brief Builds a tiny cortex with a single synapse from the neighbor to the neuron, with the given strength and pulse masks,
/// runs a single evolving tick with the given modulation, and returns the synapse strength after the tick.
/// All other kinds of plasticity are disabled, so any change can only come from reward-modulated plasticity.
static bhm_error_code_t run_single_tick(
    bhm_syn_strength_t strength,
    bhm_pulse_mask_t neighbor_pulse_mask,
    bhm_pulse_mask_t neuron_pulse_mask,
    bhm_modulation_t modulation,
    uint32_t* result_strength
) {
    bhm_error_code_t error;

    bhm_cortex2d_t* prev_cortex;
    bhm_cortex2d_t* next_cortex;
    error = bhm_crx2d_create(&prev_cortex, 4, 4, 1);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_crx2d_create(&next_cortex, 4, 4, 1);
    if (error != BHM_ERROR_NONE) return error;

    // Disable structural and functional plasticity, and make reward-modulated plasticity as likely as possible.
    prev_cortex->syngen_chance = 0x00U;
    prev_cortex->synstr_chance = 0x00U;
    prev_cortex->modstr_chance = BHM_MAX_MODSTR_CHANCE;
    prev_cortex->max_tot_strength = BHM_MAX_MAX_TOT_STRENGTH;

    // Single synapse from the neighbor to the neuron, with the given strength.
    bhm_neuron_t* neuron = &(prev_cortex->neurons[BHM_IDX2D(TEST_NEURON_X, TEST_NEURON_Y, prev_cortex->width)]);
    neuron->synac_mask = 0x01ULL << TEST_NEIGHBOR_NH_INDEX;
    neuron->synstr_mask_a = (bhm_nh_mask_t) (strength & 0x01U) << TEST_NEIGHBOR_NH_INDEX;
    neuron->synstr_mask_b = (bhm_nh_mask_t) ((strength >> 0x01U) & 0x01U) << TEST_NEIGHBOR_NH_INDEX;
    neuron->synstr_mask_c = (bhm_nh_mask_t) ((strength >> 0x02U) & 0x01U) << TEST_NEIGHBOR_NH_INDEX;
    neuron->syn_count = 0x01U;
    neuron->tot_syn_strength = strength;
    neuron->pulse_mask = neuron_pulse_mask;

    prev_cortex->neurons[BHM_IDX2D(TEST_NEIGHBOR_X, TEST_NEIGHBOR_Y, prev_cortex->width)].pulse_mask = neighbor_pulse_mask;

    crx2d_tick(prev_cortex, next_cortex, BHM_TRUE, modulation);

    bhm_neuron_t* next_neuron = &(next_cortex->neurons[BHM_IDX2D(TEST_NEURON_X, TEST_NEURON_Y, next_cortex->width)]);
    *result_strength = syn_strength_at(next_neuron, TEST_NEIGHBOR_NH_INDEX);

    // Bookkeeping must stay consistent.
    if (next_neuron->tot_syn_strength != *result_strength) return BHM_ERROR_EXTERNAL_CAUSES;

    bhm_crx2d_destroy(prev_cortex);
    bhm_crx2d_destroy(next_cortex);

    return BHM_ERROR_NONE;
}

int test_modulation_rewards_eligible_synapse() {
    uint32_t strength;

    // The neighbor fired 2 ticks ago (bit 1) and the neuron fired right after it, 1 tick ago (bit 0): eligible.
    BHM_ASSERT_EQ(run_single_tick(0x02U, 0x02U, 0x01U, BHM_MAX_MODULATION, &strength), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x03U);

    return 0;
}

int test_modulation_punishes_eligible_synapse() {
    uint32_t strength;

    // Same eligible synapse, negative modulation: weakened.
    BHM_ASSERT_EQ(run_single_tick(0x02U, 0x02U, 0x01U, -BHM_MAX_MODULATION, &strength), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x01U);

    return 0;
}

int test_modulation_ignores_non_eligible_synapses() {
    uint32_t strength;

    // No modulation: unchanged.
    BHM_ASSERT_EQ(run_single_tick(0x02U, 0x02U, 0x01U, 0x00, &strength), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x02U);

    // The neuron fired before the neighbor (wrong order, the input can't have caused the firing): unchanged.
    BHM_ASSERT_EQ(run_single_tick(0x02U, 0x01U, 0x02U, BHM_MAX_MODULATION, &strength), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x02U);

    // Both fired at the same tick (no input before the output): unchanged.
    BHM_ASSERT_EQ(run_single_tick(0x02U, 0x01U, 0x01U, BHM_MAX_MODULATION, &strength), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x02U);

    // The coincidence happened too long ago: the neuron fired BHM_DEFAULT_ELIG_WINDOW ticks ago, just outside the window.
    BHM_ASSERT_EQ(run_single_tick(
        0x02U,
        0x01ULL << (BHM_DEFAULT_ELIG_WINDOW + 1),
        0x01ULL << BHM_DEFAULT_ELIG_WINDOW,
        BHM_MAX_MODULATION,
        &strength
    ), BHM_ERROR_NONE);
    BHM_ASSERT_EQ(strength, 0x02U);

    return 0;
}

int test_modulation_add_clamps() {
    bhm_context2d_t* context;
    BHM_ASSERT_EQ(bhm_ctx2d_create(&context, 4, 4, 1), BHM_ERROR_NONE);

    // Values are summed.
    bhm_ctx2d_add_modulation(context, 0x10);
    bhm_ctx2d_add_modulation(context, -0x04);
    BHM_ASSERT_EQ(context->modulation, 0x0C);

    // The sum is clamped to the max, in both directions.
    bhm_ctx2d_add_modulation(context, 0x7FFF);
    BHM_ASSERT_EQ(context->modulation, BHM_MAX_MODULATION);
    bhm_ctx2d_add_modulation(context, -0x7FFF);
    bhm_ctx2d_add_modulation(context, -0x7FFF);
    BHM_ASSERT_EQ(context->modulation, -BHM_MAX_MODULATION);

    // The modulation is reset after an evolving tick (the first tick always evolves).
    ctx2d_tick(context);
    BHM_ASSERT_EQ(context->modulation, 0x00);

    bhm_ctx2d_destroy(context);

    return 0;
}
