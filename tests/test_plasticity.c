#include "tests.h"

#define TEST_CORTEX_WIDTH 32
#define TEST_CORTEX_HEIGHT 32
#define TEST_NH_RADIUS 2
#define TEST_INPUT_SIZE 8
#define TEST_TICKS_COUNT 1000

// Raised from the default (0x00A0U) so that many synapses change strength in the same tick, which is where the bookkeeping can go wrong.
#define TEST_SYNSTR_CHANCE 0x0800U

// High enough that reward-modulated plasticity changes many synapses, when enabled.
#define TEST_MODSTR_CHANCE 0x8000U
// Ticks between switches from reward to punishment and back, when reward-modulated plasticity is enabled.
#define TEST_MODULATION_PERIOD 50

/// @brief Computes the actual sum of the strengths of all active synapses of the given neuron, by reading its masks.
/// @param neuron The neuron to read.
/// @return The sum of the strengths of all active synapses.
static uint32_t real_tot_syn_strength(
    bhm_neuron_t* neuron
) {
    uint32_t result = 0x00U;

    for (uint32_t i = 0; i < sizeof(bhm_nh_mask_t) * 8; i++) {
        // Only active synapses count.
        if ((neuron->synac_mask >> i) & 0x01U) {
            result += ((neuron->synstr_mask_a >> i) & 0x01U) |
                      (((neuron->synstr_mask_b >> i) & 0x01U) << 0x01U) |
                      (((neuron->synstr_mask_c >> i) & 0x01U) << 0x02U);
        }
    }

    return result;
}

/// @brief Counts the bits set in the given mask.
/// @param mask The mask to read.
/// @return The amount of bits set in the mask.
static uint32_t count_set_bits(
    bhm_nh_mask_t mask
) {
    uint32_t result = 0x00U;

    while (mask) {
        result += mask & 0x01U;
        mask >>= 0x01U;
    }

    return result;
}

/// @brief Creates a small context with a constantly active input patch in its center, then runs it for [TEST_TICKS_COUNT] ticks, evolving at every tick.
/// @param context The context to create and run.
/// @param input The input to create, feeding the context.
/// @param modulated Whether to also enable reward-modulated plasticity, with a modulation switching between max reward and max punishment
/// every [TEST_MODULATION_PERIOD] ticks.
/// @return The code for the occurred error, [BHM_ERROR_NONE] if none.
static bhm_error_code_t run_evolving_context(
    bhm_context2d_t** context,
    bhm_input2d_t** input,
    bhm_bool_t modulated
) {
    bhm_error_code_t error;

    // Context init.
    error = bhm_ctx2d_create(context, TEST_CORTEX_WIDTH, TEST_CORTEX_HEIGHT, TEST_NH_RADIUS);
    if (error != BHM_ERROR_NONE) return error;

    // Context setup: evolve at every tick and make strength changes frequent.
    bhm_ctx2d_set_evol_step(*context, 0x00U);
    (*context)->even_cortex->synstr_chance = TEST_SYNSTR_CHANCE;
    (*context)->odd_cortex->synstr_chance = TEST_SYNSTR_CHANCE;
    if (modulated) {
        bhm_ctx2d_set_modstr_chance(*context, TEST_MODSTR_CHANCE);
    }

    // Input init: a square patch in the center of the cortex.
    error = bhm_i2d_create(
        input,
        (TEST_CORTEX_WIDTH / 2) - (TEST_INPUT_SIZE / 2),
        (TEST_CORTEX_HEIGHT / 2) - (TEST_INPUT_SIZE / 2),
        (TEST_CORTEX_WIDTH / 2) + (TEST_INPUT_SIZE / 2),
        (TEST_CORTEX_HEIGHT / 2) + (TEST_INPUT_SIZE / 2),
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_LINEAR
    );
    if (error != BHM_ERROR_NONE) return error;

    // Max input values, so that the input neurons are excited at every tick and activity spreads from them.
    for (bhm_cortex_size_t i = 0; i < TEST_INPUT_SIZE * TEST_INPUT_SIZE; i++) {
        (*input)->values[i] = (*context)->even_cortex->sample_window - 1;
    }

    error = bhm_ctx2d_add_input(*context, *input);
    if (error != BHM_ERROR_NONE) return error;

    // Run.
    for (uint32_t i = 0; i < TEST_TICKS_COUNT; i++) {
        if (modulated) {
            // Alternate between periods of reward and periods of punishment, so that synapses are both strengthened and weakened.
            bhm_ctx2d_add_modulation(*context, (i / TEST_MODULATION_PERIOD) % 2 ? -BHM_MAX_MODULATION : BHM_MAX_MODULATION);
        }

        error = ctx2d_tick(*context);
        if (error != BHM_ERROR_NONE) return error;
    }

    return BHM_ERROR_NONE;
}

/// @brief Returns the cortex holding the most recent state of the given context.
/// @param context The context to read.
/// @return The cortex written by the last tick.
static bhm_cortex2d_t* current_cortex(
    bhm_context2d_t* context
) {
    // Mirrors the even/odd selection in ctx2d_tick: the cortex that will be read at the next tick is the one written by the last one.
    return context->counts->ticks_count % 2 ? context->odd_cortex : context->even_cortex;
}

int test_syn_count_matches_synac_mask() {
    bhm_context2d_t* context;
    bhm_input2d_t* input;

    BHM_ASSERT_EQ(run_evolving_context(&context, &input, BHM_FALSE), BHM_ERROR_NONE);

    bhm_cortex2d_t* cortex = current_cortex(context);

    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            BHM_ASSERT_EQ(neuron->syn_count, count_set_bits(neuron->synac_mask));
        }
    }

    bhm_ctx2d_destroy(context);

    return 0;
}

int test_tot_syn_strength_matches_masks() {
    bhm_context2d_t* context;
    bhm_input2d_t* input;

    BHM_ASSERT_EQ(run_evolving_context(&context, &input, BHM_FALSE), BHM_ERROR_NONE);

    bhm_cortex2d_t* cortex = current_cortex(context);

    // Make sure plasticity actually happened, otherwise the test would pass trivially.
    uint32_t synapses_count = 0x00U;
    for (bhm_cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
        synapses_count += cortex->neurons[i].syn_count;
    }
    BHM_ASSERT(synapses_count > 0x00U);

    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            BHM_ASSERT_EQ(neuron->tot_syn_strength, real_tot_syn_strength(neuron));
        }
    }

    bhm_ctx2d_destroy(context);

    return 0;
}

int test_syn_count_within_max() {
    bhm_context2d_t* context;
    bhm_input2d_t* input;

    BHM_ASSERT_EQ(run_evolving_context(&context, &input, BHM_FALSE), BHM_ERROR_NONE);

    bhm_cortex2d_t* cortex = current_cortex(context);

    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            BHM_ASSERT(neuron->syn_count <= neuron->max_syn_count);
        }
    }

    bhm_ctx2d_destroy(context);

    return 0;
}

int test_tot_syn_strength_within_max() {
    bhm_context2d_t* context;
    bhm_input2d_t* input;

    BHM_ASSERT_EQ(run_evolving_context(&context, &input, BHM_FALSE), BHM_ERROR_NONE);

    bhm_cortex2d_t* cortex = current_cortex(context);

    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            BHM_ASSERT(neuron->tot_syn_strength <= cortex->max_tot_strength);
        }
    }

    bhm_ctx2d_destroy(context);

    return 0;
}

int test_modulation_keeps_bookkeeping() {
    bhm_context2d_t* context;
    bhm_input2d_t* input;

    BHM_ASSERT_EQ(run_evolving_context(&context, &input, BHM_TRUE), BHM_ERROR_NONE);

    bhm_cortex2d_t* cortex = current_cortex(context);

    // Same invariants as the tests above, all checked on a run with reward-modulated plasticity enabled.
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t* neuron = &(cortex->neurons[BHM_IDX2D(x, y, cortex->width)]);

            BHM_ASSERT_EQ(neuron->syn_count, count_set_bits(neuron->synac_mask));
            BHM_ASSERT_EQ(neuron->tot_syn_strength, real_tot_syn_strength(neuron));
            BHM_ASSERT(neuron->syn_count <= neuron->max_syn_count);
            BHM_ASSERT(neuron->tot_syn_strength <= cortex->max_tot_strength);
        }
    }

    bhm_ctx2d_destroy(context);

    return 0;
}
