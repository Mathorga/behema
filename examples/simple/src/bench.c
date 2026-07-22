#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <behema/behema.h>

int main(int argc, char **argv) {
    bhm_cortex_size_t cortex_width = 512;
    bhm_cortex_size_t cortex_height = 256;
    bhm_cortex_size_t input_width = 32;
    bhm_cortex_size_t input_height = 1;
    uint32_t iterations_count = 10000;
    bhm_nh_radius_t nh_radius = 2;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            iterations_count = atoi(argv[1]);
            break;
        case 3:
            iterations_count = atoi(argv[1]);
            cortex_width = atoi(argv[2]);
            break;
        case 4:
            iterations_count = atoi(argv[1]);
            cortex_width = atoi(argv[2]);
            cortex_height = atoi(argv[3]);
            break;
        default:
            printf("USAGE: bench <iterations> <width> <height>\n");
            exit(0);
            break;
    }

    bhm_error_code_t error;

    // Cortex init.
    bhm_context2d_t* bhm_ctx;
    error = ctx2d_create(&bhm_ctx, cortex_width, cortex_height, nh_radius);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing the context %d\n", error);
        return 1;
    }

    // Cortex setup.
    ctx2d_set_evol_step(bhm_ctx, 0x01U);
    ctx2d_set_pulse_mapping(bhm_ctx, BHM_PULSE_MAPPING_RPROP);
    ctx2d_set_max_syn_count(bhm_ctx, 24);
    char touchFileName[40];
    char inhexcFileName[40];
    snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    ctx2d_touch_from_map(bhm_ctx, touchFileName);
    ctx2d_inhexc_from_map(bhm_ctx, inhexcFileName);

    // Print the cortex out.
    char cortex_string[100];
    crx2d_to_string(bhm_ctx->even_cortex, cortex_string);
    printf("%s", cortex_string);

    // Input init.
    bhm_input2d_t* input;
    error = i2d_init(
        &input,
        (cortex_width / 2) - (input_width / 2),
        0,
        (cortex_width / 2) + (input_width / 2),
        input_height,
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_FPROP
    );
    if (error != BHM_ERROR_NONE) {
        printf("There was an error allocating input %d\n", error);
        return 1;
    }

    // Only set input values once.
    for (bhm_cortex_size_t i = 0; i < input_width * input_height; i++) {
        input->values[i] = bhm_ctx->even_cortex->sample_window - 1;
    }

    uint64_t start_time = millis();

    for (uint32_t i = 0; i < iterations_count; i++) {
        error = ctx2d_tick(bhm_ctx);
        if (error != BHM_ERROR_NONE) {
            printf("There was an error initializing the context %d\n", error);
            return 1;
        }

        if ((i + 1) % 100 == 0) {
            uint64_t elapsed = millis() - start_time;
            double frequency = i /(elapsed / 1000.0f);
            printf("\nPerformed %d iterations in %llums; %.2f Hz\n", i + 1, elapsed, frequency);
        }
    }

    // crx2d_to_file(even_cortex, (char*) "out/test.c2d");

    // Cleanup.
    error = ctx2d_destroy(bhm_ctx);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error destroying the context: %d\n", error);
        return 1;
    }

    return 0;
}
