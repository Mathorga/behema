#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <behema/behema.h>

int main(int argc, char **argv) {
    bhm_cortex_size_t cortex_width = 256;
    bhm_cortex_size_t cortex_height = 128;
    bhm_cortex_size_t input_width = 4;
    bhm_cortex_size_t input_height = 1;
    bhm_cortex_size_t output_width = 4;
    bhm_cortex_size_t output_height = 1;
    uint32_t iterations_count = 10000;
    bhm_nh_radius_t nh_radius = 2;
    bhm_ticks_count_t mean_output = 0;

    srand(time(NULL));

    bhm_error_code_t error;

    // Cortex init.
    bhm_cortex2d_t* even_cortex;
    bhm_cortex2d_t* odd_cortex;
    error = c2d_create(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing the even cortex %d\n", error);
        return 1;
    }
    error = c2d_create(&odd_cortex, cortex_width, cortex_height, nh_radius);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing the odd cortex %d\n", error);
        return 1;
    }

    // Cortex setup.
    c2d_set_evol_step(even_cortex, 0x01U);
    c2d_set_pulse_mapping(even_cortex, BHM_PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(even_cortex, 24);
    char touchFileName[40];
    char inhexcFileName[40];
    snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    c2d_touch_from_map(even_cortex, touchFileName);
    c2d_inhexc_from_map(even_cortex, inhexcFileName);
    c2d_copy(odd_cortex, even_cortex);

    // Print the cortex out.
    char cortex_string[100];
    c2d_to_string(even_cortex, cortex_string);
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
        printf("There was an error initializing input %d\n", error);
        return 1;
    }

    // Output init.
    bhm_output2d_t* output;
    error = o2d_init(
        &output,
        (cortex_width / 2) - (output_width / 2),
        cortex_height - 1 - output_height,
        (cortex_width / 2) + (output_width / 2),
        cortex_height - 1
    );
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing output %d\n", error);
        return 1;
    }

    // Only set input values once.
    for (bhm_cortex_size_t i = 0; i < input_width * input_height; i++) {
        input->values[i] = even_cortex->sample_window - 1;
    }

    printf("\n");

    // Main loop.
    for (uint32_t i = 0; i < iterations_count; i++) {
        bhm_cortex2d_t* prev_cortex = i % 2 ? odd_cortex : even_cortex;
        bhm_cortex2d_t* next_cortex = i % 2 ? even_cortex : odd_cortex;

        // TODO Fetch input.

        // Feed input.
        c2d_feed2d(prev_cortex, input);

        c2d_tick(prev_cortex, next_cortex);

        if ((i + 1) % 10 == 0) {
            // Read output.
            c2d_read2d(prev_cortex, output);

            // Compute output mean.
            o2d_mean(output, &mean_output);

            printf("\rPerformed %d iterations - mean output: %d", i + 1, mean_output);
            fflush(stdout);
        }
    }
    
    printf("\n");

    // Copy the cortex back to host to check the results.
    c2d_to_file(even_cortex, (char*) "out/test.c2d");

    // Cleanup.
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);
    o2d_destroy(output);

    return 0;
}
