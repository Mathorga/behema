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

    bhm_cortex_counts_t* counts = (bhm_cortex_counts_t*) malloc(sizeof(bhm_cortex_counts_t));
    counts->ticks_count = 0x00;
    counts->evols_count = 0x00;

    // Cortex setup.
    c2d_set_evol_step(even_cortex, 0x01U);
    c2d_set_pulse_mapping(even_cortex, BHM_PULSE_MAPPING_RPROP);
    c2d_set_max_syn_count(even_cortex, 24);
    // char touchFileName[40];
    // char inhexcFileName[40];
    // snprintf(touchFileName, 40, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    // snprintf(inhexcFileName, 40, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);
    // c2d_touch_from_map(even_cortex, touchFileName);
    // c2d_inhexc_from_map(even_cortex, inhexcFileName);
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
        printf("There was an error allocating input %d\n", error);
        return 1;
    }

    // Only set input values once.
    for (bhm_cortex_size_t i = 0; i < input_width * input_height; i++) {
        input->values[i] = even_cortex->sample_window - 1;
    }

    uint64_t start_time = millis();

    for (uint32_t i = 0; i < iterations_count; i++) {
        bhm_cortex2d_t* prev_cortex = i % 2 ? odd_cortex : even_cortex;
        bhm_cortex2d_t* next_cortex = i % 2 ? even_cortex : odd_cortex;

        // Defines whether to evolve or not.
        // evol_step is incremented by 1 to account for edge cases and human readable behavior:
        // 0x0000 -> 0 + 1 = 1, so the cortex evolves at every tick, meaning that there are no free ticks between evolutions.
        // 0xFFFF -> 65535 + 1 = 65536, so the cortex never evolves, meaning that there is an infinite amount of ticks between evolutions.
        bhm_bool_t evolve = (counts->ticks_count % (((bhm_evol_step_t) prev_cortex->evol_step) + 1)) == 0;

        // Feed.
        c2d_feed2d(
            prev_cortex,
            input,
            counts->ticks_count
        );

        c2d_tick(
            prev_cortex,
            next_cortex,
            evolve
        );

        counts->ticks_count++;
        // Increment evolutions count.
        if (evolve) counts->evols_count++;

        if ((i + 1) % 100 == 0) {
            uint64_t elapsed = millis() - start_time;
            double frequency = i /(elapsed / 1000.0f);
            printf("\nPerformed %d iterations in %llums; %.2f Hz\n", i + 1, elapsed, frequency);
            // c2d_to_file(even_cortex, (char*) "out/test.c2d");
        }

        // usleep(100);
    }

    // Copy the cortex back to host to check the results.
    // c2d_to_file(even_cortex, (char*) "out/test.c2d");

    // Cleanup.
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    i2d_destroy(input);

    return 0;
}
