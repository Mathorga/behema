#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <portia/portia.h>

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 1500;
    cortex_size_t cortex_height = 500;
    nh_radius_t nh_radius = 1;

    srand(time(NULL));

    error_code_t error;

    // Cortex init.
    cortex2d_t* even_cortex;
    cortex2d_t* odd_cortex;
    error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    error = c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    c2d_copy(odd_cortex, even_cortex);

    // Input init.
    // input2d_t* input;
    // i2d_init(&input, (cortex_width / 2) - 10, 0, (cortex_width / 2) + 10, 1, DEFAULT_EXC_VALUE * 2, PULSE_MAPPING_FPROP);

    uint64_t start_time = millis();

    for (int i = 0; i < 100; i++) {
        cortex2d_t* prev_cortex = i % 2 ? odd_cortex : even_cortex;
        cortex2d_t* next_cortex = i % 2 ? even_cortex : odd_cortex;

        // TODO Feed.

        printf("\npre_tick\n");
        c2d_tick(prev_cortex, next_cortex);
        printf("\npost_tick\n");

        // usleep(100);
    }

    uint64_t end_time = millis();
    printf("\nCompleted 1000 iterations in %ldms\n", end_time - start_time);

    // Cleanup.
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);

    return 0;
}
