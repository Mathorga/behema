#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <portia/portia.h>

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 150;
    cortex_size_t cortex_height = 80;
    nh_radius_t nh_radius = 2;
    cortex_size_t inputs_count = 151;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            cortex_width = atoi(argv[1]);
            break;
        case 3:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            break;
        case 4:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            break;
        case 5:
            cortex_width = atoi(argv[1]);
            cortex_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            inputs_count = atoi(argv[4]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;

    ticks_count_t* inputs = (ticks_count_t*) malloc(inputs_count * sizeof(ticks_count_t));
    ticks_count_t sample_rate = 10;
    ticks_count_t samples_count = 0;

    srand(time(NULL));

    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    odd_cortex = *c2d_copy(&even_cortex);

    for (int i = 0; i < 1000; i++) {
        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        // Only get new inputs according to the sample rate.
        if (i % sample_rate == 0) {
            // Fetch input.
            for (cortex_size_t j = 0; j < inputs_count; j++) {
                inputs[j] = 1 + (rand() % (sample_rate - 1));
            }
            samples_count = 0;
        }

        // Feed the cortex.
        for (cortex_size_t k = 0; k < inputs_count; k++) {
            if (samples_count % inputs[k]) {
                prev_cortex->neurons[k].value += DEFAULT_EXCITING_VALUE;
            }
        }

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);

        samples_count++;
    }
}
