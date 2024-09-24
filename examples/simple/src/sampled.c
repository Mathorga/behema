#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <behema/behema.h>

void print(bhm_cortex2d_t* cortex) {
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            bhm_neuron_t currentNeuron = cortex->neurons[IDX2D(x, y, cortex->width)];
            printf("%c ", currentNeuron.value >= cortex->fire_threshold ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    bhm_cortex_size_t cortex_width = 100;
    bhm_cortex_size_t cortex_height = 60;
    bhm_nh_radius_t nh_radius = 2;
    bhm_ticks_count_t sampleWindow = 10;
    bhm_ticks_count_t samplingBound = sampleWindow - 1;

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
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    bhm_cortex2d_t even_cortex;
    bhm_cortex2d_t odd_cortex;
    c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    c2d_set_evol_step(&even_cortex, 0x20U);
    c2d_set_pulse_window(&even_cortex, 0x3A);
    c2d_set_syngen_beat(&even_cortex, 0.1F);
    c2d_set_max_touch(&even_cortex, 0.2F);
    c2d_set_sample_window(&even_cortex, sampleWindow);
    c2d_copy(&odd_cortex, &even_cortex);

    bhm_cortex_size_t lInputsCoords[] = {10, 5, 40, 20};
    bhm_cortex_size_t rInputsCoords[] = {even_cortex.width - 40, 5, even_cortex.width - 10, 20};

    bhm_ticks_count_t* lInputs = (bhm_ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(bhm_ticks_count_t));
    bhm_ticks_count_t* rInputs = (bhm_ticks_count_t*) malloc((rInputsCoords[2] - rInputsCoords[0]) * (rInputsCoords[3] - rInputsCoords[1]) * sizeof(bhm_ticks_count_t));
    bhm_ticks_count_t sample_step = samplingBound;

    srand(time(NULL));

    for (int i = 0;; i++) {
        bhm_cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        bhm_cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        system("clear");

        // Only get new inputs according to the sample rate.
        if (sample_step > samplingBound) {
            // Fetch input.
            for (bhm_cortex_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (bhm_cortex_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])] = (rand() % (samplingBound));
                }
            }

            for (bhm_cortex_size_t y = rInputsCoords[1]; y < rInputsCoords[3]; y++) {
                for (bhm_cortex_size_t x = rInputsCoords[0]; x < rInputsCoords[2]; x++) {
                    rInputs[IDX2D(x - rInputsCoords[0], y - rInputsCoords[1], rInputsCoords[2] - rInputsCoords[0])] = (rand() % (samplingBound));
                }
            }

            sample_step = 0;
        }

        printf("%d\n", sample_step);

        // Feed the cortex.
        c2d_sample_sqfeed(prev_cortex, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, BHM_DEFAULT_EXC_VALUE);
        c2d_sample_sqfeed(prev_cortex, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, BHM_DEFAULT_EXC_VALUE);

        sample_step++;

        print(next_cortex);

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);

        usleep(1000000);
    }
}
