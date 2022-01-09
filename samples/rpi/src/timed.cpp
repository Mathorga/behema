#include <portia/portia.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <iostream>

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    double slope = ((double) output_end - (double) output_start) / ((double) input_end - (double) input_start);
    return (double) output_start + slope * ((double) input - (double) input_start);
}

void initPositions(cortex2d_t* cortex, float* xNeuronPositions, float* yNeuronPositions) {
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            xNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) x) + 0.5f) / (float) cortex->width;
            yNeuronPositions[IDX2D(x, y, cortex->width)] = (((float) y) + 0.5f) / (float) cortex->height;
        }
    }
}

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 100;
    cortex_size_t cortex_height = 60;
    nh_radius_t nh_radius = 2;

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
            printf("USAGE: timed <width> <height> <nh_radius>\n");
            exit(0);
            break;
    }

    srand(time(NULL));

    // Create network model.
    cortex2d_t even_cortex;
    cortex2d_t odd_cortex;
    error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    if (error != 0) {
        printf("Error %d during init\n", error);
        exit(1);
    }
    c2d_set_evol_step(&even_cortex, 0x01U);
    c2d_set_pulse_mapping(&even_cortex, PULSE_MAPPING_LINEAR);
    c2d_copy(&odd_cortex, &even_cortex);

    float* xNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));
    float* yNeuronPositions = (float*) malloc(cortex_width * cortex_height * sizeof(float));

    initPositions(&even_cortex, xNeuronPositions, yNeuronPositions);
    
    time_t startTime = time(NULL);
    time_t endTime = time(NULL);
    
    for (int i = 0;; i++) {
        cortex2d_t* prev_cortex = i % 2 ? &odd_cortex : &even_cortex;
        cortex2d_t* next_cortex = i % 2 ? &even_cortex : &odd_cortex;

        if (i % 1000 == 0) {
            endTime = time(NULL);
 
            char fileName[100];
            snprintf(fileName, 100, "out/%lu.c2d", (unsigned long) time(NULL));
            c2d_to_file(prev_cortex, fileName);
            printf("Time passed: %ld, saved file %s\n", endTime - startTime, fileName);

            // startTime = time(NULL);
        }

        c2d_sqfeed(prev_cortex, 0, 20, 1, 30, DEFAULT_EXCITING_VALUE / 2);
        c2d_sqfeed(prev_cortex, cortex_width - 1, 20, cortex_width, 30, DEFAULT_EXCITING_VALUE / 2);

        // usleep(5000);

        // Tick the cortex.
        c2d_tick(prev_cortex, next_cortex);
    }
    
    return 0;
}