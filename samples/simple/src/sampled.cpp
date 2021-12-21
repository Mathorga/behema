#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <portia/portia.h>

void print(field2d_t* field) {
    system("clear");
    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            neuron_t currentNeuron = field->neurons[IDX2D(x, y, field->width)];
            printf("%c ", currentNeuron.value >= field->fire_threshold ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    field_size_t field_width = 100;
    field_size_t field_height = 60;
    nh_radius_t nh_radius = 2;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            field_width = atoi(argv[1]);
            break;
        case 3:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            break;
        case 4:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    field2d_t even_field;
    field2d_t odd_field;
    f2d_init(&even_field, field_width, field_height, nh_radius);
    f2d_set_evol_step(&even_field, 0x20U);
    f2d_set_pulse_window(&even_field, 0x3A);
    f2d_set_syngen_beat(&even_field, 0.1F);
    f2d_set_max_touch(&even_field, 0.2F);
    f2d_set_sample_window(&even_field, 10);
    odd_field = *f2d_copy(&even_field);

    field_size_t lInputsCoords[] = {10, 5, 40, 20};
    field_size_t rInputsCoords[] = {even_field.width - 40, 5, even_field.width - 10, 20};

    ticks_count_t* lInputs = (ticks_count_t*) malloc((lInputsCoords[2] - lInputsCoords[0]) * (lInputsCoords[3] - lInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t* rInputs = (ticks_count_t*) malloc((rInputsCoords[2] - rInputsCoords[0]) * (rInputsCoords[3] - rInputsCoords[1]) * sizeof(ticks_count_t));
    ticks_count_t sample_step = even_field.sample_window;

    srand(time(NULL));

    for (int i = 0;; i++) {
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        // Only get new inputs according to the sample rate.
        if (sample_step == prev_field->sample_window) {
            // Fetch input.
            for (field_size_t y = lInputsCoords[1]; y < lInputsCoords[3]; y++) {
                for (field_size_t x = lInputsCoords[0]; x < lInputsCoords[2]; x++) {
                    lInputs[IDX2D(x - lInputsCoords[0], y - lInputsCoords[1], lInputsCoords[2] - lInputsCoords[0])] = (rand() % (prev_field->sample_window - 1));
                }
            }

            for (field_size_t y = rInputsCoords[1]; y < rInputsCoords[3]; y++) {
                for (field_size_t x = rInputsCoords[0]; x < rInputsCoords[2]; x++) {
                    rInputs[IDX2D(x - rInputsCoords[0], y - rInputsCoords[1], rInputsCoords[2] - rInputsCoords[0])] = (rand() % (prev_field->sample_window - 1));
                }
            }
            sample_step = 0;
        }

        // Feed the field.
        f2d_sample_sqfeed(prev_field, lInputsCoords[0], lInputsCoords[1], lInputsCoords[2], lInputsCoords[3], sample_step, lInputs, DEFAULT_CHARGE_VALUE);
        f2d_sample_sqfeed(prev_field, rInputsCoords[0], rInputsCoords[1], rInputsCoords[2], rInputsCoords[3], sample_step, rInputs, DEFAULT_CHARGE_VALUE);

        print(next_field);

        // Tick the field.
        f2d_tick(prev_field, next_field);

	sample_step++;

        usleep(10000);
    }
}
