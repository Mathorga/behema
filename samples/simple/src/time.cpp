#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <hal/hal.h>

void print(field2d_t* field) {
    system("clear");
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            neuron_t currentNeuron = field->neurons[IDX2D(j, i, field->width)];
            printf("%c ", currentNeuron.value > currentNeuron.threshold ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    field_size_t field_width = 150;
    field_size_t field_height = 80;
    nh_radius_t nh_radius = 2;
    field_size_t inputs_count = 151;

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
        case 5:
            field_width = atoi(argv[1]);
            field_height = atoi(argv[2]);
            nh_radius = atoi(argv[3]);
            inputs_count = atoi(argv[4]);
            break;
        default:
            printf("USAGE: sampled <width> <height> <nh_radius> <inputs_count>\n");
            exit(0);
            break;
    }

    field2d_t even_field;
    field2d_t odd_field;

    ticks_count_t* inputs = (ticks_count_t*) malloc(inputs_count * sizeof(ticks_count_t));
    ticks_count_t sample_rate = 10;
    ticks_count_t samples_count = 0;

    srand(time(NULL));

    f2d_init(&even_field, field_width, field_height, nh_radius);
    odd_field = *f2d_copy(&even_field);

    for (int i = 0; i < 1000; i++) {
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        // Only get new inputs according to the sample rate.
        if (i % sample_rate == 0) {
            // Fetch input.
            for (field_size_t j = 0; j < inputs_count; j++) {
                inputs[j] = 1 + (rand() % (sample_rate - 1));
            }
            samples_count = 0;
        }

        // Feed the field.
        for (field_size_t k = 0; k < inputs_count; k++) {
            if (samples_count % inputs[k]) {
                prev_field->neurons[k].value += NEURON_CHARGE_RATE;
            }
        }

        // print(next_field);

        // Tick the field.
        f2d_tick(prev_field, next_field, 0x0010u);

        samples_count++;

        // usleep(20000);
    }
}
