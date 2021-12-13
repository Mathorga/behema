#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <liath/liath.h>

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
    field2d_t even_field;
    field2d_t odd_field;

    field_size_t field_width = 50;
    field_size_t field_height = 30;
    nh_radius_t nh_radius = 1;
    field_size_t inputs_count = 100;
    ticks_count_t sample_rate = 10;

    ticks_count_t* inputs = (ticks_count_t*) malloc(inputs_count * sizeof(ticks_count_t));
    ticks_count_t samples_count = 0;

    srand(time(NULL));

    f2d_rinit(&even_field, field_width, field_height, nh_radius);
    odd_field = *f2d_copy(&even_field);

    for (int i = 0;; i++) {
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

        // Feed the network.
        for (field_size_t k = 0; k < inputs_count; k++) {
            if (samples_count % inputs[k]) {
                prev_field->neurons[k].value += NEURON_CHARGE_RATE;
            }
        }

        f2d_tick(prev_field, next_field, 0xFFFFu);

        print(next_field);

        samples_count++;

        usleep(40000);
    }
}
