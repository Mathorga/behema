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

    field_size_t field_width = 150;
    field_size_t field_height = 50;
    nh_radius_t nh_radius = 1;
    field_size_t inputs_count = 2000;

    srand(time(NULL));

    f2d_rinit(&even_field, field_width, field_height, nh_radius);
    odd_field = *f2d_copy(&even_field);

    for (int i = 0;; i++) {
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        if (rand() % 100 > 50) {
            f2d_rsfeed(prev_field, 0, inputs_count, NEURON_CHARGE_RATE, 2);
        }

        f2d_tick(prev_field, next_field);

        print(next_field);

        usleep(40000);
    }
}