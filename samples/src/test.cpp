#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <liath/liath.h>

void print(field2d_t* field) {
    system("clear");
    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            printf("%c", field->neurons[IDX2D(j, i, field->width)].fired ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    field2d_t even_field;
    field2d_t odd_field;

    field_size_t field_width = 300;
    field_size_t field_height = 80;
    nh_radius_t nh_radius = 1;

    srand(time(NULL));

    field2d_init(&even_field, field_width, field_height, nh_radius);
    odd_field = *field2d_copy(&even_field);

    for (int i = 0;; i++) {
        field2d_t* prev_field = i % 2 ? &odd_field : &even_field;
        field2d_t* next_field = i % 2 ? &even_field : &odd_field;

        if (!(i % 5)) {
            field2d_feed(prev_field, 0, 1000, NEURON_CHARGE_RATE);
        }

        field2d_tick(prev_field, next_field);

        print(next_field);

        usleep(40000);
    }
}