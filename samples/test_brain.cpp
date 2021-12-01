#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <liath.h>

void init(neuron_t* field, field_size_t field_width, field_size_t field_height) {
    for (field_size_t i = 0; i < field_height; i++) {
        for (field_size_t j = 0; j < field_width; j++) {
            double random = (double)rand() / (double)RAND_MAX;
            field[IDX2D(i, j, field_width)].value = random > 0.5 ? 1 : 0;
        }
    }
}

void tick(neuron_t* prev_field, neuron_t* next_field, field_size_t field_width, field_size_t field_height) {
    for (field_size_t i = 0; i < field_height; i++) {
        for (field_size_t j = 0; j < field_width; j++) {
            neuron_value_t res = prev_field[IDX2D(WRAP(j - 1, field_width), WRAP(i - 1, field_height), field_width)].value +
                                 prev_field[IDX2D(WRAP(j - 1, field_width), i,                         field_width)].value +
                                 prev_field[IDX2D(WRAP(j - 1, field_width), WRAP(i + 1, field_height), field_width)].value +
                                 prev_field[IDX2D(j,                        WRAP(i - 1, field_height), field_width)].value +
                                 prev_field[IDX2D(j,                        WRAP(i + 1, field_height), field_width)].value +
                                 prev_field[IDX2D(WRAP(j + 1, field_width), WRAP(i - 1, field_height), field_width)].value +
                                 prev_field[IDX2D(WRAP(j + 1, field_width), i,                         field_width)].value +
                                 prev_field[IDX2D(WRAP(j + 1, field_width), WRAP(i + 1, field_height), field_width)].value;
            neuron_value_t prev = prev_field[IDX2D(j, i, field_width)].value;
            next_field[IDX2D(j, i, field_width)].value = (prev && (res == 2 || res == 3)) || (!prev && res == 3);
        }
    }
}

void print(neuron_t* field, field_size_t field_width, field_size_t field_height) {
    system("clear");
    for (field_size_t i = 0; i < field_height; i++) {
        for (field_size_t j = 0; j < field_width; j++) {
            printf("%c", field[IDX2D(j, i, field_width)].value == 1 ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    neuron_t* even_field;
    neuron_t* odd_field;

    field_size_t field_width = 100;
    field_size_t field_height = 30;

    even_field = (neuron_t*) malloc(field_width * field_height * sizeof(neuron_t));
    odd_field = (neuron_t*) malloc(field_width * field_height * sizeof(neuron_t));

    init(even_field, field_width, field_height);
    print(even_field, field_width, field_height);

    for (int i = 0;; i++) {
        tick(even_field, odd_field, field_width, field_height);

        print(odd_field, field_width, field_height);

        usleep(50000);

        tick(odd_field, even_field, field_width, field_height);

        print(even_field, field_width, field_height);

        usleep(50000);
    }
}