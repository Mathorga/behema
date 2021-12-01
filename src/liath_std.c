#include "liath_std.h"

void field2d_init(field2d_t* field) {
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            field->neurons[IDX2D(j, i, field->width)].value = NEURON_STARTING_VALUE;
            memset(field->neurons[IDX2D(j, i, field->width)].input_neighbors, 0, sizeof(field->neurons[IDX2D(j, i, field->width)].input_neighbors));
        }
    }
}

void field2d_tick(field2d_t* prev_field, field2d_t* next_field) {
    for (field_size_t i = 0; i < prev_field->height; i++) {
        for (field_size_t j = 0; j < prev_field->width; j++) {
            // Retrieve the current neuron.
            neuron_t* neuron = &(prev_field->neurons[IDX2D(j, i, prev_field->width)]);

            // Compute the number of neighbors to check by [n = ((2r + 1)^2) - 1], where r is the neighborhood radius and n is the number of neighbors.
            // This only works in a square grid with a square neighborhood.
            neighbors_count_t neighbors_count = pow(2 * prev_field->neighborhood_radius + 1, 2) - 1;

            

            for (neighbors_count_t k; k < neighbors_count; k++) {

            }
            next_field->neurons[IDX2D(j, i, prev_field->width)].value = NEURON_STARTING_VALUE;
        }
    }
}