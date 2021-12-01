#include "liath_std.h"

void field2d_init(field2d_t* field) {
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            field->neurons[IDX2D(j, i, field->width)].input_neighbors = NEURON_DEFAULT_NB_MASK;
            field->neurons[IDX2D(j, i, field->width)].value = NEURON_STARTING_VALUE;
            field->neurons[IDX2D(j, i, field->width)].threshold = NEURON_DEFAULT_THRESHOLD;
        }
    }
}

void field2d_tick(field2d_t* prev_field, field2d_t* next_field) {
    for (field_size_t i = 0; i < prev_field->height; i++) {
        for (field_size_t j = 0; j < prev_field->width; j++) {
            // Retrieve the current neuron.
            neuron_t neuron = prev_field->neurons[IDX2D(j, i, prev_field->width)];

            /* Compute the neighborhood diameter:
                   d = 7
              <------------->
               r = 3
              <----->
              +-|-|-|-|-|-|-+
              |             |
              |             |
              |      X      |
              |             |
              |             |
              +-|-|-|-|-|-|-+
            */
            field_size_t nh_diameter = 2 * prev_field->neighborhood_radius + 1;

            // Compute the number of neighbors to check by [n = ((2r + 1)^2) - 1], where r is the neighborhood radius and n is the number of neighbors.
            // This only works in a square grid with a square neighborhood.
            nb_count_t neighbors_count = pow(nh_diameter, 2) - 1;

            nb_mask_t neighbor_mask = neuron.input_neighbors;

            for (nb_count_t k; k < neighbors_count; k++) {
                // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                if (neighbor_mask & 0x01 && neuron.value > neuron.threshold) {
                    next_field->neurons[IDX2D(j, i, prev_field->width)].value += NEURON_CHARGE_RATE;
                }

                // Shift the mask to check for the next neighbor.
                neighbor_mask = neighbor_mask >> 1;
            }
        }
    }
}