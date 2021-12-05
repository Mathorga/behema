#include "liath_std.h"

void field2d_init(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius) {
    field->width = width;
    field->height = height;
    field->nh_radius = nh_radius;
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            // field->neurons[IDX2D(j, i, field->width)].input_neighbors = NEURON_DEFAULT_NB_MASK;
            field->neurons[IDX2D(j, i, field->width)].input_neighbors = rand() % 0xFFFFFFFFFFFFFFFF;
            // field->neurons[IDX2D(j, i, field->width)].value = NEURON_STARTING_VALUE;
            field->neurons[IDX2D(j, i, field->width)].value = rand() % 0xFF;
            field->neurons[IDX2D(j, i, field->width)].threshold = NEURON_DEFAULT_THRESHOLD;
            field->neurons[IDX2D(j, i, field->width)].fired = 0x00;
        }
    }
}

field2d_t* field2d_copy(field2d_t* other) {
    field2d_t* field = (field2d_t*) malloc(sizeof(field2d_t));
    field->width = other->width;
    field->height = other->height;
    field->nh_radius = other->nh_radius;
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t i = 0; i < other->height; i++) {
        for (field_size_t j = 0; j < other->width; j++) {
            field->neurons[IDX2D(j, i, other->width)] = other->neurons[IDX2D(j, i, other->width)];
        }
    }

    return field;
}

void field2d_set_nh_radius(field2d_t* field, nh_radius_t radius) {
    // Only set radius if greater than zero.
    if (radius > 0) {
        field->nh_radius = radius;
    }
}

void field2d_feed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value) {
    if (starting_index + count < field->width * field->height) {
        // Loop through count.
        for (field_size_t i = 0; i < count; i++) {
            field->neurons[i].value += value;
        }
    }
}

void field2d_tick(field2d_t* prev_field, field2d_t* next_field) {
    for (field_size_t i = 0; i < prev_field->height; i++) {
        for (field_size_t j = 0; j < prev_field->width; j++) {
            // Retrieve the involved neurons.
            neuron_t prev_neuron = prev_field->neurons[IDX2D(j, i, prev_field->width)];
            neuron_t* next_neuron = &(next_field->neurons[IDX2D(j, i, prev_field->width)]);

            // Copy prev neuron values to the new one.
            *next_neuron = prev_neuron;

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
            field_size_t nh_diameter = 2 * prev_field->nh_radius + 1;

            nb_mask_t nb_mask = prev_neuron.input_neighbors;

            // Increment the current neuron value by reading its connected neighbors.
            for (nh_radius_t k = 0; k < nh_diameter; k++) {
                for (nh_radius_t l = 0; l < nh_diameter; l++) {
                    // Exclude the actual neuron from the list of neighbors.
                    if (!(k == prev_field->nh_radius && l == prev_field->nh_radius)) {
                        // Fetch the current neighbor.
                        neuron_t neighbor = prev_field->neurons[IDX2D(WRAP(j + (l - prev_field->nh_radius), prev_field->width),
                                                                      WRAP(i + (k - prev_field->nh_radius), prev_field->height),
                                                                      prev_field->width)];

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (nb_mask & 0x01 && neighbor.fired) {
                            next_neuron->value += NEURON_CHARGE_RATE;
                        }

                        // Shift the mask to check for the next neighbor.
                        nb_mask = nb_mask >> 1;
                    }
                }
            }

            // Push to equilibrium by decaying to zero, both from above and below.
            if (prev_neuron.value > 0x00) {
                next_neuron->value -= NEURON_DECAY_RATE;
            } else if (prev_neuron.value < 0x00) {
                next_neuron->value += NEURON_DECAY_RATE;
            }

            // printf("NEURON_VALUE %d %d %d %d\n", j, i, prev_neuron.value, next_neuron->value);
            // printf("FIRED %s\n", prev_neuron.fired ? "yes" : "no");

            // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
            if (prev_neuron.fired) {
                next_neuron->fired = 0x00;
                next_neuron->value = NEURON_RECOVERY_VALUE;
            } else if (next_neuron->value > prev_neuron.threshold) {
                next_neuron->fired = 0x01;
            }
        }
    }
}