#include "liath_std.h"

void f2d_init(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius) {
    field->width = width;
    field->height = height;
    field->nh_radius = nh_radius;
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            field->neurons[IDX2D(x, y, field->width)].nh_mask = NEURON_DEFAULT_NB_MASK;
            field->neurons[IDX2D(x, y, field->width)].value = NEURON_STARTING_VALUE;
            field->neurons[IDX2D(x, y, field->width)].threshold = NEURON_DEFAULT_THRESHOLD;
            field->neurons[IDX2D(x, y, field->width)].influence = NEURON_STARTING_BUSYNESS;
        }
    }
}

void f2d_rinit(field2d_t* field, field_size_t width, field_size_t height, nh_radius_t nh_radius) {
    field->width = width;
    field->height = height;
    field->nh_radius = nh_radius;
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            field->neurons[IDX2D(x, y, field->width)].nh_mask = rand() % 0xFFFFFFFFFFFFFFFF;
            field->neurons[IDX2D(x, y, field->width)].value = rand() % NEURON_DEFAULT_THRESHOLD;
            field->neurons[IDX2D(x, y, field->width)].threshold = NEURON_DEFAULT_THRESHOLD;
            field->neurons[IDX2D(x, y, field->width)].influence = NEURON_STARTING_BUSYNESS;
        }
    }
}

field2d_t* f2d_copy(field2d_t* other) {
    field2d_t* field = (field2d_t*) malloc(sizeof(field2d_t));
    field->width = other->width;
    field->height = other->height;
    field->nh_radius = other->nh_radius;
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t y = 0; y < other->height; y++) {
        for (field_size_t x = 0; x < other->width; x++) {
            field->neurons[IDX2D(x, y, other->width)] = other->neurons[IDX2D(x, y, other->width)];
        }
    }

    return field;
}

void f2d_set_nhradius(field2d_t* field, nh_radius_t radius) {
    // Only set radius if greater than zero.
    if (radius > 0) {
        field->nh_radius = radius;
    }
}

void f2d_set_nhmask(field2d_t* field, nh_mask_t mask) {
    for (field_size_t y = 0; y < field->height; y++) {
        for (field_size_t x = 0; x < field->width; x++) {
            field->neurons[IDX2D(x, y, field->width)].nh_mask = mask;
        }
    }
}

void f2d_feed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value) {
    if (starting_index + count < field->width * field->height) {
        // Loop through count.
        for (field_size_t i = starting_index; i < starting_index + count; i++) {
            field->neurons[i].value += value;
        }
    }
}

void f2d_rfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value) {
    if (starting_index + count < field->width * field->height) {
        // Loop through count.
        for (field_size_t i = starting_index; i < starting_index + count; i++) {
            field->neurons[i].value += rand() % max_value;
        }
    }
}

void f2d_sfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t value, field_size_t spread) {
    if ((starting_index + count) * spread < field->width * field->height) {
        // Loop through count.
        for (field_size_t i = starting_index; i < starting_index + count; i++) {
            field->neurons[i * spread].value += value;
        }
    }
}

void f2d_rsfeed(field2d_t* field, field_size_t starting_index, field_size_t count, neuron_value_t max_value, field_size_t spread) {
    if ((starting_index + count) * spread < field->width * field->height) {
        // Loop through count.
        for (field_size_t i = starting_index; i < starting_index + count; i++) {
            field->neurons[i * spread].value += rand() % max_value;
        }
    }
}

void f2d_tick(field2d_t* prev_field, field2d_t* next_field) {
    for (field_size_t y = 0; y < prev_field->height; y++) {
        for (field_size_t x = 0; x < prev_field->width; x++) {
            // Retrieve the involved neurons.
            neuron_t prev_neuron = prev_field->neurons[IDX2D(x, y, prev_field->width)];
            neuron_t* next_neuron = &(next_field->neurons[IDX2D(x, y, prev_field->width)]);

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

            nh_mask_t nb_mask = prev_neuron.nh_mask;

            // Increment the current neuron value by reading its connected neighbors.
            for (nh_radius_t j = 0; j < nh_diameter; j++) {
                for (nh_radius_t i = 0; i < nh_diameter; i++) {
                    // Exclude the central neuron from the list of neighbors.
                    if (!(j == prev_field->nh_radius && i == prev_field->nh_radius)) {
                        // Fetch the current neighbor.
                        neuron_t neighbor = prev_field->neurons[IDX2D(WRAP(x + (i - prev_field->nh_radius), prev_field->width),
                                                                      WRAP(y + (j - prev_field->nh_radius), prev_field->height),
                                                                      prev_field->width)];

                        // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                        if (nb_mask & 0x01 && neighbor.value > neighbor.threshold) {
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

            // Bring the neuron back to recovery if it just fired, otherwise fire it if its value is over its threshold.
            if (prev_neuron.value > prev_neuron.threshold) {
                // Fired at the previous step.
                next_neuron->value = NEURON_RECOVERY_VALUE;
            } else if (next_neuron->value > prev_neuron.threshold) {
                // Fired, increase influence.
                next_neuron->influence += NEURON_INFLUENCE_GAIN;
            } else if (prev_neuron.influence > 0) {
                // Not fired, decrease influence.
                next_neuron->influence--;
            }
        }
    }
}

void f2d_syndel(field2d_t* field) {}

void f2d_syngen(field2d_t* field) {}

void f2d_evolve(field2d_t* field) {}