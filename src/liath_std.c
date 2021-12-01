#include "liath_std.h"

void field2d_init(field2d_t* field) {
    field->neurons = (neuron_t*) malloc(field->width * field->height * sizeof(neuron_t));

    for (field_size_t i = 0; i < field->height; i++) {
        for (field_size_t j = 0; j < field->width; j++) {
            field->neurons[IDX2D(j, i, field->width)].input_neighbors = NEURON_DEFAULT_NB_MASK;
            field->neurons[IDX2D(j, i, field->width)].value = NEURON_STARTING_VALUE;
            field->neurons[IDX2D(j, i, field->width)].threshold = NEURON_DEFAULT_THRESHOLD;
            field->neurons[IDX2D(j, i, field->width)].fired = 0x00;
        }
    }
}

void field2d_tick(field2d_t* prev_field, field2d_t* next_field) {
    for (field_size_t i = 0; i < prev_field->height; i++) {
        for (field_size_t j = 0; j < prev_field->width; j++) {
            // Retrieve the involved neurons.
            neuron_t prev_neuron = prev_field->neurons[IDX2D(j, i, prev_field->width)];
            neuron_t* next_neuron = &(next_field->neurons[IDX2D(j, i, prev_field->width)]);

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

            nb_mask_t neighbor_mask = prev_neuron.input_neighbors;

            // Increment the current neuron value by reading its connected neighbors.
            for (nb_count_t k = 0; k < nh_diameter; k++) {
                for (nb_count_t l = 0; l < nh_diameter; l++) {
                    // Fetch the current neighbor.
                    neuron_t neighbor = prev_field->neurons[IDX2D(j - (l - prev_field->neighborhood_radius),
                                                                  i - (k - prev_field->neighborhood_radius),
                                                                  prev_field->width)];

                    // Check if the last bit of the mask is 1 or zero, 1 = active input, 0 = inactive input.
                    if (neighbor_mask & 0x01 && neighbor.fired) {
                        next_neuron->value += NEURON_CHARGE_RATE;
                    }

                    // Shift the mask to check for the next neighbor.
                    neighbor_mask = neighbor_mask >> 1;
                }
            }

            // Push to equilibrium by decaying to zero, both from above and below.
            if (prev_neuron.value > 0) {
                next_neuron->value -= NEURON_DECAY_RATE;
            } else if (prev_neuron.value < 0) {
                next_neuron->value += NEURON_DECAY_RATE;
            }

            // Fire if the neuron value went over its threashold, otherwise bring it back to recovery value, but only if it just fired.
            if (prev_neuron.value > prev_neuron.threshold) {
                next_neuron->fired = 0x01;
            } else if (prev_neuron.fired) {
                next_neuron->fired = 0x00;
                next_neuron->value = NEURON_RECOVERY_VALUE;
            }

        }
    }
}