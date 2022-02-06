#include "utils.h"

// The state word must be initialized to non-zero.
static uint32_t state = 0x01U;
uint32_t xorshf32() {
    // Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs".
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state = x;
}

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}

void c2d_to_file(cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* out_file = fopen(file_name, "w");

    // Write cortex metadata to the output file.
    fwrite(&(cortex->width), sizeof(cortex_size_t), 1, out_file);
    fwrite(&(cortex->height), sizeof(cortex_size_t), 1, out_file);
    fwrite(&(cortex->ticks_count), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->evol_step), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_window), sizeof(pulses_count_t), 1, out_file);

    fwrite(&(cortex->nh_radius), sizeof(nh_radius_t), 1, out_file);
    fwrite(&(cortex->fire_threshold), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->recovery_value), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->exc_value), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->decay_value), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->max_syn_count), sizeof(syn_count_t), 1, out_file);

    fwrite(&(cortex->sample_window), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_mapping), sizeof(pulse_mapping_t), 1, out_file);

    // Write all neurons.
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(neuron_t), 1, out_file);
        }
    }

    fclose(out_file);
}

void c2d_from_file(cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* in_file = fopen(file_name, "r");

    // Read cortex metadata from the output file.
    fread(&(cortex->width), sizeof(cortex_size_t), 1, in_file);
    fread(&(cortex->height), sizeof(cortex_size_t), 1, in_file);
    fread(&(cortex->ticks_count), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->evol_step), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_window), sizeof(pulses_count_t), 1, in_file);

    fread(&(cortex->nh_radius), sizeof(nh_radius_t), 1, in_file);
    fread(&(cortex->fire_threshold), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->recovery_value), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->exc_value), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->decay_value), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->max_syn_count), sizeof(syn_count_t), 1, in_file);

    fread(&(cortex->sample_window), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_mapping), sizeof(pulse_mapping_t), 1, in_file);

    // Read all neurons.
    cortex->neurons = (neuron_t*) malloc(cortex->width * cortex->height * sizeof(neuron_t));
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            fread(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(neuron_t), 1, in_file);
        }
    }

    fclose(in_file);
}