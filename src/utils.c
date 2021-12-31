#include "utils.h"

static unsigned long x = 123456789;
static unsigned long y = 362436069;
static unsigned long z = 521288629;
unsigned long xorshf96(void) {
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return z;
}

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}

void c2d_dump(cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* out_file = fopen(file_name, "w");

    // Write cortex metadata to the output file.
    fwrite(cortex, sizeof(cortex2d_t), 1, out_file);

    // Write all neurons.
    for (cortex_size_t y = 0; y < cortex->height; y++) {
        for (cortex_size_t x = 0; x < cortex->width; x++) {
            fwrite(&(cortex->neurons), sizeof(neuron_t), cortex->width * cortex->height, out_file);
        }
    }

    fclose(out_file);
}