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

void ignoreComments(FILE* fp) {
    int ch;
    char line[100];
 
    // Ignore any blank lines
    while ((ch = fgetc(fp)) != EOF && isspace(ch)) {}
 
    // Recursively ignore comments.
    // In a PGM image commented lines start with '#'.
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    } else {
        fseek(fp, -1, SEEK_CUR);
    }
}

void pgm_read(pgm_content_t* pgm, const char* filename) {
    // Open the image file in read mode.
    FILE* pgmfile = fopen(filename, "r");
 
    // If file does not exist, then return.
    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return;
    }
 
    ignoreComments(pgmfile);

    // Read file type.
    fscanf(pgmfile, "%s", pgm->pgmType);
 
    ignoreComments(pgmfile);
 
    // Read data size.
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));

    ignoreComments(pgmfile);
 
    // Read maximum value.
    fscanf(pgmfile, "%u", &(pgm->max_value));

    ignoreComments(pgmfile);
 
    // Allocate memory to store data in the struct.
    pgm->data = (uint8_t*) malloc(pgm->width * pgm->height * sizeof(uint8_t));

 
    // Store data in the struct.
    if (!strcmp(pgm->pgmType, "P2")) {
        // Plain data.
        for (uint32_t y = 0; y < pgm->height; y++) {
            for (uint32_t x = 0; x < pgm->width; x++) {
                fscanf(pgmfile, "%hhu", &(pgm->data[IDX2D(x, y, pgm->width)]));
            }
        }
    } else if (!strcmp(pgm->pgmType, "P5")) {
        // Raw data.
        fread(pgm->data, sizeof(uint8_t), pgm->width * pgm->height, pgmfile);
    } else {
        // Wrong file type.
        printf("Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
 
    // Close the file
    fclose(pgmfile);
 
    return;
}

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}


void c2d_to_file(cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* out_file = fopen(file_name, "wb");

    // Write cortex metadata to the output file.
    fwrite(&(cortex->width), sizeof(cortex_size_t), 1, out_file);
    fwrite(&(cortex->height), sizeof(cortex_size_t), 1, out_file);
    fwrite(&(cortex->ticks_count), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->evols_count), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->evol_step), sizeof(ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_window), sizeof(pulses_count_t), 1, out_file);

    fwrite(&(cortex->nh_radius), sizeof(nh_radius_t), 1, out_file);
    fwrite(&(cortex->fire_threshold), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->recovery_value), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->exc_value), sizeof(neuron_value_t), 1, out_file);
    fwrite(&(cortex->decay_value), sizeof(neuron_value_t), 1, out_file);

    fwrite(&(cortex->syngen_chance), sizeof(chance_t), 1, out_file);
    fwrite(&(cortex->synstr_chance), sizeof(chance_t), 1, out_file);

    fwrite(&(cortex->max_tot_strength), sizeof(syn_strength_t), 1, out_file);
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
    FILE* in_file = fopen(file_name, "rb");

    // Read cortex metadata from the output file.
    fread(&(cortex->width), sizeof(cortex_size_t), 1, in_file);
    fread(&(cortex->height), sizeof(cortex_size_t), 1, in_file);
    fread(&(cortex->ticks_count), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->evols_count), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->evol_step), sizeof(ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_window), sizeof(pulses_count_t), 1, in_file);

    fread(&(cortex->nh_radius), sizeof(nh_radius_t), 1, in_file);
    fread(&(cortex->fire_threshold), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->recovery_value), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->exc_value), sizeof(neuron_value_t), 1, in_file);
    fread(&(cortex->decay_value), sizeof(neuron_value_t), 1, in_file);

    fread(&(cortex->syngen_chance), sizeof(chance_t), 1, in_file);
    fread(&(cortex->synstr_chance), sizeof(chance_t), 1, in_file);

    fread(&(cortex->max_tot_strength), sizeof(syn_strength_t), 1, in_file);
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

void c2d_touch_from_map(cortex2d_t* cortex, char* map_file_name) {
    pgm_content_t pgm_content;

    // Read file.
    pgm_read(&pgm_content, map_file_name);

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height) {
        for (cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
            cortex->neurons[i].max_syn_count = map(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->max_syn_count);
        }
    } else {
        printf("\nc2d_touch_from_map file sizes do not match with cortex\n");
    }
}

void c2d_inhexc_from_map(cortex2d_t* cortex, char* map_file_name) {
    pgm_content_t pgm_content;

    // Read file.
    pgm_read(&pgm_content, map_file_name);

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height) {
        for (cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
            cortex->neurons[i].inhexc_ratio = map(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->inhexc_range);
        }
    } else {
        printf("\nc2d_inhexc_from_map file sizes do not match with cortex\n");
    }
}