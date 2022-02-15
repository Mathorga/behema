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
    fwrite(&(cortex->syndel_chance), sizeof(chance_t), 1, out_file);
    fwrite(&(cortex->synstr_chance), sizeof(chance_t), 1, out_file);
    fwrite(&(cortex->synwk_chance), sizeof(chance_t), 1, out_file);

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
    fread(&(cortex->syndel_chance), sizeof(chance_t), 1, in_file);
    fread(&(cortex->synstr_chance), sizeof(chance_t), 1, in_file);
    fread(&(cortex->synwk_chance), sizeof(chance_t), 1, in_file);

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
    // FILE* in_file = fopen(map_file_name, "r");
}






 
 
// Function to ignore any comments
// in file
void ignoreComments(FILE* fp) {
    int ch;
    char line[100];
 
    // Ignore any blank lines
    while ((ch = fgetc(fp)) != EOF && isspace(ch)) {}
 
    // Recursively ignore comments
    // in a PGM image commented lines
    // start with a '#'
    if (ch == '#') {
        fgets(line, sizeof(line), fp);
        ignoreComments(fp);
    } else {
        fseek(fp, -1, SEEK_CUR);
    }
}
 
// Function to open the input a PGM
// file and process it
void pgmb_read(pgm_content_t* pgm, const char* filename) {
    // Open the image file in the 'read binary' mode.
    FILE* pgmfile = fopen(filename, "r");
 
    // If file does not exist, then return.
    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return;
    }
 
    ignoreComments(pgmfile);

    fscanf(pgmfile, "%s", pgm->pgmType);
 
    // Check for correct PGM Binary file type.
    if (strcmp(pgm->pgmType, "P5")) {
        printf("Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
 
    ignoreComments(pgmfile);
 
    // Read the image dimensions
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));
 
    ignoreComments(pgmfile);
 
    // Read maximum gray value
    fscanf(pgmfile, "%u", &(pgm->maxValue));

    printf("\nMAX %u\n", pgm->maxValue);

    ignoreComments(pgmfile);
 
    // Allocating memory to store img info in defined struct.
    pgm->data = (unsigned char*) malloc(pgm->width * pgm->height * sizeof(unsigned char));
 
    // Storing the pixel info in the struct.
    // fgetc(pgmfile);
    fread(pgm->data, sizeof(uint8_t), pgm->width * pgm->height, pgmfile);
 
    // Close the file
    fclose(pgmfile);
 
    return;
}

void pgma_read(pgm_content_t* pgm, const char* filename) {
    // Open the image file in the 'read binary' mode.
    FILE* pgmfile = fopen(filename, "r");
 
    // If file does not exist, then return.
    if (pgmfile == NULL) {
        printf("File does not exist\n");
        return;
    }
 
    ignoreComments(pgmfile);

    fscanf(pgmfile, "%s", pgm->pgmType);
 
    // Check for correct PGM Binary file type.
    if (strcmp(pgm->pgmType, "P2")) {
        printf("Wrong file type!\n");
        exit(EXIT_FAILURE);
    }
 
    ignoreComments(pgmfile);
 
    // Read the image dimensions
    fscanf(pgmfile, "%u %u", &(pgm->width), &(pgm->height));

    printf("\nWIDTH %u\n", pgm->width);
    printf("\nHEIGHT %u\n", pgm->height);
 
    ignoreComments(pgmfile);
 
    // Read maximum gray value
    fscanf(pgmfile, "%u", &(pgm->maxValue));

    printf("\nMAX %u\n", pgm->maxValue);

    ignoreComments(pgmfile);
 
    // Allocating memory to store img info in defined struct.
    pgm->data = (unsigned char*) malloc(pgm->width * pgm->height * sizeof(unsigned char));
 
    // Storing the pixel info in the struct.
    for (uint32_t y = 0; y < pgm->height; y++) {
        for (uint32_t x = 0; x < pgm->width; x++) {
            uint8_t value;

            fscanf(pgmfile, "%hhu", &value);
            pgm->data[IDX2D(x, y, pgm->width)] = value;
        }
    }
 
    // Close the file
    fclose(pgmfile);
 
    return;
}