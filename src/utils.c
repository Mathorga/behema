#include "utils.h"

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

bhm_error_code_t pgm_read(pgm_content_t* pgm, const char* filename) {
    // Open the image file in read mode.
    FILE* pgmfile = fopen(filename, "r");
 
    // If file does not exist, then return.
    if (pgmfile == NULL) {
        printf("File does not exist: %s\n", filename);
        return BHM_ERROR_FILE_DOES_NOT_EXIST;
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
 
    return BHM_ERROR_NONE;
}

uint32_t map(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    uint32_t slope = (output_end - output_start) / (input_end - input_start);
    return output_start + slope * (input - input_start);
}

uint32_t fmap(uint32_t input, uint32_t input_start, uint32_t input_end, uint32_t output_start, uint32_t output_end) {
    double slope = ((double) output_end - (double) output_start) / ((double) input_end - (double) input_start);
    return (double) output_start + slope * ((double) input - (double) input_start);
}

uint64_t millis() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ms = S_TO_MS((uint64_t)ts.tv_sec) + NS_TO_MS((uint64_t)ts.tv_nsec);
    return ms;
}

uint64_t micros() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t us = S_TO_US((uint64_t)ts.tv_sec) + NS_TO_US((uint64_t)ts.tv_nsec);
    return us;
}

uint64_t nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    uint64_t ns = S_TO_NS((uint64_t)ts.tv_sec) + (uint64_t)ts.tv_nsec;
    return ns;
}


bhm_error_code_t c2d_to_file(bhm_cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* out_file = fopen(file_name, "wb");
    if (out_file == NULL) {
        printf("File does not exist: %s\n", file_name);
        return BHM_ERROR_FILE_DOES_NOT_EXIST;
    }

    // Write cortex metadata to the output file.
    fwrite(&(cortex->width), sizeof(bhm_cortex_size_t), 1, out_file);
    fwrite(&(cortex->height), sizeof(bhm_cortex_size_t), 1, out_file);
    fwrite(&(cortex->ticks_count), sizeof(bhm_ticks_count_t), 1, out_file);
    fwrite(&(cortex->evols_count), sizeof(bhm_ticks_count_t), 1, out_file);
    fwrite(&(cortex->evol_step), sizeof(bhm_ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_window), sizeof(bhm_ticks_count_t), 1, out_file);

    fwrite(&(cortex->nh_radius), sizeof(bhm_nh_radius_t), 1, out_file);
    fwrite(&(cortex->fire_threshold), sizeof(bhm_neuron_value_t), 1, out_file);
    fwrite(&(cortex->recovery_value), sizeof(bhm_neuron_value_t), 1, out_file);
    fwrite(&(cortex->exc_value), sizeof(bhm_neuron_value_t), 1, out_file);
    fwrite(&(cortex->decay_value), sizeof(bhm_neuron_value_t), 1, out_file);

    fwrite(&(cortex->rand_state), sizeof(bhm_rand_state_t), 1, out_file);

    fwrite(&(cortex->syngen_chance), sizeof(bhm_chance_t), 1, out_file);
    fwrite(&(cortex->synstr_chance), sizeof(bhm_chance_t), 1, out_file);

    fwrite(&(cortex->max_tot_strength), sizeof(bhm_syn_strength_t), 1, out_file);
    fwrite(&(cortex->max_syn_count), sizeof(bhm_syn_count_t), 1, out_file);
    fwrite(&(cortex->inhexc_range), sizeof(bhm_chance_t), 1, out_file);

    fwrite(&(cortex->sample_window), sizeof(bhm_ticks_count_t), 1, out_file);
    fwrite(&(cortex->pulse_mapping), sizeof(bhm_pulse_mapping_t), 1, out_file);

    // Write all neurons.
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(bhm_neuron_t), 1, out_file);
        }
    }

    fclose(out_file);

    return BHM_ERROR_NONE;
}

void c2d_from_file(bhm_cortex2d_t* cortex, char* file_name) {
    // Open output file if possible.
    FILE* in_file = fopen(file_name, "rb");

    // Read cortex metadata from the output file.
    fread(&(cortex->width), sizeof(bhm_cortex_size_t), 1, in_file);
    fread(&(cortex->height), sizeof(bhm_cortex_size_t), 1, in_file);
    fread(&(cortex->ticks_count), sizeof(bhm_ticks_count_t), 1, in_file);
    fread(&(cortex->evols_count), sizeof(bhm_ticks_count_t), 1, in_file);
    fread(&(cortex->evol_step), sizeof(bhm_ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_window), sizeof(bhm_ticks_count_t), 1, in_file);

    fread(&(cortex->nh_radius), sizeof(bhm_nh_radius_t), 1, in_file);
    fread(&(cortex->fire_threshold), sizeof(bhm_neuron_value_t), 1, in_file);
    fread(&(cortex->recovery_value), sizeof(bhm_neuron_value_t), 1, in_file);
    fread(&(cortex->exc_value), sizeof(bhm_neuron_value_t), 1, in_file);
    fread(&(cortex->decay_value), sizeof(bhm_neuron_value_t), 1, in_file);

    fread(&(cortex->rand_state), sizeof(bhm_rand_state_t), 1, in_file);

    fread(&(cortex->syngen_chance), sizeof(bhm_chance_t), 1, in_file);
    fread(&(cortex->synstr_chance), sizeof(bhm_chance_t), 1, in_file);

    fread(&(cortex->max_tot_strength), sizeof(bhm_syn_strength_t), 1, in_file);
    fread(&(cortex->max_syn_count), sizeof(bhm_syn_count_t), 1, in_file);
    fread(&(cortex->inhexc_range), sizeof(bhm_chance_t), 1, in_file);

    fread(&(cortex->sample_window), sizeof(bhm_ticks_count_t), 1, in_file);
    fread(&(cortex->pulse_mapping), sizeof(bhm_pulse_mapping_t), 1, in_file);

    // Read all neurons.
    cortex->neurons = (bhm_neuron_t*) malloc(cortex->width * cortex->height * sizeof(bhm_neuron_t));
    for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
        for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
            fread(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(bhm_neuron_t), 1, in_file);
        }
    }

    fclose(in_file);
}

void p2d_to_file(bhm_population2d_t* population, char* file_name) {
    // Open output file if possible.
    FILE* out_file = fopen(file_name, "wb");
    if (out_file == NULL) {
        printf("File does not exist: %s\n", file_name);
        return BHM_ERROR_FILE_DOES_NOT_EXIST;
    }

    // Write cortex metadata to the output file.
    fwrite(&(population->size), sizeof(bhm_population_size_t), 1, out_file);
    fwrite(&(population->selection_pool_size), sizeof(bhm_population_size_t), 1, out_file);
    fwrite(&(population->parents_count), sizeof(bhm_population_size_t), 1, out_file);
    // TODO Write other fields.

    // TODO Write all cortices.
    // for (bhm_cortex_size_t y = 0; y < cortex->height; y++) {
    //     for (bhm_cortex_size_t x = 0; x < cortex->width; x++) {
    //         fwrite(&(cortex->neurons[IDX2D(x, y, cortex->width)]), sizeof(bhm_neuron_t), 1, out_file);
    //     }
    // }

    fclose(out_file);

    return BHM_ERROR_NONE;
}

void p2d_from_file(bhm_population2d_t* population, char* file_name) {
    // TODO
}

bhm_error_code_t c2d_touch_from_map(bhm_cortex2d_t* cortex, char* map_file_name) {
    pgm_content_t pgm_content;

    // Read file.
    bhm_error_code_t error = pgm_read(&pgm_content, map_file_name);
    if (error) {
        return error;
    }

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height) {
        for (bhm_cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
            cortex->neurons[i].max_syn_count = fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->max_syn_count);
        }
    } else {
        printf("\nc2d_touch_from_map file sizes do not match with cortex\n");
        return BHM_ERROR_FILE_SIZE_WRONG;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t c2d_inhexc_from_map(bhm_cortex2d_t* cortex, char* map_file_name) {
    pgm_content_t pgm_content;

    // Read file.
    bhm_error_code_t error = pgm_read(&pgm_content, map_file_name);
    if (error) {
        return error;
    }

    // Make sure sizes are correct.
    if (cortex->width == pgm_content.width && cortex->height == pgm_content.height) {
        for (bhm_cortex_size_t i = 0; i < cortex->width * cortex->height; i++) {
            cortex->neurons[i].inhexc_ratio = fmap(pgm_content.data[i], 0, pgm_content.max_value, 0, cortex->inhexc_range);
        }
    } else {
        printf("\nc2d_inhexc_from_map file sizes do not match with cortex\n");
        return BHM_ERROR_FILE_SIZE_WRONG;
    }

    return BHM_ERROR_NONE;
}