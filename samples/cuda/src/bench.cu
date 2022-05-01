#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <portia/portia.h>

int main(int argc, char **argv) {
    cortex_size_t cortex_width = 2000;
    cortex_size_t cortex_height = 1000;
    cortex_size_t input_width = 20;
    cortex_size_t input_height = 1;
    int32_t cortex_grid_size = (cortex_width * cortex_height) / BLOCK_SIZE;
    dim3 input_grid_size(input_width, input_height);
    nh_radius_t nh_radius = 1;

    srand(time(NULL));

    error_code_t error;

    // Cortex init.
    cortex2d_t* even_cortex;
    cortex2d_t* odd_cortex;
    error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    error = c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    c2d_copy(odd_cortex, even_cortex);

    // Copy cortexes to the device.
    cortex2d_t* d_even_cortex;
    cortex2d_t* d_odd_cortex;
    cudaMalloc(&d_even_cortex, sizeof(cortex2d_t));
    cudaCheckError();
    cudaMalloc(&d_odd_cortex, sizeof(cortex2d_t));
    cudaCheckError();
    error = c2d_to_device(d_even_cortex, even_cortex);
    error = c2d_to_device(d_odd_cortex, odd_cortex);

    // Input init.
    input2d_t* input;
    i2d_init(&input, (cortex_width / 2) - (input_width / 2), 0, (cortex_width / 2) + (input_width / 2), input_height, DEFAULT_EXC_VALUE * 2, PULSE_MAPPING_FPROP);

    // Set input values.
    for (int i = 0; i < input_width * input_height; i++) {
        input->values[i] = even_cortex->sample_window / 2;
    }

    // Copy input to device.
    input2d_t* d_input;
    cudaMalloc(&d_input, sizeof(input2d_t));
    cudaCheckError();
    i2d_to_device(d_input, input);

    // Start timer.
    uint64_t start_time = millis();

    for (int i = 0; i < 1000; i++) {
        cortex2d_t* prev_cortex = i % 2 ? d_odd_cortex : d_even_cortex;
        cortex2d_t* next_cortex = i % 2 ? d_even_cortex : d_odd_cortex;

        // TODO Fetch input.

        // Copy input to device.
        i2d_to_device(d_input, input);

        // Feed.
        c2d_feed2d<<<input_grid_size, dim3(1, 1)>>>(prev_cortex, d_input);
        cudaCheckError();
        cudaDeviceSynchronize();

        c2d_tick<<<cortex_grid_size, BLOCK_SIZE>>>(prev_cortex, next_cortex);
        cudaDeviceSynchronize();

        // usleep(100);
    }

    // Stop timer.
    uint64_t end_time = millis();
    printf("\nCompleted 1000 iterations in %ldms\n", end_time - start_time);

    // Copy the cortex back to host to check the results.
    c2d_to_host(even_cortex, d_even_cortex);
    printf("\nDindolo\n");
    c2d_to_file(even_cortex, (char*) "out/test.c2d");

    // Cleanup.
    c2d_destroy(even_cortex);
    c2d_destroy(odd_cortex);
    c2d_device_destroy(d_even_cortex);
    c2d_device_destroy(d_odd_cortex);
    i2d_destroy(input);
    i2d_device_destroy(d_input);

    return 0;
}