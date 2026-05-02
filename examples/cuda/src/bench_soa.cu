#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <behema/behema.h>

bhm_error_code_t setup_cortices(
    bhm_soa_cortex_t** even_cortex,
    bhm_soa_cortex_t** odd_cortex,
    bhm_cortex_size_t cortex_width,
    bhm_cortex_size_t cortex_height,
    bhm_nh_radius_t nh_radius
) {
    bhm_error_code_t error = BHM_ERROR_NONE;

    // Initialize the first cortex.
    error = c2d_create_soa(even_cortex, cortex_width, cortex_height, nh_radius);
    if (error) return error;
    error = c2d_create_soa(odd_cortex, cortex_width, cortex_height, nh_radius);
    if (error) return error;
    error = c2d_set_evol_step_soa(*even_cortex, 0x01U);
    if (error) return error;
    error = c2d_set_pulse_mapping_soa(*even_cortex, BHM_PULSE_MAPPING_RPROP);
    if (error) return error;
    error = c2d_set_max_syn_count_soa(*even_cortex, 24);
    if (error) return error;
    // char touchFileName[40];
    // char inhexcFileName[40];
    // sprintf(touchFileName, "./res/%d_%d_touch.pgm", cortex_width, cortex_height);
    // sprintf(inhexcFileName, "./res/%d_%d_inhexc.pgm", cortex_width, cortex_height);

    // // No error check since any error in files read should not impact the program.
    // c2d_touch_from_map(*even_cortex, touchFileName);
    // c2d_inhexc_from_map(*even_cortex, inhexcFileName);

    // Copy the first cortex properties to the second one.
    error = c2d_copy_soa(*odd_cortex, *even_cortex);
    return error;
}

int main(int argc, char **argv) {
    bhm_cortex_size_t cortex_width = 512;
    bhm_cortex_size_t cortex_height = 256;
    uint32_t iterations_count = 10000;
    bhm_nh_radius_t nh_radius = 2;

    // Input handling.
    switch (argc) {
        case 1:
            break;
        case 2:
            iterations_count = atoi(argv[1]);
            break;
        case 3:
            iterations_count = atoi(argv[1]);
            cortex_width = atoi(argv[2]);
            break;
        case 4:
            iterations_count = atoi(argv[1]);
            cortex_width = atoi(argv[2]);
            cortex_height = atoi(argv[3]);
            break;
        default:
            printf("USAGE: bench <iterations> <width> <height>\n");
            exit(0);
            break;
    }

    bhm_cortex_size_t input_width = cortex_width / 4;
    bhm_cortex_size_t input_height = 1;
    dim3 input_grid_size(input_width, input_height);
    dim3 input_block_size(1, 1);

    srand(time(NULL));

    bhm_error_code_t error;

    // Cortex configuration.
    bhm_soa_cortex_t* even_cortex;
    bhm_soa_cortex_t* odd_cortex;
    error = setup_cortices(
        &even_cortex,
        &odd_cortex,
        cortex_width,
        cortex_height,
        nh_radius
    );
    if (error) return 1;

    bhm_cortex_counts_t* counts = (bhm_cortex_counts_t*) malloc(sizeof(bhm_cortex_counts_t));
    counts->ticks_count = 0x00;
    counts->evols_count = 0x00;
    
    dim3 cortex_block_size = c2d_get_block_size_soa(even_cortex);
    dim3 cortex_grid_size = c2d_get_grid_size_soa(even_cortex, cortex_block_size);

    // Print the cortex out.
    char cortex_string[100];
    c2d_to_string_soa(even_cortex, cortex_string);
    printf("%s", cortex_string);

    // Copy cortices to device.
    bhm_soa_cortex_t* d_even_cortex;
    bhm_soa_cortex_t* d_odd_cortex;
    cudaMalloc((void**) &d_even_cortex, sizeof(bhm_soa_cortex_t));
    cudaCheckError();
    cudaMalloc((void**) &d_odd_cortex, sizeof(bhm_soa_cortex_t));
    cudaCheckError();
    error = c2d_to_device_soa(d_even_cortex, even_cortex);
    error = c2d_to_device_soa(d_odd_cortex, odd_cortex);

    // Input init.
    bhm_input2d_t* input;
    i2d_init(
        &input,
        (cortex_width / 2) - (input_width / 2),
        0,
        (cortex_width / 2) + (input_width / 2),
        input_height,
        BHM_DEFAULT_EXC_VALUE * 2,
        BHM_PULSE_MAPPING_FPROP
    );

    // Set input values.
    for (int i = 0; i < input_width * input_height; i++) {
        input->values[i] = even_cortex->sample_window - 1;
    }

    // Copy input to device.
    bhm_input2d_t* d_input;
    cudaMalloc((void**) &d_input, sizeof(bhm_input2d_t));
    cudaCheckError();
    i2d_to_device(d_input, input);
    
    // Start timer.
    uint64_t start_time = millis();
    
    for (uint32_t i = 0; i < iterations_count; i++) {
        bhm_soa_cortex_t* prev_cortex = i % 2 ? d_odd_cortex : d_even_cortex;
        bhm_soa_cortex_t* next_cortex = i % 2 ? d_even_cortex : d_odd_cortex;

        // Defines whether to evolve or not.
        // evol_step is incremented by 1 to account for edge cases and human readable behavior:
        // 0x0000 -> 0 + 1 = 1, so the cortex evolves at every tick, meaning that there are no free ticks between evolutions.
        // 0xFFFF -> 65535 + 1 = 65536, so the cortex never evolves, meaning that there is an infinite amount of ticks between evolutions.
        bool evolve = (counts->ticks_count % (((bhm_evol_step_t) even_cortex->evol_step) + 1)) == 0;
        
        // TODO Fetch input.
        
        // Copy input to device.
        // i2d_to_device(d_input, input);
        
        // Feed.
        c2d_feed2d_soa<<<input_grid_size, input_block_size>>>(
            prev_cortex,
            d_input,
            counts->ticks_count
        );
        cudaCheckError();
        cudaDeviceSynchronize();
        
        c2d_tick_soa<<<cortex_grid_size, cortex_block_size>>>(
            prev_cortex,
            next_cortex,
            evolve
        );
        cudaCheckError();
        cudaDeviceSynchronize();

        counts->ticks_count++;
        // Increment evolutions count.
        if (evolve) counts->evols_count++;

        if ((i + 1) % 100 == 0) {
            uint64_t elapsed = millis() - start_time;
            double frequency = i /(elapsed / 1000.0f);
            printf("\nPerformed %d iterations in %llums; %.2f Hz\n", i + 1, elapsed, frequency);
            // c2d_to_file(even_cortex, (char*) "out/test.c2d");
        }

        // usleep(100);
    }

    // Copy the cortex back to host and save it to file.
    c2d_to_host_soa(even_cortex, d_even_cortex);
    // c2d_to_file(even_cortex, (char*) "out/test.c2d");

    // Cleanup.
    c2d_destroy_soa(even_cortex);
    c2d_destroy_soa(odd_cortex);
    c2d_device_destroy_soa(d_even_cortex);
    c2d_device_destroy_soa(d_odd_cortex);
    i2d_destroy(input);
    i2d_device_destroy(d_input);

    return 0;
}