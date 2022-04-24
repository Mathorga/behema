#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <portia/portia.h>
#include <portia/portia_cuda.h>

void print(cortex2d_t* cortex) {
    system("clear");
    for (cortex_size_t i = 0; i < cortex->height; i++) {
        for (cortex_size_t j = 0; j < cortex->width; j++) {
            neuron_t currentNeuron = cortex->neurons[IDX2D(j, i, cortex->width)];
            printf("%c ", currentNeuron.value > cortex->fire_threshold ? '@' : '.');
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    cortex2d_t* even_cortex;
    cortex2d_t* odd_cortex;

    cortex_size_t cortex_width = 150;
    cortex_size_t cortex_height = 50;
    nh_radius_t nh_radius = 1;

    srand(time(NULL));

    error_code_t error = c2d_init(&even_cortex, cortex_width, cortex_height, nh_radius);
    printf("ERROR %d", error);
    // c2d_init(&odd_cortex, cortex_width, cortex_height, nh_radius);
    // c2d_copy(odd_cortex, even_cortex);

    // for (int i = 0;; i++) {
    //     cortex2d_t* prev_cortex = i % 2 ? odd_cortex : even_cortex;
    //     cortex2d_t* next_cortex = i % 2 ? even_cortex : odd_cortex;

    //     if (rand() % 100 > 50) {
    //         c2d_rsfeed(prev_cortex, 0, inputs_count, DEFAULT_EXC_VALUE, 2);
    //     }

    //     c2d_tick(prev_cortex, next_cortex);

    //     print(next_cortex);

    //     usleep(40000);
    // }

    return 0;
}
