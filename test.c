#include <stdio.h>
#include <stdint.h>

typedef uint16_t neighbors_count_t;
typedef int16_t neuron_value_t;

typedef struct {
    neighbors_count_t connected_neighbors;
    neuron_value_t value;
} neuron_t;

int main(int argc, char **argv) {
}