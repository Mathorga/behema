#include "population.h"

// ########################################## Initialization functions ##########################################

error_code_t p2d_init(population2d_t** population) {
    // Allocate the population.
    (*population) = (population2d_t *) malloc(sizeof(cortex2d_t));
    if ((*population) == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Setup population properties.
    (*population)->size = DEFAULT_POPULATION_SIZE;
    (*population)->survivors_size = DEFAULT_SURVIVORS_SIZE;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = DEFAULT_MUT_CHANCE;

    // Allocate cortexes.
    (*population)->cortexes = (cortex2d_t *) malloc((*population)->size * sizeof(cortex2d_t));
    if ((*population)->cortexes == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Allocate fitnesses.
    (*population)->cortexes_fitness = (cortex_fitness_t *) malloc((*population)->size * sizeof(cortex_fitness_t));
    if ((*population)->cortexes_fitness == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Allocate survivors.
    (*population)->survivors = (population_size_t *) malloc((*population)->survivors_size * sizeof(population_size_t));
    if ((*population)->survivors == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    return ERROR_NONE;
}