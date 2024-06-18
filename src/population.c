#include "population.h"

// ########################################## Initialization functions ##########################################

error_code_t p2d_init(population2d_t** population, population_size_t size, population_size_t sel_pool_size, chance_t mut_chance, cortex_fitness_t (*eval_function)(cortex2d_t* cortex)) {
    // Allocate the population.
    (*population) = (population2d_t *) malloc(sizeof(cortex2d_t));
    if ((*population) == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Setup population properties.
    (*population)->size = size;
    (*population)->sel_pool_size = sel_pool_size;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = mut_chance;
    (*population)->eval_function = eval_function;

    // Allocate cortices.
    (*population)->cortices = (cortex2d_t *) malloc((*population)->size * sizeof(cortex2d_t));
    if ((*population)->cortices == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Allocate fitnesses.
    (*population)->cortices_fitness = (cortex_fitness_t *) malloc((*population)->size * sizeof(cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    // Allocate selection pool.
    (*population)->survivors = (population_size_t *) malloc((*population)->sel_pool_size * sizeof(population_size_t));
    if ((*population)->survivors == NULL) {
        return ERROR_FAILED_ALLOC;
    }

    return ERROR_NONE;
}

error_code_t p2d_populate(population2d_t* population, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius) {
    for (population_size_t i = 0; i < population->size; i++) {
        // Allocate a temporary pointer to the ith cortex.
        cortex2d_t* cortex = &(population->cortices[i]);

        // Init the ith cortex.
        error_code_t error = c2d_init(&cortex, width, height, nh_radius);

        if (error != ERROR_NONE) {
            // There was an error initializing a cortex, so abort population setup, clean what's been initialized up to now and return the error.
            for (population_size_t j = 0; j < i - 1; j++) {
                // Destroy the jth cortex.
                c2d_destroy(&(population->cortices[j]));
            }
            return error;
        }
    }

    return ERROR_NONE;
}

// ########################################## Setter functions ##################################################

error_code_t p2d_set_mut_rate(population2d_t* population, chance_t mut_chance) {
    population->mut_chance = mut_chance;

    return ERROR_NONE;
}

// ########################################## Action functions ##################################################

error_code_t p2d_evaluate(population2d_t* population) {
    // Loop through all cortices to evaluate each of them.
    for (int i = 0; i < population->size; i++) {
        // Evaluate the current cortex by using the population evaluation function.
        // The computed fitness is stored in the population itself.
        population->cortices_fitness[i] = population->eval_function(&(population->cortices[i]));
    }

    return ERROR_NONE;
}

error_code_t p2d_select(population2d_t* population) {
    // TODO Sort cortices by fitness.
    population_size_t* sorted_indexes = (population_size_t*) malloc(population->size * sizeof(population_size_t));

    // Pick the best-fitting cortices and store them as survivors.
    population->survivors = sorted_indexes;

    // Free up temp indexes array.
    free(sorted_indexes);

    return ERROR_NONE;
}