#include "population.h"


// ########################################## Initialization functions ##########################################

int idf_compare(const void* a, const void* b) {
    return (*(bhm_indexed_fitness_t*)a).fitness - (*(bhm_indexed_fitness_t*)b).fitness;
}


// ########################################## Initialization functions ##########################################

bhm_error_code_t p2d_init(bhm_population2d_t** population, bhm_population_size_t size, bhm_population_size_t sel_pool_size, bhm_chance_t mut_chance, bhm_cortex_fitness_t (*eval_function)(bhm_cortex2d_t* cortex)) {
    // Allocate the population.
    (*population) = (bhm_population2d_t *) malloc(sizeof(bhm_cortex2d_t));
    if ((*population) == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Setup population properties.
    (*population)->size = size;
    (*population)->sel_pool_size = sel_pool_size;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = mut_chance;
    (*population)->eval_function = eval_function;

    // Allocate cortices.
    (*population)->cortices = (bhm_cortex2d_t *) malloc((*population)->size * sizeof(bhm_cortex2d_t));
    if ((*population)->cortices == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Allocate fitnesses.
    (*population)->cortices_fitness = (bhm_cortex_fitness_t *) malloc((*population)->size * sizeof(bhm_cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Allocate selection pool.
    (*population)->survivors = (bhm_population_size_t *) malloc((*population)->sel_pool_size * sizeof(bhm_population_size_t));
    if ((*population)->survivors == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_populate(bhm_population2d_t* population, bhm_cortex_size_t width, bhm_cortex_size_t height, bhm_nh_radius_t nh_radius) {
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Allocate a temporary pointer to the ith cortex.
        bhm_cortex2d_t* cortex = &(population->cortices[i]);

        // Init the ith cortex.
        bhm_error_code_t error = c2d_init(&cortex, width, height, nh_radius);

        if (error != BHM_ERROR_NONE) {
            // There was an error initializing a cortex, so abort population setup, clean what's been initialized up to now and return the error.
            for (bhm_population_size_t j = 0; j < i - 1; j++) {
                // Destroy the jth cortex.
                c2d_destroy(&(population->cortices[j]));
            }
            return error;
        }
    }

    return BHM_ERROR_NONE;
}


// ########################################## Setter functions ##################################################

bhm_error_code_t p2d_set_mut_rate(bhm_population2d_t* population, bhm_chance_t mut_chance) {
    population->mut_chance = mut_chance;

    return BHM_ERROR_NONE;
}


// ########################################## Action functions ##################################################

bhm_error_code_t p2d_evaluate(bhm_population2d_t* population) {
    // Loop through all cortices to evaluate each of them.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Evaluate the current cortex by using the population evaluation function.
        // The computed fitness is stored in the population itself.
        population->cortices_fitness[i] = population->eval_function(&(population->cortices[i]));
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_select(bhm_population2d_t* population) {
    // Allocate temporary fitnesses.
    bhm_indexed_fitness_t* sorted_indexes = (bhm_indexed_fitness_t*) malloc(population->size * sizeof(bhm_indexed_fitness_t));

    // Populate temp indexes.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        sorted_indexes[i].index = i;
        sorted_indexes[i].fitness = population->cortices_fitness[i];
    }

    // Sort cortex fitnesses.
    qsort(sorted_indexes, population->size, sizeof(bhm_indexed_fitness_t), idf_compare);

    // Pick the best-fitting cortices and store them as survivors.
    // Survivors are by definition the cortices correspondint to the first elements in the sorted list of fitnesses.
    for (bhm_population_size_t i = 0; i < population->sel_pool_size; i++) {
        population->survivors[i] = sorted_indexes[i].index;
    }

    // Free up temp indexes array.
    free(sorted_indexes);

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_crossover(bhm_population2d_t* population) {
    // TODO.

    return BHM_ERROR_NONE;
}