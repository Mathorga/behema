#include "population.h"


// ##########################################
// Utility functions.
// ##########################################

int idf_compare_desc(const void* a, const void* b) {
    return (*(bhm_indexed_fitness_t*)b).fitness - (*(bhm_indexed_fitness_t*)a).fitness;
}

int idf_compare_asc(const void* a, const void* b) {
    return (*(bhm_indexed_fitness_t*)a).fitness - (*(bhm_indexed_fitness_t*)b).fitness;
}

// ##########################################
// ##########################################


// ##########################################
// Initialization functions.
// ##########################################

bhm_error_code_t p2d_init(
    bhm_population2d_t** population,
    bhm_population_size_t size,
    bhm_population_size_t selection_pool_size,
    bhm_chance_t mut_chance,
    bhm_error_code_t (*eval_function)(bhm_cortex2d_t* cortex, bhm_cortex_fitness_t* fitness)
) {
    // Allocate the population.
    (*population) = (bhm_population2d_t *) malloc(sizeof(bhm_cortex2d_t));
    if ((*population) == NULL) return BHM_ERROR_FAILED_ALLOC;

    // Make sure the selection pool size does not exceed the total population size
    // since it would make no sense.
    if (selection_pool_size > size) return BHM_ERROR_SIZE_WRONG;

    // Setup population properties.
    (*population)->size = size;
    (*population)->selection_pool_size = selection_pool_size;
    (*population)->parents_count = DEFAULT_PARENTS_COUNT;
    (*population)->mut_chance = mut_chance;
    (*population)->rand_state = BHM_STARTING_RAND;
    (*population)->eval_function = eval_function;

    // Allocate cortices.
    (*population)->cortices = (bhm_cortex2d_t*) malloc((*population)->size * sizeof(bhm_cortex2d_t));
    if ((*population)->cortices == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Allocate fitnesses.
    (*population)->cortices_fitness = (bhm_cortex_fitness_t*) malloc((*population)->size * sizeof(bhm_cortex_fitness_t));
    if ((*population)->cortices_fitness == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Allocate selection pool.
    (*population)->selection_pool = (bhm_population_size_t*) malloc((*population)->selection_pool_size * sizeof(bhm_population_size_t));
    if ((*population)->selection_pool == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_populate(
    bhm_population2d_t* population,
    bhm_cortex_size_t width,
    bhm_cortex_size_t height,
    bhm_nh_radius_t nh_radius
) {
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Randomly init the ith cortex.
        bhm_error_code_t error = c2d_init(&(population->cortices[i]), width, height, nh_radius);
        if (error != BHM_ERROR_NONE) {
            // There was an error initializing a cortex, so abort population setup, clean what's been initialized up to now and return the error.
            for (bhm_population_size_t j = 0; j < i - 1; j++) {
                // Destroy the jth cortex.
                c2d_destroy(&(population->cortices[j]));
            }
            return error;
        }

        population->cortices[i].rand_state = population->rand_state + BHM_STARTING_RAND * i;
    }

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_rand_populate(
    bhm_population2d_t* population,
    bhm_cortex_size_t width,
    bhm_cortex_size_t height,
    bhm_nh_radius_t nh_radius
) {
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Randomly init the ith cortex.
        bhm_error_code_t error = c2d_rand_init(&(population->cortices[i]), width, height, nh_radius);
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

bhm_error_code_t p2d_destroy_cortices(bhm_population2d_t* population) {
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        free(population->cortices[i].neurons);
    }

    free(population->cortices);

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_destroy(bhm_population2d_t* population) {
    bhm_error_code_t error = p2d_destroy_cortices(population);
    if (error != BHM_ERROR_NONE) return error;

    free(population->cortices_fitness);
    free(population->selection_pool);
    free(population);

    return BHM_ERROR_NONE;
}

// ##########################################
// ##########################################


// ##########################################
// Setter functions.
// ##########################################

bhm_error_code_t p2d_set_mut_rate(bhm_population2d_t* population, bhm_chance_t mut_chance) {
    population->mut_chance = mut_chance;

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_set_eval_function(bhm_population2d_t* population, bhm_error_code_t (*eval_function)(bhm_cortex2d_t* cortex, bhm_cortex_fitness_t* fitness)) {
    population->eval_function = eval_function;

    return BHM_ERROR_NONE;
}

// ##########################################
// ##########################################


// ##########################################
// Action functions.
// ##########################################

bhm_error_code_t p2d_evaluate(bhm_population2d_t* population) {
    // Loop through all cortices to evaluate each of them.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Evaluate the current cortex by using the population evaluation function.
        // The computed fitness is stored in the population itself.
        bhm_error_code_t error = population->eval_function(&(population->cortices[i]), &(population->cortices_fitness[i]));
        if (error != BHM_ERROR_NONE) {
            return error;
        }
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

    // Sort cortex fitnesses descending.
    qsort(sorted_indexes, population->size, sizeof(bhm_indexed_fitness_t), idf_compare_desc);

    // Pick the best-fitting cortices and store them as selection_pool.
    // Survivors are by definition the cortices correspondint to the first elements in the sorted list of fitnesses.
    for (bhm_population_size_t i = 0; i < population->selection_pool_size; i++) {
        population->selection_pool[i] = sorted_indexes[i].index;
    }

    // Free up temp indexes array.
    free(sorted_indexes);

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_breed(bhm_population2d_t* population, bhm_cortex2d_t** child) {
    // Allocate parents.
    bhm_cortex2d_t* parents = (bhm_cortex2d_t*) malloc(population->parents_count * sizeof(bhm_cortex2d_t));
    if (parents == NULL) return BHM_ERROR_FAILED_ALLOC;
    bhm_population_size_t* parents_indexes = (bhm_population_size_t*) malloc(population->parents_count * sizeof(bhm_population_size_t));
    if (parents_indexes == NULL) return BHM_ERROR_FAILED_ALLOC;

    // Pick parents from the selection pool.
    for (bhm_population_size_t i = 0; i < population->parents_count; i++) {
        bhm_population_size_t parent_index;
        bhm_bool_t index_is_valid;

        do {
            // Pick a random parent.
            population->rand_state = xorshf32(population->rand_state);
            parent_index = population->selection_pool[population->rand_state % population->selection_pool_size];
            index_is_valid = BHM_TRUE;

            // Make sure the selected index is not already been selected.
            for (bhm_population_size_t j = 0; j < i; j++) {
                if (parents_indexes[j] == parent_index) {
                    index_is_valid = BHM_FALSE;
                }
            }
        } while (!index_is_valid);

        parents_indexes[i] = parent_index;
        parents[i] = population->cortices[parent_index];
    }

    bhm_population_size_t winner_parent_index;

    // Pick width and height from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    bhm_cortex_size_t child_width = parents[winner_parent_index].width;
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    bhm_cortex_size_t child_height = parents[winner_parent_index].height;

    // Init child with default values.
    bhm_error_code_t error = c2d_create(
        child,
        child_width,
        child_height,
        parents[0].nh_radius
    );
    if (error != BHM_ERROR_NONE) return error;

    // Pick pulse window from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_window(*child, parents[winner_parent_index].pulse_window);
    if (error != BHM_ERROR_NONE) return error;

    // Pick fire threshold from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_fire_threshold(*child, parents[winner_parent_index].fire_threshold);
    if (error != BHM_ERROR_NONE) return error;

    // TODO Set recovery value and exc/decay values.

    // Pick syngen chance from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_syngen_chance(*child, parents[winner_parent_index].syngen_chance);
    if (error != BHM_ERROR_NONE) return error;

    // Pick synstrength chance from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_synstr_chance(*child, parents[winner_parent_index].synstr_chance);
    if (error != BHM_ERROR_NONE) return error;

    // TODO Set max tot strength.

    // Pick max syn count from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_max_syn_count(*child, parents[winner_parent_index].max_tot_strength);
    if (error != BHM_ERROR_NONE) return error;

    // Pick inhexc range from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_inhexc_range(*child, parents[winner_parent_index].inhexc_range);
    if (error != BHM_ERROR_NONE) return error;

    // Pick sample window from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_sample_window(*child, parents[winner_parent_index].sample_window);
    if (error != BHM_ERROR_NONE) return error;

    // Pick pulse mapping from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    error = c2d_set_pulse_mapping(*child, parents[winner_parent_index].pulse_mapping);
    if (error != BHM_ERROR_NONE) return error;

    // Pick neurons' max syn count from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    bhm_cortex2d_t msc_parent = parents[winner_parent_index];

    // Pick neurons' inhexc ratio from a random parent.
    population->rand_state = xorshf32(population->rand_state);
    winner_parent_index = population->rand_state % population->parents_count;
    bhm_cortex2d_t inhexc_parent = parents[winner_parent_index];

    // Pick neuron values from parents.
    for (bhm_cortex_size_t y = 0; y < (*child)->height; y++) {
        for (bhm_cortex_size_t x = 0; x < (*child)->width; x++) {
            (*child)->neurons[IDX2D(x, y, (*child)->width)].max_syn_count = msc_parent.neurons[IDX2D(x, y, (*child)->width)].max_syn_count;
            (*child)->neurons[IDX2D(x, y, (*child)->width)].inhexc_ratio = inhexc_parent.neurons[IDX2D(x, y, (*child)->width)].inhexc_ratio;
        }
    }

    // Free up temp arrays.
    free(parents);
    free(parents_indexes);

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_crossover(bhm_population2d_t* population, bhm_bool_t mutate) {
    bhm_error_code_t error;

    // Create a temp population to hold the new generation.
    bhm_cortex2d_t* offspring = (bhm_cortex2d_t*) malloc(population->size * sizeof(bhm_cortex2d_t));
    if (offspring == NULL) {
        return BHM_ERROR_FAILED_ALLOC;
    }

    // Breed the selection pool and create children for the new generation.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Create a new child by breeding parents from the population's selection pool.
        bhm_cortex2d_t* child;
        error = p2d_breed(population, &child);
        if (error != BHM_ERROR_NONE) {
            return error;
        }

        child->rand_state = population->rand_state + BHM_STARTING_RAND * i;

        // Mutate the newborn if so specified.
        if (mutate) {
            error = c2d_mutate(child, population->mut_chance);
            if (error != BHM_ERROR_NONE) {
                return error;
            }
        }

        // Store the produced child.
        offspring[i] = *child;
    }

    // Replace the old generation with the new one.
    error = p2d_destroy_cortices(population);
    population->cortices = offspring;

    return BHM_ERROR_NONE;
}

bhm_error_code_t p2d_mutate(bhm_population2d_t* population) {
    // Mutate each cortex in the population.
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        bhm_error_code_t error = c2d_mutate(&(population->cortices[i]), population->mut_chance);
        if (error != BHM_ERROR_NONE) {
            return error;
        }
    }

    return BHM_ERROR_NONE;
}

// ##########################################
// ##########################################