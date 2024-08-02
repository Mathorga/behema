/*
*****************************************************************
population.h

Copyright (C) 2024 Luka Micheletti
*****************************************************************
*/

#ifndef __CORTEX_POP__
#define __CORTEX_POP__

#include "cortex.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DEFAULT_POPULATION_SIZE 0x00FFU
#define DEFAULT_SURVIVORS_SIZE 0x0014U
#define DEFAULT_PARENTS_COUNT 0x0002U
#define DEFAULT_MUT_CHANCE 0x000002A0U

typedef uint16_t cortex_fitness_t;
typedef uint16_t population_size_t;

/// @brief Utility struct used to keep index consistency while working with fitness arrays.
typedef struct {
    population_size_t index;
    cortex_fitness_t fitness;
} indexed_fitness_t;

/// @brief Population of 2D cortices.
typedef struct {
    // Size of the population (number of contained cortices).
    population_size_t size;

    // Size of the pool of fittest individuals to be selected as reproductors.
    population_size_t sel_pool_size;

    // Amount of parents needed to generate offspring during crossover.
    population_size_t parents_count;

    // Chance of mutation during the evolution step.
    chance_t mut_chance;

    // Evaluation function.
    cortex_fitness_t (*eval_function)(cortex2d_t* cortex);

    // List of all cortices in the population.
    cortex2d_t* cortices;

    // cortices' fitness.
    cortex_fitness_t* cortices_fitness;

    // Indexes of all survivors to the current round of selection.
    population_size_t* survivors;
} population2d_t;


// ########################################## Utility functions ##########################################

/// @brief Compares the provided indexed fitnesses by fitness value.
/// @param a The first fitness to compare.
/// @param b The second fitness to compare.
/// @return 0 if a == b, a strictly negative number if a < b, a strictly positive if a > b.
int idf_compare(const void* a, const void* b);

// ########################################## Initialization functions ##########################################

/// @brief Initializes the provided population with default values.
/// @param population The population to initialize.
/// @param size The population size to start with.
/// @param sel_pool_size The size of the pool of fittest individuals.
/// @param mut_chance The probability of mutation for each evolution step.
/// @param eval_function The function used to evaluate each cortex.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_init(population2d_t** population, population_size_t size, population_size_t sel_pool_size, chance_t mut_chance, cortex_fitness_t (*eval_function)(cortex2d_t* cortex));

/// @brief Populates the starting pool of cortices with the provided values.
/// @param population The population whose cortices to setup.
/// @param width The width of the cortex.
/// @param height The height of the cortex.
/// @param nh_radius The neighborhood radius for each individual cortex neuron.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_populate(population2d_t* population, cortex_size_t width, cortex_size_t height, nh_radius_t nh_radius);

/// @brief Destroys the given cortex2d and frees memory for it and its neurons.
/// @param cortex The cortex to destroy
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_destroy(population2d_t* population);


// ########################################## Setter functions ##################################################

/// @brief Sets the provided population the appropriate mutation rate
/// @param population The population to apply the new mutation rate to.
/// @param mut_chance The mutation rate to apply to the population.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_set_mut_rate(population2d_t* population, chance_t mut_chance);


// ########################################## Action functions ##################################################

/// @brief Evaluates the provided population by individually evaluating each cortex and then populating their fitnes values.
/// @param population The population to evaluate.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_evaluate(population2d_t* population);

/// @brief Selects the fittest individuals in the given population and stores them for crossover.
/// @param population The population to select.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_select(population2d_t* population);

/// @brief Breeds the currently selected survivors and generates a new population starting from them.
/// @param population The population to breed.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_crossover(population2d_t* population);

/// @brief Mutates the given population in order to provide variability in the pool.
/// @param population the population to mutate.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_mutate(population2d_t* population);


#ifdef __cplusplus
}
#endif

#endif
