/*
*****************************************************************
cortex_pop.h

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

/// Population of 2D cortexes.
typedef struct {
    // Size of the population (number of contained cortexes).
    population_size_t size;

    // Size of the pool of fittest individuals.
    population_size_t survivors_size;

    // Amount of parents needed to generate offspring during crossover.
    population_size_t parents_count;

    // Chance of mutation during the evolution step.
    chance_t mut_chance;

    // Evaluation function.
    // TODO is this correct??
    cortex_fitness_t (*eval_function)(cortex2d_t* cortex);

    // List of all cortexes in the population.
    cortex2d_t* cortexes;

    // Cortexes' fitness.
    cortex_fitness_t* cortexes_fitness;

    // Indexes of all survivors to the current round of selection.
    population_size_t* survivors;
} population2d_t;

// ########################################## Initialization functions ##########################################

/// @brief Initializes the provided population with default values.
/// @param population The population to initialize.
/// @return The code for the occurred error, [ERROR_NONE] if none.
error_code_t p2d_init(population2d_t** population);


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