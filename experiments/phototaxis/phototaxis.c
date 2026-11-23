/*
*****************************************************************
phototaxis.c

Phototaxis experiment, step 1: can the genetic algorithm evolve cortices that steer a simulated robot towards a light?
Evolution only: there is no reward signal during the robot's life.

The world is a 2D plane with a light and a round robot with two wheels and two light sensors, angled left and right.
Each sensor feeds an input patch of the cortex (more light -> higher firing rate), and each wheel's speed is read from an
output patch (higher firing rate -> faster wheel). Every control step the sensors are read, the cortex runs for one
sample window, and the robot moves according to its wheels.

Cortex layout (not to scale):
    +------------------------+
    | [sensor L]   [wheel L] |
    |                        |
    | [sensor R]   [wheel R] |
    +------------------------+
Each sensor is close to the wheel on its own side, so plain excitation spreading from a sensor to its nearest wheel makes
the robot turn away from the light. Turning towards it needs the opposite (crossed excitation, or uncrossed inhibition),
which evolution has to find.

Copyright (C) 2026 Luka Micheletti
*****************************************************************
*/

#include <stdio.h>
#include <math.h>

#include "behema.h"

#define PT_PI 3.14159265358979323846

// ##########################################
// World.
// ##########################################

// Distance between the robot and the light at the start of each episode.
#define PT_LIGHT_DISTANCE 30.0
// Distance at which the robot is considered to have reached the light.
#define PT_REACH_DISTANCE 1.5
// Control steps per episode.
#define PT_STEPS_COUNT 150
// Max distance a wheel can travel in a single control step.
#define PT_MAX_SPEED 0.5
// Distance between the two wheels: the larger it is, the slower the robot turns.
#define PT_WHEEL_BASE 2.0
// Angle between the robot's heading and each sensor's direction.
#define PT_SENSOR_ANGLE (PT_PI / 4.0)
// Distance at which light intensity halves.
#define PT_LIGHT_FALLOFF 30.0

// Light directions relative to the robot's initial heading, one per episode. Symmetric, so that always turning the same way doesn't pay off.
static const double pt_light_angles[] = {PT_PI / 2.0, PT_PI / 4.0, -PT_PI / 4.0, -PT_PI / 2.0};
#define PT_EPISODES_COUNT (sizeof(pt_light_angles) / sizeof(double))

typedef struct {
    double x;
    double y;
    double heading;
} pt_robot_t;

/// @brief Computes the light intensity read by one of the robot's sensors, between 0 and 1.
/// @param side +1 for the left sensor, -1 for the right one.
static double pt_sensor_intensity(
    pt_robot_t* robot,
    double light_x,
    double light_y,
    double side
) {
    double dx = light_x - robot->x;
    double dy = light_y - robot->y;
    double distance_squared = dx * dx + dy * dy;

    // Sensors only see light in front of them: full intensity when pointing straight at it, none at 90 degrees or more.
    double sensor_direction = robot->heading + side * PT_SENSOR_ANGLE;
    double alignment = cos(atan2(dy, dx) - sensor_direction);
    if (alignment < 0.0) alignment = 0.0;

    // Intensity decreases with distance, halving at PT_LIGHT_FALLOFF.
    return alignment * (PT_LIGHT_FALLOFF * PT_LIGHT_FALLOFF) / (PT_LIGHT_FALLOFF * PT_LIGHT_FALLOFF + distance_squared);
}

/// @brief Moves the robot according to its wheels' speeds (differential drive). Speeds are between 0 and 1.
static void pt_move_robot(
    pt_robot_t* robot,
    double left_speed,
    double right_speed
) {
    double left = left_speed * PT_MAX_SPEED;
    double right = right_speed * PT_MAX_SPEED;

    // A faster right wheel turns the robot left (counterclockwise), and vice versa.
    robot->heading += (right - left) / PT_WHEEL_BASE;
    robot->x += cos(robot->heading) * (left + right) / 2.0;
    robot->y += sin(robot->heading) * (left + right) / 2.0;
}


// ##########################################
// Controllers.
// ##########################################

/// @brief A controller maps sensor intensities to wheel speeds (all between 0 and 1). [state] is controller-specific.
typedef bhm_error_code_t (*pt_controller_t)(void* state, double left_intensity, double right_intensity, double* left_speed, double* right_speed);

// ########## Cortex controller ##########

// Input and output patches, see the layout in the header.
#define PT_CORTEX_WIDTH 20
#define PT_CORTEX_HEIGHT 12
#define PT_NH_RADIUS 2
#define PT_PATCH_SIZE 4
#define PT_PATCH_MARGIN 1

typedef struct {
    bhm_context2d_t* context;
    bhm_input2d_t* left_sensor;
    bhm_input2d_t* right_sensor;
    bhm_output2d_t* left_wheel;
    bhm_output2d_t* right_wheel;
} pt_brain_t;

/// @brief Creates a brain (a context with its inputs and outputs) from the given genome cortex.
/// The genome is copied, so every brain starts from the same "newborn" state.
static bhm_error_code_t pt_brain_create(
    pt_brain_t* brain,
    bhm_cortex2d_t* genome
) {
    bhm_error_code_t error;

    error = bhm_ctx2d_create(&(brain->context), genome->width, genome->height, genome->nh_radius);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_crx2d_copy(brain->context->even_cortex, genome);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_crx2d_copy(brain->context->odd_cortex, genome);
    if (error != BHM_ERROR_NONE) return error;

    bhm_cortex_size_t left_x0 = PT_PATCH_MARGIN;
    bhm_cortex_size_t right_x0 = genome->width - PT_PATCH_MARGIN - PT_PATCH_SIZE;
    bhm_cortex_size_t top_y0 = PT_PATCH_MARGIN;
    bhm_cortex_size_t bottom_y0 = genome->height - PT_PATCH_MARGIN - PT_PATCH_SIZE;

    // Sensors on the left side of the cortex: left sensor on top, right sensor at the bottom.
    error = bhm_i2d_create(&(brain->left_sensor), left_x0, top_y0, left_x0 + PT_PATCH_SIZE, top_y0 + PT_PATCH_SIZE, BHM_DEFAULT_EXC_VALUE * 2, BHM_PULSE_MAPPING_FPROP);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_i2d_create(&(brain->right_sensor), left_x0, bottom_y0, left_x0 + PT_PATCH_SIZE, bottom_y0 + PT_PATCH_SIZE, BHM_DEFAULT_EXC_VALUE * 2, BHM_PULSE_MAPPING_FPROP);
    if (error != BHM_ERROR_NONE) return error;

    // Wheels on the right side of the cortex: left wheel on top, right wheel at the bottom.
    error = bhm_o2d_init(&(brain->left_wheel), right_x0, top_y0, right_x0 + PT_PATCH_SIZE, top_y0 + PT_PATCH_SIZE);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_o2d_init(&(brain->right_wheel), right_x0, bottom_y0, right_x0 + PT_PATCH_SIZE, bottom_y0 + PT_PATCH_SIZE);
    if (error != BHM_ERROR_NONE) return error;

    // The context owns its inputs and outputs from here on.
    error = bhm_ctx2d_add_input(brain->context, brain->left_sensor);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_add_input(brain->context, brain->right_sensor);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_add_output(brain->context, brain->left_wheel);
    if (error != BHM_ERROR_NONE) return error;
    error = bhm_ctx2d_add_output(brain->context, brain->right_wheel);
    if (error != BHM_ERROR_NONE) return error;

    return BHM_ERROR_NONE;
}

/// @brief Sets all of an input's values from an intensity between 0 and 1.
static void pt_set_input(
    bhm_input2d_t* input,
    bhm_ticks_count_t sample_window,
    double intensity
) {
    // Values go from 0 to sample_window - 1. Values equal to the sample window never fire (see value_to_pulse),
    // which is used for darkness: 0 would still fire once per window.
    bhm_ticks_count_t value = (bhm_ticks_count_t) round(intensity * (sample_window - 1));
    if (value == 0) value = sample_window;

    for (bhm_cortex_size_t i = 0; i < (input->x1 - input->x0) * (input->y1 - input->y0); i++) {
        input->values[i] = value;
    }
}

/// @brief Reads an output's mean firing rate, between 0 (never fires) and 1 (fires at every tick of the pulse window).
static double pt_read_output(
    bhm_output2d_t* output,
    bhm_ticks_count_t pulse_window
) {
    double sum = 0.0;
    bhm_cortex_size_t count = (output->x1 - output->x0) * (output->y1 - output->y0);

    for (bhm_cortex_size_t i = 0; i < count; i++) {
        sum += output->values[i];
    }

    double rate = sum / count / (pulse_window + 1);
    return rate > 1.0 ? 1.0 : rate;
}

static bhm_error_code_t pt_cortex_controller(
    void* state,
    double left_intensity,
    double right_intensity,
    double* left_speed,
    double* right_speed
) {
    pt_brain_t* brain = (pt_brain_t*) state;
    bhm_cortex2d_t* cortex = brain->context->even_cortex;

    pt_set_input(brain->left_sensor, cortex->sample_window, left_intensity);
    pt_set_input(brain->right_sensor, cortex->sample_window, right_intensity);

    // Run the cortex for one full sample window, so that inputs are fully encoded.
    for (bhm_ticks_count_t t = 0; t < cortex->sample_window; t++) {
        bhm_error_code_t error = ctx2d_tick(brain->context);
        if (error != BHM_ERROR_NONE) return error;
    }

    *left_speed = pt_read_output(brain->left_wheel, cortex->pulse_window);
    *right_speed = pt_read_output(brain->right_wheel, cortex->pulse_window);

    return BHM_ERROR_NONE;
}

// ########## Fixed controllers, used as baselines ##########

static bhm_error_code_t pt_stationary_controller(void* state, double left_intensity, double right_intensity, double* left_speed, double* right_speed) {
    *left_speed = 0.0;
    *right_speed = 0.0;
    return BHM_ERROR_NONE;
}

static bhm_error_code_t pt_straight_controller(void* state, double left_intensity, double right_intensity, double* left_speed, double* right_speed) {
    *left_speed = 1.0;
    *right_speed = 1.0;
    return BHM_ERROR_NONE;
}

// Braitenberg vehicles: each wheel's speed is a base speed plus a sensor's intensity.
#define PT_BRAITENBERG_GAIN 3.0
#define PT_BRAITENBERG_BASE 0.2

/// @brief Braitenberg's vehicle 2a ("fear"): each sensor drives the wheel on its own side, so the robot turns away from light.
static bhm_error_code_t pt_uncrossed_controller(void* state, double left_intensity, double right_intensity, double* left_speed, double* right_speed) {
    *left_speed = fmin(1.0, PT_BRAITENBERG_BASE + PT_BRAITENBERG_GAIN * left_intensity);
    *right_speed = fmin(1.0, PT_BRAITENBERG_BASE + PT_BRAITENBERG_GAIN * right_intensity);
    return BHM_ERROR_NONE;
}

/// @brief Braitenberg's vehicle 2b ("aggression"): each sensor drives the opposite wheel, so the robot turns towards light.
static bhm_error_code_t pt_crossed_controller(void* state, double left_intensity, double right_intensity, double* left_speed, double* right_speed) {
    *left_speed = fmin(1.0, PT_BRAITENBERG_BASE + PT_BRAITENBERG_GAIN * right_intensity);
    *right_speed = fmin(1.0, PT_BRAITENBERG_BASE + PT_BRAITENBERG_GAIN * left_intensity);
    return BHM_ERROR_NONE;
}


// ##########################################
// Episodes and fitness.
// ##########################################

/// @brief Runs a single episode with the light at the given angle, and computes the episode's score: the mean closeness to
/// the light over all steps, where closeness goes from 0 (initial distance or farther) to 1 (light reached).
/// Reaching the light early counts as being on it for the remaining steps, so faster robots score higher.
/// @param trace If not NULL, the robot's position at each step is written to it as CSV rows.
static bhm_error_code_t pt_run_episode(
    pt_controller_t controller,
    void* state,
    double light_angle,
    FILE* trace,
    uint32_t trace_episode,
    double* score
) {
    pt_robot_t robot = {0.0, 0.0, 0.0};
    double light_x = PT_LIGHT_DISTANCE * cos(light_angle);
    double light_y = PT_LIGHT_DISTANCE * sin(light_angle);
    double closeness_sum = 0.0;

    for (uint32_t step = 0; step < PT_STEPS_COUNT; step++) {
        double distance = hypot(light_x - robot.x, light_y - robot.y);

        if (distance < PT_REACH_DISTANCE) {
            closeness_sum += PT_STEPS_COUNT - step;
            break;
        }

        double closeness = 1.0 - distance / PT_LIGHT_DISTANCE;
        closeness_sum += closeness > 0.0 ? closeness : 0.0;

        if (trace != NULL) {
            fprintf(trace, "%u,%u,%.3f,%.3f,%.3f,%.3f\n", trace_episode, step, robot.x, robot.y, light_x, light_y);
        }

        double left_speed;
        double right_speed;
        bhm_error_code_t error = controller(
            state,
            pt_sensor_intensity(&robot, light_x, light_y, 1.0),
            pt_sensor_intensity(&robot, light_x, light_y, -1.0),
            &left_speed,
            &right_speed
        );
        if (error != BHM_ERROR_NONE) return error;

        pt_move_robot(&robot, left_speed, right_speed);
    }

    *score = closeness_sum / PT_STEPS_COUNT;

    return BHM_ERROR_NONE;
}

/// @brief Runs all episodes with a fixed controller and returns the mean score.
static double pt_evaluate_fixed(
    pt_controller_t controller
) {
    double total = 0.0;

    for (uint32_t e = 0; e < PT_EPISODES_COUNT; e++) {
        double score;
        pt_run_episode(controller, NULL, pt_light_angles[e], NULL, 0, &score);
        total += score;
    }

    return total / PT_EPISODES_COUNT;
}

/// @brief Runs all episodes with a brain grown from the given genome (a new one for each episode) and returns the mean score.
static bhm_error_code_t pt_evaluate_genome(
    bhm_cortex2d_t* genome,
    FILE* trace,
    double* scores,
    double* mean_score
) {
    double total = 0.0;

    for (uint32_t e = 0; e < PT_EPISODES_COUNT; e++) {
        pt_brain_t brain;
        bhm_error_code_t error = pt_brain_create(&brain, genome);
        if (error != BHM_ERROR_NONE) return error;

        double score;
        error = pt_run_episode(pt_cortex_controller, &brain, pt_light_angles[e], trace, e, &score);
        if (error != BHM_ERROR_NONE) return error;

        if (scores != NULL) scores[e] = score;
        total += score;

        bhm_ctx2d_destroy(brain.context);
    }

    *mean_score = total / PT_EPISODES_COUNT;

    return BHM_ERROR_NONE;
}

/// @brief Population evaluation function. Thread-safe: everything it uses is local.
static bhm_error_code_t pt_fitness(
    bhm_cortex2d_t* genome,
    bhm_cortex_fitness_t* fitness
) {
    double score;
    bhm_error_code_t error = pt_evaluate_genome(genome, NULL, NULL, &score);
    if (error != BHM_ERROR_NONE) return error;

    // Fitness is a 16-bit integer, scores are between 0 and 1.
    *fitness = (bhm_cortex_fitness_t) (score * 0xFFFFU);

    return BHM_ERROR_NONE;
}


// ##########################################
// Evolution.
// ##########################################

#define PT_POPULATION_SIZE 48
#define PT_SELECTION_POOL_SIZE 8
#define PT_MUT_CHANCE 0x1000U
#define PT_GENERATIONS_COUNT 40
// Rounds of mutation applied to the initial population, so that it starts with some diversity.
#define PT_INITIAL_MUTATIONS 8

static double pt_percent(bhm_cortex_fitness_t fitness) {
    return 100.0 * fitness / 0xFFFFU;
}

int main(int argc, char **argv) {
    bhm_error_code_t error;

    // Baselines.
    printf("Baselines (mean closeness to the light, 0-100%%):\n");
    printf("    stationary:                       %5.1f%%\n", 100.0 * pt_evaluate_fixed(pt_stationary_controller));
    printf("    straight ahead:                   %5.1f%%\n", 100.0 * pt_evaluate_fixed(pt_straight_controller));
    printf("    Braitenberg 2a (uncrossed, fear): %5.1f%%\n", 100.0 * pt_evaluate_fixed(pt_uncrossed_controller));
    printf("    Braitenberg 2b (crossed, seeks):  %5.1f%%\n\n", 100.0 * pt_evaluate_fixed(pt_crossed_controller));

    // Population init.
    bhm_population2d_t* population;
    error = p2d_init(&population, PT_POPULATION_SIZE, PT_SELECTION_POOL_SIZE, PT_MUT_CHANCE, pt_fitness);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error initializing the population: %d\n", error);
        return 1;
    }
    error = p2d_populate(population, PT_CORTEX_WIDTH, PT_CORTEX_HEIGHT, PT_NH_RADIUS);
    if (error != BHM_ERROR_NONE) {
        printf("There was an error populating the population: %d\n", error);
        return 1;
    }
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        // Proportional mapping, so that firing rates grow roughly linearly with light intensity.
        bhm_crx2d_set_pulse_mapping(&(population->cortices[i]), BHM_PULSE_MAPPING_FPROP);
    }
    for (uint32_t i = 0; i < PT_INITIAL_MUTATIONS; i++) {
        p2d_mutate(population, BHM_FALSE);
    }

    // Evolve.
    printf(" gen |  best |  mean | best's syngen synstr threshold pulse_w max_syn max_str\n");
    printf("-----+-------+-------+----------------------------------------------------\n");
    for (uint32_t generation = 0; generation < PT_GENERATIONS_COUNT; generation++) {
        error = p2d_evaluate(population);
        if (error != BHM_ERROR_NONE) {
            printf("There was an error evaluating the population: %d\n", error);
            return 1;
        }

        bhm_population_size_t best = 0;
        double fitness_sum = 0.0;
        for (bhm_population_size_t i = 0; i < population->size; i++) {
            if (population->cortices_fitness[i] > population->cortices_fitness[best]) best = i;
            fitness_sum += population->cortices_fitness[i];
        }
        bhm_cortex2d_t* best_cortex = &(population->cortices[best]);
        printf(" %3u | %4.1f%% | %4.1f%% |   %6u %6u %9d %7u %7u %7u\n",
            generation,
            pt_percent(population->cortices_fitness[best]),
            100.0 * fitness_sum / population->size / 0xFFFFU,
            best_cortex->syngen_chance,
            best_cortex->synstr_chance,
            best_cortex->fire_threshold,
            best_cortex->pulse_window,
            best_cortex->max_syn_count,
            best_cortex->max_tot_strength);
        fflush(stdout);

        // The last generation is only evaluated, so that its best individual can be inspected below.
        if (generation == PT_GENERATIONS_COUNT - 1) break;

        error = p2d_select(population);
        if (error != BHM_ERROR_NONE) return 1;
        error = p2d_crossover(population, BHM_TRUE, BHM_FALSE);
        if (error != BHM_ERROR_NONE) return 1;
    }

    // Inspect the best individual of the last generation.
    bhm_population_size_t best = 0;
    for (bhm_population_size_t i = 0; i < population->size; i++) {
        if (population->cortices_fitness[i] > population->cortices_fitness[best]) best = i;
    }

    // Its trajectories are written to a CSV file, for plotting.
    FILE* trace = fopen("bin/phototaxis_best.csv", "w");
    if (trace != NULL) fprintf(trace, "episode,step,robot_x,robot_y,light_x,light_y\n");

    double scores[PT_EPISODES_COUNT];
    double mean_score;
    error = pt_evaluate_genome(&(population->cortices[best]), trace, scores, &mean_score);
    if (trace != NULL) fclose(trace);
    if (error != BHM_ERROR_NONE) return 1;

    printf("\nBest of the last generation: %.1f%%\n", 100.0 * mean_score);
    for (uint32_t e = 0; e < PT_EPISODES_COUNT; e++) {
        printf("    light at %+4.0f degrees: %5.1f%%\n", pt_light_angles[e] * 180.0 / PT_PI, 100.0 * scores[e]);
    }
    printf("Trajectories written to bin/phototaxis_best.csv\n");

    p2d_destroy(population);

    return 0;
}
