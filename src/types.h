#include <stdint.h>

#ifndef __BHM_TYPES__
#define __BHM_TYPES__

typedef enum {
    BHM_FALSE = 0,
    BHM_TRUE = 1
} bhm_bool_t;

typedef enum {
    // Values are forced to 32 bit integers by using big enough values: 100000 is 17 bits long, so 32 bits are automatically allocated.
    // Linear.
    BHM_PULSE_MAPPING_LINEAR = 0x100000U,
    // Floored proportional.
    BHM_PULSE_MAPPING_FPROP = 0x100001U,
    // Rounded proportional.
    BHM_PULSE_MAPPING_RPROP = 0x100002U,
    // Double floored proportional.
    BHM_PULSE_MAPPING_DFPROP = 0x100003U,
} bhm_pulse_mapping_t;

typedef uint8_t bhm_byte_t;

typedef int16_t bhm_neuron_value_t;

// A mask made of 8 bytes can hold up to 48 neighbors (i.e. radius = 3).
// Using 16 bytes the radius can be up to 5 (120 neighbors).
typedef uint64_t bhm_nh_mask_t;
typedef int8_t bhm_nh_radius_t;
typedef uint8_t bhm_syn_count_t;
typedef uint8_t bhm_syn_strength_t;
typedef uint16_t bhm_ticks_count_t;
typedef uint32_t bhm_evol_step_t;
typedef uint64_t bhm_pulse_mask_t;
typedef uint32_t bhm_chance_t;
typedef uint32_t bhm_rand_state_t;

typedef int32_t bhm_cortex_size_t;

#endif