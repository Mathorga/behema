#include "tests.h"

typedef struct {
    const char* name;
    int (*function)();
} bhm_test_t;

int main(int argc, char **argv) {
    bhm_test_t tests[] = {
        {"BHM_ROTR_MASK rotates masks", test_rotr_mask},
        {"syn_count matches synac_mask", test_syn_count_matches_synac_mask},
        {"tot_syn_strength matches strength masks", test_tot_syn_strength_matches_masks},
        {"syn_count within max_syn_count", test_syn_count_within_max},
        {"tot_syn_strength within max_tot_strength", test_tot_syn_strength_within_max},
        {"modulation keeps bookkeeping consistent", test_modulation_keeps_bookkeeping},
        {"modulation rewards eligible synapse", test_modulation_rewards_eligible_synapse},
        {"modulation punishes eligible synapse", test_modulation_punishes_eligible_synapse},
        {"modulation ignores non-eligible synapses", test_modulation_ignores_non_eligible_synapses},
        {"modulation is summed, clamped and reset", test_modulation_add_clamps},
    };
    uint32_t tests_count = sizeof(tests) / sizeof(bhm_test_t);
    uint32_t failed_count = 0;

    for (uint32_t i = 0; i < tests_count; i++) {
        printf("[%d/%d] %s\n", i + 1, tests_count, tests[i].name);

        if (tests[i].function() != 0) {
            printf("    FAILED\n");
            failed_count++;
        } else {
            printf("    OK\n");
        }
    }

    printf("\n%d passed, %d failed\n", tests_count - failed_count, failed_count);

    // A non-zero exit code makes make (and CI) aware of the failure.
    return failed_count > 0 ? 1 : 0;
}
