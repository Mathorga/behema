#include "tests.h"

int test_rotr_mask() {
    // No rotation: the mask is returned unchanged.
    BHM_ASSERT_EQ(BHM_ROTR_MASK(0x15ULL, 0, 5), 0x15ULL);

    // Example from macros.h, size = 5 and shift = 2: edcba -> baedc.
    // 0b10110 (e=1, d=0, c=1, b=1, a=0) -> 0b10101 (b=1, a=0, e=1, d=0, c=1).
    BHM_ASSERT_EQ(BHM_ROTR_MASK(0x16ULL, 2, 5), 0x15ULL);

    // The lowest bit re-enters as the highest of the [size] bits.
    BHM_ASSERT_EQ(BHM_ROTR_MASK(0x01ULL, 1, 25), 0x01ULL << 24);

    // Bits above [size] are cleared.
    BHM_ASSERT_EQ(BHM_ROTR_MASK(0xFF00ULL | 0x03ULL, 1, 8), 0x81ULL);

    // Rotating [size] times by one gives the original mask back, for the largest neighborhood (radius 3, 49 cells).
    bhm_nh_mask_t mask = 0x0001234567890ABCULL & ((0x01ULL << 49) - 1);
    bhm_nh_mask_t rotated = mask;
    for (uint32_t i = 0; i < 49; i++) {
        rotated = BHM_ROTR_MASK(rotated, 1, 49);
    }
    BHM_ASSERT_EQ(rotated, mask);

    return 0;
}
