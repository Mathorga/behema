static unsigned long x = 123456789;
static unsigned long y = 362436069;
static unsigned long z = 521288629;
unsigned long xorshf96(void) {
    unsigned long t;
    x ^= x << 16;
    x ^= x >> 5;
    x ^= x << 1;

    t = x;
    x = y;
    y = z;
    z = t ^ x ^ y;

    return z;
}