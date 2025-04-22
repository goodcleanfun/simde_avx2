#include "greatest/greatest.h"
#include "avx2.h"

TEST test_avx2 (void) {
    const struct {
        simde__m256i a;
        simde__m256i r;
    } test_vec[8] = {
        { simde_mm256_set_epi8(INT8_C( -27), INT8_C(  88), INT8_C(-122), INT8_C(  -6),
                            INT8_C( -23), INT8_C( 108), INT8_C(-103), INT8_C(  32),
                            INT8_C(  43), INT8_C( 116), INT8_C(  -6), INT8_C( -98),
                            INT8_C( -62), INT8_C( -87), INT8_C(  90), INT8_C(  82),
                            INT8_C(  86), INT8_C(   8), INT8_C(-126), INT8_C( -22),
                            INT8_C( -80), INT8_C(-125), INT8_C(  -5), INT8_C(-101),
                            INT8_C(  36), INT8_C( 114), INT8_C( -51), INT8_C(  59),
                            INT8_C( -97), INT8_C( 124), INT8_C(  25), INT8_C(  90)),
        simde_mm256_set_epi8(INT8_C(  27), INT8_C(  88), INT8_C( 122), INT8_C(   6),
                            INT8_C(  23), INT8_C( 108), INT8_C( 103), INT8_C(  32),
                            INT8_C(  43), INT8_C( 116), INT8_C(   6), INT8_C(  98),
                            INT8_C(  62), INT8_C(  87), INT8_C(  90), INT8_C(  82),
                            INT8_C(  86), INT8_C(   8), INT8_C( 126), INT8_C(  22),
                            INT8_C(  80), INT8_C( 125), INT8_C(   5), INT8_C( 101),
                            INT8_C(  36), INT8_C( 114), INT8_C(  51), INT8_C(  59),
                            INT8_C(  97), INT8_C( 124), INT8_C(  25), INT8_C(  90)) },
        { simde_mm256_set_epi8(INT8_C( 111), INT8_C(  46), INT8_C( -44), INT8_C(  36),
                            INT8_C( -79), INT8_C( 101), INT8_C(   0), INT8_C(   2),
                            INT8_C( -69), INT8_C(  31), INT8_C( -68), INT8_C( -82),
                            INT8_C( -45), INT8_C( 120), INT8_C(  39), INT8_C(  46),
                            INT8_C(  66), INT8_C(  30), INT8_C(-106), INT8_C( 118),
                            INT8_C(  61), INT8_C(  98), INT8_C( -61), INT8_C(  98),
                            INT8_C(  49), INT8_C( -12), INT8_C(-117), INT8_C(-115),
                            INT8_C(  63), INT8_C( -92), INT8_C(-102), INT8_C(-110)),
        simde_mm256_set_epi8(INT8_C( 111), INT8_C(  46), INT8_C(  44), INT8_C(  36),
                            INT8_C(  79), INT8_C( 101), INT8_C(   0), INT8_C(   2),
                            INT8_C(  69), INT8_C(  31), INT8_C(  68), INT8_C(  82),
                            INT8_C(  45), INT8_C( 120), INT8_C(  39), INT8_C(  46),
                            INT8_C(  66), INT8_C(  30), INT8_C( 106), INT8_C( 118),
                            INT8_C(  61), INT8_C(  98), INT8_C(  61), INT8_C(  98),
                            INT8_C(  49), INT8_C(  12), INT8_C( 117), INT8_C( 115),
                            INT8_C(  63), INT8_C(  92), INT8_C( 102), INT8_C( 110)) },
        { simde_mm256_set_epi8(INT8_C(  64), INT8_C( -84), INT8_C(  54), INT8_C(-102),
                            INT8_C( -69), INT8_C(  12), INT8_C(-119), INT8_C( -19),
                            INT8_C(  19), INT8_C( -55), INT8_C( -11), INT8_C(-117),
                            INT8_C( -68), INT8_C( -51), INT8_C(  26), INT8_C(  72),
                            INT8_C( -15), INT8_C( 108), INT8_C( -66), INT8_C( -24),
                            INT8_C( -97), INT8_C( -48), INT8_C(  75), INT8_C(  35),
                            INT8_C(  48), INT8_C( -25), INT8_C( -43), INT8_C(   2),
                            INT8_C( -75), INT8_C(  28), INT8_C(-108), INT8_C( -43)),
        simde_mm256_set_epi8(INT8_C(  64), INT8_C(  84), INT8_C(  54), INT8_C( 102),
                            INT8_C(  69), INT8_C(  12), INT8_C( 119), INT8_C(  19),
                            INT8_C(  19), INT8_C(  55), INT8_C(  11), INT8_C( 117),
                            INT8_C(  68), INT8_C(  51), INT8_C(  26), INT8_C(  72),
                            INT8_C(  15), INT8_C( 108), INT8_C(  66), INT8_C(  24),
                            INT8_C(  97), INT8_C(  48), INT8_C(  75), INT8_C(  35),
                            INT8_C(  48), INT8_C(  25), INT8_C(  43), INT8_C(   2),
                            INT8_C(  75), INT8_C(  28), INT8_C( 108), INT8_C(  43)) },
        { simde_mm256_set_epi8(INT8_C(   8), INT8_C( -54), INT8_C(  -1), INT8_C(-128),
                            INT8_C( 118), INT8_C( -15), INT8_C( 125), INT8_C(  76),
                            INT8_C(  47), INT8_C(  33), INT8_C(  69), INT8_C(  21),
                            INT8_C(-116), INT8_C(  34), INT8_C(  36), INT8_C(  31),
                            INT8_C( -32), INT8_C( -84), INT8_C(  23), INT8_C( -76),
                            INT8_C(  82), INT8_C(-115), INT8_C(  74), INT8_C(-110),
                            INT8_C( -46), INT8_C( 125), INT8_C( -52), INT8_C( -99),
                            INT8_C(  30), INT8_C(-106), INT8_C(  66), INT8_C(   5)),
        simde_mm256_set_epi8(INT8_C(   8), INT8_C(  54), INT8_C(   1), INT8_C(-128),
                            INT8_C( 118), INT8_C(  15), INT8_C( 125), INT8_C(  76),
                            INT8_C(  47), INT8_C(  33), INT8_C(  69), INT8_C(  21),
                            INT8_C( 116), INT8_C(  34), INT8_C(  36), INT8_C(  31),
                            INT8_C(  32), INT8_C(  84), INT8_C(  23), INT8_C(  76),
                            INT8_C(  82), INT8_C( 115), INT8_C(  74), INT8_C( 110),
                            INT8_C(  46), INT8_C( 125), INT8_C(  52), INT8_C(  99),
                            INT8_C(  30), INT8_C( 106), INT8_C(  66), INT8_C(   5)) },
        { simde_mm256_set_epi8(INT8_C( 122), INT8_C(  42), INT8_C(-121), INT8_C(-106),
                            INT8_C( 122), INT8_C(  -8), INT8_C(  81), INT8_C(-109),
                            INT8_C( 124), INT8_C(  32), INT8_C(  63), INT8_C( -21),
                            INT8_C( -51), INT8_C( -42), INT8_C(   1), INT8_C( -78),
                            INT8_C(  74), INT8_C(   8), INT8_C(  25), INT8_C(  10),
                            INT8_C( 113), INT8_C( -75), INT8_C( -32), INT8_C( 126),
                            INT8_C( -87), INT8_C(  67), INT8_C(  78), INT8_C( -64),
                            INT8_C(   7), INT8_C( -40), INT8_C( -46), INT8_C( -59)),
        simde_mm256_set_epi8(INT8_C( 122), INT8_C(  42), INT8_C( 121), INT8_C( 106),
                            INT8_C( 122), INT8_C(   8), INT8_C(  81), INT8_C( 109),
                            INT8_C( 124), INT8_C(  32), INT8_C(  63), INT8_C(  21),
                            INT8_C(  51), INT8_C(  42), INT8_C(   1), INT8_C(  78),
                            INT8_C(  74), INT8_C(   8), INT8_C(  25), INT8_C(  10),
                            INT8_C( 113), INT8_C(  75), INT8_C(  32), INT8_C( 126),
                            INT8_C(  87), INT8_C(  67), INT8_C(  78), INT8_C(  64),
                            INT8_C(   7), INT8_C(  40), INT8_C(  46), INT8_C(  59)) },
        { simde_mm256_set_epi8(INT8_C(  10), INT8_C( 120), INT8_C(  81), INT8_C(-105),
                            INT8_C(  73), INT8_C( -95), INT8_C(  79), INT8_C( -86),
                            INT8_C( -93), INT8_C( -54), INT8_C( -43), INT8_C( -88),
                            INT8_C(  59), INT8_C( -27), INT8_C(  12), INT8_C(  10),
                            INT8_C(  73), INT8_C( -48), INT8_C( 112), INT8_C(  27),
                            INT8_C(-113), INT8_C( -31), INT8_C( -56), INT8_C( -96),
                            INT8_C(  48), INT8_C( -94), INT8_C(-111), INT8_C(  60),
                            INT8_C(-116), INT8_C( -77), INT8_C( -70), INT8_C(  17)),
        simde_mm256_set_epi8(INT8_C(  10), INT8_C( 120), INT8_C(  81), INT8_C( 105),
                            INT8_C(  73), INT8_C(  95), INT8_C(  79), INT8_C(  86),
                            INT8_C(  93), INT8_C(  54), INT8_C(  43), INT8_C(  88),
                            INT8_C(  59), INT8_C(  27), INT8_C(  12), INT8_C(  10),
                            INT8_C(  73), INT8_C(  48), INT8_C( 112), INT8_C(  27),
                            INT8_C( 113), INT8_C(  31), INT8_C(  56), INT8_C(  96),
                            INT8_C(  48), INT8_C(  94), INT8_C( 111), INT8_C(  60),
                            INT8_C( 116), INT8_C(  77), INT8_C(  70), INT8_C(  17)) },
        { simde_mm256_set_epi8(INT8_C(  61), INT8_C( -57), INT8_C( -99), INT8_C(   0),
                            INT8_C(  98), INT8_C(-121), INT8_C(  67), INT8_C( -20),
                            INT8_C(  44), INT8_C(  53), INT8_C(-128), INT8_C(  44),
                            INT8_C( 127), INT8_C(  53), INT8_C(-127), INT8_C(  58),
                            INT8_C(  35), INT8_C(  83), INT8_C( -56), INT8_C(  22),
                            INT8_C(  -4), INT8_C(  -6), INT8_C(  -7), INT8_C( 121),
                            INT8_C( -22), INT8_C( -32), INT8_C( -52), INT8_C( 124),
                            INT8_C( -93), INT8_C(  55), INT8_C( -23), INT8_C( -62)),
        simde_mm256_set_epi8(INT8_C(  61), INT8_C(  57), INT8_C(  99), INT8_C(   0),
                            INT8_C(  98), INT8_C( 121), INT8_C(  67), INT8_C(  20),
                            INT8_C(  44), INT8_C(  53), INT8_C(-128), INT8_C(  44),
                            INT8_C( 127), INT8_C(  53), INT8_C( 127), INT8_C(  58),
                            INT8_C(  35), INT8_C(  83), INT8_C(  56), INT8_C(  22),
                            INT8_C(   4), INT8_C(   6), INT8_C(   7), INT8_C( 121),
                            INT8_C(  22), INT8_C(  32), INT8_C(  52), INT8_C( 124),
                            INT8_C(  93), INT8_C(  55), INT8_C(  23), INT8_C(  62)) },
        { simde_mm256_set_epi8(INT8_C(  71), INT8_C( -58), INT8_C(  24), INT8_C( 117),
                            INT8_C(   2), INT8_C( -31), INT8_C( -86), INT8_C( 101),
                            INT8_C(   3), INT8_C(  63), INT8_C(   2), INT8_C( -30),
                            INT8_C( -33), INT8_C(  51), INT8_C(  60), INT8_C(  81),
                            INT8_C( -91), INT8_C( -73), INT8_C(  66), INT8_C(  67),
                            INT8_C(  72), INT8_C(  -7), INT8_C(  44), INT8_C( -32),
                            INT8_C( -80), INT8_C( 101), INT8_C( -98), INT8_C(  89),
                            INT8_C(  89), INT8_C(  94), INT8_C( 109), INT8_C(-109)),
        simde_mm256_set_epi8(INT8_C(  71), INT8_C(  58), INT8_C(  24), INT8_C( 117),
                            INT8_C(   2), INT8_C(  31), INT8_C(  86), INT8_C( 101),
                            INT8_C(   3), INT8_C(  63), INT8_C(   2), INT8_C(  30),
                            INT8_C(  33), INT8_C(  51), INT8_C(  60), INT8_C(  81),
                            INT8_C(  91), INT8_C(  73), INT8_C(  66), INT8_C(  67),
                            INT8_C(  72), INT8_C(   7), INT8_C(  44), INT8_C(  32),
                            INT8_C(  80), INT8_C( 101), INT8_C(  98), INT8_C(  89),
                            INT8_C(  89), INT8_C(  94), INT8_C( 109), INT8_C( 109)) }
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        simde__m256i r = simde_mm256_abs_epi8(test_vec[i].a);
        for (int j = 0; j < (sizeof(r) / sizeof(int8_t)); j++) {
            ASSERT_EQ(r[j], test_vec[i].r[j]);
        }
    }
    PASS();
}

SUITE(avx2_suite) {
    RUN_TEST(test_avx2);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();
    RUN_SUITE(avx2_suite);
    GREATEST_MAIN_END();
}

