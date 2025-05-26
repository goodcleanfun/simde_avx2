#include "greatest/greatest.h"
#include "avx2.h"

TEST test_simde_mm256_abs_epi8(void) {
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
            int8_t a = ((int8_t*)&r)[j];
            int8_t b = ((int8_t*)&test_vec[i].r)[j];
            ASSERT_EQ(a, b);
        }
    }
    PASS();
}

TEST test_simde_mm256_and_si256(void) {
    const struct {
        simde__m256i a;
        simde__m256i b;
        simde__m256i r;
    } test_vec[8] = {
        { simde_mm256_set_epi64x(INT64_C( 8722470578646828517), INT64_C(  891261850847437783),
                                INT64_C( 8698554819020653857), INT64_C(-7282900013878242954)),
        simde_mm256_set_epi64x(INT64_C(-8128142018056442141), INT64_C( 5559182722028422309),
                                INT64_C( 2093267872519066825), INT64_C(-7117023562774970023)),
        simde_mm256_set_epi64x(INT64_C(  648519197013312737), INT64_C(  866420841735143557),
                                INT64_C( 1730587322060899329), INT64_C(-7482378910948097712)) },
        { simde_mm256_set_epi64x(INT64_C(-2297219683620407228), INT64_C(-2314825045857877411),
                                INT64_C(-2223407797787304327), INT64_C( 5408595704702705619)),
        simde_mm256_set_epi64x(INT64_C( 1902387556947256757), INT64_C(-4636290958455233996),
                                INT64_C( -193279292138890017), INT64_C( 2387678637527501964)),
        simde_mm256_set_epi64x(INT64_C(    1867272746704900), INT64_C(-6944527661819330028),
                                INT64_C(-2233693047608222631), INT64_C(   72674428659436672)) },
        { simde_mm256_set_epi64x(INT64_C(-8083909718117301567), INT64_C(   11995607010100125),
                                INT64_C(-6068617776224060223), INT64_C(-6387203967446836987)),
        simde_mm256_set_epi64x(INT64_C(-8320376883848651160), INT64_C(-4950145821323384534),
                                INT64_C(-7969688999974624617), INT64_C(  659904372446782737)),
        simde_mm256_set_epi64x(INT64_C(-8322647438183611840), INT64_C(    2406350531494152),
                                INT64_C(-9131628786599059327), INT64_C(   74330855942160641)) },
        { simde_mm256_set_epi64x(INT64_C(-7862557356832127783), INT64_C(-5197238245936512816),
                                INT64_C(-1440736387308233171), INT64_C( -422437923560182700)),
        simde_mm256_set_epi64x(INT64_C( 4501573497311276896), INT64_C( 1568099047173454230),
                                INT64_C( 6784671475384752865), INT64_C(-5901872067663085826)),
        simde_mm256_set_epi64x(INT64_C( 1324204786773460032), INT64_C( 1568098471546732688),
                                INT64_C( 5476791399028365857), INT64_C(-6196932668584612780)) },
        { simde_mm256_set_epi64x(INT64_C(  -83457062575009429), INT64_C(-7222721162513873213),
                                INT64_C( 8275972355230696496), INT64_C( 5685146925209815999)),
        simde_mm256_set_epi64x(INT64_C( 7621095561231011691), INT64_C(-1384347240916299959),
                                INT64_C( 8784701942784527649), INT64_C(-6329984144489188000)),
        simde_mm256_set_epi64x(INT64_C( 7549018173429252459), INT64_C(-8592431562369268159),
                                INT64_C( 8126746635764630560), INT64_C(  586910516468318496)) },
        { simde_mm256_set_epi64x(INT64_C( 5973184558080946927), INT64_C(-1786695518880322601),
                                INT64_C(  564422817571527071), INT64_C( 4038585732338755869)),
        simde_mm256_set_epi64x(INT64_C(-8901168232869945121), INT64_C( 8118630853720063073),
                                INT64_C( -228868271804772649), INT64_C(-6456700929251086932)),
        simde_mm256_set_epi64x(INT64_C(   27024505729917135), INT64_C( 6926573216261613633),
                                INT64_C(  346814025888696471), INT64_C( 2306177340255840524)) },
        { simde_mm256_set_epi64x(INT64_C( 4967668340414178010), INT64_C(-2410168209476403592),
                                INT64_C(-3019436090811439415), INT64_C(-6965119139859890192)),
        simde_mm256_set_epi64x(INT64_C(-5120337331222163918), INT64_C(-1589564432494918546),
                                INT64_C( 5292723257474752308), INT64_C( 2511807878775255697)),
        simde_mm256_set_epi64x(INT64_C(   67729921108361746), INT64_C(-3999169530918599576),
                                INT64_C( 4616337787987166720), INT64_C(  167381957966049936)) },
        { simde_mm256_set_epi64x(INT64_C(-6179811667909625694), INT64_C(-2471055444546593648),
                                INT64_C( 7540412455883833292), INT64_C( 6654843089135720963)),
        simde_mm256_set_epi64x(INT64_C( -939588147635733509), INT64_C(-1340596046637757449),
                                INT64_C(-1662948605324253370), INT64_C(  817158485966988858)),
        simde_mm256_set_epi64x(INT64_C(-6758530821969135454), INT64_C(-3664509346923870064),
                                INT64_C( 7540157231680104260), INT64_C(  599541701488411138)) }
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        simde__m256i r = simde_mm256_and_si256(test_vec[i].a, test_vec[i].b);
        for (int j = 0; j < (sizeof(r) / sizeof(int64_t)); j++) {
            int64_t a = ((int64_t*)&r)[j];
            int64_t b = ((int64_t*)&test_vec[i].r)[j];
            ASSERT_EQ(a, b);
        }
    }
    PASS();
}

SUITE(avx2_suite) {
    RUN_TEST(test_simde_mm256_abs_epi8);
    RUN_TEST(test_simde_mm256_and_si256);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();
    RUN_SUITE(avx2_suite);
    GREATEST_MAIN_END();
}

