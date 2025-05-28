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


TEST test_simde_mm256_srli_epi16(void) {
    const struct {
        simde__m256i a;
        simde__m256i r1;
        simde__m256i r3;
        simde__m256i r5;
        simde__m256i r11;
        simde__m256i r13;
        simde__m256i r15;
        simde__m256i r16;
        simde__m256i r24;
    } test_vec[8] = {
      { simde_mm256_set_epi16(INT16_C(-13208), INT16_C( 32518), INT16_C(-12083), INT16_C( -4650),
                              INT16_C( 32616), INT16_C(-23415), INT16_C(-12219), INT16_C(-11043),
                              INT16_C( 17138), INT16_C( 18141), INT16_C( 29257), INT16_C(-17957),
                              INT16_C( -2929), INT16_C(-12343), INT16_C( -8291), INT16_C(-11958)),
        simde_mm256_set_epi16(INT16_C( 26164), INT16_C( 16259), INT16_C( 26726), INT16_C( 30443),
                              INT16_C( 16308), INT16_C( 21060), INT16_C( 26658), INT16_C( 27246),
                              INT16_C(  8569), INT16_C(  9070), INT16_C( 14628), INT16_C( 23789),
                              INT16_C( 31303), INT16_C( 26596), INT16_C( 28622), INT16_C( 26789)),
        simde_mm256_set_epi16(INT16_C(  6541), INT16_C(  4064), INT16_C(  6681), INT16_C(  7610),
                              INT16_C(  4077), INT16_C(  5265), INT16_C(  6664), INT16_C(  6811),
                              INT16_C(  2142), INT16_C(  2267), INT16_C(  3657), INT16_C(  5947),
                              INT16_C(  7825), INT16_C(  6649), INT16_C(  7155), INT16_C(  6697)),
        simde_mm256_set_epi16(INT16_C(  1635), INT16_C(  1016), INT16_C(  1670), INT16_C(  1902),
                              INT16_C(  1019), INT16_C(  1316), INT16_C(  1666), INT16_C(  1702),
                              INT16_C(   535), INT16_C(   566), INT16_C(   914), INT16_C(  1486),
                              INT16_C(  1956), INT16_C(  1662), INT16_C(  1788), INT16_C(  1674)),
        simde_mm256_set_epi16(INT16_C(    25), INT16_C(    15), INT16_C(    26), INT16_C(    29),
                              INT16_C(    15), INT16_C(    20), INT16_C(    26), INT16_C(    26),
                              INT16_C(     8), INT16_C(     8), INT16_C(    14), INT16_C(    23),
                              INT16_C(    30), INT16_C(    25), INT16_C(    27), INT16_C(    26)),
        simde_mm256_set_epi16(INT16_C(     6), INT16_C(     3), INT16_C(     6), INT16_C(     7),
                              INT16_C(     3), INT16_C(     5), INT16_C(     6), INT16_C(     6),
                              INT16_C(     2), INT16_C(     2), INT16_C(     3), INT16_C(     5),
                              INT16_C(     7), INT16_C(     6), INT16_C(     6), INT16_C(     6)),
        simde_mm256_set_epi16(INT16_C(     1), INT16_C(     0), INT16_C(     1), INT16_C(     1),
                              INT16_C(     0), INT16_C(     1), INT16_C(     1), INT16_C(     1),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     1),
                              INT16_C(     1), INT16_C(     1), INT16_C(     1), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C(  9810), INT16_C( 24519), INT16_C(-20641), INT16_C( 29186),
                              INT16_C(-23300), INT16_C( -6682), INT16_C(-18375), INT16_C( 30920),
                              INT16_C( 29283), INT16_C( 14293), INT16_C( -6612), INT16_C( 11040),
                              INT16_C(-31748), INT16_C( -6890), INT16_C( 12929), INT16_C(-16870)),
        simde_mm256_set_epi16(INT16_C(  4905), INT16_C( 12259), INT16_C( 22447), INT16_C( 14593),
                              INT16_C( 21118), INT16_C( 29427), INT16_C( 23580), INT16_C( 15460),
                              INT16_C( 14641), INT16_C(  7146), INT16_C( 29462), INT16_C(  5520),
                              INT16_C( 16894), INT16_C( 29323), INT16_C(  6464), INT16_C( 24333)),
        simde_mm256_set_epi16(INT16_C(  1226), INT16_C(  3064), INT16_C(  5611), INT16_C(  3648),
                              INT16_C(  5279), INT16_C(  7356), INT16_C(  5895), INT16_C(  3865),
                              INT16_C(  3660), INT16_C(  1786), INT16_C(  7365), INT16_C(  1380),
                              INT16_C(  4223), INT16_C(  7330), INT16_C(  1616), INT16_C(  6083)),
        simde_mm256_set_epi16(INT16_C(   306), INT16_C(   766), INT16_C(  1402), INT16_C(   912),
                              INT16_C(  1319), INT16_C(  1839), INT16_C(  1473), INT16_C(   966),
                              INT16_C(   915), INT16_C(   446), INT16_C(  1841), INT16_C(   345),
                              INT16_C(  1055), INT16_C(  1832), INT16_C(   404), INT16_C(  1520)),
        simde_mm256_set_epi16(INT16_C(     4), INT16_C(    11), INT16_C(    21), INT16_C(    14),
                              INT16_C(    20), INT16_C(    28), INT16_C(    23), INT16_C(    15),
                              INT16_C(    14), INT16_C(     6), INT16_C(    28), INT16_C(     5),
                              INT16_C(    16), INT16_C(    28), INT16_C(     6), INT16_C(    23)),
        simde_mm256_set_epi16(INT16_C(     1), INT16_C(     2), INT16_C(     5), INT16_C(     3),
                              INT16_C(     5), INT16_C(     7), INT16_C(     5), INT16_C(     3),
                              INT16_C(     3), INT16_C(     1), INT16_C(     7), INT16_C(     1),
                              INT16_C(     4), INT16_C(     7), INT16_C(     1), INT16_C(     5)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     1), INT16_C(     0),
                              INT16_C(     1), INT16_C(     1), INT16_C(     1), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     1), INT16_C(     0),
                              INT16_C(     1), INT16_C(     1), INT16_C(     0), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C(  4687), INT16_C( -4828), INT16_C(  9674), INT16_C(  8229),
                              INT16_C(-28519), INT16_C( 24429), INT16_C(-25708), INT16_C(-15646),
                              INT16_C( 27606), INT16_C(  -993), INT16_C( 27866), INT16_C(-11890),
                              INT16_C( 25757), INT16_C( -1957), INT16_C( 24727), INT16_C(-30230)),
        simde_mm256_set_epi16(INT16_C(  2343), INT16_C( 30354), INT16_C(  4837), INT16_C(  4114),
                              INT16_C( 18508), INT16_C( 12214), INT16_C( 19914), INT16_C( 24945),
                              INT16_C( 13803), INT16_C( 32271), INT16_C( 13933), INT16_C( 26823),
                              INT16_C( 12878), INT16_C( 31789), INT16_C( 12363), INT16_C( 17653)),
        simde_mm256_set_epi16(INT16_C(   585), INT16_C(  7588), INT16_C(  1209), INT16_C(  1028),
                              INT16_C(  4627), INT16_C(  3053), INT16_C(  4978), INT16_C(  6236),
                              INT16_C(  3450), INT16_C(  8067), INT16_C(  3483), INT16_C(  6705),
                              INT16_C(  3219), INT16_C(  7947), INT16_C(  3090), INT16_C(  4413)),
        simde_mm256_set_epi16(INT16_C(   146), INT16_C(  1897), INT16_C(   302), INT16_C(   257),
                              INT16_C(  1156), INT16_C(   763), INT16_C(  1244), INT16_C(  1559),
                              INT16_C(   862), INT16_C(  2016), INT16_C(   870), INT16_C(  1676),
                              INT16_C(   804), INT16_C(  1986), INT16_C(   772), INT16_C(  1103)),
        simde_mm256_set_epi16(INT16_C(     2), INT16_C(    29), INT16_C(     4), INT16_C(     4),
                              INT16_C(    18), INT16_C(    11), INT16_C(    19), INT16_C(    24),
                              INT16_C(    13), INT16_C(    31), INT16_C(    13), INT16_C(    26),
                              INT16_C(    12), INT16_C(    31), INT16_C(    12), INT16_C(    17)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     7), INT16_C(     1), INT16_C(     1),
                              INT16_C(     4), INT16_C(     2), INT16_C(     4), INT16_C(     6),
                              INT16_C(     3), INT16_C(     7), INT16_C(     3), INT16_C(     6),
                              INT16_C(     3), INT16_C(     7), INT16_C(     3), INT16_C(     4)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     0),
                              INT16_C(     1), INT16_C(     0), INT16_C(     1), INT16_C(     1),
                              INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     1),
                              INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C( 16592), INT16_C( -9654), INT16_C( -8076), INT16_C( 10592),
                              INT16_C( 20644), INT16_C( 25911), INT16_C( -1061), INT16_C( 18172),
                              INT16_C( 22556), INT16_C(-19191), INT16_C( 28031), INT16_C(  -883),
                              INT16_C(  5347), INT16_C( -3724), INT16_C(-32544), INT16_C(-24989)),
        simde_mm256_set_epi16(INT16_C(  8296), INT16_C( 27941), INT16_C( 28730), INT16_C(  5296),
                              INT16_C( 10322), INT16_C( 12955), INT16_C( 32237), INT16_C(  9086),
                              INT16_C( 11278), INT16_C( 23172), INT16_C( 14015), INT16_C( 32326),
                              INT16_C(  2673), INT16_C( 30906), INT16_C( 16496), INT16_C( 20273)),
        simde_mm256_set_epi16(INT16_C(  2074), INT16_C(  6985), INT16_C(  7182), INT16_C(  1324),
                              INT16_C(  2580), INT16_C(  3238), INT16_C(  8059), INT16_C(  2271),
                              INT16_C(  2819), INT16_C(  5793), INT16_C(  3503), INT16_C(  8081),
                              INT16_C(   668), INT16_C(  7726), INT16_C(  4124), INT16_C(  5068)),
        simde_mm256_set_epi16(INT16_C(   518), INT16_C(  1746), INT16_C(  1795), INT16_C(   331),
                              INT16_C(   645), INT16_C(   809), INT16_C(  2014), INT16_C(   567),
                              INT16_C(   704), INT16_C(  1448), INT16_C(   875), INT16_C(  2020),
                              INT16_C(   167), INT16_C(  1931), INT16_C(  1031), INT16_C(  1267)),
        simde_mm256_set_epi16(INT16_C(     8), INT16_C(    27), INT16_C(    28), INT16_C(     5),
                              INT16_C(    10), INT16_C(    12), INT16_C(    31), INT16_C(     8),
                              INT16_C(    11), INT16_C(    22), INT16_C(    13), INT16_C(    31),
                              INT16_C(     2), INT16_C(    30), INT16_C(    16), INT16_C(    19)),
        simde_mm256_set_epi16(INT16_C(     2), INT16_C(     6), INT16_C(     7), INT16_C(     1),
                              INT16_C(     2), INT16_C(     3), INT16_C(     7), INT16_C(     2),
                              INT16_C(     2), INT16_C(     5), INT16_C(     3), INT16_C(     7),
                              INT16_C(     0), INT16_C(     7), INT16_C(     4), INT16_C(     4)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     1), INT16_C(     1), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     1), INT16_C(     0),
                              INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     1),
                              INT16_C(     0), INT16_C(     1), INT16_C(     1), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C( -4839), INT16_C(  3174), INT16_C(  7509), INT16_C( 28795),
                              INT16_C( -1732), INT16_C(-26609), INT16_C(-11656), INT16_C(  3035),
                              INT16_C(-10865), INT16_C(  2405), INT16_C( 29471), INT16_C( 19828),
                              INT16_C( 29576), INT16_C( 23078), INT16_C( 11200), INT16_C( 26322)),
        simde_mm256_set_epi16(INT16_C( 30348), INT16_C(  1587), INT16_C(  3754), INT16_C( 14397),
                              INT16_C( 31902), INT16_C( 19463), INT16_C( 26940), INT16_C(  1517),
                              INT16_C( 27335), INT16_C(  1202), INT16_C( 14735), INT16_C(  9914),
                              INT16_C( 14788), INT16_C( 11539), INT16_C(  5600), INT16_C( 13161)),
        simde_mm256_set_epi16(INT16_C(  7587), INT16_C(   396), INT16_C(   938), INT16_C(  3599),
                              INT16_C(  7975), INT16_C(  4865), INT16_C(  6735), INT16_C(   379),
                              INT16_C(  6833), INT16_C(   300), INT16_C(  3683), INT16_C(  2478),
                              INT16_C(  3697), INT16_C(  2884), INT16_C(  1400), INT16_C(  3290)),
        simde_mm256_set_epi16(INT16_C(  1896), INT16_C(    99), INT16_C(   234), INT16_C(   899),
                              INT16_C(  1993), INT16_C(  1216), INT16_C(  1683), INT16_C(    94),
                              INT16_C(  1708), INT16_C(    75), INT16_C(   920), INT16_C(   619),
                              INT16_C(   924), INT16_C(   721), INT16_C(   350), INT16_C(   822)),
        simde_mm256_set_epi16(INT16_C(    29), INT16_C(     1), INT16_C(     3), INT16_C(    14),
                              INT16_C(    31), INT16_C(    19), INT16_C(    26), INT16_C(     1),
                              INT16_C(    26), INT16_C(     1), INT16_C(    14), INT16_C(     9),
                              INT16_C(    14), INT16_C(    11), INT16_C(     5), INT16_C(    12)),
        simde_mm256_set_epi16(INT16_C(     7), INT16_C(     0), INT16_C(     0), INT16_C(     3),
                              INT16_C(     7), INT16_C(     4), INT16_C(     6), INT16_C(     0),
                              INT16_C(     6), INT16_C(     0), INT16_C(     3), INT16_C(     2),
                              INT16_C(     3), INT16_C(     2), INT16_C(     1), INT16_C(     3)),
        simde_mm256_set_epi16(INT16_C(     1), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     1), INT16_C(     1), INT16_C(     1), INT16_C(     0),
                              INT16_C(     1), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C(-25851), INT16_C(  6707), INT16_C(-23633), INT16_C( -4351),
                              INT16_C(  -641), INT16_C(-22303), INT16_C(  6727), INT16_C(  9129),
                              INT16_C(  1286), INT16_C(-28152), INT16_C(-22922), INT16_C(  -950),
                              INT16_C( -1798), INT16_C(-15465), INT16_C(   910), INT16_C(-23243)),
        simde_mm256_set_epi16(INT16_C( 19842), INT16_C(  3353), INT16_C( 20951), INT16_C( 30592),
                              INT16_C( 32447), INT16_C( 21616), INT16_C(  3363), INT16_C(  4564),
                              INT16_C(   643), INT16_C( 18692), INT16_C( 21307), INT16_C( 32293),
                              INT16_C( 31869), INT16_C( 25035), INT16_C(   455), INT16_C( 21146)),
        simde_mm256_set_epi16(INT16_C(  4960), INT16_C(   838), INT16_C(  5237), INT16_C(  7648),
                              INT16_C(  8111), INT16_C(  5404), INT16_C(   840), INT16_C(  1141),
                              INT16_C(   160), INT16_C(  4673), INT16_C(  5326), INT16_C(  8073),
                              INT16_C(  7967), INT16_C(  6258), INT16_C(   113), INT16_C(  5286)),
        simde_mm256_set_epi16(INT16_C(  1240), INT16_C(   209), INT16_C(  1309), INT16_C(  1912),
                              INT16_C(  2027), INT16_C(  1351), INT16_C(   210), INT16_C(   285),
                              INT16_C(    40), INT16_C(  1168), INT16_C(  1331), INT16_C(  2018),
                              INT16_C(  1991), INT16_C(  1564), INT16_C(    28), INT16_C(  1321)),
        simde_mm256_set_epi16(INT16_C(    19), INT16_C(     3), INT16_C(    20), INT16_C(    29),
                              INT16_C(    31), INT16_C(    21), INT16_C(     3), INT16_C(     4),
                              INT16_C(     0), INT16_C(    18), INT16_C(    20), INT16_C(    31),
                              INT16_C(    31), INT16_C(    24), INT16_C(     0), INT16_C(    20)),
        simde_mm256_set_epi16(INT16_C(     4), INT16_C(     0), INT16_C(     5), INT16_C(     7),
                              INT16_C(     7), INT16_C(     5), INT16_C(     0), INT16_C(     1),
                              INT16_C(     0), INT16_C(     4), INT16_C(     5), INT16_C(     7),
                              INT16_C(     7), INT16_C(     6), INT16_C(     0), INT16_C(     5)),
        simde_mm256_set_epi16(INT16_C(     1), INT16_C(     0), INT16_C(     1), INT16_C(     1),
                              INT16_C(     1), INT16_C(     1), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     1), INT16_C(     1), INT16_C(     1),
                              INT16_C(     1), INT16_C(     1), INT16_C(     0), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C(  7674), INT16_C( 18879), INT16_C( 27446), INT16_C(-29225),
                              INT16_C( -2725), INT16_C( 23364), INT16_C( 12045), INT16_C(-28927),
                              INT16_C(-14599), INT16_C(-16964), INT16_C(   660), INT16_C( 23234),
                              INT16_C(-21987), INT16_C(-30631), INT16_C( 26152), INT16_C(-28363)),
        simde_mm256_set_epi16(INT16_C(  3837), INT16_C(  9439), INT16_C( 13723), INT16_C( 18155),
                              INT16_C( 31405), INT16_C( 11682), INT16_C(  6022), INT16_C( 18304),
                              INT16_C( 25468), INT16_C( 24286), INT16_C(   330), INT16_C( 11617),
                              INT16_C( 21774), INT16_C( 17452), INT16_C( 13076), INT16_C( 18586)),
        simde_mm256_set_epi16(INT16_C(   959), INT16_C(  2359), INT16_C(  3430), INT16_C(  4538),
                              INT16_C(  7851), INT16_C(  2920), INT16_C(  1505), INT16_C(  4576),
                              INT16_C(  6367), INT16_C(  6071), INT16_C(    82), INT16_C(  2904),
                              INT16_C(  5443), INT16_C(  4363), INT16_C(  3269), INT16_C(  4646)),
        simde_mm256_set_epi16(INT16_C(   239), INT16_C(   589), INT16_C(   857), INT16_C(  1134),
                              INT16_C(  1962), INT16_C(   730), INT16_C(   376), INT16_C(  1144),
                              INT16_C(  1591), INT16_C(  1517), INT16_C(    20), INT16_C(   726),
                              INT16_C(  1360), INT16_C(  1090), INT16_C(   817), INT16_C(  1161)),
        simde_mm256_set_epi16(INT16_C(     3), INT16_C(     9), INT16_C(    13), INT16_C(    17),
                              INT16_C(    30), INT16_C(    11), INT16_C(     5), INT16_C(    17),
                              INT16_C(    24), INT16_C(    23), INT16_C(     0), INT16_C(    11),
                              INT16_C(    21), INT16_C(    17), INT16_C(    12), INT16_C(    18)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     2), INT16_C(     3), INT16_C(     4),
                              INT16_C(     7), INT16_C(     2), INT16_C(     1), INT16_C(     4),
                              INT16_C(     6), INT16_C(     5), INT16_C(     0), INT16_C(     2),
                              INT16_C(     5), INT16_C(     4), INT16_C(     3), INT16_C(     4)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     1),
                              INT16_C(     1), INT16_C(     0), INT16_C(     0), INT16_C(     1),
                              INT16_C(     1), INT16_C(     1), INT16_C(     0), INT16_C(     0),
                              INT16_C(     1), INT16_C(     1), INT16_C(     0), INT16_C(     1)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
      { simde_mm256_set_epi16(INT16_C(-13197), INT16_C( 24881), INT16_C(-10578), INT16_C(-21298),
                              INT16_C( 16303), INT16_C( -8332), INT16_C( 25558), INT16_C( 12717),
                              INT16_C( 18247), INT16_C(-30759), INT16_C(  9647), INT16_C( 18112),
                              INT16_C( -4632), INT16_C(  7524), INT16_C(-32339), INT16_C( 28325)),
        simde_mm256_set_epi16(INT16_C( 26169), INT16_C( 12440), INT16_C( 27479), INT16_C( 22119),
                              INT16_C(  8151), INT16_C( 28602), INT16_C( 12779), INT16_C(  6358),
                              INT16_C(  9123), INT16_C( 17388), INT16_C(  4823), INT16_C(  9056),
                              INT16_C( 30452), INT16_C(  3762), INT16_C( 16598), INT16_C( 14162)),
        simde_mm256_set_epi16(INT16_C(  6542), INT16_C(  3110), INT16_C(  6869), INT16_C(  5529),
                              INT16_C(  2037), INT16_C(  7150), INT16_C(  3194), INT16_C(  1589),
                              INT16_C(  2280), INT16_C(  4347), INT16_C(  1205), INT16_C(  2264),
                              INT16_C(  7613), INT16_C(   940), INT16_C(  4149), INT16_C(  3540)),
        simde_mm256_set_epi16(INT16_C(  1635), INT16_C(   777), INT16_C(  1717), INT16_C(  1382),
                              INT16_C(   509), INT16_C(  1787), INT16_C(   798), INT16_C(   397),
                              INT16_C(   570), INT16_C(  1086), INT16_C(   301), INT16_C(   566),
                              INT16_C(  1903), INT16_C(   235), INT16_C(  1037), INT16_C(   885)),
        simde_mm256_set_epi16(INT16_C(    25), INT16_C(    12), INT16_C(    26), INT16_C(    21),
                              INT16_C(     7), INT16_C(    27), INT16_C(    12), INT16_C(     6),
                              INT16_C(     8), INT16_C(    16), INT16_C(     4), INT16_C(     8),
                              INT16_C(    29), INT16_C(     3), INT16_C(    16), INT16_C(    13)),
        simde_mm256_set_epi16(INT16_C(     6), INT16_C(     3), INT16_C(     6), INT16_C(     5),
                              INT16_C(     1), INT16_C(     6), INT16_C(     3), INT16_C(     1),
                              INT16_C(     2), INT16_C(     4), INT16_C(     1), INT16_C(     2),
                              INT16_C(     7), INT16_C(     0), INT16_C(     4), INT16_C(     3)),
        simde_mm256_set_epi16(INT16_C(     1), INT16_C(     0), INT16_C(     1), INT16_C(     1),
                              INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     1), INT16_C(     0), INT16_C(     0),
                              INT16_C(     1), INT16_C(     0), INT16_C(     1), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)),
        simde_mm256_set_epi16(INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0),
                              INT16_C(     0), INT16_C(     0), INT16_C(     0), INT16_C(     0)) },
    };

    for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
        simde__m256i r;
        r = simde_mm256_srli_epi16(test_vec[i].a, 0);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].a[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 1);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r1[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 3);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r3[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 5);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r5[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 11);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r11[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 13);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r13[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 15);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r15[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 16);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r16[j]);
        }
        r = simde_mm256_srli_epi16(test_vec[i].a, 24);
        for (size_t j = 0; j < 16; j++) {
            ASSERT_EQ(r[j], test_vec[i].r24[j]);
        }
    }

    PASS();
}


SUITE(avx2_suite) {
    RUN_TEST(test_simde_mm256_abs_epi8);
    RUN_TEST(test_simde_mm256_and_si256);
    RUN_TEST(test_simde_mm256_srli_epi16);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
    GREATEST_MAIN_BEGIN();
    RUN_SUITE(avx2_suite);
    GREATEST_MAIN_END();
}

