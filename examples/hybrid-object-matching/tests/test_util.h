/* Minimal assertion helpers.
 *
 * Deliberately not a framework: a C project with four test binaries does not
 * need an external dependency, and CTest already provides discovery, parallel
 * execution and reporting. Each test file is a plain `main` that returns
 * non-zero on failure.
 */
#ifndef MSEARCH_TEST_UTIL_H
#define MSEARCH_TEST_UTIL_H

#include <stdio.h>
#include <string.h>

static int g_checks = 0;
static int g_failures = 0;
static const char *g_current_case = "";

#define TEST_CASE(name)                                                                            \
    do {                                                                                           \
        g_current_case = (name);                                                                   \
        printf("-- %s\n", (name));                                                                 \
    } while (0)

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        g_checks++;                                                                                \
        if (!(cond)) {                                                                             \
            g_failures++;                                                                          \
            printf("   FAIL %s:%d [%s]: %s\n", __FILE__, __LINE__, g_current_case, #cond);         \
        }                                                                                          \
    } while (0)

#define CHECK_EQ_INT(actual, expected)                                                             \
    do {                                                                                           \
        g_checks++;                                                                                \
        const long long actual_ = (long long)(actual);                                             \
        const long long expected_ = (long long)(expected);                                         \
        if (actual_ != expected_) {                                                                \
            g_failures++;                                                                          \
            printf("   FAIL %s:%d [%s]: %s == %lld, expected %lld\n", __FILE__, __LINE__,          \
                   g_current_case, #actual, actual_, expected_);                                   \
        }                                                                                          \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                    \
    do {                                                                                           \
        g_checks++;                                                                                \
        const double diff_ = (double)(actual) - (double)(expected);                                \
        if (!(diff_ < (tolerance) && diff_ > -(tolerance))) {                                      \
            g_failures++;                                                                          \
            printf("   FAIL %s:%d [%s]: %s == %g, expected %g +/- %g\n", __FILE__, __LINE__,       \
                   g_current_case, #actual, (double)(actual), (double)(expected),                  \
                   (double)(tolerance));                                                           \
        }                                                                                          \
    } while (0)

#define CHECK_STR_CONTAINS(haystack, needle)                                                       \
    do {                                                                                           \
        g_checks++;                                                                                \
        if (strstr((haystack), (needle)) == NULL) {                                                \
            g_failures++;                                                                          \
            printf("   FAIL %s:%d [%s]: \"%s\" does not contain \"%s\"\n", __FILE__, __LINE__,     \
                   g_current_case, (haystack), (needle));                                          \
        }                                                                                          \
    } while (0)

#define TEST_REPORT()                                                                              \
    (printf("%s: %d checks, %d failures\n", g_failures == 0 ? "PASS" : "FAIL", g_checks,           \
            g_failures),                                                                           \
     g_failures == 0 ? 0 : 1)

/* Reproducible PRNG: tests that depend on random input must fail the same way
 * on every machine and every run. */
typedef struct {
    unsigned int state;
} TestRng;

static inline void rng_seed(TestRng *rng, unsigned int seed)
{
    rng->state = seed != 0 ? seed : 0x9e3779b9u;
}

static inline unsigned int rng_next(TestRng *rng)
{
    unsigned int x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static inline int rng_range(TestRng *rng, int lo, int hi)
{
    return lo + (int)(rng_next(rng) % (unsigned int)(hi - lo + 1));
}

#endif /* MSEARCH_TEST_UTIL_H */
