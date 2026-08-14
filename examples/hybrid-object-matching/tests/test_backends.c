/* Backend equivalence and the determinism contract.
 *
 * These are the tests the redesign exists to make possible. The original
 * implementation returned whichever match a thread reached first, so there was
 * nothing stable to assert; here every backend must agree with the serial
 * reference exactly, on the same input, every run.
 */
#include <stdlib.h>
#include <string.h>

#include "msearch/backend.h"
#include "msearch/runtime.h"
#include "support.h"
#include "test_util.h"

#define PICTURE_BASE 1000 /* filler value; far from any object value */

static Config test_config(void)
{
    Config config;
    msearch_config_defaults(&config);
    config.log_level = LOG_ERROR;
    return config;
}

static int run_backend(const char *name, const Problem *problem, Match *matches)
{
    const MatchBackend *backend = msearch_backend_find(name);
    char err[256] = "";
    const Config config = test_config();
    if (backend == NULL) {
        return 0;
    }
    const Status status = msearch_run_local(problem, backend, &config, matches, err, sizeof(err));
    if (status != MSEARCH_OK) {
        printf("   backend %s failed: %s\n", name, err);
        return 0;
    }
    return 1;
}

/* Run `problem` on every available backend and require byte equality with the
 * serial reference. */
static void check_all_backends_agree(const Problem *problem, const char *label)
{
    const int count = problem->num_pictures;
    Match *reference = calloc((size_t)count, sizeof(Match));
    Match *candidate = calloc((size_t)count, sizeof(Match));
    CHECK(reference != NULL && candidate != NULL);
    if (reference == NULL || candidate == NULL) {
        free(reference);
        free(candidate);
        return;
    }

    CHECK(run_backend("serial", problem, reference));

    int backend_count = 0;
    const MatchBackend *const *backends = msearch_backend_all(&backend_count);
    for (int b = 0; b < backend_count; ++b) {
        char reason[128] = "";
        if (strcmp(backends[b]->name, "serial") == 0 ||
            !backends[b]->available(reason, sizeof(reason))) {
            continue;
        }
        memset(candidate, 0, (size_t)count * sizeof(Match));
        CHECK(run_backend(backends[b]->name, problem, candidate));
        for (int i = 0; i < count; ++i) {
            if (memcmp(&reference[i], &candidate[i], sizeof(Match)) != 0) {
                printf("   [%s] %s disagrees on picture %d: serial=(%d,%d,%d) %s=(%d,%d,%d)\n",
                       label, backends[b]->name, reference[i].picture_id, reference[i].object_id,
                       reference[i].row, reference[i].col, backends[b]->name,
                       candidate[i].object_id, candidate[i].row, candidate[i].col);
            }
            CHECK(memcmp(&reference[i], &candidate[i], sizeof(Match)) == 0);
        }
    }

    free(reference);
    free(candidate);
}

static void test_random_equivalence(void)
{
    TEST_CASE("all backends agree on random problems");

    /* A spread of shapes: objects smaller than, equal to and larger than the
     * picture, and object areas on both sides of the CUDA kernel-selection
     * threshold (m*m >= 256) so that both kernels are exercised.
     *
     * Sizes are listed largest-first because support_problem_create assigns
     * ascending ids in array order, and the search stops at the lowest
     * matching id. A 1x1 object matches almost anywhere, so putting it last
     * makes it the fallback rather than a short-circuit that would leave every
     * larger object unexercised. */
    const int picture_ns[] = {8, 17, 32, 64, 3};
    const int object_ms[] = {33, 20, 16, 5, 3, 2, 1};
    const int num_pictures = (int)(sizeof(picture_ns) / sizeof(picture_ns[0]));
    const int num_objects = (int)(sizeof(object_ms) / sizeof(object_ms[0]));

    TestRng rng;
    rng_seed(&rng, 12345u);

    Problem problem;
    CHECK(support_problem_create(&problem, 4.0, picture_ns, num_pictures, object_ms, num_objects));

    for (int i = 0; i < num_pictures; ++i) {
        support_fill_random(problem.pictures[i].data,
                            (size_t)picture_ns[i] * (size_t)picture_ns[i], &rng, 1, 50);
    }
    for (int k = 0; k < num_objects; ++k) {
        support_fill_random(problem.objects[k].data, (size_t)object_ms[k] * (size_t)object_ms[k],
                            &rng, 1, 50);
    }
    /* Plant a guaranteed match so the "found" path is covered, not just the
     * exhaustive-scan path. */
    support_plant(&problem.pictures[2], &problem.objects[3], 7, 11);

    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);
    check_all_backends_agree(&problem, "random");
    msearch_problem_free(&problem);
}

static void test_lowest_object_id_wins(void)
{
    TEST_CASE("the lowest matching object id wins, regardless of position");

    const int picture_ns[] = {8};
    const int object_ms[] = {2, 2};
    Problem problem;
    CHECK(support_problem_create(&problem, 0.01, picture_ns, 1, object_ms, 2));

    for (int i = 0; i < 8 * 8; ++i) {
        problem.pictures[0].data[i] = PICTURE_BASE;
    }
    /* Give the higher-id object the earlier position: a "first match found
     * wins" implementation would return it. */
    problem.objects[0].id = 500;
    problem.objects[1].id = 400;
    const int early_pattern[4] = {1, 2, 3, 4};
    const int late_pattern[4] = {5, 6, 7, 8};
    memcpy(problem.objects[0].data, early_pattern, sizeof(early_pattern));
    memcpy(problem.objects[1].data, late_pattern, sizeof(late_pattern));
    support_plant(&problem.pictures[0], &problem.objects[0], 0, 0); /* id 500 at (0,0) */
    support_plant(&problem.pictures[0], &problem.objects[1], 5, 5); /* id 400 at (5,5) */

    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);

    Match match;
    CHECK(run_backend("serial", &problem, &match));
    CHECK_EQ_INT(match.object_id, 400);
    CHECK_EQ_INT(match.row, 5);
    CHECK_EQ_INT(match.col, 5);

    check_all_backends_agree(&problem, "lowest-id");
    msearch_problem_free(&problem);
}

static void test_row_major_first_placement_wins(void)
{
    TEST_CASE("ties within one object resolve to the row-major-first placement");

    const int picture_ns[] = {10};
    const int object_ms[] = {2};
    Problem problem;
    CHECK(support_problem_create(&problem, 0.01, picture_ns, 1, object_ms, 1));

    for (int i = 0; i < 10 * 10; ++i) {
        problem.pictures[0].data[i] = PICTURE_BASE;
    }
    const int pattern[4] = {11, 22, 33, 44};
    memcpy(problem.objects[0].data, pattern, sizeof(pattern));
    /* Three occurrences; (2,7) is first in row-major order. */
    support_plant(&problem.pictures[0], &problem.objects[0], 6, 1);
    support_plant(&problem.pictures[0], &problem.objects[0], 2, 7);
    support_plant(&problem.pictures[0], &problem.objects[0], 4, 4);

    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);

    Match match;
    CHECK(run_backend("serial", &problem, &match));
    CHECK_EQ_INT(match.row, 2);
    CHECK_EQ_INT(match.col, 7);

    check_all_backends_agree(&problem, "row-major-first");
    msearch_problem_free(&problem);
}

static void test_zero_elements_can_match(void)
{
    TEST_CASE("a zero-valued region matches a zero-valued object");

    /* Regression test for the original |(p-o)/p| division by zero: with p == 0
     * the old code produced NaN, `NaN < threshold` was false, and a genuine
     * match was silently discarded. */
    const int picture_ns[] = {6};
    const int object_ms[] = {2};
    Problem problem;
    CHECK(support_problem_create(&problem, 0.5, picture_ns, 1, object_ms, 1));

    for (int i = 0; i < 6 * 6; ++i) {
        problem.pictures[0].data[i] = PICTURE_BASE;
    }
    memset(problem.objects[0].data, 0, 4 * sizeof(int));
    support_plant(&problem.pictures[0], &problem.objects[0], 1, 3);

    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);

    Match match;
    CHECK(run_backend("serial", &problem, &match));
    CHECK(msearch_match_found(&match));
    CHECK_EQ_INT(match.row, 1);
    CHECK_EQ_INT(match.col, 3);

    check_all_backends_agree(&problem, "zero-region");
    msearch_problem_free(&problem);
}

static void test_degenerate_shapes(void)
{
    TEST_CASE("objects larger than the picture, and problems with no objects");

    const int picture_ns[] = {3};
    const int object_ms[] = {5};
    Problem problem;
    CHECK(support_problem_create(&problem, 1.0, picture_ns, 1, object_ms, 1));
    for (int i = 0; i < 9; ++i) {
        problem.pictures[0].data[i] = 1;
    }
    for (int i = 0; i < 25; ++i) {
        problem.objects[0].data[i] = 1;
    }
    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);

    Match match;
    CHECK(run_backend("serial", &problem, &match));
    CHECK(!msearch_match_found(&match));
    CHECK_EQ_INT(match.picture_id, problem.pictures[0].id);
    check_all_backends_agree(&problem, "oversized-object");
    msearch_problem_free(&problem);

    CHECK(support_problem_create(&problem, 1.0, picture_ns, 1, NULL, 0));
    for (int i = 0; i < 9; ++i) {
        problem.pictures[0].data[i] = 1;
    }
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);
    CHECK(run_backend("serial", &problem, &match));
    CHECK(!msearch_match_found(&match));
    check_all_backends_agree(&problem, "no-objects");
    msearch_problem_free(&problem);
}

static void test_repeated_runs_are_identical(void)
{
    TEST_CASE("every backend is deterministic across repeated runs");

    const int picture_ns[] = {40, 40};
    const int object_ms[] = {3, 4, 6};
    Problem problem;
    CHECK(support_problem_create(&problem, 6.0, picture_ns, 2, object_ms, 3));

    TestRng rng;
    rng_seed(&rng, 777u);
    for (int i = 0; i < 2; ++i) {
        support_fill_random(problem.pictures[i].data, 40 * 40, &rng, 1, 20);
    }
    for (int k = 0; k < 3; ++k) {
        support_fill_random(problem.objects[k].data, (size_t)object_ms[k] * (size_t)object_ms[k],
                            &rng, 1, 20);
    }
    char err[256];
    CHECK_EQ_INT(msearch_problem_validate(&problem, err, sizeof(err)), MSEARCH_OK);

    int backend_count = 0;
    const MatchBackend *const *backends = msearch_backend_all(&backend_count);
    for (int b = 0; b < backend_count; ++b) {
        char reason[128] = "";
        if (!backends[b]->available(reason, sizeof(reason))) {
            continue;
        }
        Match first[2];
        Match again[2];
        CHECK(run_backend(backends[b]->name, &problem, first));
        for (int repeat = 0; repeat < 4; ++repeat) {
            CHECK(run_backend(backends[b]->name, &problem, again));
            CHECK(memcmp(first, again, sizeof(first)) == 0);
        }
    }
    msearch_problem_free(&problem);
}

int main(void)
{
    test_random_equivalence();
    test_lowest_object_id_wins();
    test_row_major_first_placement_wins();
    test_zero_elements_can_match();
    test_degenerate_shapes();
    test_repeated_runs_are_identical();
    return TEST_REPORT();
}
