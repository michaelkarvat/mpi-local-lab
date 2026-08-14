/* Helpers for building Problem instances in memory, so backend tests do not
 * have to go through the file system. */
#ifndef MSEARCH_TEST_SUPPORT_H
#define MSEARCH_TEST_SUPPORT_H

#include <stdlib.h>

#include "msearch/problem.h"
#include "test_util.h"

static inline void support_fill_random(int *data, size_t count, TestRng *rng, int lo, int hi)
{
    for (size_t i = 0; i < count; ++i) {
        data[i] = rng_range(rng, lo, hi);
    }
}

/* Copy `object` into `picture` at (row, col), creating an exact match there
 * (score 0, below any positive threshold). */
static inline void support_plant(Picture *picture, const Object *object, int row, int col)
{
    for (int y = 0; y < object->m; ++y) {
        for (int x = 0; x < object->m; ++x) {
            picture->data[(size_t)(row + y) * picture->n + (col + x)] =
                object->data[(size_t)y * object->m + x];
        }
    }
}

/* Allocate a problem with the given shapes; matrices are left uninitialised. */
static inline int support_problem_create(Problem *problem, double threshold, const int *picture_ns,
                                         int num_pictures, const int *object_ms, int num_objects)
{
    char err[256];
    msearch_problem_init(problem);
    problem->threshold = threshold;
    if (msearch_problem_alloc(problem, num_pictures, num_objects, err, sizeof(err)) != MSEARCH_OK) {
        return 0;
    }
    for (int i = 0; i < num_pictures; ++i) {
        if (msearch_picture_alloc(&problem->pictures[i], 100 + i, picture_ns[i], err,
                                  sizeof(err)) != MSEARCH_OK) {
            return 0;
        }
    }
    for (int k = 0; k < num_objects; ++k) {
        if (msearch_object_alloc(&problem->objects[k], 200 + k, object_ms[k], err, sizeof(err)) !=
            MSEARCH_OK) {
            return 0;
        }
    }
    return 1;
}

#endif /* MSEARCH_TEST_SUPPORT_H */
