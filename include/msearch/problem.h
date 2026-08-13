#ifndef MSEARCH_PROBLEM_H
#define MSEARCH_PROBLEM_H

#include <stddef.h>

#include "msearch/types.h"

#ifdef __cplusplus
extern "C" {
#endif

void msearch_problem_init(Problem *problem);
void msearch_problem_free(Problem *problem);

/* Allocate the picture/object arrays (elements zeroed). */
Status msearch_problem_alloc(Problem *problem, int num_pictures, int num_objects, char *err,
                             size_t err_len);
Status msearch_picture_alloc(Picture *picture, int id, int n, char *err, size_t err_len);
Status msearch_object_alloc(Object *object, int id, int m, char *err, size_t err_len);

/* Check semantic invariants and establish the ordering the determinism
 * contract depends on (objects sorted by ascending id, stable). Call once
 * after loading; the backends assume it has run. */
Status msearch_problem_validate(Problem *problem, char *err, size_t err_len);

/* Number of top-left placements of an m x m object inside an n x n picture,
 * or 0 when the object does not fit. Returned as long long because n = 50000
 * already overflows int. */
long long msearch_placement_count(int n, int m);

/* Largest n*n over all pictures -- used to size reusable buffers once. */
size_t msearch_max_picture_elems(const Problem *problem);

#ifdef __cplusplus
}
#endif
#endif /* MSEARCH_PROBLEM_H */
